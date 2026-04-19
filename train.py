"""
NestVision v2 — Main Training Pipeline
=========================================
Dataset : Roboflow  aulias-workspace/indoor-obstacle
          https://universe.roboflow.com/aulias-workspace/indoor-obstacle

Distillation method:
  CrossKD  (Wang et al., CVPR 2024)
  + PKD Pearson neck features  (Cao et al., NeurIPS 2022)
  + Localization Distillation  (Zheng et al., CVPR 2022)

Usage:
    # Full pipeline (teacher → student → benchmark)
    python train.py --phase all

    # Individual phases
    python train.py --phase teacher   --epochs_teacher 50
    python train.py --phase student   --epochs_student 100
    python train.py --phase benchmark

    # Custom dataset key / version
    python train.py --phase all --rf_api_key YOUR_KEY --rf_version 1

    # Use a pre-downloaded local yaml
    python train.py --phase all --data indoor-obstacle/data.yaml
"""

import os
import sys
import argparse
import logging
import shutil
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("NestVision")


# ──────────────────────────────────────────────────────────────────────────────
#  Callbacks — per-epoch metrics print + checkpoint every N epochs
# ──────────────────────────────────────────────────────────────────────────────

def make_callbacks(role: str, weights_dir: str, save_every: int = 10):
    """
    Returns a dict of Ultralytics trainer callbacks that:
      1. Print loss + lr after every train epoch.
      2. Print val metrics after every val step.
      3. Every `save_every` epochs: overwrite ONE rolling checkpoint file
         AND mirror it to /kaggle/working/ so Kaggle can commit it even
         if the session is interrupted mid-training.
    """
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Single rolling checkpoint — always the same filename, overwritten in-place
    ckpt_path = weights_dir / f"{role}_checkpoint.pt"

    # Detect Kaggle environment
    on_kaggle = Path("/kaggle/working").exists()
    kaggle_mirror = Path("/kaggle/working") / f"{role}_checkpoint.pt" if on_kaggle else None

    _epoch_start_time = {}

    # ── Callback 1: record epoch start time ───────────────────────────────────
    def on_train_epoch_start(trainer):
        _epoch_start_time["t"] = time.perf_counter()

    # ── Callback 2: print train loss every epoch ──────────────────────────────
    def on_train_epoch_end(trainer):
        epoch    = trainer.epoch + 1
        total_ep = trainer.epochs
        elapsed  = time.perf_counter() - _epoch_start_time.get("t", time.perf_counter())

        loss_names = getattr(trainer, "loss_names", ["loss"])
        loss_items = trainer.tloss
        if hasattr(loss_items, "tolist"):
            loss_vals = loss_items.tolist()
        elif hasattr(loss_items, "__iter__"):
            loss_vals = list(loss_items)
        else:
            loss_vals = [float(loss_items)]
        while len(loss_vals) < len(loss_names):
            loss_vals.append(float("nan"))

        lr = 0.0
        if trainer.optimizer and trainer.optimizer.param_groups:
            lr = trainer.optimizer.param_groups[0]["lr"]

        SEP = "─" * 72
        print(f"\n{SEP}")
        print(f"  [{role.upper()}]  Epoch {epoch:>4d}/{total_ep}   elapsed {elapsed:5.1f}s   lr={lr:.2e}")
        print(SEP)
        for name, val in zip(loss_names, loss_vals):
            print(f"  {'loss/' + name:<22} {val:.6f}")
        print(SEP)

    # ── Callback 3: print val metrics + save checkpoint ───────────────────────
    def on_fit_epoch_end(trainer):
        epoch    = trainer.epoch + 1
        total_ep = trainer.epochs
        metrics  = trainer.metrics

        # Print validation metrics
        if metrics:
            SEP = "─" * 72
            print(f"\n  [{role.upper()}]  Epoch {epoch}/{total_ep}  — Validation Metrics")
            print(SEP)
            for k, v in metrics.items():
                short_k = k.replace("metrics/", "").replace("(B)", "")
                try:
                    print(f"  {short_k:<22} {float(v):.4f}")
                except (TypeError, ValueError):
                    print(f"  {short_k:<22} {v}")
            print(SEP + "\n")

        # Rolling checkpoint every N epochs
        if epoch % save_every == 0:
            src = Path(trainer.last)   # last.pt always written by Ultralytics
            if not src.exists():
                logger.warning(f"[{role}] last.pt not found at {src}, skipping checkpoint")
                return

            # ── Step 1: delete old checkpoint BEFORE copying new one ──────────
            #    (free disk first — important on Kaggle's limited storage)
            if ckpt_path.exists():
                ckpt_path.unlink()
            shutil.copy2(src, ckpt_path)
            logger.info(f"[{role}] ✅ Checkpoint saved → {ckpt_path}  (epoch {epoch}/{total_ep})")

            # ── Step 2: mirror to /kaggle/working/ so it survives a crash ─────
            #    On Kaggle you can click "Save & Run All" or the session auto-
            #    commits /kaggle/working output even on timeout.
            if kaggle_mirror is not None:
                if kaggle_mirror.exists():
                    kaggle_mirror.unlink()
                shutil.copy2(ckpt_path, kaggle_mirror)
                logger.info(f"[{role}] 📦 Mirrored to Kaggle output → {kaggle_mirror}")

    return {
        "on_train_epoch_start": on_train_epoch_start,
        "on_train_epoch_end":   on_train_epoch_end,
        "on_fit_epoch_end":     on_fit_epoch_end,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Resume helper — find latest checkpoint to continue from
# ──────────────────────────────────────────────────────────────────────────────

def find_resume_checkpoint(role: str, weights_dir: str = "weights") -> str | None:
    """
    Look for a rolling checkpoint to resume from, in priority order:
      1. weights/<role>_checkpoint.pt   (local)
      2. /kaggle/working/<role>_checkpoint.pt  (Kaggle mirror from previous session)
    Returns the path string if found, else None.
    """
    candidates = [
        Path(weights_dir) / f"{role}_checkpoint.pt",
        Path("/kaggle/working") / f"{role}_checkpoint.pt",
    ]
    for p in candidates:
        if p.exists():
            logger.info(f"[{role}] 🔄 Resume checkpoint found → {p}")
            return str(p)
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset setup — Roboflow  aulias-workspace/indoor-obstacle
# ──────────────────────────────────────────────────────────────────────────────

def setup_dataset(args) -> str:
    """
    Download the Roboflow indoor-obstacle dataset and return path to data.yaml.
    If --data points to an existing yaml, skip download and use it directly.
    """
    if args.data and Path(args.data).exists():
        logger.info(f"[Dataset] Using existing dataset yaml: {args.data}")
        return args.data

    if not args.rf_api_key:
        raise ValueError(
            "Roboflow API key required to download indoor-obstacle dataset.\n"
            "Pass --rf_api_key YOUR_KEY  or  --data path/to/data.yaml\n"
            "Get your key at: https://app.roboflow.com/settings/api"
        )

    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError(
            "roboflow package not installed.\n"
            "Run: pip install roboflow"
        )

    logger.info("[Dataset] Downloading indoor-obstacle from Roboflow …")
    rf = Roboflow(api_key=args.rf_api_key)

    project = rf.workspace("aulias-workspace").project("indoor-obstacle")
    version  = project.version(args.rf_version)

    dataset = version.download(
        model_format="yolov8",
        location="datasets/indoor-obstacle",
        overwrite=False,
    )

    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        # Some Roboflow exports put it one level up
        data_yaml = Path("datasets/indoor-obstacle/data.yaml")

    logger.info(f"[Dataset] data.yaml → {data_yaml}")
    return str(data_yaml)


# ──────────────────────────────────────────────────────────────────────────────
#  Phase 1 — Teacher Training  (YOLOv8L)
# ──────────────────────────────────────────────────────────────────────────────

def train_teacher(args, data_yaml: str) -> str:
    from ultralytics import YOLO

    logger.info("=" * 60)
    logger.info("PHASE 1  ·  Teacher Training  (YOLOv8L)  —  indoor-obstacle")
    logger.info("=" * 60)

    # ── Resume from checkpoint if available ───────────────────────────────────
    resume_ckpt = find_resume_checkpoint("teacher", weights_dir="weights")
    if resume_ckpt:
        logger.info(f"[Teacher] 🔄 Resuming from: {resume_ckpt}")
        model = YOLO(resume_ckpt)
    else:
        logger.info("[Teacher] Starting fresh from yolov8l.pt")
        model = YOLO("yolov8l.pt")

    # ── Register per-epoch callbacks ──────────────────────────────────────────
    cbs = make_callbacks("teacher", weights_dir="weights", save_every=args.save_every)
    for event, fn in cbs.items():
        model.add_callback(event, fn)

    model.train(
        data=data_yaml,
        epochs=args.epochs_teacher,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/NestVision",
        name="Teacher_L",
        patience=20,
        save=True,
        save_period=-1,       # -1 = disable per-epoch saving (we handle it via callbacks)
        exist_ok=True,
        verbose=True,
        resume=resume_ckpt is not None,   # continue epoch count from checkpoint
        # Augmentation — aggressive for teacher
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    # teacher_checkpoint.pt được callback lưu sau mỗi save_every epochs
    ckpt = Path("weights/teacher_checkpoint.pt")
    if not ckpt.exists():
        # fallback: copy best.pt hoặc last.pt nếu callback chưa kịp chạy
        weights_base = Path("runs/NestVision/Teacher_L/weights")
        for candidate in ("best.pt", "last.pt"):
            src = weights_base / candidate
            if src.exists():
                ckpt.parent.mkdir(exist_ok=True)
                shutil.copy2(src, ckpt)
                logger.warning(f"[Phase 1] checkpoint not found, copied {candidate} → {ckpt}")
                break
        else:
            raise FileNotFoundError(
                f"teacher_checkpoint.pt not found at {ckpt} and no fallback "
                f"weights exist under runs/NestVision/Teacher_L/weights/"
            )

    logger.info(f"[Phase 1] Teacher weights ready → {ckpt}")
    return str(ckpt)


# ──────────────────────────────────────────────────────────────────────────────
#  Phase 2 — Student Training with CrossKD
# ──────────────────────────────────────────────────────────────────────────────

def train_student(args, data_yaml: str) -> str:
    from core.trainer import NestVisionTrainer

    logger.info("=" * 60)
    logger.info("PHASE 2  ·  Student Training  (YOLOv8N + CrossKD)")
    logger.info("         Based on: Wang et al., CVPR 2024")
    logger.info("=" * 60)

    # ── Resume from checkpoint if available ───────────────────────────────────
    resume_ckpt = find_resume_checkpoint("student", weights_dir="weights")
    if resume_ckpt:
        logger.info(f"[Student] 🔄 Resuming from: {resume_ckpt}")
        base_model = resume_ckpt
    else:
        logger.info("[Student] Starting fresh from yolov8n.pt")
        base_model = "yolov8n.pt"

    trainer_args = dict(
        model=base_model,
        data=data_yaml,
        epochs=args.epochs_student,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/NestVision",
        name="Student_N_CrossKD",
        patience=25,
        save=True,
        save_period=-1,       # -1 = disable per-epoch saving (we handle it via callbacks)
        exist_ok=True,
        verbose=True,
        # resume không dùng ở đây — Ultralytics sẽ set model=None khi resume=True
        # thay vào đó truyền checkpoint path trực tiếp vào model= ở trên
        # Lighter augmentation for student
        mosaic=0.8,
        mixup=0.05,
        degrees=5.0,
        translate=0.1,
        scale=0.4,
        fliplr=0.5,
        # CrossKD-specific
        teacher_weights=args.teacher_weights,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        cross_split=args.cross_split,
    )

    trainer = NestVisionTrainer(overrides=trainer_args)

    # ── Register per-epoch callbacks ──────────────────────────────────────────
    cbs = make_callbacks("student", weights_dir="weights", save_every=args.save_every)
    for event, fn in cbs.items():
        trainer.add_callback(event, fn)

    trainer.train()

    # Prefer best.pt; fall back to last.pt if best.pt was not written
    weights_base = Path("runs/NestVision/Student_N_CrossKD/weights")
    src = weights_base / "best.pt"
    if not src.exists():
        fallback = weights_base / "last.pt"
        if fallback.exists():
            logger.warning(
                f"[Phase 2] best.pt not found at {src}; "
                f"falling back to last.pt → {fallback}"
            )
            src = fallback
        else:
            raise FileNotFoundError(
                f"Neither best.pt nor last.pt found under {weights_base}. "
                "Check that 'save=True' is set and training completed at least one epoch."
            )

    dst = Path("weights/student_best.pt")
    dst.parent.mkdir(exist_ok=True)
    shutil.copy2(src, dst)
    logger.info(f"[Phase 2] Student weights saved → {dst}  (source: {src.name})")

    return str(dst)


# ──────────────────────────────────────────────────────────────────────────────
#  Phase 3 — Benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark_phase(args, data_yaml: str):
    from core.benchmark import run_benchmark, print_benchmark_table

    logger.info("=" * 60)
    logger.info("PHASE 3  ·  Benchmark  (Teacher vs Student — CrossKD)")
    logger.info("=" * 60)

    results = run_benchmark(
        teacher_path=args.teacher_weights,
        student_path=args.student_weights,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        save_json="results/benchmark.json",
    )
    print_benchmark_table(results)
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "NestVision v2 — CrossKD Knowledge Distillation Pipeline\n"
            "Dataset: Roboflow aulias-workspace/indoor-obstacle"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument(
        "--phase",
        choices=["all", "teacher", "student", "benchmark"],
        default="all",
        help="Which phase to run  [default: all]",
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--data",
        default=None,
        help="Path to data.yaml. If omitted, dataset is downloaded via Roboflow API.",
    )
    p.add_argument(
        "--rf_api_key",
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key  (or set ROBOFLOW_API_KEY env var)",
    )
    p.add_argument(
        "--rf_version",
        type=int,
        default=1,
        help="Roboflow dataset version  [default: 1]",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--imgsz",          type=int,   default=640)
    p.add_argument("--batch",          type=int,   default=16)
    p.add_argument(
        "--device",
        default="0" if __import__("torch").cuda.is_available() else "cpu",
    )
    p.add_argument("--epochs_teacher", type=int,   default=50)
    p.add_argument("--epochs_student", type=int,   default=100)
    p.add_argument(
        "--save_every", type=int, default=10,
        help="Save a checkpoint every N epochs  [default: 10]",
    )

    # ── CrossKD hyperparams ───────────────────────────────────────────────────
    p.add_argument(
        "--temperature", type=float, default=3.0,
        help="Softening temperature for CrossKD & LD  [default: 3.0]",
    )
    p.add_argument(
        "--alpha", type=float, default=0.4,
        help="Task (hard-label) loss weight  [default: 0.4]",
    )
    p.add_argument(
        "--beta", type=float, default=0.3,
        help="CrossKD cross-head KD loss weight  [default: 0.3]",
    )
    p.add_argument(
        "--gamma", type=float, default=0.2,
        help="PKD neck feature loss weight  [default: 0.2]",
    )
    p.add_argument(
        "--delta", type=float, default=0.1,
        help="Localization distillation loss weight  [default: 0.1]",
    )
    p.add_argument(
        "--cross_split", type=int, default=1,
        help="Head layer index where student hands off to teacher head  [default: 1]",
    )

    # ── Weight paths ──────────────────────────────────────────────────────────
    p.add_argument("--teacher_weights", default="weights/teacher_checkpoint.pt")
    p.add_argument("--student_weights", default="weights/student_best.pt")

    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("runs",    exist_ok=True)

    # Resolve dataset
    data_yaml = setup_dataset(args)

    phase = args.phase

    if phase in ("all", "teacher"):
        teacher_path = train_teacher(args, data_yaml)
        # Cập nhật args để student phase dùng đúng checkpoint vừa train
        args.teacher_weights = teacher_path

    if phase in ("all", "student"):
        train_student(args, data_yaml)

    if phase in ("all", "benchmark"):
        run_benchmark_phase(args, data_yaml)

    logger.info("NestVision v2 pipeline complete ✓")


if __name__ == "__main__":
    main()
