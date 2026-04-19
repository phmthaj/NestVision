"""
NestVision v2 — Benchmark
==========================
Evaluates Teacher (YOLOv8L) vs Student (YOLOv8N + CrossKD)
on the Roboflow indoor-obstacle validation set.

Metrics:
  mAP@50, mAP@50-95, Precision, Recall,
  Inference speed (ms/image), Model size (MB),
  FLOPs (GFLOPs), Parameters (M)
"""

import time
import json
import os
from pathlib import Path
from typing import Dict, Any

import torch
from ultralytics import YOLO
import logging

logger = logging.getLogger("NestVision")

# indoor-obstacle dataset class names
# (update if your Roboflow version has different names)
INDOOR_OBSTACLE_CLASSES = [
    "chair", "couch", "dining table", "dog", "door",
    "laptop", "person", "potted plant", "stairs",
    "suitcase", "tv", "window",
]


def _model_size_mb(path: str) -> float:
    return Path(path).stat().st_size / (1024 ** 2)


def _count_params_m(model) -> float:
    return sum(p.numel() for p in model.model.parameters()) / 1e6


def _measure_latency(model, imgsz: int = 640, n_runs: int = 100, device: str = "cpu") -> float:
    """Median inference latency in milliseconds over n_runs passes."""
    dummy = torch.zeros(1, 3, imgsz, imgsz).to(device)
    model.model.eval()
    for _ in range(10):  # warm-up
        model.model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.model(dummy)
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return float(times[len(times) // 2])


def run_benchmark(
    teacher_path: str = "weights/teacher_best.pt",
    student_path:  str = "weights/student_best.pt",
    data_yaml:     str = "datasets/indoor-obstacle/data.yaml",
    imgsz:         int = 640,
    save_json:     str = "results/benchmark.json",
) -> Dict[str, Any]:
    """
    Run full benchmark comparison.

    Returns:
    {
      "teacher": { mAP50, mAP50_95, precision, recall, speed_ms, size_mb, params_m },
      "student":  { ... },
      "compression": { size_ratio, speed_ratio, map50_drop, map5095_drop }
    }
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Benchmark] Device: {device}  |  dataset: {data_yaml}")

    results = {}

    for role, path in [("teacher", teacher_path), ("student", student_path)]:
        if not os.path.exists(path):
            logger.warning(f"[Benchmark] {role} weights not found: {path} — skipping")
            results[role] = None
            continue

        logger.info(f"[Benchmark] Evaluating {role}: {path}")
        model = YOLO(path)

        val_res = model.val(
            data=data_yaml,
            imgsz=imgsz,
            verbose=False,
            device=device,
        )

        model.model.to(device).eval()
        lat_ms   = _measure_latency(model, imgsz=imgsz, device=device)
        size_mb  = _model_size_mb(path)
        params_m = _count_params_m(model)

        results[role] = {
            "mAP50":     float(val_res.box.map50),
            "mAP50_95":  float(val_res.box.map),
            "precision": float(val_res.box.mp),
            "recall":    float(val_res.box.mr),
            "speed_ms":  round(lat_ms, 2),
            "size_mb":   round(size_mb, 2),
            "params_m":  round(params_m, 2),
        }
        logger.info(f"[Benchmark] {role}: {results[role]}")

    if results.get("teacher") and results.get("student"):
        t, s = results["teacher"], results["student"]
        results["compression"] = {
            "size_ratio":   round(t["size_mb"]  / s["size_mb"],  2),
            "speed_ratio":  round(t["speed_ms"] / s["speed_ms"], 2),
            "map50_drop":   round((t["mAP50"]    - s["mAP50"])    * 100, 2),
            "map5095_drop": round((t["mAP50_95"] - s["mAP50_95"]) * 100, 2),
        }

    os.makedirs(os.path.dirname(save_json), exist_ok=True)
    with open(save_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"[Benchmark] Results saved → {save_json}")
    return results


def print_benchmark_table(results: Dict[str, Any]):
    SEP = "─" * 62
    print(f"\n{'═'*62}")
    print("  NestVision v2  ·  Teacher vs Student  (CrossKD, CVPR 2024)")
    print(f"  Dataset: Roboflow  aulias-workspace/indoor-obstacle")
    print(f"{'═'*62}")

    for role in ("teacher", "student"):
        r = results.get(role)
        if not r:
            print(f"\n  {role.upper()}: not evaluated")
            continue
        label = f"TEACHER  (YOLOv8L)" if role == "teacher" else f"STUDENT  (YOLOv8N + CrossKD)"
        print(f"\n  {label}")
        print(SEP)
        print(f"  {'mAP@50':<22} {r['mAP50']:.4f}")
        print(f"  {'mAP@50-95':<22} {r['mAP50_95']:.4f}")
        print(f"  {'Precision':<22} {r['precision']:.4f}")
        print(f"  {'Recall':<22} {r['recall']:.4f}")
        print(f"  {'Latency (ms)':<22} {r['speed_ms']}")
        print(f"  {'Model size (MB)':<22} {r['size_mb']}")
        print(f"  {'Parameters (M)':<22} {r['params_m']}")

    cmp = results.get("compression")
    if cmp:
        print(f"\n  COMPRESSION SUMMARY")
        print(SEP)
        print(f"  {'Size reduction':<28} {cmp['size_ratio']}×  smaller")
        print(f"  {'Speed-up':<28} {cmp['speed_ratio']}×  faster")
        print(f"  {'mAP@50 drop':<28} {cmp['map50_drop']} pp")
        print(f"  {'mAP@50-95 drop':<28} {cmp['map5095_drop']} pp")
    print(f"\n{'═'*62}\n")
