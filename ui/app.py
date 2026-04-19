"""
NestVision v2 — Inference UI
==============================
Gradio interface for indoor obstacle detection.
Dataset: Roboflow  aulias-workspace/indoor-obstacle

Run:
    python ui/app.py
    python ui/app.py --share
    python ui/app.py --port 7860
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import numpy as np
from PIL import Image
import torch

# ── Class names for indoor-obstacle dataset (aulias-workspace/indoor-obstacle) ─
# {0: 'bed', 1: 'chair', 2: 'couch', 3: 'door', 4: 'nightstand',
#  5: 'person', 6: 'stair', 7: 'table'}
INDOOR_CLASSES = [
    "bed", "chair", "couch", "door", "nightstand",
    "person", "stair", "table",
]

LABEL_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F",
]

_loaded_models = {}


def load_model(path: str):
    if path not in _loaded_models:
        if not os.path.exists(path):
            return None
        from ultralytics import YOLO
        _loaded_models[path] = YOLO(path)
    return _loaded_models[path]


def run_inference(model, frame_rgb: np.ndarray, conf: float = 0.25, iou: float = 0.45):
    """
    Args:
        frame_rgb: numpy array in RGB format (từ Gradio hoặc webcam)
    Returns:
        annotated_rgb (np.ndarray), elapsed_ms (float), detections (list)
    """
    # YOLO predict() nhận BGR và tự flip BGR→RGB bên trong (preprocess).
    # Gradio trả về RGB → đổi sang BGR trước khi pass vào YOLO.
    frame_bgr = frame_rgb[:, :, ::-1].copy()

    t0 = time.perf_counter()
    results = model.predict(
        source=frame_bgr,   # ✅ BGR — đúng format YOLO expect
        conf=conf,
        iou=iou,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    res = results[0]

    # res.plot(pil=True) trả về PIL Image RGB — không cần đảo channel thủ công
    annotated_pil = res.plot(pil=True)      # PIL RGB
    annotated_rgb = np.array(annotated_pil) # numpy RGB

    # Dùng model.names làm source of truth — luôn khớp với lúc train
    model_names = model.names  # {0: 'bed', 1: 'chair', ...}

    detections = []
    if res.boxes is not None:
        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf_v = float(box.conf[0])
            xyxy   = box.xyxy[0].tolist()
            label  = model_names.get(cls_id, f"cls{cls_id}")
            detections.append({
                "label":      label,
                "confidence": round(conf_v, 3),
                "bbox":       [round(x, 1) for x in xyxy],
            })

    return annotated_rgb, round(elapsed_ms, 1), detections


def format_detections_md(detections, model_name: str, elapsed_ms: float) -> str:
    lines = [f"### {model_name}  ·  {elapsed_ms:.1f} ms\n"]
    if not detections:
        lines.append("_No objects detected_")
    else:
        lines.append(f"**{len(detections)} object(s) found**\n")
        for i, d in enumerate(detections, 1):
            x1, y1, x2, y2 = d["bbox"]
            lines.append(
                f"{i}. **{d['label']}**  conf={d['confidence']:.2f}  "
                f"bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]"
            )
    return "\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui(
    teacher_path: str = "weights/teacher_checkpoint.pt",
    student_path:  str = "weights/student_checkpoint.pt",
    benchmark_json: str = "results/benchmark.json",
):
    # ── Tab 1: Compare Images ──────────────────────────────────────────────────
    def compare(image, conf, iou):
        if image is None:
            return None, None, "Upload an image first.", "Upload an image first."

        # image từ Gradio type="numpy" đã là RGB
        frame_rgb = image

        t_model = load_model(teacher_path)
        s_model = load_model(student_path)

        t_img = t_det = t_ms = None
        s_img = s_det = s_ms = None

        if t_model:
            t_arr, t_ms, t_det = run_inference(t_model, frame_rgb, conf, iou)
            t_img = Image.fromarray(t_arr)
        if s_model:
            s_arr, s_ms, s_det = run_inference(s_model, frame_rgb, conf, iou)
            s_img = Image.fromarray(s_arr)

        t_md = format_detections_md(t_det or [], "Teacher (YOLOv8L)", t_ms or 0)
        s_md = format_detections_md(s_det or [], "Student (YOLOv8N + CrossKD)", s_ms or 0)

        return t_img, s_img, t_md, s_md

    # ── Tab 2: Real-time Video ─────────────────────────────────────────────────
    def video_stream(frame, conf, iou, model_choice):
        """
        Callback cho gr.Image(streaming=True) — nhận từng frame từ webcam.
        Trả về frame đã annotate (numpy RGB).
        """
        if frame is None:
            return None

        model_path = teacher_path if model_choice == "Teacher (YOLOv8L)" else student_path
        model = load_model(model_path)
        if model is None:
            return frame  # model chưa có → trả lại frame gốc

        annotated_rgb, elapsed_ms, detections = run_inference(model, frame, conf, iou)

        # Vẽ thêm FPS / latency lên góc trái ảnh
        try:
            import cv2
            fps_text = f"{model_choice}  {elapsed_ms:.0f}ms  {len(detections)} obj"
            cv2.putText(
                annotated_rgb, fps_text,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA,
            )
        except ImportError:
            pass  # nếu không có opencv thì bỏ qua overlay text

        return annotated_rgb

    # ── Tab 3: Benchmark ──────────────────────────────────────────────────────
    def load_benchmark():
        if not os.path.exists(benchmark_json):
            return "Benchmark not run yet. Run `python train.py --phase benchmark`."
        with open(benchmark_json) as f:
            data = json.load(f)

        lines = ["## NestVision v2 — Benchmark Results\n"]
        lines.append("**Distillation method:** CrossKD (Wang et al., CVPR 2024)  \n")
        lines.append("**Dataset:** Roboflow `aulias-workspace/indoor-obstacle`\n\n")

        for role, label in [("teacher", "Teacher (YOLOv8L)"), ("student", "Student (YOLOv8N + CrossKD)")]:
            r = data.get(role)
            if not r:
                lines.append(f"### {label}\n_Not evaluated_\n")
                continue
            lines.append(f"### {label}\n")
            lines.append(f"| Metric | Value |\n|---|---|\n")
            lines.append(f"| mAP@50 | {r['mAP50']:.4f} |\n")
            lines.append(f"| mAP@50-95 | {r['mAP50_95']:.4f} |\n")
            lines.append(f"| Precision | {r['precision']:.4f} |\n")
            lines.append(f"| Recall | {r['recall']:.4f} |\n")
            lines.append(f"| Latency (ms) | {r['speed_ms']} |\n")
            lines.append(f"| Model size (MB) | {r['size_mb']} |\n")
            lines.append(f"| Parameters (M) | {r['params_m']} |\n\n")

        cmp = data.get("compression")
        if cmp:
            lines.append("### Compression Summary\n")
            lines.append(f"| | |\n|---|---|\n")
            lines.append(f"| Size reduction | **{cmp['size_ratio']}×** smaller |\n")
            lines.append(f"| Speed-up | **{cmp['speed_ratio']}×** faster |\n")
            lines.append(f"| mAP@50 drop | {cmp['map50_drop']} pp |\n")
            lines.append(f"| mAP@50-95 drop | {cmp['map5095_drop']} pp |\n")

        return "".join(lines)

    # ── Layout ────────────────────────────────────────────────────────────────
    with gr.Blocks(title="NestVision v2 — Indoor Obstacle Detection") as demo:
        gr.Markdown(
            "# 🏠 NestVision v2 — Indoor Obstacle Detection\n"
            "**CrossKD** (CVPR 2024) distillation · "
            "Dataset: [`aulias-workspace/indoor-obstacle`](https://universe.roboflow.com/aulias-workspace/indoor-obstacle)\n\n"
            f"**Classes:** {', '.join(INDOOR_CLASSES)}"
        )

        # ── Tab 1: So sánh ảnh ──────────────────────────────────────────────
        with gr.Tab("🔍 Compare Models"):
            with gr.Row():
                img_input = gr.Image(label="Input Image", type="numpy")
                with gr.Column():
                    conf_slider = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confidence threshold")
                    iou_slider  = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="IoU threshold")
                    run_btn     = gr.Button("▶  Run Comparison", variant="primary")

            with gr.Row():
                teacher_out = gr.Image(label="Teacher (YOLOv8L)", type="pil")
                student_out = gr.Image(label="Student (YOLOv8N + CrossKD)", type="pil")

            with gr.Row():
                teacher_md = gr.Markdown()
                student_md = gr.Markdown()

            run_btn.click(
                fn=compare,
                inputs=[img_input, conf_slider, iou_slider],
                outputs=[teacher_out, student_out, teacher_md, student_md],
            )

        # ── Tab 2: Real-time Webcam ──────────────────────────────────────────
        with gr.Tab("🎥 Real-time Video"):
            gr.Markdown(
                "### Real-time inference từ webcam\n"
                "Chọn model, bật webcam, inference chạy trên từng frame.\n"
                "> **Lưu ý:** Teacher (YOLOv8L) nặng hơn → FPS thấp hơn Student."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    rt_conf   = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confidence threshold")
                    rt_iou    = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="IoU threshold")
                    rt_model  = gr.Radio(
                        choices=["Teacher (YOLOv8L)", "Student (YOLOv8N + CrossKD)"],
                        value="Teacher (YOLOv8L)",
                        label="Model",
                    )

                with gr.Column(scale=3):
                    # streaming=True → Gradio gọi video_stream() cho mỗi frame webcam
                    webcam_in  = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="Webcam Input",
                        type="numpy",
                    )
                    webcam_out = gr.Image(
                        label="Annotated Output",
                        type="numpy",
                        streaming=True,
                    )

            # Mỗi frame mới từ webcam → gọi video_stream
            webcam_in.stream(
                fn=video_stream,
                inputs=[webcam_in, rt_conf, rt_iou, rt_model],
                outputs=webcam_out,
            )

        # ── Tab 3: Benchmark ────────────────────────────────────────────────
        with gr.Tab("📊 Benchmark"):
            bench_btn = gr.Button("Load Benchmark Results")
            bench_out = gr.Markdown()
            bench_btn.click(fn=load_benchmark, outputs=bench_out)

        # ── Tab 4: About ────────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                "## NestVision v2 — Distillation Method\n\n"
                "### CrossKD  (CVPR 2024)\n"
                "> Wang et al. *CrossKD: Cross-Head Knowledge Distillation for Object Detection.*  \n"
                "> IEEE/CVF CVPR 2024, pp. 16520–16530.\n\n"
                "**Key insight:** Conventional prediction-mimicking KD creates a *target conflict* "
                "where the student head receives contradictory signals from GT labels and soft teacher "
                "targets simultaneously.  \n\n"
                "CrossKD resolves this by feeding the student's *intermediate* detection-head features "
                "into the **teacher's** head layers, producing *cross-head predictions* that are then "
                "aligned to the teacher.  This isolates the GT supervision to the student's own head "
                "and the KD signal to the cross-head path — eliminating the conflict.\n\n"
                "### Additional losses\n"
                "| Loss | Reference |\n|---|---|\n"
                "| PKD Pearson neck feature loss | Cao et al., NeurIPS 2022 |\n"
                "| Localization distillation (box DFL) | Zheng et al., CVPR 2022 |\n\n"
                "### Total loss\n"
                "```\n"
                "L = α · L_task  +  β · L_CrossKD  +  γ · L_PKD_feat  +  δ · L_loc\n"
                "```\n"
                "Default: α=0.4  β=0.3  γ=0.2  δ=0.1\n"
            )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share",   action="store_true")
    parser.add_argument("--port",    type=int, default=7860)
    parser.add_argument("--teacher", default="weights/teacher_best.pt")
    parser.add_argument("--student", default="weights/student_best.pt")
    args = parser.parse_args()

    demo = build_ui(teacher_path=args.teacher, student_path=args.student)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()