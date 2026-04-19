# 🏠 NestVision v2 — Knowledge Distillation for Indoor Obstacle Detection

> YOLOv8L (Teacher) → YOLOv8N (Student) using **CrossKD** (CVPR 2024)  
> Dataset: [`aulias-workspace/indoor-obstacle`](https://universe.roboflow.com/aulias-workspace/indoor-obstacle) on Roboflow

---

## Distillation Method — CrossKD (CVPR 2024)

> Wang et al. *"CrossKD: Cross-Head Knowledge Distillation for Object Detection."*  
> IEEE/CVF CVPR 2024, pp. 16520–16530.  arXiv: [2306.11369](https://arxiv.org/abs/2306.11369)

### The problem with vanilla prediction mimicking

Standard KD on detection heads creates a **target conflict**: the student head receives two contradictory supervision signals at the same time:
- GT annotations pushing it toward ground-truth boxes/labels
- Teacher soft targets pushing it toward the teacher's (often different) predictions

This conflict limits performance gains compared to feature imitation methods.

### CrossKD solution

```
Student backbone → Student neck → Student feat (P3/P4/P5)
                                       │
                    Student head       │   CrossKD path
                    Layer 0            │
                         ↓             │
                    Student head       └──► Teacher head layers 1-end
                    Layer 1-end              ↓
                         ↓             Cross-head predictions
                    Student preds       ↓
                    (trained with   KL-div vs Teacher preds
                     GT labels)    (no GT conflict here!)
```

The student's head layer-0 features are **re-forwarded through the teacher's remaining head layers**, producing *cross-head predictions*. These are aligned to the teacher via KL-divergence. The student's own head path is only supervised by GT — no conflicting teacher signal.

### Complete loss function

```
L_total = α · L_task  +  β · L_CrossKD  +  γ · L_PKD_feat  +  δ · L_loc

  L_task     : standard YOLOv8 detection loss  (hard GT labels)
  L_CrossKD  : KL-div between cross-head preds and teacher preds
  L_PKD_feat : Pearson correlation loss on neck feature maps  (NeurIPS 2022)
  L_loc      : KL-div on DFL box regression distributions     (CVPR 2022)
```

Default weights: **α=0.4  β=0.3  γ=0.2  δ=0.1**

---

## Project Structure

```
NestVision_v2/
├── train.py                 # Main CLI entry point
├── README.md
│
├── core/
│   ├── distillation.py      # CrossKD + PKD + LD loss implementation
│   ├── trainer.py           # NestVisionTrainer (injects KD loss)
│   └── benchmark.py         # Teacher vs Student benchmark
│
├── ui/
│   └── app.py               # Gradio inference UI
│
├── weights/                 # Saved checkpoints (auto-created)
│   ├── teacher_best.pt
│   └── student_best.pt

---

## Quick Start

### 1. Install dependencies

```bash
pip install ultralytics roboflow gradio
```

### 2. Get your Roboflow API key

Sign up at [app.roboflow.com](https://app.roboflow.com) and copy your API key from **Settings → API**.

### 3. Run full pipeline

```bash
# Downloads dataset automatically, trains teacher → student → benchmark
python train.py --phase all --rf_api_key YOUR_KEY

# Or export key as env var
export ROBOFLOW_API_KEY=YOUR_KEY
python train.py --phase all
```

### 4. Step-by-step

```bash
# Phase 1: Train teacher (YOLOv8L)
python train.py --phase teacher --rf_api_key YOUR_KEY --epochs_teacher 50

# Phase 2: Train student with CrossKD
python train.py --phase student --epochs_student 100

# Phase 3: Benchmark
python train.py --phase benchmark
```

### 5. Use a pre-downloaded dataset

```bash
# If you already have the data.yaml
python train.py --phase all --data datasets/indoor-obstacle/data.yaml
```

### 6. Launch inference UI 

```bash
python ui/app.py --teacher weights/teacher_checkpoint.pt --student weights/student_checkpoint.pt
# Open http://localhost:7860
python ui/app.py --share    # public Gradio URL
```

---

## CLI Reference

```
python train.py [OPTIONS]

Dataset:
  --data            Path to data.yaml (skips Roboflow download if provided)
  --rf_api_key      Roboflow API key  (or set ROBOFLOW_API_KEY env var)
  --rf_version      Roboflow dataset version  [default: 1]

Training:
  --phase           all | teacher | student | benchmark  [default: all]
  --imgsz           Image size                           [default: 640]
  --batch           Batch size                           [default: 16]
  --device          CUDA device or 'cpu'                 [default: auto]
  --epochs_teacher  Teacher training epochs              [default: 50]
  --epochs_student  Student training epochs              [default: 100]

CrossKD Hyperparameters:
  --temperature     Softening temperature (CrossKD & LD) [default: 3.0]
  --alpha           Task loss weight                     [default: 0.4]
  --beta            CrossKD cross-head loss weight       [default: 0.3]
  --gamma           PKD neck feature loss weight         [default: 0.2]
  --delta           Localization distillation weight     [default: 0.1]
  --cross_split     Head layer split for CrossKD         [default: 1]

Weights:
  --teacher_weights                                      [default: weights/teacher_best.pt]
  --student_weights                                      [default: weights/student_best.pt]
```

---

## Distillation Architecture (full)

```
Image ──┬──► Teacher (YOLOv8L)
        │         │
        │     ┌───┴──────────────────┐
        │     │ Neck feats (P3/P4/P5)│──► PKD Pearson Loss (γ)
        │     │ Head cv2/cv3 layers  │──► Teacher cls/reg preds
        │     └──────────────────────┘             │
        │                                          │ KL-div
        │                                          ▼
        └──► Student (YOLOv8N)           CrossKD Loss (β)
                  │                               ▲
              ┌───┴──────────────────┐            │
              │ Neck feats (P3/P4/P5)│──► PKD ───┘
              │ Head layer-0 feats   │──────────────► Teacher head layers 1-end
              │ Head predictions     │                      (cross-head path)
              │ DFL box distributions│──► Localization KD (δ)
              └──────────────────────┘
              │ GT detection labels  │──► Task loss (α)
              └──────────────────────┘
```

---

## Dataset — Roboflow indoor-obstacle

| | |
|---|---|
| Workspace | `aulias-workspace` |
| Project | `indoor-obstacle` |
| URL | https://universe.roboflow.com/aulias-workspace/indoor-obstacle |
| Format | YOLOv8 |

---

## References

| Paper | Venue | Role in NestVision v2 |
|---|---|---|
| Wang et al., *CrossKD: Cross-Head Knowledge Distillation for Object Detection* | CVPR 2024 | Main distillation method |

## Teacher Model link: https://drive.google.com/drive/folders/1xkxtZWwoLNFKjCGqFJmmQxtMSX-5FTj-?usp=sharing
Teacher model was trained in 50 epoch while student was trained in 40 epochs. Both were recorded in file student-train-stats and teacher-trained-stats
