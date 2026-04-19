"""
NestVision v2 — Knowledge Distillation Core
=============================================
Implements CrossKD (CVPR 2024) for YOLOv8:

    Wang et al. "CrossKD: Cross-Head Knowledge Distillation for Object Detection"
    IEEE/CVF CVPR 2024, pp. 16520-16530.  arXiv: 2306.11369

Key idea (vs original NestVision):
    Original: student features -> MSE to teacher features  (feature imitation)
                + student predictions -> KL to teacher predictions  (prediction mimicking)

    CrossKD:  student intermediate head features -> fed into TEACHER's detection head
              -> cross-head predictions -> forced to mimic teacher's own predictions
              This removes the "target conflict" where GT labels and soft teacher
              targets push the student head in opposite directions simultaneously.

Additionally retains:
    - Multi-scale FPN feature alignment via Pearson-correlation loss (PKD-style,
      NeurIPS 2022) rather than plain MSE -- normalises across channels to be
      scale-invariant.
    - Localization distillation (LD) on box regression distributions (Zheng et al.
      CVPR 2022) -- transfers bounding-box coordinate knowledge not captured by
      classification-only KL.

Total loss:
    L = alpha * L_task
      + beta  * L_cross_head   (CrossKD -- cls & reg branches)
      + gamma * L_feat_pkd     (PKD-style Pearson neck distillation)
      + delta * L_loc           (Localization distillation)

────────────────────────────────────────────────────────────────────────────────
ARCHITECTURE NOTES: YOLOv8 Detect head (per FPN level i)
────────────────────────────────────────────────────────────────────────────────
cv2[i] : Sequential(Conv(x,c2,3), Conv(c2,c2,3), Conv2d(c2, 4*reg_max, 1))
         regression branch — output [B, 4*reg_max, H, W]  (raw DFL logits)

cv3[i] : Sequential(Conv(x,c3,3), Conv(c3,c3,3), Conv2d(c3, nc, 1))
         classification branch — output [B, nc, H, W]  (raw logits, sigmoid in loss)

Training mode: Detect.forward returns list x where
    x[i] = cat(cv2[i](feat), cv3[i](feat))  shape [B, 4*reg_max+nc, H_i, W_i]
v8DetectionLoss splits x[i] back into box/cls internally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss
import logging
import copy

logger = logging.getLogger("NestVision")


# ──────────────────────────────────────────────────────────────────────────────
#  Channel Alignment: 1x1 conv  teacher-channels -> student-channels
# ──────────────────────────────────────────────────────────────────────────────
class ChannelProjection(nn.Module):
    """Learnable 1x1 convolution to align teacher->student channel dims."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ──────────────────────────────────────────────────────────────────────────────
#  PKD-style Pearson Correlation Feature Loss  (NeurIPS 2022)
# ──────────────────────────────────────────────────────────────────────────────
class PKDFeatureLoss(nn.Module):
    """
    Pearson Correlation Coefficient loss on neck feature maps.

    Ref: Cao et al. "PKD: General Distillation Framework for Object Detectors
    via Pearson Correlation Coefficient." NeurIPS 2022.

    Per-sample normalisation: flatten [B, C, H, W] -> [B, C*H*W], then
    zero-mean unit-variance across the full C*H*W dimension (dim=1).
    This measures cosine similarity across the entire feature map vector,
    which is the Pearson correlation formulation from the paper.
    """

    def forward(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor
    ) -> torch.Tensor:
        # [B, C, H, W] -> [B, C*H*W]: flatten entire feature map per sample
        s = student_feat.flatten(1)
        t = teacher_feat.detach().flatten(1)

        # Pearson normalisation: per-sample zero-mean unit-variance over C*H*W
        s = (s - s.mean(dim=1, keepdim=True)) / (s.std(dim=1, keepdim=True) + 1e-6)
        t = (t - t.mean(dim=1, keepdim=True)) / (t.std(dim=1, keepdim=True) + 1e-6)

        return F.mse_loss(s, t)


# ──────────────────────────────────────────────────────────────────────────────
#  CrossKD Head Wrapper
# ──────────────────────────────────────────────────────────────────────────────
class CrossKDHeadWrapper(nn.Module):
    """
    Wraps teacher head cv2/cv3 layers so student intermediate features can be
    forwarded through them to produce cross-head predictions.

    CrossKD (paper Fig.1c, Section 3.2):
      Student feat after its own head layers 0..split-1  ->  teacher head layers split..n
      -> cross-head predictions  ->  KL-div vs teacher's own predictions

    Key correctness constraint:
      cls branch (cv3) and reg branch (cv2) MUST use their OWN separate intermediate
      features from the student. They are NOT interchangeable — their channels and
      semantics differ entirely.
    """

    def __init__(self, teacher_head, split_layer: int = 1):
        super().__init__()
        self.teacher_cv2 = copy.deepcopy(teacher_head.cv2)   # reg branches
        self.teacher_cv3 = copy.deepcopy(teacher_head.cv3)   # cls branches
        self.split = split_layer

        # Freeze: distillation loss must not update teacher weights
        for p in self.parameters():
            p.requires_grad = False

    def forward_cross(
        self,
        student_cls_feat: torch.Tensor,   # output of student cv3[level][0..split-1]
        student_reg_feat: torch.Tensor,   # output of student cv2[level][0..split-1]
        level: int,
    ):
        """
        Returns:
            cross_reg: [B, 4*reg_max, H, W]  raw DFL logits
            cross_cls: [B, nc, H, W]          raw cls logits
        """
        # Regression branch: student reg intermediate -> teacher cv2[level][split:]
        x_reg = student_reg_feat
        for i, layer in enumerate(self.teacher_cv2[level]):
            if i >= self.split:
                x_reg = layer(x_reg)

        # Classification branch: student cls intermediate -> teacher cv3[level][split:]
        x_cls = student_cls_feat
        for i, layer in enumerate(self.teacher_cv3[level]):
            if i >= self.split:
                x_cls = layer(x_cls)

        return x_reg, x_cls


# ──────────────────────────────────────────────────────────────────────────────
#  Localization Distillation Loss  (Zheng et al., CVPR 2022)
# ──────────────────────────────────────────────────────────────────────────────
class LocalizationDistillationLoss(nn.Module):
    """
    Transfer box regression knowledge via KL divergence on DFL distributions.

    Input: raw cv2[i] output [B, 4*reg_max, H, W] -- NOT the concatenated preds.

    YOLOv8 DFL: each anchor has 4 coordinate predictions, each represented as
    a probability distribution over reg_max bins.

    Correct reshape for per-coordinate KL-div:
        [B, 4*reg_max, H, W]
        -> reshape(B, 4, reg_max, H*W)
        -> permute(0,3,1,2) -> [B, H*W, 4, reg_max]
        -> reshape(-1, reg_max)  -> [B*H*W*4, reg_max]
        -> softmax on dim=-1     -> each row = prob dist over reg_max bins
        -> KL-div(student || teacher) with temperature scaling
    """

    def forward(
        self,
        student_reg: torch.Tensor,   # [B, 4*reg_max, H, W]
        teacher_reg: torch.Tensor,   # [B, 4*reg_max, H, W]
        T: float = 2.0,
    ) -> torch.Tensor:
        B, C, H, W = student_reg.shape
        reg_max = C // 4

        # Reshape to isolate each coordinate's distribution: [B*H*W*4, reg_max]
        s = (student_reg
             .reshape(B, 4, reg_max, H * W)
             .permute(0, 3, 1, 2)          # [B, H*W, 4, reg_max]
             .reshape(-1, reg_max))

        t = (teacher_reg.detach()
             .reshape(B, 4, reg_max, H * W)
             .permute(0, 3, 1, 2)
             .reshape(-1, reg_max))

        s_log  = F.log_softmax(s / T, dim=-1)
        t_soft = F.softmax(t / T, dim=-1)
        return F.kl_div(s_log, t_soft, reduction="batchmean") * (T ** 2)


# ──────────────────────────────────────────────────────────────────────────────
#  NestVision CrossKD Distillation Loss  (main class)
# ──────────────────────────────────────────────────────────────────────────────
class NestVisionDistillationLoss(v8DetectionLoss):
    """
    Combined CrossKD loss for NestVision:

        L = alpha * L_task
          + beta  * L_cross_head   (CrossKD  -- CVPR 2024)
          + gamma * L_feat_pkd     (PKD Pearson neck -- NeurIPS 2022)
          + delta * L_loc          (Localization distillation -- CVPR 2022)
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        temperature: float = 3.0,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.2,
        delta: float = 0.1,
        cross_split: int = 1,
        device: str = "cpu",
    ):
        super().__init__(student_model)
        self.teacher = teacher_model
        self.T = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.device = device

        self._feat_projections: nn.ModuleList | None = None
        self._pkd_loss = PKDFeatureLoss()
        self._ld_loss = LocalizationDistillationLoss()

        teacher_detect = self._get_detect_head(teacher_model)
        if teacher_detect is not None:
            self._crossKD = CrossKDHeadWrapper(teacher_detect, split_layer=cross_split).to(device)
            logger.info(f"[NestVision] CrossKD head wrapper built (split_layer={cross_split})")
        else:
            self._crossKD = None
            logger.warning("[NestVision] Could not extract teacher detection head -- CrossKD disabled")

        logger.info(
            f"[NestVision] CrossKD loss init | T={temperature} "
            f"a={alpha} b={beta} g={gamma} d={delta}"
        )

    @staticmethod
    def _get_detect_head(model):
        from ultralytics.nn.modules.head import Detect
        for m in model.modules():
            if isinstance(m, Detect):
                return m
        return None

    def _build_projections(self, student_feats, teacher_feats):
        projs = []
        for s_f, t_f in zip(student_feats, teacher_feats):
            t_ch, s_ch = t_f.shape[1], s_f.shape[1]
            projs.append(
                ChannelProjection(t_ch, s_ch).to(self.device) if t_ch != s_ch
                else nn.Identity().to(self.device)
            )
        self._feat_projections = nn.ModuleList(projs)
        logger.info(
            f"[NestVision] Neck projections: "
            f"teacher {[t.shape[1] for t in teacher_feats]} -> "
            f"student {[s.shape[1] for s in student_feats]}"
        )

    # ── PKD Neck Feature Loss ─────────────────────────────────────────────────
    def _compute_feat_loss(self, student_feats, teacher_feats) -> torch.Tensor:
        if self._feat_projections is None:
            self._build_projections(student_feats, teacher_feats)

        loss = torch.tensor(0.0, device=self.device)
        for proj, s_f, t_f in zip(self._feat_projections, student_feats, teacher_feats):
            if t_f.shape[2:] != s_f.shape[2:]:
                t_f = F.interpolate(t_f, size=s_f.shape[2:], mode="bilinear", align_corners=False)
            t_f_proj = proj(t_f.detach())
            loss = loss + self._pkd_loss(s_f, t_f_proj)
        return loss / max(len(student_feats), 1)

    # ── CrossKD Cross-Head Loss ───────────────────────────────────────────────
    def _compute_crossKD_loss(
        self,
        student_cls_feats,   # list[Tensor [B,c3,H,W]]: student cv3[i][0] output
        student_reg_feats,   # list[Tensor [B,c2,H,W]]: student cv2[i][0] output
        teacher_cls_preds,   # list[Tensor [B,nc,H,W]]: teacher cv3[i](t_feat)
        teacher_reg_preds,   # list[Tensor [B,4*rm,H,W]]: teacher cv2[i](t_feat)
    ) -> torch.Tensor:
        """
        CrossKD loss per FPN level.

        Classification loss (L_CrossKD_cls):
          YOLOv8 cv3 uses independent sigmoids per class (multi-label), NOT softmax.
          Loss = BCE(cross_cls_logits, sigmoid(teacher_cls_logits))
          Using softmax+KL here would be WRONG for YOLOv8's cls head.

        Regression loss (L_CrossKD_reg):
          YOLOv8 cv2 outputs DFL logits [B, 4*reg_max, H, W].
          Per-coordinate KL-div with temperature scaling, matching LD paper formulation.
          Loss = KL(softmax(cross_reg/T) || softmax(teacher_reg/T)) * T^2
        """
        if self._crossKD is None or not student_cls_feats:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        n_levels = min(len(student_cls_feats), len(teacher_cls_preds))
        T = self.T

        for i in range(n_levels):
            try:
                cross_reg, cross_cls = self._crossKD.forward_cross(
                    student_cls_feats[i], student_reg_feats[i], level=i
                )

                # Classification: sigmoid BCE (YOLOv8 multi-label cls head)
                t_cls_soft = torch.sigmoid(teacher_cls_preds[i].detach())
                loss_cls = F.binary_cross_entropy_with_logits(
                    cross_cls, t_cls_soft, reduction="mean"
                )
                loss = loss + loss_cls

                # Regression: per-coordinate KL-div on DFL distributions
                t_reg = teacher_reg_preds[i].detach()
                B, C, H, W = cross_reg.shape
                reg_max = C // 4

                cr = (cross_reg.reshape(B, 4, reg_max, H * W)
                               .permute(0, 3, 1, 2).reshape(-1, reg_max))
                tr = (t_reg.reshape(B, 4, reg_max, H * W)
                           .permute(0, 3, 1, 2).reshape(-1, reg_max))

                loss_reg = F.kl_div(
                    F.log_softmax(cr / T, dim=-1),
                    F.softmax(tr / T, dim=-1),
                    reduction="batchmean"
                ) * (T ** 2)
                loss = loss + loss_reg

            except Exception as e:
                logger.debug(f"[CrossKD] Level {i} skipped: {e}")

        return loss / max(n_levels, 1)

    # ── Main __call__ ─────────────────────────────────────────────────────────
    def __call__(self, preds, batch):
        """
        preds[0]: list of per-level [B, 4*reg_max+nc, H_i, W_i] tensors
        preds[1]: list of neck feature maps [P3, P4, P5]
        """
        # ── 1. Task loss (GT hard labels) ─────────────────────────────────────
        task_loss, loss_items = super().__call__(preds, batch)

        # ── 2. Teacher forward pass (no grad) ─────────────────────────────────
        with torch.no_grad():
            teacher_out = self.teacher(batch["img"])

        # ── 3. Extract neck feature maps ──────────────────────────────────────
        student_feats = preds[1] if isinstance(preds, (list, tuple)) and len(preds) > 1 else []
        teacher_feats = (
            teacher_out[1]
            if isinstance(teacher_out, (list, tuple)) and len(teacher_out) > 1
            else []
        )

        # ── 4. PKD Neck Feature Loss ──────────────────────────────────────────
        feat_loss = torch.tensor(0.0, device=self.device)
        if student_feats and teacher_feats and len(student_feats) == len(teacher_feats):
            feat_loss = self._compute_feat_loss(student_feats, teacher_feats)

        # ── 5. CrossKD Cross-Head Loss ─────────────────────────────────────────
        crossKD_loss = torch.tensor(0.0, device=self.device)
        try:
            student_detect = self._get_detect_head(self.model)
            teacher_detect = self._get_detect_head(self.teacher)

            if (student_detect is not None and teacher_detect is not None
                    and student_feats and teacher_feats
                    and len(student_feats) == len(teacher_feats)):

                student_cls_feats = []
                student_reg_feats = []
                teacher_cls_preds = []
                teacher_reg_preds = []

                for i, (s_feat, t_feat) in enumerate(zip(student_feats, teacher_feats)):
                    # Student: extract intermediate feat after layer 0 of each branch
                    if i < len(student_detect.cv3) and i < len(student_detect.cv2):
                        student_cls_feats.append(student_detect.cv3[i][0](s_feat))
                        student_reg_feats.append(student_detect.cv2[i][0](s_feat))

                    # Teacher: full head pass on teacher's OWN neck features (t_feat).
                    # These are the distillation targets (right side of KL-div).
                    # MUST use t_feat not s_feat.
                    if i < len(teacher_detect.cv3) and i < len(teacher_detect.cv2):
                        with torch.no_grad():
                            teacher_cls_preds.append(teacher_detect.cv3[i](t_feat))
                            teacher_reg_preds.append(teacher_detect.cv2[i](t_feat))

                crossKD_loss = self._compute_crossKD_loss(
                    student_cls_feats, student_reg_feats,
                    teacher_cls_preds, teacher_reg_preds,
                )
        except Exception as e:
            logger.debug(f"[CrossKD] Cross-head loss skipped: {e}")

        # ── 6. Localization Distillation Loss ─────────────────────────────────
        # Run cv2[i] explicitly per level on both student and teacher neck feats.
        # Do NOT use preds[0] (concatenated multi-level tensor) -- it mixes cls+reg
        # and cannot be correctly reshaped for per-coordinate DFL KL-div.
        ld_loss = torch.tensor(0.0, device=self.device)
        try:
            student_detect = self._get_detect_head(self.model)
            teacher_detect = self._get_detect_head(self.teacher)

            if (student_detect is not None and teacher_detect is not None
                    and student_feats and teacher_feats
                    and len(student_feats) == len(teacher_feats)):

                ld_total = torch.tensor(0.0, device=self.device)
                n_valid = 0

                for i, (s_feat, t_feat) in enumerate(zip(student_feats, teacher_feats)):
                    if i >= len(student_detect.cv2) or i >= len(teacher_detect.cv2):
                        continue
                    s_reg = student_detect.cv2[i](s_feat)          # [B, 4*reg_max, H, W]
                    with torch.no_grad():
                        t_reg = teacher_detect.cv2[i](t_feat)       # [B, 4*reg_max, H, W]
                    ld_total = ld_total + self._ld_loss(s_reg, t_reg, T=self.T)
                    n_valid += 1

                if n_valid > 0:
                    ld_loss = ld_total / n_valid

        except Exception as e:
            logger.debug(f"[LD] Localization distillation skipped: {e}")

        # ── 7. Aggregate ───────────────────────────────────────────────────────
        total_loss = (
            self.alpha * task_loss
            + self.beta  * crossKD_loss
            + self.gamma * feat_loss
            + self.delta * ld_loss
        )

        return total_loss, loss_items
