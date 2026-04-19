"""
NestVision v2 - Trainer
========================
Subclasses DetectionTrainer to inject the CrossKD loss after
the standard training setup is complete.
"""

import os
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.cfg import get_cfg, DEFAULT_CFG

from core.distillation import NestVisionDistillationLoss
import logging

logger = logging.getLogger("NestVision")


class NestVisionTrainer(DetectionTrainer):
    """
    YOLOv8 DetectionTrainer extended with CrossKD (CVPR 2024) distillation.
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        overrides = overrides or {}

        # Pop NestVision-specific kwargs truoc khi truyen vao Ultralytics
        self.teacher_weights = overrides.pop("teacher_weights", "weights/teacher_checkpoint.pt")
        self.kd_temperature  = overrides.pop("temperature",  3.0)
        self.kd_alpha        = overrides.pop("alpha",        0.4)
        self.kd_beta         = overrides.pop("beta",         0.3)
        self.kd_gamma        = overrides.pop("gamma",        0.2)
        self.kd_delta        = overrides.pop("delta",        0.1)
        self.kd_cross_split  = overrides.pop("cross_split",  1)

        # FIX: Ultralytics moi yeu cau cfg phai la dict hop le, khong duoc None.
        # Truyen DEFAULT_CFG thay vi None de check_dict_alignment() khong bi loi
        # AttributeError: 'NoneType' object has no attribute 'keys'
        if cfg is None:
            cfg = DEFAULT_CFG

        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def _setup_train(self, *args, **kwargs):
        # Ultralytics cu: _setup_train(self, world_size)
        # Ultralytics moi: _setup_train(self)
        # Dung *args/**kwargs de tuong thich ca hai version
        super()._setup_train(*args, **kwargs)

        logger.info(f"[NestVision] Loading teacher -> {self.teacher_weights}")
        if not os.path.exists(self.teacher_weights):
            raise FileNotFoundError(
                f"Teacher weights not found: {self.teacher_weights}\n"
                "Run phase 1 first, or supply --teacher_weights."
            )

        teacher_model = YOLO(self.teacher_weights).model.to(self.device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

        logger.info(
            f"[NestVision] Teacher loaded & frozen | "
            f"CrossKD params: T={self.kd_temperature} "
            f"a={self.kd_alpha} b={self.kd_beta} "
            f"g={self.kd_gamma} d={self.kd_delta} "
            f"split={self.kd_cross_split}"
        )

        self.compute_loss = NestVisionDistillationLoss(
            student_model=self.model,
            teacher_model=teacher_model,
            temperature=self.kd_temperature,
            alpha=self.kd_alpha,
            beta=self.kd_beta,
            gamma=self.kd_gamma,
            delta=self.kd_delta,
            cross_split=self.kd_cross_split,
            device=str(self.device),
        )
