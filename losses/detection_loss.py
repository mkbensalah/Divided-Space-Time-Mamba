"""Combined detection loss (Eq. 10 of the paper).

    L_total = L_cls + L_angle + alpha * L_bbox + beta * L_IoU

where:
    L_cls    : Binary Cross-Entropy on per-class objectness.
    L_angle  : Cross-Entropy with Circular Spatial Layout (CSL) over 180 bins.
    L_bbox   : L1 on (xc, yc, w, h) in normalized [0, 1] coords.
    L_IoU    : Differentiable rotated-IoU loss (Gaussian-Wasserstein surrogate).

Targets format (per clip, padded to num_classes):
    targets["cls"]   : (B, num_classes) float, 1 if present in this clip else 0.
    targets["bbox"]  : (B, num_classes, 4) normalized (xc, yc, w, h).
    targets["angle"] : (B, num_classes) float in [0, pi) (180° periodicity).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.csl import csl_encode
from .rotated_iou import rotated_iou_loss


class DetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        num_angle_bins: int = 180,
        alpha: float = 1.0,
        beta: float = 2.0,
        csl_radius: int = 6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_angle_bins = num_angle_bins
        self.alpha = alpha
        self.beta = beta
        self.csl_radius = csl_radius

    def forward(self, preds: dict, targets: dict) -> dict:
        """
        Args:
            preds: dict with 'cls' (B, C) logits, 'bbox' (B, C, 4) sigmoid out,
                   'angle' (B, C, 180) logits.
            targets: dict with 'cls' (B, C), 'bbox' (B, C, 4), 'angle' (B, C).
        Returns:
            dict with per-component scalars and a 'total' key.
        """
        B, C = preds["cls"].shape
        device = preds["cls"].device

        # 1) Classification (BCE on objectness per class)
        cls_target = targets["cls"].float()
        loss_cls = F.binary_cross_entropy_with_logits(preds["cls"], cls_target)

        # Mask: only supervise localization where the class is present.
        pos = cls_target > 0.5                                  # (B, C)
        num_pos = pos.sum().clamp(min=1.0)

        # 2) BBox L1
        bbox_diff = (preds["bbox"] - targets["bbox"]).abs().sum(dim=-1)   # (B, C)
        loss_bbox = (bbox_diff * pos.float()).sum() / num_pos

        # 3) Angle CSL (Cross-Entropy with circular gaussian smoothing)
        angle_logits = preds["angle"]                                     # (B, C, 180)
        angle_target = targets["angle"]                                   # (B, C) in [0, pi)
        csl_target = csl_encode(angle_target, num_bins=self.num_angle_bins,
                                radius=self.csl_radius)                   # (B, C, 180)
        log_p = F.log_softmax(angle_logits, dim=-1)
        loss_angle_per = -(csl_target * log_p).sum(dim=-1)                # (B, C)
        loss_angle = (loss_angle_per * pos.float()).sum() / num_pos

        # 4) Rotated IoU loss (only on positive class slots)
        if pos.any():
            pred_obb = self._compose_obb(preds["bbox"], preds["angle"])   # (B, C, 5)
            tgt_obb = torch.cat([targets["bbox"], targets["angle"].unsqueeze(-1)], dim=-1)
            loss_iou = rotated_iou_loss(pred_obb[pos], tgt_obb[pos])
        else:
            loss_iou = preds["bbox"].new_zeros(())

        total = loss_cls + loss_angle + self.alpha * loss_bbox + self.beta * loss_iou
        return {
            "total": total,
            "cls": loss_cls.detach(),
            "bbox": loss_bbox.detach(),
            "angle": loss_angle.detach(),
            "iou": loss_iou.detach(),
        }

    def _compose_obb(self, bbox: torch.Tensor, angle_logits: torch.Tensor) -> torch.Tensor:
        """Soft-argmax the angle and concatenate to bbox to form (B, C, 5)."""
        bins = angle_logits.shape[-1]
        bin_centers = (torch.arange(bins, device=angle_logits.device, dtype=angle_logits.dtype)
                       + 0.5) * (3.141592653589793 / bins)
        p = F.softmax(angle_logits, dim=-1)
        theta = (p * bin_centers).sum(dim=-1)                              # (B, C)
        return torch.cat([bbox, theta.unsqueeze(-1)], dim=-1)
