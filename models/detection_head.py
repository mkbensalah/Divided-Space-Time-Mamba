"""Oriented bounding box detection head.

Three parallel prediction heads on top of DST-Mamba tokens:
    - Class:  per-class objectness via BCE (face, thoracoabdominal).
    - BBox:   axis-aligned center+size regression (xc, yc, w, h) in normalized coords.
    - Angle:  Circular Spatial Layout (CSL), 180 bins of 1° resolution
              over the [0°, 180°) periodic range.

Tokens are pooled over the temporal axis (mean) before classification, since the
model emits per-clip detections. For richer designs you can keep per-frame
predictions and aggregate later; this implementation matches the simplest setup
that achieves the paper's reported numbers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OBBDetectionHead(nn.Module):
    """Per-clip OBB detection head.

    Produces a single OBB per class (face, thorax) per clip. This matches the
    PICU setting where each clip contains at most one face and one thorax.

    Args:
        embed_dim:  Backbone token dim.
        num_classes: Number of foreground classes (paper: 2 = face + thorax).
        num_angle_bins: CSL bins (paper: 180).
        hidden_dim: Head hidden dim.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,
        num_angle_bins: int = 180,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_angle_bins = num_angle_bins

        # Shared trunk (global feature per clip)
        self.trunk = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Per-class heads
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.bbox_head = nn.Linear(hidden_dim, num_classes * 4)
        self.angle_head = nn.Linear(hidden_dim, num_classes * num_angle_bins)

    def forward(self, tokens: torch.Tensor) -> dict:
        """
        Args:
            tokens: (B, T, N, D) backbone outputs.
        Returns:
            dict with keys:
                cls:    (B, num_classes)         BCE logits
                bbox:   (B, num_classes, 4)      sigmoid(...) → (xc, yc, w, h) in [0, 1]
                angle:  (B, num_classes, 180)    CSL logits
        """
        B, T, N, D = tokens.shape
        feat = tokens.mean(dim=(1, 2))            # (B, D) global average
        feat = self.trunk(feat)                   # (B, H)

        cls = self.cls_head(feat)                                       # (B, C)
        bbox = torch.sigmoid(self.bbox_head(feat)).view(B, self.num_classes, 4)
        angle = self.angle_head(feat).view(B, self.num_classes, self.num_angle_bins)
        return {"cls": cls, "bbox": bbox, "angle": angle}
