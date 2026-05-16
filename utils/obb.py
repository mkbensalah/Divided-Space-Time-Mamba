"""Oriented bounding box (OBB) utilities."""

from __future__ import annotations

import math
import numpy as np
import torch


def obb_to_corners_np(obb: np.ndarray) -> np.ndarray:
    """(N, 5) → (N, 4, 2) corner coordinates. theta in radians."""
    xc, yc, w, h, t = obb.T
    cos_t, sin_t = np.cos(t), np.sin(t)
    hw, hh = w / 2.0, h / 2.0
    dx = np.stack([hw, -hw, -hw, hw], axis=-1)
    dy = np.stack([hh, hh, -hh, -hh], axis=-1)
    cx = cos_t[:, None] * dx - sin_t[:, None] * dy
    cy = sin_t[:, None] * dx + cos_t[:, None] * dy
    return np.stack([xc[:, None] + cx, yc[:, None] + cy], axis=-1)


def denormalize_obb(obb_norm: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
    """Convert normalized OBB (xc, yc, w, h in [0, 1], theta in rad) to pixel coords."""
    out = obb_norm.clone()
    out[..., 0] *= img_w
    out[..., 1] *= img_h
    out[..., 2] *= img_w
    out[..., 3] *= img_h
    return out


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angle into [0, pi)."""
    return theta % math.pi
