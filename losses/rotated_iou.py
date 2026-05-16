"""Rotated IoU for oriented bounding boxes.

Two implementations:
    - rotated_iou_shapely:   exact convex-polygon intersection via Shapely.
                              Used for evaluation (non-differentiable, CPU).
    - rotated_iou_loss:       smooth differentiable surrogate for training.
                              Uses Gaussian-Wasserstein distance approximation
                              ("GWD-IoU") which is monotonic in true rIoU and
                              produces stable gradients for OBB regression.
"""

from __future__ import annotations

from typing import Tuple

import math
import torch
import numpy as np

try:
    from shapely.geometry import Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


def obb_to_corners(obb: torch.Tensor) -> torch.Tensor:
    """(xc, yc, w, h, theta) → (4, 2) corner coordinates.

    Args:
        obb: (..., 5) tensor. theta in radians.
    Returns:
        (..., 4, 2) corners in (x, y).
    """
    xc, yc, w, h, theta = obb.unbind(-1)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    # Half extents
    hw, hh = w / 2, h / 2
    # Local corners (clockwise from top-right)
    dx = torch.stack([hw, -hw, -hw, hw], dim=-1)
    dy = torch.stack([hh, hh, -hh, -hh], dim=-1)
    # Rotate
    cx = cos_t[..., None] * dx - sin_t[..., None] * dy
    cy = sin_t[..., None] * dx + cos_t[..., None] * dy
    corners_x = xc[..., None] + cx
    corners_y = yc[..., None] + cy
    return torch.stack([corners_x, corners_y], dim=-1)


def rotated_iou_shapely(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Exact rotated IoU for evaluation.

    Args:
        box1, box2: (N, 5) tensors of (xc, yc, w, h, theta_rad).
    Returns:
        (N,) IoU values in [0, 1].
    """
    if not HAS_SHAPELY:
        raise RuntimeError("Shapely is required for exact rotated IoU.")
    c1 = obb_to_corners(box1).detach().cpu().numpy()
    c2 = obb_to_corners(box2).detach().cpu().numpy()
    ious = np.zeros(c1.shape[0], dtype=np.float32)
    for i in range(c1.shape[0]):
        try:
            p1 = Polygon(c1[i]).buffer(0)
            p2 = Polygon(c2[i]).buffer(0)
            if not (p1.is_valid and p2.is_valid):
                continue
            inter = p1.intersection(p2).area
            union = p1.union(p2).area
            ious[i] = inter / union if union > 0 else 0.0
        except Exception:
            ious[i] = 0.0
    return torch.from_numpy(ious).to(box1.device)


# --------------------------------------------------------------------- #
# Differentiable rotated IoU loss (Gaussian-Wasserstein Distance, GWD)
# --------------------------------------------------------------------- #

def _obb_to_gaussian(obb: torch.Tensor):
    """Convert OBB to 2D Gaussian (mu, Sigma)."""
    xc, yc, w, h, theta = obb.unbind(-1)
    mu = torch.stack([xc, yc], dim=-1)                                 # (..., 2)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t,  cos_t], dim=-1),
    ], dim=-2)                                                         # (..., 2, 2)
    S = torch.zeros_like(R)
    S[..., 0, 0] = (w / 2) ** 2
    S[..., 1, 1] = (h / 2) ** 2
    Sigma = R @ S @ R.transpose(-1, -2)                                # (..., 2, 2)
    return mu, Sigma


def _trace(M: torch.Tensor) -> torch.Tensor:
    return M[..., 0, 0] + M[..., 1, 1]


def _sqrtm_2x2_psd(M: torch.Tensor) -> torch.Tensor:
    """Principal square root of a 2x2 PSD matrix in closed form."""
    a = M[..., 0, 0]
    b = M[..., 0, 1]
    d = M[..., 1, 1]
    s = (a * d - b * b).clamp(min=1e-8).sqrt()                         # det
    t = (a + d + 2 * s).clamp(min=1e-8).sqrt()                          # trace+2*det
    out = torch.zeros_like(M)
    out[..., 0, 0] = (a + s) / t
    out[..., 1, 1] = (d + s) / t
    out[..., 0, 1] = b / t
    out[..., 1, 0] = b / t
    return out


def rotated_iou_loss(pred: torch.Tensor, target: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Differentiable rotated-IoU loss via Gaussian-Wasserstein distance.

    L = 1 - 1 / (1 + log(1 + GWD/tau))   (monotonic surrogate, in [0, 1]).

    Args:
        pred, target: (N, 5) tensors of (xc, yc, w, h, theta_rad).
        tau: temperature.
    Returns:
        scalar loss (mean over N).
    """
    mu1, S1 = _obb_to_gaussian(pred)
    mu2, S2 = _obb_to_gaussian(target)

    d_mu = ((mu1 - mu2) ** 2).sum(dim=-1)
    sqrt_S2 = _sqrtm_2x2_psd(S2)
    cross = _sqrtm_2x2_psd(sqrt_S2 @ S1 @ sqrt_S2)
    d_cov = _trace(S1) + _trace(S2) - 2 * _trace(cross)
    gwd = (d_mu + d_cov).clamp(min=0.0)

    iou_surrogate = 1.0 / (1.0 + torch.log1p(gwd / tau))
    return (1.0 - iou_surrogate).mean()
