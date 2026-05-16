"""Circular Spatial Layout (CSL) encoding for orientation prediction.

Following Yang et al., CSL re-frames angle regression as a classification problem
over `num_bins` bins covering [0, 180°), with a circular Gaussian window around
the ground-truth bin. This naturally handles the 180° periodicity (1° and 179°
are close, not maximally different).
"""

from __future__ import annotations

import math
import torch


PI = math.pi


def angle_to_bin(angle_rad: torch.Tensor, num_bins: int = 180) -> torch.Tensor:
    """Map angle in [0, pi) (radians) to bin index [0, num_bins).

    Wraps any input into the [0, pi) range first.
    """
    angle = angle_rad % PI
    return (angle / PI * num_bins).long().clamp(max=num_bins - 1)


def bin_to_angle(bin_idx: torch.Tensor, num_bins: int = 180) -> torch.Tensor:
    """Inverse of angle_to_bin, returns bin centers in radians."""
    return (bin_idx.float() + 0.5) * (PI / num_bins)


def csl_encode(angle_rad: torch.Tensor, num_bins: int = 180, radius: int = 6) -> torch.Tensor:
    """Encode an angle into a circular Gaussian soft-label over `num_bins`.

    Args:
        angle_rad: (...,) angles in [0, pi). Wrapped if outside.
        num_bins:  number of CSL bins (paper: 180).
        radius:    Gaussian std (in bins). Paper uses ~6.
    Returns:
        (..., num_bins) soft labels summing to 1 along last dim.
    """
    shape = angle_rad.shape
    angle = (angle_rad % PI).unsqueeze(-1)                                # (..., 1)
    centers = bin_to_angle(torch.arange(num_bins, device=angle_rad.device),
                           num_bins=num_bins)                              # (num_bins,)
    centers = centers.view(*([1] * len(shape)), num_bins)                 # (..., num_bins)

    # Circular distance in bin units.
    diff = (angle - centers).abs()
    diff = torch.minimum(diff, PI - diff)
    diff_bins = diff / (PI / num_bins)

    sigma2 = float(radius ** 2)
    w = torch.exp(-(diff_bins ** 2) / (2 * sigma2))
    w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return w


def csl_decode(logits: torch.Tensor) -> torch.Tensor:
    """Decode CSL logits to an angle in radians via circular mean.

    Uses the unit-vector trick at 2*theta to respect the 180° periodicity
    (so bin 0 and bin 179 are correctly adjacent).

    Args:
        logits: (..., num_bins)
    Returns:
        (...,) angle in [0, pi).
    """
    num_bins = logits.shape[-1]
    p = torch.softmax(logits, dim=-1)
    centers = bin_to_angle(torch.arange(num_bins, device=logits.device), num_bins=num_bins)
    two_t = 2 * centers
    cos_sum = (p * torch.cos(two_t)).sum(dim=-1)
    sin_sum = (p * torch.sin(two_t)).sum(dim=-1)
    return 0.5 * torch.atan2(sin_sum, cos_sum) % PI
