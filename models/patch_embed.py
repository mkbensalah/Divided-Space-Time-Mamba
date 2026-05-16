"""Patch embedding for video clips.

Supports both RGB (3-channel) and RGB-D (4-channel) input. The depth map is
fused with RGB via early channel concatenation prior to tokenization (Sec. 3.5
of the paper), so the only difference is the conv input channel count.

Two flavors:
    - PatchEmbed2D:  per-frame 2D conv, used when tubelet_size = 1.
    - PatchEmbed3D:  3D conv with temporal stride = tubelet_size.

Output shape is (B, T', N, D) where:
    T' = T // tubelet_size
    N  = (H // P) * (W // P)
    D  = embed_dim
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbed2D(nn.Module):
    """Per-frame 2D patch embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) input clip (C=3 for RGB, 4 for RGB-D).
        Returns:
            (B, T, N, D)
        """
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.proj(x)                          # (B*T, D, H/P, W/P)
        x = rearrange(x, "(b t) d h w -> b t (h w) d", b=B, t=T)
        return x


class PatchEmbed3D(nn.Module):
    """3D conv patch embedding with tubelet temporal stride."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = img_size // patch_size
        self.num_spatial_patches = self.grid_size ** 2
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W).
        Returns:
            (B, T', N, D) with T' = T // tubelet_size, N = grid_size**2.
        """
        x = self.proj(x)                                # (B, D, T', H/P, W/P)
        x = rearrange(x, "b d t h w -> b t (h w) d")
        return x


def build_patch_embed(
    img_size: int = 224,
    patch_size: int = 16,
    tubelet_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 768,
):
    """Factory: returns 2D or 3D patch embed depending on tubelet_size."""
    if tubelet_size == 1:
        return PatchEmbed2D(img_size, patch_size, in_chans, embed_dim)
    return PatchEmbed3D(img_size, patch_size, tubelet_size, in_chans, embed_dim)
