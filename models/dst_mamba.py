"""Divided Space-Time Mamba (DST-Mamba) backbone.

Implements Fig. 1 of the paper. Each DST encoder block performs:

    1) Spatial Bi-Mamba scan within each frame:
           X^space ∈ R^{(B*T) x N x D}    →   y^space  (Eq. 9)
       Add & LayerNorm.

    2) Temporal Bi-Mamba scan across frames at each spatial position:
           X^time  ∈ R^{(B*N) x T x D}    →   y^time
       Add & LayerNorm.

    3) MLP, Add & LayerNorm.

The backbone stacks L such blocks (default L=12) and returns tokens for
downstream pretraining or detection heads.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from .mamba_block import BiMambaBlock
from .patch_embed import build_patch_embed


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        h = int(dim * hidden_ratio)
        self.fc1 = nn.Linear(dim, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DSTMambaBlock(nn.Module):
    """One Divided Space-Time encoder block (Fig. 1b)."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.norm_space = nn.LayerNorm(dim)
        self.spatial_mamba = BiMambaBlock(d_model=dim, d_state=d_state,
                                          d_conv=d_conv, expand=expand)

        self.norm_time = nn.LayerNorm(dim)
        self.temporal_mamba = BiMambaBlock(d_model=dim, d_state=d_state,
                                           d_conv=d_conv, expand=expand)

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, D) tokens.
        Returns:
            (B, T, N, D)
        """
        B, T, N, D = x.shape

        # ---- Spatial Bi-Mamba (within each frame) ----
        x_s = rearrange(x, "b t n d -> (b t) n d")
        x_s = x_s + self.spatial_mamba(self.norm_space(x_s))
        x = rearrange(x_s, "(b t) n d -> b t n d", b=B, t=T)

        # ---- Temporal Bi-Mamba (across frames, per spatial location) ----
        x_t = rearrange(x, "b t n d -> (b n) t d")
        x_t = x_t + self.temporal_mamba(self.norm_time(x_t))
        x = rearrange(x_t, "(b n) t d -> b t n d", b=B, n=N)

        # ---- MLP ----
        x = x + self.mlp(self.norm_mlp(x))
        return x


class DSTMamba(nn.Module):
    """Divided Space-Time Mamba backbone.

    Args:
        img_size:       Spatial side length (square).
        patch_size:     Spatial patch size.
        num_frames:     Number of input frames T.
        tubelet_size:   Temporal patch size (1 = per-frame 2D embed).
        in_chans:       3 for RGB, 4 for RGB-D.
        embed_dim:      Token dimension.
        depth:          Number of DST-Mamba blocks (L).
        d_state, d_conv, expand: SSM hyperparameters.
        mlp_ratio:      MLP hidden expansion.
        cls_token:      If True, prepends a learnable CLS token per frame.
                        Detection mode uses cls_token=False.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        cls_token: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_cls = cls_token

        self.patch_embed = build_patch_embed(
            img_size=img_size, patch_size=patch_size,
            tubelet_size=tubelet_size, in_chans=in_chans, embed_dim=embed_dim
        )
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.T_eff = num_frames // tubelet_size

        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        # Separate spatial and temporal positional embeddings (Fig. 1a).
        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, 1, self.num_patches, embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.T_eff, 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            DSTMambaBlock(dim=embed_dim, d_state=d_state, d_conv=d_conv,
                          expand=expand, mlp_ratio=mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        if self.use_cls:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                # Skip biases flagged by BiMambaBlock: dt_proj bias is initialized
                # to inv_softplus(dt) to keep SSM timescales in a useful range.
                if m.bias is not None and not getattr(m, "_no_reinit_bias", False):
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(
        self,
        x: torch.Tensor,
        keep_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W).
            keep_idx: optional (B, num_keep) int64 tensor of visible spatial indices
                      in [0, N). When provided, only those spatial positions enter
                      the encoder (asymmetric MAE). Requires cls_token=False.
        Returns:
            (B, T_eff, N_out, D) where N_out = num_keep if keep_idx given, else N.
        """
        tokens = self.patch_embed(x)                                  # (B, T', N, D)
        tokens = tokens + self.pos_embed_spatial + self.pos_embed_temporal

        if self.use_cls:
            assert keep_idx is None, "keep_idx (asymmetric MAE) requires cls_token=False"
            B, T, N, D = tokens.shape
            cls = self.cls_token.expand(B, T, 1, D)
            tokens = torch.cat([cls, tokens], dim=2)                  # (B, T, N+1, D)

        tokens = self.pos_drop(tokens)

        if keep_idx is not None:
            # Select only the visible spatial positions for the asymmetric encoder.
            # This is the true VideoMAE-style approach: masked patches never enter
            # the encoder, so compute cost scales with (1 - mask_ratio).
            B, T, _, D = tokens.shape
            K = keep_idx.shape[1]
            idx = keep_idx.unsqueeze(1).unsqueeze(-1).expand(B, T, K, D)
            tokens = torch.gather(tokens, 2, idx)                     # (B, T, K, D)

        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)

    def forward(self, x: torch.Tensor, keep_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward_features(x, keep_idx=keep_idx)


def dst_mamba_base(in_chans: int = 3, **kw) -> DSTMamba:
    """Base config from the paper: depth=12, dim=768."""
    defaults = dict(
        img_size=224, patch_size=16, num_frames=16, tubelet_size=1,
        in_chans=in_chans, embed_dim=768, depth=12,
        d_state=16, d_conv=4, expand=2, mlp_ratio=4.0,
    )
    defaults.update(kw)
    return DSTMamba(**defaults)


def dst_mamba_small(in_chans: int = 3, **kw) -> DSTMamba:
    """Smaller variant for ablation."""
    defaults = dict(
        img_size=224, patch_size=16, num_frames=16, tubelet_size=1,
        in_chans=in_chans, embed_dim=384, depth=8,
        d_state=16, d_conv=4, expand=2, mlp_ratio=4.0,
    )
    defaults.update(kw)
    return DSTMamba(**defaults)
