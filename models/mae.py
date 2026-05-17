"""VideoMAE-style self-supervised pretraining for DST-Mamba.

Implements:
    - Tube masking (Sec. 2.5): the same spatial mask is applied across all
      temporal positions in a clip, yielding consistent spatial regions masked
      over time. Default ratio: 80%.
    - Asymmetric encoder/decoder: encoder sees only visible tokens; lightweight
      decoder reconstructs the masked patches.
    - Pixel-MSE loss on normalized targets.

The encoder reuses the DST-Mamba backbone. The decoder is a smaller stack of
DST-Mamba blocks plus a linear pixel-projection head.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .dst_mamba import DSTMamba, DSTMambaBlock


def tube_mask(
    B: int, T: int, N: int, mask_ratio: float, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a tube mask: same spatial positions masked across all T.

    Returns:
        mask:     (B, T, N) bool — True = visible, False = masked.
        keep_idx: (B, num_keep) int64 — sorted spatial indices of visible patches.
    """
    num_keep = int(N * (1 - mask_ratio))
    rand = torch.rand(B, N, device=device)
    keep_idx = rand.topk(num_keep, dim=1, largest=True).indices   # (B, num_keep)
    keep_idx, _ = keep_idx.sort(dim=1)                            # deterministic order
    spatial_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    spatial_mask.scatter_(1, keep_idx, True)
    full_mask = spatial_mask.unsqueeze(1).expand(B, T, N).contiguous()
    return full_mask, keep_idx


def normalize_patches(target: torch.Tensor) -> torch.Tensor:
    """Per-patch normalize (mean/std), matching the VideoMAE reconstruction target."""
    mean = target.mean(dim=-1, keepdim=True)
    std = target.var(dim=-1, unbiased=False, keepdim=True).add(1e-6).sqrt()
    return (target - mean) / std


class DSTMambaMAE(nn.Module):
    """MAE wrapper around the DST-Mamba encoder.

    Args:
        encoder: a DSTMamba instance configured WITHOUT a CLS token.
        decoder_dim: decoder hidden dim (paper: 384).
        decoder_depth: number of decoder Mamba blocks (paper: 4).
        in_chans: 3 for RGB, 4 for RGB-D.
        norm_target: per-patch normalize reconstruction target.
    """

    def __init__(
        self,
        encoder: DSTMamba,
        decoder_dim: int = 384,
        decoder_depth: int = 4,
        in_chans: int = 3,
        norm_target: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.in_chans = in_chans
        self.norm_target = norm_target

        ed = encoder.embed_dim
        self.encoder_to_decoder = nn.Linear(ed, decoder_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.dec_pos_spatial = nn.Parameter(torch.zeros(1, 1, encoder.num_patches, decoder_dim))
        self.dec_pos_temporal = nn.Parameter(torch.zeros(1, encoder.T_eff, 1, decoder_dim))
        nn.init.trunc_normal_(self.dec_pos_spatial, std=0.02)
        nn.init.trunc_normal_(self.dec_pos_temporal, std=0.02)

        self.decoder_blocks = nn.ModuleList([
            DSTMambaBlock(dim=decoder_dim) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # Pixel reconstruction: predict tubelet_size * patch^2 * C values per token.
        out_dim = encoder.tubelet_size * (encoder.patch_size ** 2) * in_chans
        self.decoder_pred = nn.Linear(decoder_dim, out_dim)

    # ------------------------------------------------------------------ #

    def _patchify_target(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (B, C, T, H, W) → (B, T', N, tubelet*P*P*C) patch tokens.

        Patches must be aligned with the encoder's patch embedding.
        """
        B, C, T, H, W = x.shape
        P = self.encoder.patch_size
        ts = self.encoder.tubelet_size
        x = rearrange(
            x,
            "b c (t ts) (h p1) (w p2) -> b t (h w) (ts p1 p2 c)",
            ts=ts, p1=P, p2=P,
        )
        return x

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.8) -> dict:
        """
        Args:
            x: (B, C, T, H, W) input clip (ImageNet-normalized).
            mask_ratio: fraction of tokens to mask (paper default 0.8).
        Returns:
            dict with 'loss', 'pred', 'mask'.
        """
        B, C, T_in, H, W = x.shape
        T = self.encoder.T_eff
        N = self.encoder.num_patches

        # 1) Tube mask: same spatial positions dropped across all T frames.
        mask, keep_idx = tube_mask(B, T, N, mask_ratio, x.device)   # (B,T,N), (B, num_keep)
        num_keep = keep_idx.shape[1]
        dec_dim = self.mask_token.shape[-1]

        # 2) Asymmetric encoder: only the ~20% visible tokens are processed.
        #    Compute cost is O(num_keep * T) instead of O(N * T).
        enc_vis = self.encoder.forward_features(x, keep_idx=keep_idx)  # (B, T, num_keep, D)
        vis_dec = self.encoder_to_decoder(enc_vis)                      # (B, T, num_keep, dec_dim)

        # 3) Build full decoder input: start with mask tokens at all N positions,
        #    then scatter the projected visible tokens back to their original locations.
        dec = self.mask_token.expand(B, T, N, dec_dim).clone()
        idx = keep_idx.unsqueeze(1).unsqueeze(-1).expand(B, T, num_keep, dec_dim)
        dec.scatter_(2, idx, vis_dec)
        dec = dec + self.dec_pos_spatial + self.dec_pos_temporal

        # 4) Decode full sequence (all N positions).
        for blk in self.decoder_blocks:
            dec = blk(dec)
        dec = self.decoder_norm(dec)

        # 5) Pixel projection and MSE loss on masked positions only.
        pred = self.decoder_pred(dec)        # (B, T, N, ts*P*P*C)
        target = self._patchify_target(x)    # (B, T, N, ts*P*P*C)
        if self.norm_target:
            target = normalize_patches(target)

        masked = ~mask                                                   # True = needs reconstruction
        loss_per_token = ((pred - target) ** 2).mean(dim=-1)            # (B, T, N)
        loss = (loss_per_token * masked.float()).sum() / (masked.float().sum() + 1e-6)

        return {"loss": loss, "pred": pred, "mask": mask}
