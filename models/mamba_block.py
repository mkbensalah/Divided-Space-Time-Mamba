"""Bidirectional Mamba block (Fig. 2 of the paper).

The input sequence is processed by two parallel streams (forward and backward)
to capture dependencies in both temporal directions. Each stream consists of a
linear projection, a 1D causal convolution, and a selective state-space scan.
The two outputs are summed and projected back to the model dimension.

This module is direction-agnostic: the same block is reused for the spatial scan
(within a frame) and the temporal scan (across frames) in the DST-Mamba backbone.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from causal_conv1d import causal_conv1d_fn
    HAS_MAMBA_KERNELS = True
except ImportError:
    HAS_MAMBA_KERNELS = False


class BiMambaBlock(nn.Module):
    """Bidirectional selective state-space block.

    Args:
        d_model:    Hidden dimension of input/output tokens.
        d_state:    SSM state dimension N. Default 16 (Mamba paper default).
        d_conv:     Causal-conv kernel size. Default 4.
        expand:     Inner expansion factor E. Inner dim = expand * d_model.
        dt_rank:    Rank of the data-dependent timescale parameter. "auto" → ceil(d_model/16).
        bias:       Whether the input/output linears carry bias.
        conv_bias:  Whether the 1D conv carries bias.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str | int = "auto",
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        # Shared input projection: produces [x, z] for both directions.
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Forward stream
        self.conv1d_fwd = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        self.x_proj_fwd = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj_fwd = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log_fwd = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                                                 .repeat(self.d_inner, 1)))
        self.D_fwd = nn.Parameter(torch.ones(self.d_inner))

        # Backward stream
        self.conv1d_bwd = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        self.x_proj_bwd = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj_bwd = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log_bwd = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                                                 .repeat(self.d_inner, 1)))
        self.D_bwd = nn.Parameter(torch.ones(self.d_inner))

        # Output projection (post-merge)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        # Initialize dt projections so dt is bounded in a reasonable range.
        # Mark with _no_reinit_bias so DSTMamba._init_weights skips these biases;
        # zeroing them would destroy the calibrated inv_softplus(dt) initialization.
        for proj in (self.dt_proj_fwd, self.dt_proj_bwd):
            nn.init.uniform_(proj.weight, -0.001, 0.001)
            with torch.no_grad():
                dt = torch.exp(
                    torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
                ).clamp(min=1e-4)
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                proj.bias.copy_(inv_dt)
            proj._no_reinit_bias = True

    # ------------------------------------------------------------------ #

    def _ssm_stream(
        self,
        u: torch.Tensor,
        conv1d: nn.Conv1d,
        x_proj: nn.Linear,
        dt_proj: nn.Linear,
        A_log: torch.Tensor,
        D: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """One direction of the bidirectional scan.

        Args:
            u: (B, L, d_inner) input embedding for this direction.
            z: (B, L, d_inner) gating tensor (shared between directions).
        Returns:
            (B, L, d_inner)
        """
        B, L, _ = u.shape
        u_c = u.transpose(1, 2)  # (B, d_inner, L)

        # Causal 1D conv with kernel d_conv, then SiLU
        if HAS_MAMBA_KERNELS:
            u_c = causal_conv1d_fn(
                x=u_c,
                weight=conv1d.weight.squeeze(1),
                bias=conv1d.bias,
                activation="silu",
            )
        else:
            u_c = conv1d(u_c)[..., :L]
            u_c = F.silu(u_c)

        # Data-dependent dt, B, C
        x_dbl = x_proj(u_c.transpose(1, 2))  # (B, L, dt_rank + 2*d_state)
        dt, Bp, Cp = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt_proj(dt))  # (B, L, d_inner)

        A = -torch.exp(A_log)  # (d_inner, d_state)

        if HAS_MAMBA_KERNELS:
            y = selective_scan_fn(
                u=u_c,
                delta=dt.transpose(1, 2),
                A=A,
                B=Bp.transpose(1, 2),
                C=Cp.transpose(1, 2),
                D=D,
                z=z.transpose(1, 2),
                delta_bias=None,
                delta_softplus=False,
                return_last_state=False,
            )
            return y.transpose(1, 2)

        # Fallback: naive selective scan (slow; use only without Mamba kernels).
        return self._selective_scan_naive(u_c, dt, A, Bp, Cp, D, z)

    def _selective_scan_naive(self, u_c, dt, A, Bp, Cp, D, z):
        """Naive Python selective scan. u_c: (B, d_inner, L). z: (B, L, d_inner).

        Returns: (B, L, d_inner) to match the mamba_ssm kernel's output layout.
        """
        B, d_inner, L = u_c.shape
        N = A.shape[-1]
        h = u_c.new_zeros(B, d_inner, N)
        ys = []
        # Discretization: A_bar = exp(dt * A); B_bar = dt * B (zero-order hold approx.)
        for t in range(L):
            dt_t = dt[:, t]                               # (B, d_inner)
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A)     # (B, d_inner, N)
            B_bar = dt_t.unsqueeze(-1) * Bp[:, t].unsqueeze(1)  # (B, d_inner, N)
            h = A_bar * h + B_bar * u_c[:, :, t].unsqueeze(-1)
            y_t = torch.einsum("bdn,bn->bd", h, Cp[:, t])  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=2)                          # (B, d_inner, L)
        y = y + D.unsqueeze(0).unsqueeze(-1) * u_c
        y = y.transpose(1, 2)                               # (B, L, d_inner)
        y = y * F.silu(z)                                   # gate
        return y

    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) input tokens.
        Returns:
            (B, L, d_model)
        """
        B, L, _ = x.shape
        xz = self.in_proj(x)                          # (B, L, 2*d_inner)
        u, z = xz.chunk(2, dim=-1)                    # each (B, L, d_inner)

        y_fwd = self._ssm_stream(u, self.conv1d_fwd, self.x_proj_fwd, self.dt_proj_fwd,
                                 self.A_log_fwd, self.D_fwd, z)

        # Backward scan: flip along sequence, run, flip back
        u_b = torch.flip(u, dims=[1])
        z_b = torch.flip(z, dims=[1])
        y_bwd = self._ssm_stream(u_b, self.conv1d_bwd, self.x_proj_bwd, self.dt_proj_bwd,
                                 self.A_log_bwd, self.D_bwd, z_b)
        y_bwd = torch.flip(y_bwd, dims=[1])

        y = y_fwd + y_bwd
        return self.out_proj(y)
