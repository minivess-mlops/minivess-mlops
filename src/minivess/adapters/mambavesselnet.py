"""MambaVesselNet++ adapter — ModelAdapter ABC implementation.

Architecture: Chen et al. 2024 (ACM MM) / Xu & Chen et al. 2025 (ACM TOMM)
Reference:    https://github.com/CC0117/MambaVesselNet  (MIT license)
DOI MM2024:   https://doi.org/10.1145/3696409.3700231
DOI TOMM2025: https://doi.org/10.1145/3757324

Design decisions:
- D01: Inline backbone (MIT-licensed code adapted into this module)
- D02: BidirMamba = two standard Mamba modules (fwd + bwd.flip), no vendored fork
- D03: No permutation — native (B,C,H,W,D) MONAI convention throughout
- D04: AMP conservative — train ON, val OFF (mirrors SAM3, MONAI #4243)
- D05: eval_roi_size = (64,64,64) — matches training patch
- D06: All MONAI blocks require norm_name="instance"
- D07: LayerNorm INSIDE BidirectionalMambaLayer only (no double-LN)
- D08: _mamba_available() lives in model_builder, re-exported here for tests
- D09: GPU tests in tests/gpu_instance/ (not tests/v2/unit/)
- D10: ONNX export raises NotImplementedError (mamba selective scan not traceable)

PLATFORM PAPER — NOT SOTA RACE.
MambaVesselNet demonstrates the SSM family integrates via ModelAdapter ABC.
BANNED: "outperforms / surpasses / beats DynUNet"
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from monai.networks.blocks import (  # type: ignore[attr-defined]
    UnetOutBlock,
    UnetrBasicBlock,
    UnetrUpBlock,
)
from torch import Tensor

# Re-export availability check for test imports (D08)
from minivess.adapters.model_builder import _mamba_available as _mamba_available

__all__ = [
    "_mamba_available",
    "MlpChannel",
    "BidirectionalMambaLayer",
    "MambaBlock",
    "MambaVesselNetBackbone",
]

# Default feature dimensions for MambaVesselNet++
_FEATURE_DIMS = (48, 96, 192, 384, 768)


# ── T02: Channel MLP ──────────────────────────────────────────────────────────


class MlpChannel(nn.Module):
    """Channel MLP: pointwise Conv3d with 4x inner expansion and GELU activation.

    Architecture: Conv3d(dim, dim*4, 1) → GELU → Conv3d(dim*4, dim, 1)
    Pure PyTorch — no mamba-ssm dependency.
    Input / output shape: (B, C, H, W, D) — MONAI convention.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = dim * 4
        self.fc1 = nn.Conv3d(dim, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(hidden, dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return torch.as_tensor(self.fc2(self.act(self.fc1(x))))


# ── T03: Bidirectional Mamba Layer ────────────────────────────────────────────


class BidirectionalMambaLayer(nn.Module):
    """Bidirectional Mamba layer that processes 3D volumes as flattened sequences.

    Architecture (D07 — LN lives HERE, not in MambaBlock):
      1. Flatten (B,C,H,W,D) → (B,L,C) where L = H*W*D
      2. Apply LayerNorm on the sequence dimension
      3. Forward Mamba scan: fwd(x_norm)
      4. Backward Mamba scan: bwd(x_norm.flip(1)).flip(1)
      5. Sum fwd + bwd, unflatten back to (B,C,H,W,D)
      6. Add residual: return x + out

    D02: BidirMamba equivalent to bimamba_type="v2" — two standard Mamba modules,
    no vendored fork required.

    Parameters
    ----------
    dim:
        Channel dimension (C in the input tensor).
    d_state:
        Mamba state space dimension.
    d_conv:
        Mamba local convolution width.
    expand:
        Mamba inner-dimension expansion factor.
    mamba_cls:
        Optional Mamba class override. Pass a MockMamba for CPU unit tests.
        If None, imports mamba_ssm.Mamba at instantiation time (requires CUDA).
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mamba_cls: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if mamba_cls is None:
            from mamba_ssm import Mamba  # noqa: PLC0415

            mamba_cls = Mamba
        self.norm = nn.LayerNorm(dim)
        self.fwd: nn.Module = mamba_cls(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.bwd: nn.Module = mamba_cls(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, x: Tensor) -> Tensor:
        """Process (B,C,H,W,D) through bidirectional Mamba scan.

        Parameters
        ----------
        x:
            Input tensor (B, C, H, W, D) — MONAI convention, depth LAST.

        Returns
        -------
        Tensor of same shape as input, with residual applied.
        """
        B, C, *spatial = x.shape
        L = math.prod(spatial)
        x_seq = x.reshape(B, C, L).permute(0, 2, 1)  # (B, L, C)
        x_n = self.norm(x_seq)  # (B, L, C)
        fwd = self.fwd(x_n)  # (B, L, C)
        bwd = self.bwd(x_n.flip(1)).flip(1)  # (B, L, C)
        out = (fwd + bwd).permute(0, 2, 1)  # (B, C, L)
        out = out.reshape(B, C, *spatial)  # (B, C, H, W, D)
        return torch.as_tensor(x + out)  # residual


# ── T04: Mamba Block ──────────────────────────────────────────────────────────


class MambaBlock(nn.Module):
    """Stacked Mamba layers with pre-norm MLP channel mixing.

    Architecture per depth level (D07 — no LN before BidirMamba):
      x = BidirectionalMambaLayer(x)   # LN inside BidirMamba
      x = x + MlpChannel(LN(x))        # LN only for MLP branch

    Parameters
    ----------
    dim:
        Channel dimension.
    depth:
        Number of BidirMamba+MLP pairs. Default=1 (8 blocks at bottleneck).
    d_state, d_conv, expand:
        Mamba hyperparameters forwarded to BidirectionalMambaLayer.
    mamba_cls:
        Optional Mamba class override for CPU testing. See BidirectionalMambaLayer.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mamba_cls: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BidirectionalMambaLayer(
                    dim, d_state, d_conv, expand, mamba_cls=mamba_cls
                )
                for _ in range(depth)
            ]
        )
        self.mlp_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.mlps = nn.ModuleList([MlpChannel(dim) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        """Apply bidirectional Mamba + channel MLP to (B,C,H,W,D) input."""
        for bidir, ln, mlp in zip(self.layers, self.mlp_norms, self.mlps, strict=True):
            x = bidir(x)  # LN inside BidirMamba (D07)
            # LayerNorm for MLP needs (..., C); reshape to (B, L, C), apply, reshape back
            B, C, *sp = x.shape
            L = math.prod(sp)
            x_seq = x.reshape(B, C, L).permute(0, 2, 1)  # (B, L, C)
            x = x + mlp(ln(x_seq).permute(0, 2, 1).reshape(B, C, *sp))
        return x


# ── T05: Full Backbone ────────────────────────────────────────────────────────


class MambaVesselNetBackbone(nn.Module):
    """MambaVesselNet++ backbone (equivalent to mvnNet in reference repo).

    Architecture (adapted from Xu & Chen et al. 2025, ACM TOMM):
    - Hi-Encoder: 5 UnetrBasicBlock levels + stride-2 Conv3d downsampling
    - Bottleneck: 8 MambaBlocks (4 enc + 4 dec) at 768 channels
    - Decoder: 4 UnetrUpBlock levels + UnetrBasicBlock + UnetOutBlock

    Input:  (B, in_chans, H, W, D) — MONAI convention, depth LAST.
    Output: (B, out_chans, H, W, D) — raw logits (no softmax).

    Divisor: 16 (4 downsampling levels, 2^4 = 16).
    Required patch size: multiple of 16. Default: 64³.

    Note (D06): ALL MONAI block calls include norm_name="instance" — it has no default.
    Note (D07): LN lives inside BidirMamba. No double-LN.

    Parameters
    ----------
    in_chans:
        Number of input channels (e.g., 1 for grayscale MRI).
    out_chans:
        Number of output classes (e.g., 2 for foreground + background).
    feature_dims:
        Channel widths per encoder level. Default: (48, 96, 192, 384, 768).
    d_state, d_conv, expand:
        Mamba hyperparameters forwarded to MambaBlock.
    mamba_cls:
        Optional Mamba class override for CPU testing.
    """

    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 2,
        feature_dims: tuple[int, ...] = _FEATURE_DIMS,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mamba_cls: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        f = feature_dims
        mamba_kw: dict[str, Any] = {
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            "mamba_cls": mamba_cls,
        }

        # ── Encoder ──────────────────────────────────────────────────────
        self.enc_conv1 = UnetrBasicBlock(
            3,
            in_chans,
            f[0],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.down1 = nn.Conv3d(f[0], f[0], kernel_size=2, stride=2)
        self.enc_conv2 = UnetrBasicBlock(
            3, f[0], f[1], kernel_size=3, stride=1, norm_name="instance", res_block=True
        )
        self.down2 = nn.Conv3d(f[1], f[1], kernel_size=2, stride=2)
        self.enc_conv3 = UnetrBasicBlock(
            3, f[1], f[2], kernel_size=3, stride=1, norm_name="instance", res_block=True
        )
        self.down3 = nn.Conv3d(f[2], f[2], kernel_size=2, stride=2)
        self.enc_conv4 = UnetrBasicBlock(
            3, f[2], f[3], kernel_size=3, stride=1, norm_name="instance", res_block=True
        )
        self.down4 = nn.Conv3d(f[3], f[3], kernel_size=2, stride=2)
        self.enc_conv5 = UnetrBasicBlock(
            3, f[3], f[4], kernel_size=3, stride=1, norm_name="instance", res_block=True
        )

        # ── Bottleneck: 8 MambaBlocks at f[4] channels ───────────────────
        self.enc_mamba = nn.ModuleList(
            [MambaBlock(f[4], depth=1, **mamba_kw) for _ in range(4)]
        )
        self.dec_mamba = nn.ModuleList(
            [MambaBlock(f[4], depth=1, **mamba_kw) for _ in range(4)]
        )

        # ── Decoder ──────────────────────────────────────────────────────
        # UnetrUpBlock(spatial, in_ch, out_ch, kernel, upsample_kernel, norm)
        # After transp_conv: in_ch → out_ch; then cat with skip (out_ch), so conv_block sees 2*out_ch → out_ch
        self.dec_conv5 = UnetrUpBlock(
            3,
            f[4],
            f[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.dec_conv4 = UnetrUpBlock(
            3,
            f[3],
            f[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.dec_conv3 = UnetrUpBlock(
            3,
            f[2],
            f[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.dec_conv2 = UnetrUpBlock(
            3,
            f[1],
            f[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.dec_conv1 = UnetrBasicBlock(
            3, f[0], f[0], kernel_size=3, stride=1, norm_name="instance", res_block=True
        )
        self.out = UnetOutBlock(3, f[0], out_chans)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through encoder, bottleneck, decoder.

        Parameters
        ----------
        x:
            Input tensor (B, in_chans, H, W, D).
            H, W, D must each be divisible by 16 (4 downsampling levels).

        Returns
        -------
        Tensor (B, out_chans, H, W, D) — raw logits, no softmax.
        """
        # Encoder — save skip connections
        enc1 = self.enc_conv1(x)
        x = self.down1(enc1)
        enc2 = self.enc_conv2(x)
        x = self.down2(enc2)
        enc3 = self.enc_conv3(x)
        x = self.down3(enc3)
        enc4 = self.enc_conv4(x)
        x = self.down4(enc4)
        z = self.enc_conv5(x)

        # Bottleneck — 4 enc MambaBlocks + 4 dec MambaBlocks
        for m in self.enc_mamba:
            z = m(z)
        for m in self.dec_mamba:
            z = m(z)

        # Decoder — UnetrUpBlock.forward(inp, skip)
        x = self.dec_conv5(z, enc4)
        x = self.dec_conv4(x, enc3)
        x = self.dec_conv3(x, enc2)
        x = self.dec_conv2(x, enc1)
        x = self.dec_conv1(x)
        return torch.as_tensor(self.out(x))  # (B, out_chans, H, W, D) raw logits
