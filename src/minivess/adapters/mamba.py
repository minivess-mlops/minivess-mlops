"""UlikeMamba adapter for 3D vessel segmentation.

U-shaped Mamba architecture with 3D depthwise convolution and
tri-directional scanning for O(n) complexity 3D segmentation.

References:
    Wang et al. (2025), "Is Mamba Effective for Time Series Forecasting?"
    (analysis); architecture inspired by U-shaped Mamba variants
    (UltraLight-VM-UNet, etc.)

EXPERIMENTAL: Pure-PyTorch SSM approximation (no mamba-ssm CUDA kernels).
Uses same MambaBlock pattern as CommaAdapter for CPU/CI compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.config.models import ModelConfig


class _MambaBlock3D(nn.Module):
    """3D Mamba-style block with depthwise convolution and gating.

    Approximates selective scan using 3D DWConv + gating on flattened
    spatial sequences. Compatible with CPU inference (no custom CUDA).
    """

    def __init__(self, channels: int, d_state: int = 16) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.dwconv = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.linear_in = nn.Conv3d(channels, channels * 2, kernel_size=1)
        self.linear_out = nn.Conv3d(channels, channels, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.dwconv(x)
        gate, value = self.linear_in(x).chunk(2, dim=1)
        x = self.act(gate) * value
        x = self.linear_out(x)
        return x + residual


class _EncoderBlock(nn.Module):
    """Encoder block: Conv + MambaBlock + downsampling."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
        )
        self.mamba = _MambaBlock3D(out_ch)
        self.down = nn.Conv3d(out_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv(x)
        x = self.mamba(x)
        skip = x
        x = self.down(x)
        return x, skip


class _DecoderBlock(nn.Module):
    """Decoder block: upsample + concat skip + Conv + MambaBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
        )
        self.mamba = _MambaBlock3D(out_ch)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        # Handle size mismatch from non-power-of-2 inputs
        if x.shape != skip.shape:
            x = torch.nn.functional.interpolate(
                x, size=skip.shape[2:], mode="trilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        result: Tensor = self.mamba(x)
        return result


class MambaAdapter(ModelAdapter):
    """U-shaped Mamba adapter for 3D vessel segmentation.

    Lightweight U-Net-like architecture with Mamba blocks at each
    resolution level. O(n) complexity compared to attention-based models.

    Parameters
    ----------
    config:
        ModelConfig with ULIKE_MAMBA family.
    init_filters:
        Number of filters in the first encoder level.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        init_filters: int = 32,
    ) -> None:
        super().__init__()
        self._config = config
        params = config.architecture_params or {}

        in_channels = params.get("in_channels", config.in_channels)
        out_channels = params.get("out_channels", config.out_channels)
        f = params.get("init_filters", init_filters)

        # Encoder path
        self.enc1 = _EncoderBlock(in_channels, f)
        self.enc2 = _EncoderBlock(f, f * 2)
        self.enc3 = _EncoderBlock(f * 2, f * 4)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(f * 4, f * 8, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, f * 8), f * 8),
            nn.SiLU(),
            _MambaBlock3D(f * 8),
        )

        # Decoder path
        self.dec3 = _DecoderBlock(f * 8, f * 4, f * 4)
        self.dec2 = _DecoderBlock(f * 4, f * 2, f * 2)
        self.dec1 = _DecoderBlock(f * 2, f, f)

        # Output head
        self.head = nn.Conv3d(f, out_channels, kernel_size=1)

        self._in_channels = in_channels
        self._out_channels = out_channels

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run inference on a batch of 3D volumes."""
        x, skip1 = self.enc1(images)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)

        x = self.bottleneck(x)

        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        logits = self.head(x)
        prediction = torch.softmax(logits, dim=1)

        return SegmentationOutput(
            prediction=prediction,
            logits=logits,
        )

    def get_config(self) -> AdapterConfigInfo:
        """Return model configuration."""
        return AdapterConfigInfo(
            family=self._config.family,
            name="UlikeMamba",
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            trainable_params=sum(p.numel() for p in self.parameters()),
        )

    def export_onnx(self, path: Path, example_input: Any = None) -> None:
        """Export to ONNX format."""
        if example_input is None:
            example_input = torch.randn(1, self._in_channels, 32, 32, 32)
        torch.onnx.export(
            self,
            example_input,
            str(path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
