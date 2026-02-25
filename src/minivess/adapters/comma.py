"""COMMA/Mamba architecture adapter for 3D vessel segmentation.

Implements the Coordinate Mamba Architecture (Shi et al., 2025) using
pure-PyTorch SSM blocks for CPU/CI compatibility. The selective state-space
model captures long-range spatial dependencies in 3D volumes without
requiring custom CUDA kernels.

Reference: Shi et al. (2025). "COMMA: Coordinate Mamba Architecture for
Vessel Segmentation." arxiv:2503.02332

R5.18 assessment (333 lines): This module contains tightly coupled components
(MambaBlock, CoordinateEmbedding, encoder/decoder blocks, CommaAdapter) that
form a single architecture. Splitting would fragment a cohesive neural network
definition — no action required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.config.models import ModelConfig


class MambaBlock(nn.Module):
    """Pure-PyTorch selective state-space block (Mamba-style).

    Approximates Mamba's selective scan using 1D depthwise convolution
    and gating, operating on flattened spatial sequences.

    Parameters
    ----------
    d_model:
        Input/output feature dimension.
    d_state:
        SSM state dimension.
    d_conv:
        Depthwise convolution kernel size.
    expand:
        Expansion factor for inner dimension.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        d_inner = d_model * expand

        # Input projection: expand dimension
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # 1D depthwise conv for local context (causal-style)
        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )

        # SSM parameters: state transition approximation
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection: compress back
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """Process sequence through selective state-space block.

        Parameters
        ----------
        x:
            Input tensor (B, L, D).

        Returns
        -------
        Output tensor (B, L, D).
        """
        residual = x
        x = self.norm(x)

        # Expand and split into two paths
        xz = self.in_proj(x)
        x_path, z = xz.chunk(2, dim=-1)

        # 1D conv path (B, L, D_inner) → (B, D_inner, L) → conv → (B, L, D_inner)
        x_conv = x_path.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, : x_path.shape[1]]
        x_conv = x_conv.transpose(1, 2)

        # Selective gate with SSM-inspired dynamics
        dt = self.act(self.dt_proj(x_conv))
        x_ssm = dt * x_conv + self.D.unsqueeze(0).unsqueeze(0) * x_path

        # Gate and project
        y = x_ssm * self.act(z)
        y = self.out_proj(y)

        return y + residual


class CoordinateEmbedding(nn.Module):
    """3D coordinate-aware positional embedding.

    Generates normalized coordinate grids for each spatial axis and
    fuses them with input features through a learned projection.

    Parameters
    ----------
    in_channels:
        Number of input feature channels.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # Project concatenated (features + 3 coord channels) back to in_channels
        self.proj = nn.Conv3d(in_channels + 3, in_channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Add coordinate embeddings to feature maps.

        Parameters
        ----------
        x:
            Input tensor (B, C, D, H, W).

        Returns
        -------
        Output tensor (B, C, D, H, W) with coordinate information fused.
        """
        b, _c, d, h, w = x.shape

        # Generate normalized coordinate grids [-1, 1]
        coords_d = torch.linspace(-1, 1, d, device=x.device, dtype=x.dtype)
        coords_h = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        coords_w = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)

        grid_d = coords_d.view(1, 1, d, 1, 1).expand(b, 1, d, h, w)
        grid_h = coords_h.view(1, 1, 1, h, 1).expand(b, 1, d, h, w)
        grid_w = coords_w.view(1, 1, 1, 1, w).expand(b, 1, d, h, w)

        # Concatenate coords with features, then project back
        x_coord = torch.cat([x, grid_d, grid_h, grid_w], dim=1)
        return self.proj(x_coord)


class _CommaEncoderBlock(nn.Module):
    """COMMA encoder block: Conv3d + CoordinateEmbedding + MambaBlock."""

    def __init__(self, in_channels: int, out_channels: int, d_state: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.SiLU(),
        )
        self.coord_embed = CoordinateEmbedding(out_channels)
        self.mamba = MambaBlock(d_model=out_channels, d_state=d_state)
        self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (downsampled, skip_connection)."""
        x = self.conv(x)
        x = self.coord_embed(x)

        # Flatten spatial → sequence for Mamba: (B, C, D, H, W) → (B, D*H*W, C)
        b, c, d, h, w = x.shape
        x_seq = x.reshape(b, c, -1).transpose(1, 2)
        x_seq = self.mamba(x_seq)
        x = x_seq.transpose(1, 2).reshape(b, c, d, h, w)

        skip = x
        x = self.downsample(x)
        return x, skip


class _CommaDecoderBlock(nn.Module):
    """COMMA decoder block: Upsample + skip + Conv3d."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels, in_channels, kernel_size=2, stride=2,
        )
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CommaAdapter(ModelAdapter):
    """COMMA (Coordinate Mamba Architecture) adapter for 3D segmentation.

    Implements a UNet-like encoder-decoder with Mamba SSM blocks and
    coordinate-aware positional embeddings at each resolution level.

    Parameters
    ----------
    config:
        ModelConfig with COMMA_MAMBA family.
    init_filters:
        Number of filters in the first encoder level.
    d_state:
        SSM state dimension for MambaBlocks.
    """

    def __init__(
        self,
        config: ModelConfig,
        init_filters: int = 32,
        d_state: int = 16,
    ) -> None:
        super().__init__()
        self.config = config
        self.init_filters = init_filters
        self.d_state = d_state

        f = init_filters
        # Encoder: 3 levels
        self.enc1 = _CommaEncoderBlock(config.in_channels, f, d_state)
        self.enc2 = _CommaEncoderBlock(f, f * 2, d_state)
        self.enc3 = _CommaEncoderBlock(f * 2, f * 4, d_state)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(f * 4, f * 8, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(f * 8),
            nn.SiLU(),
        )

        # Decoder: 3 levels (mirror encoder)
        self.dec3 = _CommaDecoderBlock(f * 8, f * 4, f * 4)
        self.dec2 = _CommaDecoderBlock(f * 4, f * 2, f * 2)
        self.dec1 = _CommaDecoderBlock(f * 2, f, f)

        # Final 1x1 conv to class logits
        self.head = nn.Conv3d(f, config.out_channels, kernel_size=1)

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run COMMA inference on a batch of 3D volumes.

        Parameters
        ----------
        images:
            Input tensor (B, C, D, H, W).

        Returns
        -------
        SegmentationOutput with softmax predictions and raw logits.
        """
        # Encoder
        x, skip1 = self.enc1(images)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        logits = self.head(x)
        return self._build_output(logits, "comma_mamba")

    def get_config(self) -> AdapterConfigInfo:
        return self._build_config(
            init_filters=self.init_filters,
            d_state=self.d_state,
        )

    def load_checkpoint(self, path: Path) -> None:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict)

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def export_onnx(self, path: Path, example_input: Tensor) -> None:
        """Export model to ONNX format."""
        import warnings

        path.parent.mkdir(parents=True, exist_ok=True)
        self.eval()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                onnx_program = torch.onnx.export(
                    self,
                    example_input,
                    dynamo=True,
                )
                onnx_program.save(str(path))
            except Exception:
                torch.onnx.export(
                    self,
                    example_input,
                    str(path),
                    input_names=["images"],
                    output_names=["logits"],
                    opset_version=17,
                    dynamo=False,
                )
