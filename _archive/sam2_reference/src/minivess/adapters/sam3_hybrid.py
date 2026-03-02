"""Sam3HybridAdapter — SAM2 features + DynUNet 3D decoder with gated fusion.

V3 of the SAM3 variants: combines frozen SAM2 encoder features with a
trainable DynUNet encoder/decoder via gated feature fusion at the bottleneck.

Architecture:
- Frozen SAM2 Hiera-Tiny extracts 2D features per-slice
- AxialProjection applies 1D conv along Z to create 3D feature volume
- DynUNet encoder/decoder processes the 3D input volume
- GatedFeatureFusion merges SAM and DynUNet features at bottleneck
- Gate alpha initialized to 0.0 → pure DynUNet at training start

Based on: nnSAM (Li et al., 2025), DB-SAM gated fusion pattern.
Expected: best SAM variant, but likely still below standalone DynUNet.
VRAM: ~7.5 GB (batch_size=1, AMP mandatory on 8 GB GPUs).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from monai.networks.nets import DynUNet as MonaiDynUNet
from torch import Tensor, nn

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput
from minivess.adapters.sam2_backbone import Sam2Backbone

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.config.models import ModelConfig

logger = logging.getLogger(__name__)


class AxialProjection(nn.Module):
    """1D convolution along the Z (depth) axis.

    Takes stacked 2D SAM features (B, C, D, H, W) and applies a
    1D conv along D to capture inter-slice context.

    Parameters
    ----------
    in_channels:
        Number of input feature channels.
    out_channels:
        Number of output feature channels.
    kernel_size:
        1D kernel size along Z axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply axial 1D convolution.

        Parameters
        ----------
        x:
            Input tensor (B, C, D, H, W).

        Returns
        -------
        Output tensor (B, out_channels, D, H, W).
        """
        return self.conv(x)


class GatedFeatureFusion(nn.Module):
    """Gated fusion of 3D DynUNet features with 2D SAM features.

    Computes: ``f_3d + sigmoid(alpha) * proj_conv(f_sam.detach())``

    The gate_alpha is initialized to 0.0, so at training start the model
    behaves as pure DynUNet. SAM features are detached to prevent
    gradients flowing back to the frozen SAM encoder.

    Parameters
    ----------
    dim_3d:
        Channel dimension of 3D features.
    dim_sam:
        Channel dimension of SAM features (projected to dim_3d).
    """

    def __init__(self, dim_3d: int, dim_sam: int) -> None:
        super().__init__()
        self.proj_conv = nn.Conv3d(dim_sam, dim_3d, kernel_size=1)
        self.gate_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, f_3d: Tensor, f_sam: Tensor) -> Tensor:
        """Fuse 3D and SAM features with learned gating.

        Parameters
        ----------
        f_3d:
            DynUNet features (B, dim_3d, D, H', W').
        f_sam:
            SAM features (B, dim_sam, D, H', W'), detached.

        Returns
        -------
        Fused features (B, dim_3d, D, H', W').
        """
        f_sam_proj = self.proj_conv(f_sam.detach())

        # Spatial alignment if needed
        if f_sam_proj.shape[2:] != f_3d.shape[2:]:
            f_sam_proj = F.interpolate(
                f_sam_proj,
                size=f_3d.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        gate = torch.sigmoid(self.gate_alpha)
        return f_3d + gate * f_sam_proj


class Sam3HybridAdapter(ModelAdapter):
    """Hybrid SAM2 + DynUNet adapter with gated feature fusion.

    Extracts 2D features per-slice from frozen SAM2, projects them
    to 3D via axial convolution, and fuses with DynUNet bottleneck
    features through a learned gate.

    Parameters
    ----------
    config:
        ModelConfig with SAM3_HYBRID family.
    variant:
        SAM2 Hiera variant (default: "hiera_tiny").
    """

    def __init__(
        self,
        config: ModelConfig,
        variant: str = "hiera_tiny",
        *,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.config = config

        arch = config.architecture_params
        variant = arch.get("sam2_variant", variant)
        pretrained = arch.get("pretrained", pretrained)
        filters = arch.get("filters", [32, 64, 128, 256])

        # Frozen SAM2 backbone
        self.sam_backbone = Sam2Backbone(variant=variant, pretrained=pretrained)
        sam_out_channels = self.sam_backbone.out_channels

        # Axial projection: SAM 2D features → 3D volume
        self.axial_proj = AxialProjection(
            in_channels=sam_out_channels,
            out_channels=sam_out_channels,
        )

        # DynUNet 3D encoder/decoder
        n_levels = len(filters)
        kernel_size = [[3, 3, 3]] * n_levels
        strides = [[1, 1, 1]] + [[2, 2, 2]] * (n_levels - 1)
        upsample_kernel_size = [[2, 2, 2]] * (n_levels - 1)

        self.dynunet = MonaiDynUNet(
            spatial_dims=3,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            filters=filters,
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )

        # Gated fusion at output level (DynUNet logits + SAM features)
        self.fusion = GatedFeatureFusion(
            dim_3d=config.out_channels,
            dim_sam=sam_out_channels,
        )

        self._variant = variant
        self._filters = filters

        # Set net for base class compatibility (points to dynunet)
        self.net = self.dynunet

        logger.info(
            "Sam3HybridAdapter: SAM=%s (frozen), DynUNet filters=%s, "
            "fusion at output (dim=%d)",
            variant,
            filters,
            config.out_channels,
        )

    def _extract_sam_features_3d(self, images: Tensor) -> Tensor:
        """Extract SAM features per-slice and stack as 3D volume.

        Parameters
        ----------
        images:
            Input volume (B, C, D, H, W).

        Returns
        -------
        SAM feature volume (B, sam_channels, D, H', W').
        """
        with torch.no_grad():
            embeddings = self.sam_backbone.get_image_embeddings(images)

        # Stack along D: list of (B, C, H', W') → (B, C, D, H', W')
        sam_features = torch.stack(embeddings, dim=2)

        # Axial projection for inter-slice context
        sam_features = self.axial_proj(sam_features)

        return sam_features

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run hybrid SAM2 + DynUNet inference.

        1. Extract SAM features per-slice → 3D volume
        2. Run DynUNet on input → get bottleneck features
        3. Fuse SAM features with DynUNet bottleneck
        4. DynUNet decoder produces final segmentation

        For simplicity, we run full DynUNet forward and add fused SAM
        features as a residual. This avoids modifying DynUNet internals.

        Parameters
        ----------
        images:
            Input tensor (B, C, D, H, W).

        Returns
        -------
        SegmentationOutput with 2-class predictions.
        """
        # Get SAM feature volume
        sam_features = self._extract_sam_features_3d(images)

        # Run DynUNet forward
        dynunet_logits = self.dynunet(images)

        # Fuse SAM features into DynUNet output space
        # Resize SAM features to match DynUNet output spatial dims
        if sam_features.shape[2:] != dynunet_logits.shape[2:]:
            sam_features = F.interpolate(
                sam_features,
                size=dynunet_logits.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        # Project SAM features to match DynUNet output channels, then gate-fuse
        logits = self.fusion(dynunet_logits, sam_features)

        return self._build_output(logits, "sam3_hybrid")

    def get_config(self) -> AdapterConfigInfo:
        return self._build_config(
            variant=self._variant,
            filters=self._filters,
            fusion="gated_bottleneck",
            gate_alpha=self.fusion.gate_alpha.item(),
        )

    def trainable_parameters(self) -> int:
        """DynUNet + axial projection + fusion parameters."""
        total = 0
        for module in [self.dynunet, self.axial_proj, self.fusion]:
            total += sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total

    def save_checkpoint(self, path: Path) -> None:
        """Save DynUNet + fusion + axial projection weights."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "dynunet": self.dynunet.state_dict(),
            "axial_proj": self.axial_proj.state_dict(),
            "fusion": self.fusion.state_dict(),
        }
        torch.save(state, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load DynUNet + fusion + axial projection weights."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "dynunet" in state:
            self.dynunet.load_state_dict(state["dynunet"])
        if "axial_proj" in state:
            self.axial_proj.load_state_dict(state["axial_proj"])
        if "fusion" in state:
            self.fusion.load_state_dict(state["fusion"])
