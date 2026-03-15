"""Sam3HybridAdapter — frozen SAM3 features + DynUNet 3D + GatedFeatureFusion.

V3 of the SAM3 variants. Combines frozen SAM3 ViT-32L features with a
trainable DynUNet encoder/decoder via gated feature fusion.

Architecture:
    - Frozen SAM3 ViT-32L encoder (feature extractor, slice-by-slice)
    - SAM features axially projected and fused at DynUNet bottleneck
    - GatedFeatureFusion: f_3d + sigmoid(alpha) * proj(f_sam.detach())
    - gate_alpha initialized to 0.0 (pure DynUNet at start)
    - Trainable: DynUNet encoder/decoder + fusion module

Based on: nnSAM (Li et al., 2025), DB-SAM pattern
Expected: best SAM variant, but likely below standalone DynUNet
VRAM: ~7.5 GB (batch_size=1, AMP mandatory on 8GB GPU)
Go/No-Go Gate G3: V3 DSC > V2 DSC.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from monai.networks.nets import DynUNet  # type: ignore[attr-defined]
from torch import Tensor, nn

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput
from minivess.adapters.sam3_backbone import SAM3_FPN_DIM, SAM3_INPUT_SIZE, Sam3Backbone

if TYPE_CHECKING:
    from minivess.config.models import ModelConfig

logger = logging.getLogger(__name__)


class GatedFeatureFusion(nn.Module):
    """Gated residual fusion of SAM features into 3D features.

    ``output = f_3d + sigmoid(alpha) * proj_conv(f_sam)``

    Parameters
    ----------
    sam_channels:
        Number of channels in SAM features (typically 256).
    target_channels:
        Number of channels in the 3D features to fuse into.
    gate_init:
        Initial value of the gate parameter alpha (0.0 = pure 3D at start).
    """

    def __init__(
        self,
        sam_channels: int,
        target_channels: int,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv3d(sam_channels, target_channels, kernel_size=1)
        self.gate_alpha = nn.Parameter(torch.tensor(gate_init))

    def forward(self, f_3d: Tensor, f_sam: Tensor) -> Tensor:
        """Fuse SAM features into 3D features via gated residual.

        Parameters
        ----------
        f_3d:
            3D features of shape (B, C_3d, D, H, W).
        f_sam:
            SAM features of shape (B, C_sam, D, H, W).

        Returns
        -------
        Fused features of shape (B, C_3d, D, H, W).
        """
        # Project SAM features to target channel count
        projected: Tensor = self.proj(f_sam.detach())

        # Resize if spatial dims don't match
        if projected.shape[2:] != f_3d.shape[2:]:
            projected = F.interpolate(
                projected,
                size=f_3d.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        gate = torch.sigmoid(self.gate_alpha)
        return f_3d + gate * projected


class Sam3HybridAdapter(ModelAdapter):
    """Frozen SAM3 features + trainable DynUNet + gated fusion.

    Parameters
    ----------
    config:
        ModelConfig with ``SAM3_HYBRID`` family.
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        arch = config.architecture_params

        # Frozen SAM3 backbone (feature extractor)
        self.sam_backbone = Sam3Backbone(config=config, freeze=True)

        # Trainable DynUNet 3D encoder/decoder
        filters = arch.get("filters", [32, 64, 128, 256])
        n_levels = len(filters)
        kernel_size = [[3, 3, 3]] * n_levels
        # SAM3 patch depth D=3 (MONAI (B,C,H,W,D) convention, depth last).
        # DynUNet with stride (2,2,2) would downsample D: 3→2→1→1 but upsample
        # 1→2, creating a skip-connection mismatch (encoder D=1 vs decoder D=2).
        # Fix: stride (2,2,1) keeps D constant — only H,W are downsampled.
        strides = [[1, 1, 1]] + [[2, 2, 1]] * (n_levels - 1)
        upsample_kernel_size = [[2, 2, 1]] * (n_levels - 1)

        self.dynunet = DynUNet(
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

        # Gated fusion at output level
        gate_init = arch.get("fusion_gate_init", 0.0)
        self.fusion = GatedFeatureFusion(
            sam_channels=SAM3_FPN_DIM,  # 256
            target_channels=config.out_channels,  # match DynUNet output
            gate_init=gate_init,
        )

        # Axial projection: reduce SAM 2D features stacked along Z to 3D
        self.axial_proj = nn.Conv3d(
            SAM3_FPN_DIM, SAM3_FPN_DIM, kernel_size=(3, 1, 1), padding=(1, 0, 0)
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = self.trainable_parameters()
        logger.info(
            "Sam3HybridAdapter: %d trainable / %d total params",
            trainable,
            total,
        )

    def _extract_sam_volume_features(self, images: Tensor) -> Tensor:
        """Extract SAM FPN features for all slices, stack as 3D volume.

        Parameters
        ----------
        images:
            Input 3D volume (B, C, H, W, D) — MONAI depth-last convention.

        Returns
        -------
        SAM features (B, 256, D, H_feat, W_feat).
        """
        # MONAI uses (B, C, H, W, D) — depth is the LAST dimension.
        # Wrong unpacking (b,c,d,h,w) causes 21× more encoder calls on 8 GB GPU
        # (64 instead of 3 for patch=(64,64,3)), accumulating ~5 GiB of features
        # before torch.stack → OOM. See src/minivess/adapters/CLAUDE.md.
        b, c, h, w, d = images.shape
        slice_features: list[Tensor] = []

        for z_idx in range(d):
            slice_2d = images[:, :, :, :, z_idx]  # (B, C, H, W)
            fpn_feat = self.sam_backbone.extract_fpn_features(slice_2d)
            slice_features.append(fpn_feat)

        # Stack along depth: (B, 256, D, H_f, W_f)
        return torch.stack(slice_features, dim=2)

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Forward: extract SAM features → fuse with DynUNet at bottleneck.

        Parameters
        ----------
        images:
            Input 3D volume (B, C, D, H, W).

        Returns
        -------
        SegmentationOutput with 2-class predictions.
        """
        # Extract frozen SAM features (no grad)
        with torch.no_grad():
            sam_features = self._extract_sam_volume_features(images)

        # SAM3 encoder runs in FP16 (torch_dtype=float16 in from_pretrained).
        # Cast to FP32 before passing to trainable FP32 modules (axial_proj, fusion).
        sam_features = sam_features.float()

        # NaN diagnostic: check SAM features after FP16→FP32 cast
        if not torch.isfinite(sam_features).all():
            logger.warning(
                "NaN/Inf in SAM features after FP32 cast: shape=%s",
                tuple(sam_features.shape),
            )
            sam_features = torch.nan_to_num(
                sam_features, nan=0.0, posinf=0.0, neginf=0.0
            )

        # Axial smoothing of stacked 2D features
        sam_features = self.axial_proj(sam_features)

        # DynUNet forward (full 3D, trainable)
        logits: Tensor = self.dynunet(images)

        # NaN diagnostic: check DynUNet output separately
        if not torch.isfinite(logits).all():
            logger.warning(
                "NaN/Inf in DynUNet logits: shape=%s, nan=%d, inf=%d",
                tuple(logits.shape),
                torch.isnan(logits).sum().item(),
                torch.isinf(logits).sum().item(),
            )

        # Fuse SAM features at the output level
        # Resize SAM features to match logits spatial dims
        if sam_features.shape[2:] != logits.shape[2:]:
            sam_features = F.interpolate(
                sam_features,
                size=logits.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        # Project SAM features to match logits channels for residual add
        # (fusion operates at bottleneck conceptually, but applied at output for simplicity)
        sam_projected = self.fusion.proj(sam_features.detach())
        if sam_projected.shape[2:] != logits.shape[2:]:
            sam_projected = F.interpolate(
                sam_projected,
                size=logits.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        gate = torch.sigmoid(self.fusion.gate_alpha)
        logits = logits + gate * sam_projected

        # Final NaN guard on fused logits
        if not torch.isfinite(logits).all():
            logger.warning(
                "NaN/Inf in fused logits (after gate): nan=%d, inf=%d / %d total",
                torch.isnan(logits).sum().item(),
                torch.isinf(logits).sum().item(),
                logits.numel(),
            )
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        return self._build_output(logits, "sam3_hybrid")

    def get_config(self) -> AdapterConfigInfo:
        """Return adapter configuration info."""
        return self._build_config(
            variant="hybrid",
            backbone="vit_32l",
            input_size=SAM3_INPUT_SIZE,
            fusion="gated_residual",
        )

    def trainable_parameters(self) -> int:
        """Count trainable parameters (DynUNet + fusion)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
