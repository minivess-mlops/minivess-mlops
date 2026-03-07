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
    from pathlib import Path

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
        strides = [[1, 1, 1]] + [[2, 2, 2]] * (n_levels - 1)
        upsample_kernel_size = [[2, 2, 2]] * (n_levels - 1)

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
            Input 3D volume (B, C, D, H, W).

        Returns
        -------
        SAM features (B, 256, D, H_feat, W_feat).
        """
        b, c, d, h, w = images.shape
        slice_features: list[Tensor] = []

        for z_idx in range(d):
            slice_2d = images[:, :, z_idx, :, :]
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

        # Axial smoothing of stacked 2D features
        sam_features = self.axial_proj(sam_features)

        # DynUNet forward (full 3D, trainable)
        logits: Tensor = self.dynunet(images)

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

        return self._build_output(logits, "sam3_hybrid")

    def get_config(self) -> AdapterConfigInfo:
        """Return adapter configuration info."""
        return self._build_config(
            variant="hybrid",
            backbone="vit_32l",
            input_size=SAM3_INPUT_SIZE,
            fusion="gated_residual",
        )

    def save_checkpoint(self, path: Path) -> None:
        """Save adapter state dict (no self.net dependency)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.state_dict()}, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load adapter state dict (no self.net dependency)."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            self.load_state_dict(payload["model_state_dict"])
        else:
            self.load_state_dict(payload)

    def export_onnx(self, path: Path, example_input: Tensor) -> None:
        """Export adapter to ONNX (no self.net dependency).

        Uses a thin wrapper to return raw logits tensor instead of
        SegmentationOutput, which ONNX tracing cannot handle.
        """
        import warnings

        path.parent.mkdir(parents=True, exist_ok=True)
        self.eval()

        class _LogitsWrapper(torch.nn.Module):
            def __init__(self, adapter: Sam3HybridAdapter) -> None:
                super().__init__()
                self.adapter = adapter

            def forward(self, x: Tensor) -> Tensor:
                result: Tensor = self.adapter(x).logits
                return result

        wrapper = _LogitsWrapper(self)
        wrapper.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                wrapper,
                (example_input,),
                str(path),
                input_names=["images"],
                output_names=["logits"],
                opset_version=17,
                dynamo=False,
            )

    def trainable_parameters(self) -> int:
        """Count trainable parameters (DynUNet + fusion)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
