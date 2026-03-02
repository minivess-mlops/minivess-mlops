"""Sam3VanillaAdapter — frozen SAM3 ViT-32L + trainable decoder.

V1 of the SAM3 variants. Demonstrates how badly vanilla SAM3 performs
on 3D microvessel segmentation without any adaptation.

Architecture:
    - Frozen SAM3 ViT-32L perception encoder (1024-dim → 256-dim FPN)
    - Trainable mask decoder (~2-4M params)
    - Slice-by-slice 2D inference on 3D volumes
    - Null prompt mode (fully automatic, no interactive prompts)
    - binary_to_2class for cross-entropy compatibility

Expected results: DSC ~0.35-0.55, clDice ~0.3-0.5
Go/No-Go Gate G1: DSC >= 0.10 or abandon SAM for segmentation.

References:
    - Ravi et al. (2025). "SAM 3." arXiv:2511.16719
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput
from minivess.adapters.sam3_backbone import SAM3_INPUT_SIZE, Sam3Backbone
from minivess.adapters.sam3_decoder import Sam3MaskDecoder

if TYPE_CHECKING:
    from minivess.config.models import ModelConfig

logger = logging.getLogger(__name__)


class Sam3VanillaAdapter(ModelAdapter):  # type: ignore[misc]
    """Frozen SAM3 encoder + trainable decoder for segmentation.

    Parameters
    ----------
    config:
        ModelConfig with ``SAM3_VANILLA`` family.
    use_stub:
        If True, use stub encoder/decoder for testing.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        use_stub: bool = False,
    ) -> None:
        super().__init__()
        self.config = config

        # Frozen SAM3 backbone (encoder + FPN neck)
        self.backbone = Sam3Backbone(config=config, use_stub=use_stub, freeze=True)

        # Trainable mask decoder
        self.decoder = Sam3MaskDecoder(config=config, use_stub=use_stub)

        logger.info(
            "Sam3VanillaAdapter: encoder=%d params (frozen), decoder=%d params (trainable)",
            sum(p.numel() for p in self.backbone.parameters()),
            sum(p.numel() for p in self.decoder.parameters()),
        )

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Slice-by-slice forward through SAM3 encoder → decoder.

        Parameters
        ----------
        images:
            Input 3D volume of shape (B, C, D, H, W).

        Returns
        -------
        SegmentationOutput with 2-class predictions.
        """
        b, c, d, h, w = images.shape
        slice_logits: list[Tensor] = []

        for z_idx in range(d):
            slice_2d = images[:, :, z_idx, :, :]  # (B, C, H, W)

            # Extract FPN features (frozen, no grad)
            fpn_features = self.backbone.extract_fpn_features(slice_2d)

            # Decode to binary mask (trainable)
            binary_logits = self.decoder(fpn_features)  # (B, 1, H_f, W_f)

            # Resize back to original spatial dims
            if binary_logits.shape[2] != h or binary_logits.shape[3] != w:
                binary_logits = F.interpolate(
                    binary_logits,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )

            # Convert to 2-class: [-logits, logits]
            two_class = self.decoder.binary_to_2class(binary_logits)  # (B, 2, H, W)
            slice_logits.append(two_class)

        # Stack along depth: (B, 2, D, H, W)
        logits_3d = torch.stack(slice_logits, dim=2)

        return self._build_output(logits_3d, "sam3_vanilla")

    def get_config(self) -> AdapterConfigInfo:
        """Return adapter configuration info."""
        return self._build_config(
            variant="vanilla",
            backbone="vit_32l",
            input_size=SAM3_INPUT_SIZE,
            encoder_frozen=True,
        )

    def trainable_parameters(self) -> int:
        """Count trainable parameters (decoder only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
