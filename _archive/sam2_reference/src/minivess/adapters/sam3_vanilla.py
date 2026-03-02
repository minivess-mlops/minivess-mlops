"""Sam3VanillaAdapter — frozen SAM2 encoder + trainable decoder.

V1 of the SAM3 variants: tests how badly vanilla SAM2 works on
3D microvessel segmentation. Uses slice-by-slice 2D inference with
null prompt embeddings (fully automatic, no user prompts).

Expected performance: DSC ~0.35-0.55, clDice ~0.3-0.5 (well below
DynUNet baseline of 0.824 DSC / 0.906 clDice).

Reference: Ravi et al. (2024). "SAM 2: Segment Anything in Images
and Videos." Meta FAIR.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput
from minivess.adapters.sam2_backbone import Sam2Backbone
from minivess.adapters.sam2_decoder import Sam2MaskDecoder
from minivess.adapters.slice_inference import slice_by_slice_forward, unresize_from_sam

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.config.models import ModelConfig

logger = logging.getLogger(__name__)


class _Sam2SliceModel(torch.nn.Module):
    """Combines SAM2 backbone + decoder for per-slice processing.

    Used internally by Sam3VanillaAdapter to enable slice_by_slice_forward
    to call a single model that handles resize → encode → decode → unresize.
    """

    def __init__(
        self,
        backbone: Sam2Backbone,
        decoder: Sam2MaskDecoder,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, slice_2d: Tensor) -> Tensor:
        """Process a single 2D slice.

        Parameters
        ----------
        slice_2d:
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        Logits tensor of shape (B, 2, H, W).
        """
        original_h, original_w = slice_2d.shape[2], slice_2d.shape[3]

        # Replicate grayscale to 3 channels for SAM2
        if slice_2d.shape[1] == 1:
            slice_3ch = slice_2d.expand(-1, 3, -1, -1)
        else:
            slice_3ch = slice_2d

        # Resize to SAM's expected 1024x1024
        import torch.nn.functional as F

        if slice_3ch.shape[2] != 1024 or slice_3ch.shape[3] != 1024:
            slice_3ch = F.interpolate(
                slice_3ch,
                size=(1024, 1024),
                mode="bilinear",
                align_corners=False,
            )

        # Encode
        features = self.backbone.extract_features(slice_3ch)

        # Decode (produces 2-class logits)
        logits_2d = self.decoder(features)

        # Unresize back to original spatial dims
        logits_2d = unresize_from_sam(logits_2d, original_h, original_w)

        return logits_2d


class Sam3VanillaAdapter(ModelAdapter):
    """Vanilla SAM2 adapter: frozen encoder + trainable decoder.

    Processes 3D volumes slice-by-slice through the SAM2 Hiera encoder
    and a lightweight trainable mask decoder with null prompts.

    Parameters
    ----------
    config:
        ModelConfig with SAM3_VANILLA family.
    variant:
        SAM2 Hiera variant (default: "hiera_tiny").
    pretrained:
        Whether to load official Meta SAM2 weights.
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

        self.backbone = Sam2Backbone(variant=variant, pretrained=pretrained)
        self.decoder = Sam2MaskDecoder(
            embed_dim=self.backbone.out_channels,
        )
        self._variant = variant

        # Combine backbone + decoder for slice-by-slice forward
        self._slice_model = _Sam2SliceModel(self.backbone, self.decoder)

        # Set the net attribute for base class compatibility
        self.net = self._slice_model

        n_encoder = sum(p.numel() for p in self.backbone.parameters())
        n_decoder = sum(p.numel() for p in self.decoder.parameters())
        logger.info(
            "Sam3VanillaAdapter: encoder=%d params (frozen), decoder=%d params (trainable)",
            n_encoder,
            n_decoder,
        )

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run SAM2 inference slice-by-slice on 3D volume.

        Parameters
        ----------
        images:
            Input tensor (B, C, D, H, W).

        Returns
        -------
        SegmentationOutput with 2-class softmax predictions.
        """
        logits = slice_by_slice_forward(self._slice_model, images)
        return self._build_output(logits, "sam3_vanilla")

    def get_config(self) -> AdapterConfigInfo:
        return self._build_config(
            variant=self._variant,
            encoder_frozen=True,
            decoder_trainable=True,
        )

    def trainable_parameters(self) -> int:
        """Only decoder parameters are trainable."""
        return sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)

    def save_checkpoint(self, path: Path) -> None:
        """Save decoder weights only (encoder is frozen/pretrained)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.decoder.state_dict(), path)

    def load_checkpoint(self, path: Path) -> None:
        """Load decoder weights."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.decoder.load_state_dict(state_dict)
