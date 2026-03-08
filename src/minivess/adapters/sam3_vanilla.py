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

IMPORTANT: Real pretrained SAM3 weights are required (GPU VRAM ≥6 GB, frozen encoder).
No stub/fallback mode exists — use pytest.mark.skipif for CI tests.

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
    from pathlib import Path

    from minivess.config.models import ModelConfig

logger = logging.getLogger(__name__)


class Sam3VanillaAdapter(ModelAdapter):
    """Frozen SAM3 encoder + trainable decoder for segmentation.

    Uses SDPA (Scaled Dot-Product Attention) to avoid materializing
    5184×5184 attention matrices, reducing encoder peak VRAM from ~7 GB
    to ~1.1 GB. This makes training feasible on 8 GB consumer GPUs.

    Parameters
    ----------
    config:
        ModelConfig with ``SAM3_VANILLA`` family.
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config

        # Frozen SAM3 backbone (encoder + FPN neck)
        self.backbone = Sam3Backbone(config=config, freeze=True)

        # Trainable mask decoder
        self.decoder = Sam3MaskDecoder(config=config)

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
            Input 3D volume of shape (B, C, H, W, D) — MONAI convention
            where D (depth/Z) is the last spatial dimension.

        Returns
        -------
        SegmentationOutput with 2-class predictions.
        """
        b, c, h, w, d = images.shape
        slice_logits: list[Tensor] = []

        for z_idx in range(d):
            slice_2d = images[:, :, :, :, z_idx]  # (B, C, H, W)

            # Extract FPN features (frozen encoder with SDPA, FP16)
            fpn_features = self.backbone.extract_fpn_features(slice_2d)

            # Cast to FP32 for trainable decoder (encoder outputs FP16)
            fpn_features = fpn_features.float()

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

        # Stack along depth (last dim): (B, 2, H, W, D) — MONAI convention
        logits_3d = torch.stack(slice_logits, dim=4)

        return self._build_output(logits_3d, "sam3_vanilla")

    def get_eval_roi_size(self) -> tuple[int, int, int]:
        """Return the sliding-window ROI size for full-volume evaluation.

        SAM3 ViT-32L resizes every input to 1008×1008 regardless of spatial
        size, so larger ROI windows cost the same encoder FLOPS as small ones.
        Using (512, 512, 3) reduces windows from ~3300 to ~27 per 512×512×61
        volume, cutting evaluation time from ~6 hours to ~4 minutes.
        """
        return (512, 512, 3)

    def get_config(self) -> AdapterConfigInfo:
        """Return adapter configuration info."""
        return self._build_config(
            variant="vanilla",
            backbone="vit_32l",
            input_size=SAM3_INPUT_SIZE,
            encoder_frozen=True,
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
            def __init__(self, adapter: Sam3VanillaAdapter) -> None:
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
        """Count trainable parameters (decoder only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
