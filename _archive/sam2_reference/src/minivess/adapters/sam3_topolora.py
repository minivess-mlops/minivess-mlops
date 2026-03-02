"""Sam3TopoLoraAdapter — SAM2 with LoRA + topology-aware loss.

V2 of the SAM3 variants: tests whether topology-aware loss (cbdice_cldice)
combined with LoRA fine-tuning improves SAM2's vessel segmentation.

Based on TopoLoRA-SAM (arXiv:2601.02273) which showed +15% clDice with
LoRA + clDice loss. We use r=16, alpha=32 on encoder attention (q, v projections).

Expected improvement: +10-20% clDice over V1 (vanilla).

Reference: Ravi et al. (2024). "SAM 2." + Xiang et al. (2025). "TopoLoRA-SAM."
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from peft import LoraConfig, get_peft_model
from torch import Tensor

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput
from minivess.adapters.sam2_backbone import Sam2Backbone
from minivess.adapters.sam2_decoder import Sam2MaskDecoder
from minivess.adapters.slice_inference import slice_by_slice_forward, unresize_from_sam

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.config.models import ModelConfig

logger = logging.getLogger(__name__)


def _find_sam_attention_modules(backbone: Sam2Backbone) -> list[str]:
    """Find nn.Linear layers in the SAM2 Hiera encoder for LoRA.

    Targets attention projection layers (q_proj, v_proj style) which
    are nn.Linear in Hiera, NOT Conv3d.

    Parameters
    ----------
    backbone:
        Sam2Backbone instance to inspect.

    Returns
    -------
    List of module names suitable for LoRA target_modules.
    """
    targets: list[str] = []
    for name, module in backbone._encoder.named_modules():
        if isinstance(module, torch.nn.Linear) and name:
            targets.append(name)
    return targets


class _LoraSliceModel(torch.nn.Module):
    """Combines LoRA-adapted SAM2 backbone + decoder for per-slice processing."""

    def __init__(
        self,
        backbone_encoder: torch.nn.Module,
        decoder: Sam2MaskDecoder,
    ) -> None:
        super().__init__()
        self.encoder = backbone_encoder
        self.decoder = decoder

    def forward(self, slice_2d: Tensor) -> Tensor:
        """Process a single 2D slice through LoRA encoder + decoder."""
        original_h, original_w = slice_2d.shape[2], slice_2d.shape[3]

        # Replicate grayscale to 3 channels
        if slice_2d.shape[1] == 1:
            slice_3ch = slice_2d.expand(-1, 3, -1, -1)
        else:
            slice_3ch = slice_2d

        # Resize to 1024x1024 for SAM2
        import torch.nn.functional as F

        if slice_3ch.shape[2] != 1024 or slice_3ch.shape[3] != 1024:
            slice_3ch = F.interpolate(
                slice_3ch,
                size=(1024, 1024),
                mode="bilinear",
                align_corners=False,
            )

        # Encode with LoRA-adapted encoder
        features: Tensor = self.encoder(slice_3ch)

        # Decode
        logits_2d = self.decoder(features)

        # Unresize
        logits_2d = unresize_from_sam(logits_2d, original_h, original_w)

        return logits_2d


class Sam3TopoLoraAdapter(ModelAdapter):
    """SAM2 adapter with LoRA fine-tuning for topology-aware training.

    Applies PEFT LoRA to the SAM2 Hiera encoder's attention layers,
    enabling efficient fine-tuning with topology-aware losses like
    cbdice_cldice.

    Parameters
    ----------
    config:
        ModelConfig with SAM3_TOPOLORA family and LoRA hyperparameters.
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

        # Create backbone (initially frozen)
        self.backbone = Sam2Backbone(variant=variant, pretrained=pretrained)

        # Apply LoRA to encoder attention layers
        lora_rank = config.lora_rank
        lora_alpha = config.lora_alpha
        lora_dropout = config.lora_dropout

        target_modules = _find_sam_attention_modules(self.backbone)
        if target_modules:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
            )
            self._lora_encoder = get_peft_model(self.backbone._encoder, lora_config)
            # Freeze non-LoRA params, unfreeze LoRA params
            for name, param in self._lora_encoder.named_parameters():
                param.requires_grad = "lora_" in name
        else:
            logger.warning(
                "No Linear layers found for LoRA in SAM2 %s encoder; "
                "using backbone as-is",
                variant,
            )
            self._lora_encoder = self.backbone._encoder
            # Unfreeze all params since we can't do LoRA
            for param in self._lora_encoder.parameters():
                param.requires_grad = True

        # Create decoder (always trainable)
        self.decoder = Sam2MaskDecoder(
            embed_dim=self.backbone.out_channels,
        )

        self._variant = variant
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha

        # Combined model for slice-by-slice forward
        self._slice_model = _LoraSliceModel(self._lora_encoder, self.decoder)
        self.net = self._slice_model

        n_lora = sum(
            p.numel() for p in self._lora_encoder.parameters() if p.requires_grad
        )
        n_decoder = sum(p.numel() for p in self.decoder.parameters())
        logger.info(
            "Sam3TopoLoraAdapter: LoRA params=%d, decoder params=%d",
            n_lora,
            n_decoder,
        )

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run LoRA-adapted SAM2 inference slice-by-slice.

        Parameters
        ----------
        images:
            Input tensor (B, C, D, H, W).

        Returns
        -------
        SegmentationOutput with 2-class softmax predictions.
        """
        logits = slice_by_slice_forward(self._slice_model, images)
        return self._build_output(logits, "sam3_topolora")

    def get_config(self) -> AdapterConfigInfo:
        return self._build_config(
            variant=self._variant,
            lora_rank=self._lora_rank,
            lora_alpha=self._lora_alpha,
            encoder_lora=True,
            decoder_trainable=True,
        )

    def trainable_parameters(self) -> int:
        """LoRA encoder params + all decoder params."""
        n_lora = sum(
            p.numel() for p in self._lora_encoder.parameters() if p.requires_grad
        )
        n_decoder = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        return n_lora + n_decoder

    def save_checkpoint(self, path: Path) -> None:
        """Save LoRA weights + decoder weights (not frozen encoder base)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "lora_encoder": {
                k: v for k, v in self._lora_encoder.state_dict().items() if "lora_" in k
            },
            "decoder": self.decoder.state_dict(),
        }
        torch.save(state, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load LoRA + decoder weights."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "lora_encoder" in state:
            self._lora_encoder.load_state_dict(state["lora_encoder"], strict=False)
        if "decoder" in state:
            self.decoder.load_state_dict(state["decoder"])
