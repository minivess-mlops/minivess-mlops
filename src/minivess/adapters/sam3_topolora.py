"""Sam3TopoLoraAdapter — SAM3 + LoRA on FFN + topology-aware loss.

V2 of the SAM3 variants. Tests whether topology-aware loss (cbdice_cldice)
combined with LoRA fine-tuning improves SAM3 on microvessel segmentation.

Architecture:
    - SAM3 ViT-32L encoder with LoRA adapters on FFN (mlp.lin1, mlp.lin2)
    - Trainable mask decoder
    - LoRA: low-rank adaptation (r=16, alpha=32 default)
    - Loss: cbdice_cldice (topology-aware)

Based on: TopoLoRA-SAM (Khazem, 2026, arXiv:2601.02273)
Expected: +10-20% clDice over V1 (vanilla)
Go/No-Go Gate G2: clDice improvement >= 2% over V1.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput
from minivess.adapters.sam3_backbone import SAM3_INPUT_SIZE, Sam3Backbone
from minivess.adapters.sam3_decoder import Sam3MaskDecoder

if TYPE_CHECKING:
    from minivess.config.models import ModelConfig

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer wrapping an existing nn.Linear layer.

    Adds a low-rank decomposition: output = original(x) + (B @ A)(x) * scaling.
    SAM3 ViT-32L is pure Linear (no Conv2d in the transformer blocks), so only
    nn.Linear is supported. Conv2d LoRA would require a different decomposition.

    Parameters
    ----------
    original:
        The original nn.Linear layer to wrap.
    rank:
        Rank of the low-rank decomposition.
    alpha:
        Scaling factor (alpha / rank).
    dropout:
        Dropout probability for LoRA path.
    """

    def __init__(
        self,
        original: nn.Module,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(original, nn.Linear):
            msg = (
                f"LoRALinear only supports nn.Linear, got {type(original).__name__}. "
                f"SAM3 ViT-32L uses only Linear layers in its transformer blocks."
            )
            raise TypeError(msg)

        self.original = original
        self.rank = rank
        self.scaling = alpha / rank

        # Freeze the original layer
        for param in original.parameters():
            param.requires_grad = False

        in_features = original.in_features
        out_features = original.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def weight(self) -> Tensor:
        """Expose original layer's weight for HF code that accesses .weight.dtype."""
        return self.original.weight

    @property
    def bias(self) -> Tensor | None:
        """Expose original layer's bias for HF code that accesses .bias."""
        return self.original.bias

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: original(x) + LoRA(x).

        Handles FP16/FP32 dtype mismatch: SAM3 encoder runs in FP16
        but LoRA params are initialized in FP32. Cast input to match
        LoRA dtype for matmul, then cast result back to input dtype.
        """
        original_out: Tensor = self.original(x)
        input_dtype = x.dtype

        x_dropped = self.lora_dropout(x).to(self.lora_A.dtype)
        lora_out = (x_dropped @ self.lora_A.T @ self.lora_B.T) * self.scaling
        lora_out = lora_out.to(input_dtype)

        result: Tensor = original_out + lora_out
        return result


def _apply_lora_to_encoder(
    encoder: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    """Apply LoRA adapters to Linear layers in the encoder.

    SAM3 ViT-32L transformer blocks use only nn.Linear (no Conv2d in FFN).
    Conv2d layers (e.g., patch embedding) are skipped — LoRALinear only
    supports nn.Linear. See P2 issue for future LoRAConv2d implementation.

    Returns list of module names that received LoRA.
    """
    lora_targets: list[str] = []

    for name, module in list(encoder.named_modules()):
        # Target only nn.Linear layers — Conv2d is skipped (Glitch #9 fix).
        # LoRALinear raises TypeError on Conv2d; LoRAConv2d is a future P2 item.
        if isinstance(module, nn.Linear):
            # Skip very small layers (< rank features)
            if module.in_features < rank:
                continue

            lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)

            # Replace the module in the parent
            parts = name.split(".")
            parent = encoder
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], lora_layer)
            lora_targets.append(name)

    return lora_targets


class Sam3TopoLoraAdapter(ModelAdapter):
    """SAM3 + LoRA on encoder FFN + trainable decoder.

    Parameters
    ----------
    config:
        ModelConfig with ``SAM3_TOPOLORA`` family and LoRA params.
    """

    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self._lora_rank = config.lora_rank
        self._lora_alpha = config.lora_alpha

        # SAM3 backbone (NOT frozen yet — LoRA applied first)
        self.backbone = Sam3Backbone(config=config, freeze=False)

        # Apply LoRA to encoder, then freeze base weights
        lora_targets = _apply_lora_to_encoder(
            self.backbone.encoder,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        logger.info("LoRA applied to %d layers: %s", len(lora_targets), lora_targets)

        # Freeze FPN neck (no LoRA there)
        for param in self.backbone.fpn_neck.parameters():
            param.requires_grad = False

        # Trainable mask decoder
        self.decoder = Sam3MaskDecoder(config=config)

        trainable = self.trainable_parameters()
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "Sam3TopoLoraAdapter: %d trainable / %d total params (%.1f%%)",
            trainable,
            total,
            100 * trainable / max(total, 1),
        )

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Slice-by-slice forward with LoRA-adapted encoder.

        Parameters
        ----------
        images:
            Input 3D volume of shape (B, C, H, W, D) — MONAI dimension order.

        Returns
        -------
        SegmentationOutput with 2-class predictions.
        """
        # MONAI dimension order: (B, C, H, W, D) — depth is LAST
        b, c, h, w, d = images.shape
        slice_logits: list[Tensor] = []

        for z_idx in range(d):
            slice_2d = images[:, :, :, :, z_idx]  # (B, C, H, W)

            # FPN features (LoRA adapters are trainable, rest is frozen)
            fpn_features = self.backbone.extract_fpn_features(slice_2d)
            # Cast FP16 encoder output to FP32 for decoder (#680)
            fpn_features = fpn_features.float()

            # Decode to binary mask
            binary_logits = self.decoder(fpn_features)

            # Resize back to original spatial dims
            if binary_logits.shape[2] != h or binary_logits.shape[3] != w:
                binary_logits = F.interpolate(
                    binary_logits,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )

            two_class = self.decoder.binary_to_2class(binary_logits)
            slice_logits.append(two_class)

        logits_3d = torch.stack(slice_logits, dim=2)
        return self._build_output(logits_3d, "sam3_topolora")

    def get_config(self) -> AdapterConfigInfo:
        """Return adapter configuration info."""
        return self._build_config(
            variant="topolora",
            backbone="vit_32l",
            input_size=SAM3_INPUT_SIZE,
            lora_rank=self._lora_rank,
            lora_alpha=self._lora_alpha,
        )

    def trainable_parameters(self) -> int:
        """Count trainable parameters (LoRA + decoder)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
