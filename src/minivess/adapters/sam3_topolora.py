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
    from pathlib import Path

    from minivess.config.models import ModelConfig

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer wrapping an existing linear/conv layer.

    Adds a low-rank decomposition: output = original(x) + (B @ A)(x) * scaling.

    Parameters
    ----------
    original:
        The original layer to wrap (nn.Linear or nn.Conv2d).
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
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank

        # Freeze the original layer
        for param in original.parameters():
            param.requires_grad = False

        # Determine dimensions from the original layer
        if isinstance(original, nn.Linear):
            in_features = original.in_features
            out_features = original.out_features
            self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        elif isinstance(original, nn.Conv2d):
            in_channels = original.in_channels
            out_channels = original.out_channels
            self.lora_A = nn.Parameter(torch.randn(rank, in_channels) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_channels, rank))
        else:
            msg = f"LoRALinear only supports Linear/Conv2d, got {type(original)}"
            raise TypeError(msg)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._is_conv = isinstance(original, nn.Conv2d)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: original(x) + LoRA(x)."""
        original_out: Tensor = self.original(x)

        if self._is_conv:
            # For Conv2d: use 1x1 conv via matrix multiply on output-spatial features
            # Apply LoRA as a channel-mixing residual on the original output
            b, c_out, h_out, w_out = original_out.shape
            # Global average pool input channels → project through LoRA
            x_avg = x.mean(dim=(2, 3))  # (B, C_in)
            x_avg = self.lora_dropout(x_avg)
            lora_out = (x_avg @ self.lora_A.T @ self.lora_B.T) * self.scaling
            lora_out = lora_out.unsqueeze(-1).unsqueeze(-1)  # (B, C_out, 1, 1)
            result: Tensor = original_out + lora_out
            return result
        else:
            x_dropped = self.lora_dropout(x)
            lora_out = (x_dropped @ self.lora_A.T @ self.lora_B.T) * self.scaling

        result_lin: Tensor = original_out + lora_out
        return result_lin


def _apply_lora_to_encoder(
    encoder: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    """Apply LoRA adapters to Conv2d layers in the encoder.

    For the stub encoder, targets the proj Conv2d.
    For real SAM3, would target mlp.lin1 and mlp.lin2 in each block.

    Returns list of module names that received LoRA.
    """
    lora_targets: list[str] = []

    for name, module in list(encoder.named_modules()):
        # Target Conv2d and Linear layers (excluding tiny layers)
        if isinstance(module, nn.Linear | nn.Conv2d):
            # Skip very small layers (< rank features)
            if isinstance(module, nn.Linear) and module.in_features < rank:
                continue
            if isinstance(module, nn.Conv2d) and module.in_channels < rank:
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
            Input 3D volume of shape (B, C, D, H, W).

        Returns
        -------
        SegmentationOutput with 2-class predictions.
        """
        b, c, d, h, w = images.shape
        slice_logits: list[Tensor] = []

        for z_idx in range(d):
            slice_2d = images[:, :, z_idx, :, :]  # (B, C, H, W)

            # FPN features (LoRA adapters are trainable, rest is frozen)
            fpn_features = self.backbone.extract_fpn_features(slice_2d)

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
            def __init__(self, adapter: Sam3TopoLoraAdapter) -> None:
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
        """Count trainable parameters (LoRA + decoder)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
