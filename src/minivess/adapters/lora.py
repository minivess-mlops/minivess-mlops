"""LoRA fine-tuning wrapper for ModelAdapter instances.

Uses PEFT (Parameter-Efficient Fine-Tuning) to apply Low-Rank
Adaptation to any ModelAdapter's underlying network, reducing
the number of trainable parameters for efficient fine-tuning.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from peft import LoraConfig, get_peft_model
from torch import Tensor

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class LoraModelAdapter(ModelAdapter):
    """LoRA wrapper that applies PEFT LoRA to any ModelAdapter.

    Parameters
    ----------
    base_model:
        An existing ModelAdapter to wrap with LoRA.
    lora_rank:
        Rank of the low-rank decomposition (default: 16).
    lora_alpha:
        LoRA scaling factor (default: 32.0).
    lora_dropout:
        Dropout probability for LoRA layers (default: 0.1).
    target_modules:
        Module names to apply LoRA to. If None, targets all
        nn.Linear and nn.Conv3d layers automatically.
    """

    def __init__(
        self,
        base_model: ModelAdapter,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        target_modules: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._base_model = base_model
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha
        self._lora_dropout = lora_dropout

        # Find target modules if not specified
        if target_modules is None:
            target_modules = self._find_target_modules(base_model)

        if not target_modules:
            logger.warning(
                "No suitable target modules found for LoRA; "
                "using base model without adaptation"
            )
            self._peft_model = None
            return

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )

        # Apply PEFT to the inner network
        self._peft_model = get_peft_model(base_model.net, lora_config)

        # Freeze base model parameters, only LoRA params trainable
        for name, param in self._peft_model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        n_trainable = sum(
            p.numel() for p in self._peft_model.parameters() if p.requires_grad
        )
        n_total = sum(p.numel() for p in self._peft_model.parameters())
        logger.info(
            "LoRA applied: %d/%d trainable params (%.1f%%)",
            n_trainable,
            n_total,
            100.0 * n_trainable / max(n_total, 1),
        )

    @staticmethod
    def _find_target_modules(model: ModelAdapter) -> list[str]:
        """Find linear and conv3d layer names suitable for LoRA."""
        targets: list[str] = []
        for name, module in model.net.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv3d)) and name:
                targets.append(name)
        return targets

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run forward pass through LoRA-adapted model.

        Note: Uses custom logic (PEFT model wrapping) rather than the
        standard _build_output helper for the non-PEFT fallback path.
        """
        if self._peft_model is not None:
            output = self._peft_model(images)
            logits = output[0] if isinstance(output, (list, tuple)) else output
        else:
            result = self._base_model(images, **kwargs)
            return result

        return self._build_output(logits, "lora_adapted")

    def get_config(self) -> AdapterConfigInfo:
        base_config = self._base_model.get_config()
        base_config.extras.update(
            {
                "lora_rank": self._lora_rank,
                "lora_alpha": self._lora_alpha,
                "lora_dropout": self._lora_dropout,
                "lora_applied": self._peft_model is not None,
            }
        )
        return base_config

    def load_checkpoint(self, path: Path) -> None:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if self._peft_model is not None:
            self._peft_model.load_state_dict(state_dict, strict=False)
        else:
            self._base_model.load_checkpoint(path)

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._peft_model is not None:
            # Save only LoRA adapter weights
            lora_state = {
                k: v for k, v in self._peft_model.named_parameters() if "lora_" in k
            }
            torch.save(lora_state, path)
        else:
            self._base_model.save_checkpoint(path)

    def trainable_parameters(self) -> int:
        if self._peft_model is not None:
            return sum(
                p.numel() for p in self._peft_model.parameters() if p.requires_grad
            )
        return self._base_model.trainable_parameters()

    def export_onnx(self, path: Path, example_input: Tensor) -> None:
        """Export LoRA-merged model to ONNX.

        Merges LoRA weights into the base model before export.
        """
        if self._peft_model is not None:
            merged = self._peft_model.merge_and_unload()
            path.parent.mkdir(parents=True, exist_ok=True)
            merged.eval()
            torch.onnx.export(
                merged,
                example_input,
                str(path),
                input_names=["images"],
                output_names=["logits"],
                opset_version=17,
                dynamo=False,
            )
        else:
            self._base_model.export_onnx(path, example_input)
