"""Integration smoke tests for SAM3 training loop (SAM-11).

Validates that each SAM3 adapter can:
1. Execute one training step (forward + backward)
2. Have gradients flow to trainable parameters
3. Work with AMP (mixed precision)
"""

from __future__ import annotations

import torch
from torch import nn

from minivess.config.models import ModelConfig, ModelFamily


def _one_training_step(adapter: nn.Module, volume: torch.Tensor, label: torch.Tensor):
    """Execute one training step and return loss + grad info."""
    adapter.train()
    output = adapter(volume)
    logits = output.logits

    # Simple cross-entropy-style loss (spatial softmax + NLL)
    loss = nn.functional.cross_entropy(logits, label)
    loss.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in adapter.parameters()
        if p.requires_grad
    )
    return loss.item(), has_grad


class TestSam3VanillaTrainingStep:
    """Sam3VanillaAdapter: one training step."""

    def test_training_step(self) -> None:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        config = ModelConfig(
            family=ModelFamily.SAM3_VANILLA,
            name="vanilla-integ",
            in_channels=1,
            out_channels=2,
        )
        adapter = Sam3VanillaAdapter(config)
        volume = torch.randn(1, 1, 2, 32, 32)
        label = torch.randint(0, 2, (1, 2, 32, 32))  # (B, D, H, W)
        loss_val, has_grad = _one_training_step(adapter, volume, label)
        assert loss_val > 0
        assert has_grad, "No gradients in trainable params"

    def test_amp_compatible(self) -> None:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        config = ModelConfig(
            family=ModelFamily.SAM3_VANILLA,
            name="vanilla-amp",
            in_channels=1,
            out_channels=2,
        )
        adapter = Sam3VanillaAdapter(config)
        adapter.train()
        volume = torch.randn(1, 1, 2, 32, 32)
        with torch.amp.autocast("cpu"):
            output = adapter(volume)
        assert output.logits.shape[1] == 2


class TestSam3TopoLoraTrainingStep:
    """Sam3TopoLoraAdapter: one training step."""

    def test_training_step(self) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        config = ModelConfig(
            family=ModelFamily.SAM3_TOPOLORA,
            name="topolora-integ",
            in_channels=1,
            out_channels=2,
            lora_rank=4,
            lora_alpha=8.0,
        )
        adapter = Sam3TopoLoraAdapter(config)
        volume = torch.randn(1, 1, 2, 32, 32)
        label = torch.randint(0, 2, (1, 2, 32, 32))
        loss_val, has_grad = _one_training_step(adapter, volume, label)
        assert loss_val > 0
        assert has_grad, "No gradients in LoRA/decoder params"

    def test_amp_compatible(self) -> None:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        config = ModelConfig(
            family=ModelFamily.SAM3_TOPOLORA,
            name="topolora-amp",
            in_channels=1,
            out_channels=2,
            lora_rank=4,
        )
        adapter = Sam3TopoLoraAdapter(config)
        adapter.train()
        volume = torch.randn(1, 1, 2, 32, 32)
        with torch.amp.autocast("cpu"):
            output = adapter(volume)
        assert output.logits.shape[1] == 2


class TestSam3HybridTrainingStep:
    """Sam3HybridAdapter: one training step."""

    def test_training_step(self) -> None:
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        config = ModelConfig(
            family=ModelFamily.SAM3_HYBRID,
            name="hybrid-integ",
            in_channels=1,
            out_channels=2,
            architecture_params={"filters": [16, 32, 64, 128]},
        )
        adapter = Sam3HybridAdapter(config)
        # D=8 for DynUNet skip connection alignment
        volume = torch.randn(1, 1, 8, 32, 32)
        label = torch.randint(0, 2, (1, 8, 32, 32))
        loss_val, has_grad = _one_training_step(adapter, volume, label)
        assert loss_val > 0
        assert has_grad, "No gradients in DynUNet/fusion params"

    def test_amp_compatible(self) -> None:
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        config = ModelConfig(
            family=ModelFamily.SAM3_HYBRID,
            name="hybrid-amp",
            in_channels=1,
            out_channels=2,
            architecture_params={"filters": [16, 32, 64, 128]},
        )
        adapter = Sam3HybridAdapter(config)
        adapter.train()
        volume = torch.randn(1, 1, 8, 32, 32)
        with torch.amp.autocast("cpu"):
            output = adapter(volume)
        assert output.logits.shape[1] == 2
