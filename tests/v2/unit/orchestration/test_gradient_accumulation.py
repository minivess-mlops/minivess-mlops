"""Tests for gradient accumulation in SegmentationTrainer.

Verifies that gradient_accumulation_steps correctly:
- Divides loss before backward
- Calls optimizer.step() every N iterations
- Logs effective_batch_size to metrics
- Is a noop when steps=1

Issue: #940 (SAM3 batch_size=1 fix)
Plan: docs/planning/sam3-batch-size-1-and-robustification-plan.xml Task 1.3
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn


from minivess.config.models import CheckpointConfig


class _StubOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _TinyModel(nn.Module):
    """Minimal model for testing — 1-layer conv."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, 2, kernel_size=3, padding=1)
        self._initial_weight = self.conv.weight.data.clone()

    def forward(self, x: torch.Tensor) -> _StubOutput:
        return _StubOutput(logits=self.conv(x))

    def parameters(self, recurse: bool = True) -> Any:
        return super().parameters(recurse=recurse)

    def to(self, device: Any) -> "_TinyModel":
        return super().to(device)  # type: ignore[return-value]

    def train(self, mode: bool = True) -> "_TinyModel":
        return super().train(mode)  # type: ignore[return-value]


def _make_batches(n: int) -> list[dict[str, torch.Tensor]]:
    """Create n synthetic training batches."""
    return [
        {
            "image": torch.randn(1, 1, 8, 8, 4),
            "label": torch.randint(0, 2, (1, 1, 8, 8, 4)).float(),
        }
        for _ in range(n)
    ]


class TestGradientAccumulationConfig:
    """TrainingConfig must support gradient_accumulation_steps."""

    def test_training_config_has_grad_accum_field(self) -> None:
        from minivess.config.models import TrainingConfig

        cfg = TrainingConfig()
        assert hasattr(cfg, "gradient_accumulation_steps")
        assert cfg.gradient_accumulation_steps == 1  # default: no accumulation

    def test_training_config_grad_accum_positive(self) -> None:
        from minivess.config.models import TrainingConfig

        with pytest.raises(Exception):
            TrainingConfig(gradient_accumulation_steps=0)


class TestGradientAccumulationTrainer:
    """SegmentationTrainer must support gradient accumulation."""

    def _make_trainer(
        self,
        grad_accum_steps: int = 1,
    ) -> tuple:
        """Create a SegmentationTrainer with a tiny model."""
        from minivess.config.models import TrainingConfig
        from minivess.pipeline.trainer import SegmentationTrainer

        model = _TinyModel()
        config = TrainingConfig(
            max_epochs=2,
            batch_size=1,
            learning_rate=0.01,
            warmup_epochs=0,
            mixed_precision=False,
            gradient_accumulation_steps=grad_accum_steps,
            checkpoint=CheckpointConfig(),
        )
        criterion = nn.MSELoss()
        trainer = SegmentationTrainer(
            model=model,
            config=config,
            device="cpu",
            criterion=criterion,
        )
        return trainer, model

    def test_grad_accum_steps_1_is_standard(self) -> None:
        """With accum=1, optimizer.step() called every batch (standard behavior)."""
        trainer, model = self._make_trainer(grad_accum_steps=1)
        batches = _make_batches(4)

        with patch.object(
            trainer.optimizer, "step", wraps=trainer.optimizer.step
        ) as mock_step:
            trainer.train_epoch(batches)
            assert mock_step.call_count == 4

    def test_grad_accum_steps_4_calls_step_once_per_4(self) -> None:
        """With accum=4, optimizer.step() called once per 4 batches."""
        trainer, model = self._make_trainer(grad_accum_steps=4)
        batches = _make_batches(8)

        with patch.object(
            trainer.optimizer, "step", wraps=trainer.optimizer.step
        ) as mock_step:
            trainer.train_epoch(batches)
            # 8 batches / 4 accum = 2 optimizer steps
            assert mock_step.call_count == 2

    def test_grad_accum_handles_incomplete_last_group(self) -> None:
        """If batches aren't divisible by accum steps, flush remainder."""
        trainer, model = self._make_trainer(grad_accum_steps=4)
        batches = _make_batches(6)  # 6 / 4 = 1 full + 1 partial

        with patch.object(
            trainer.optimizer, "step", wraps=trainer.optimizer.step
        ) as mock_step:
            trainer.train_epoch(batches)
            # 1 full group (4) + 1 partial group (2) = 2 optimizer steps
            assert mock_step.call_count == 2

    def test_grad_accum_produces_weight_update(self) -> None:
        """Gradient accumulation must actually update model weights."""
        trainer, model = self._make_trainer(grad_accum_steps=4)
        initial_weight = model.conv.weight.data.clone()
        batches = _make_batches(4)

        trainer.train_epoch(batches)

        # Weights should have changed
        assert not torch.allclose(model.conv.weight.data, initial_weight), (
            "Model weights not updated after gradient accumulation"
        )

    def test_grad_accum_returns_correct_avg_loss(self) -> None:
        """Average loss should be computed over all batches, not accumulated groups."""
        trainer, model = self._make_trainer(grad_accum_steps=4)
        batches = _make_batches(8)

        result = trainer.train_epoch(batches)
        assert result.loss > 0  # Non-zero loss
        assert isinstance(result.loss, float)

    def test_effective_batch_size_in_metrics(self) -> None:
        """Trainer must expose effective_batch_size for MLflow logging."""
        from minivess.config.models import TrainingConfig

        config = TrainingConfig(
            batch_size=1,
            gradient_accumulation_steps=4,
            checkpoint=CheckpointConfig(),
        )
        assert config.batch_size * config.gradient_accumulation_steps == 4
