"""Tests for SegmentationTrainer NaN handling in validation.

Phase 4 from docs/planning/sam3-nan-loss-fix.md:
  Validation pipeline hardening — validate_epoch should log NaN but not crash.
  Training loss NaN should still hard-fail (existing behavior preserved).

Issue: #715
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from minivess.adapters.base import ModelAdapter, SegmentationOutput
from minivess.config.models import TrainingConfig
from minivess.pipeline.trainer import SegmentationTrainer


class _DummyAdapter(ModelAdapter):
    """Minimal adapter for testing trainer NaN handling."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        b = images.shape[0]
        logits = torch.randn(b, 2, 4, 4, 3)
        return self._build_output(logits, "dummy")

    def get_config(self) -> Any:
        return self._build_config(variant="dummy")

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _NaNCriterion(nn.Module):
    """Loss function that always returns NaN."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return torch.tensor(float("nan"), device=pred.device)


class _FiniteCriterion(nn.Module):
    """Loss function that returns a finite value."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return torch.tensor(0.5, device=pred.device)


def _make_val_batch() -> list[dict[str, Tensor]]:
    """Create a minimal validation batch."""
    return [
        {
            "image": torch.randn(1, 1, 4, 4, 3),
            "label": torch.randint(0, 2, (1, 1, 4, 4, 3)).float(),
        }
    ]


def _make_trainer(
    criterion: nn.Module,
    *,
    mixed_precision: bool = False,
) -> SegmentationTrainer:
    """Create a SegmentationTrainer with injected criterion."""
    # MINIVESS_ALLOW_HOST is set by session-level fixture in conftest.py

    config = TrainingConfig(
        max_epochs=2,
        batch_size=1,
        learning_rate=1e-4,
        mixed_precision=mixed_precision,
        warmup_epochs=0,
        val_interval=1,
    )
    model = _DummyAdapter()

    return SegmentationTrainer(
        model=model,
        config=config,
        criterion=criterion,
        device="cpu",
    )


class TestValidateEpochNaNHandling:
    """validate_epoch should handle NaN loss gracefully."""

    def test_nan_loss_does_not_raise(self) -> None:
        """validate_epoch should return NaN loss, not raise an exception."""
        trainer = _make_trainer(_NaNCriterion())
        loader = _make_val_batch()

        # Should NOT raise — NaN in validation is logged, not fatal
        result = trainer.validate_epoch(loader)

        assert math.isnan(result.loss), f"Expected NaN loss, got {result.loss}"

    def test_nan_loss_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """validate_epoch should log a warning when val_loss is NaN."""
        trainer = _make_trainer(_NaNCriterion())
        loader = _make_val_batch()

        with caplog.at_level(logging.WARNING, logger="minivess.pipeline.trainer"):
            trainer.validate_epoch(loader)

        assert any(
            "non-finite" in record.message.lower() or "nan" in record.message.lower()
            for record in caplog.records
        ), f"Expected NaN warning, got: {[r.message for r in caplog.records]}"

    def test_finite_loss_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """validate_epoch should not warn when loss is finite."""
        trainer = _make_trainer(_FiniteCriterion())
        loader = _make_val_batch()

        with caplog.at_level(logging.WARNING, logger="minivess.pipeline.trainer"):
            trainer.validate_epoch(loader)

        nan_warnings = [
            r
            for r in caplog.records
            if "non-finite" in r.message.lower() or "nan" in r.message.lower()
        ]
        assert not nan_warnings, f"Unexpected NaN warning: {nan_warnings}"

    def test_finite_loss_returns_correct_value(self) -> None:
        """validate_epoch should return the correct finite loss."""
        trainer = _make_trainer(_FiniteCriterion())
        loader = _make_val_batch()

        result = trainer.validate_epoch(loader)

        assert abs(result.loss - 0.5) < 1e-5, f"Expected 0.5 loss, got {result.loss}"


class TestTrainEpochNaNStillFails:
    """Training NaN should still hard-fail (preserve existing guard)."""

    def test_train_nan_loss_raises(self) -> None:
        """train_epoch should raise ValueError on NaN training loss."""
        trainer = _make_trainer(_NaNCriterion())
        loader = _make_val_batch()  # Same format works for train

        with pytest.raises(ValueError, match="Non-finite loss"):
            trainer.train_epoch(loader)
