from __future__ import annotations

from unittest.mock import MagicMock

import torch
from monai.losses import DiceCELoss

from minivess.adapters.base import SegmentationOutput
from minivess.config.models import TrainingConfig
from minivess.pipeline.metrics import SegmentationMetrics
from minivess.pipeline.trainer import EpochResult, SegmentationTrainer


def _make_fake_model(num_classes: int = 2) -> MagicMock:
    """Create a mock ModelAdapter that returns random logits."""
    model = MagicMock()
    model.parameters.return_value = [torch.randn(2, 2, requires_grad=True)]
    model.train.return_value = None
    model.eval.return_value = None
    model.to.return_value = model

    def _forward(images: torch.Tensor) -> SegmentationOutput:
        b = images.shape[0]
        logits = torch.randn(b, num_classes, 4, 4, 4, requires_grad=True)
        pred = torch.softmax(logits, dim=1)
        return SegmentationOutput(prediction=pred, logits=logits, metadata={})

    model.side_effect = _forward
    model.__call__ = _forward
    return model


def _make_loader(num_batches: int = 2, batch_size: int = 1) -> list[dict[str, torch.Tensor]]:
    """Create a fake data loader (list of batch dicts)."""
    return [
        {
            "image": torch.randn(batch_size, 1, 4, 4, 4),
            "label": torch.randint(0, 2, (batch_size, 1, 4, 4, 4)),
        }
        for _ in range(num_batches)
    ]


def _make_config(**overrides) -> TrainingConfig:
    """Create a minimal TrainingConfig for testing."""
    defaults = {
        "max_epochs": 2,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "warmup_epochs": 0,
        "mixed_precision": False,
        "gradient_clip_val": 0.0,
        "early_stopping_patience": 10,
        "num_folds": 2,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


class TestTrainerWithMetrics:
    """Tests for SegmentationTrainer with injected SegmentationMetrics."""

    def test_train_epoch_returns_metrics(self) -> None:
        model = _make_fake_model()
        config = _make_config()
        metrics = SegmentationMetrics(num_classes=2, device="cpu")
        criterion = DiceCELoss(softmax=True, to_onehot_y=True)

        trainer = SegmentationTrainer(
            model,
            config,
            device="cpu",
            metrics=metrics,
            criterion=criterion,
        )
        loader = _make_loader(num_batches=2)
        result = trainer.train_epoch(loader)

        assert isinstance(result, EpochResult)
        assert len(result.metrics) > 0, "Metrics should be populated"
        assert "dice" in result.metrics

    def test_validate_epoch_returns_metrics(self) -> None:
        model = _make_fake_model()
        config = _make_config()
        metrics = SegmentationMetrics(num_classes=2, device="cpu")
        criterion = DiceCELoss(softmax=True, to_onehot_y=True)

        trainer = SegmentationTrainer(
            model,
            config,
            device="cpu",
            metrics=metrics,
            criterion=criterion,
        )
        loader = _make_loader(num_batches=2)
        result = trainer.validate_epoch(loader)

        assert isinstance(result, EpochResult)
        assert len(result.metrics) > 0
        assert "dice" in result.metrics

    def test_metrics_reset_between_epochs(self) -> None:
        model = _make_fake_model()
        config = _make_config()
        metrics = SegmentationMetrics(num_classes=2, device="cpu")
        criterion = DiceCELoss(softmax=True, to_onehot_y=True)

        trainer = SegmentationTrainer(
            model,
            config,
            device="cpu",
            metrics=metrics,
            criterion=criterion,
        )
        loader = _make_loader(num_batches=2)

        r1 = trainer.train_epoch(loader)
        r2 = trainer.train_epoch(loader)

        # Both should have metrics (reset worked - no accumulation error)
        assert len(r1.metrics) > 0
        assert len(r2.metrics) > 0

    def test_no_metrics_when_not_injected(self) -> None:
        model = _make_fake_model()
        config = _make_config()
        criterion = DiceCELoss(softmax=True, to_onehot_y=True)

        trainer = SegmentationTrainer(
            model,
            config,
            device="cpu",
            criterion=criterion,
        )
        loader = _make_loader(num_batches=2)
        result = trainer.train_epoch(loader)

        assert result.metrics == {}

    def test_fit_logs_metrics_to_tracker(self) -> None:
        model = _make_fake_model()
        config = _make_config(max_epochs=2, early_stopping_patience=10)
        metrics = SegmentationMetrics(num_classes=2, device="cpu")
        criterion = DiceCELoss(softmax=True, to_onehot_y=True)
        tracker = MagicMock()

        trainer = SegmentationTrainer(
            model,
            config,
            device="cpu",
            metrics=metrics,
            criterion=criterion,
            tracker=tracker,
        )
        train_loader = _make_loader(num_batches=2)
        val_loader = _make_loader(num_batches=2)

        trainer.fit(train_loader, val_loader)

        # Tracker should have been called with metric values including train_dice, val_dice
        assert tracker.log_epoch_metrics.call_count == 2
        first_call_metrics = tracker.log_epoch_metrics.call_args_list[0][0][0]
        assert "train_dice" in first_call_metrics
        assert "val_dice" in first_call_metrics
