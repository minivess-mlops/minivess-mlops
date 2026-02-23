from __future__ import annotations

import pytest
import torch

from minivess.adapters.segresnet import SegResNetAdapter
from minivess.config.models import ModelConfig, ModelFamily, TrainingConfig
from minivess.pipeline.loss_functions import build_loss_function
from minivess.pipeline.metrics import MetricResult, SegmentationMetrics
from minivess.pipeline.trainer import EpochResult, SegmentationTrainer


class TestLossFunctions:
    """Test loss function factory."""

    def test_dice_ce(self) -> None:
        loss = build_loss_function("dice_ce")
        pred = torch.randn(2, 2, 8, 8, 4)
        target = torch.randint(0, 2, (2, 1, 8, 8, 4))
        result = loss(pred, target)
        assert result.ndim == 0  # scalar

    def test_dice(self) -> None:
        loss = build_loss_function("dice")
        pred = torch.randn(2, 2, 8, 8, 4)
        target = torch.randint(0, 2, (2, 1, 8, 8, 4))
        result = loss(pred, target)
        assert result.ndim == 0

    def test_focal(self) -> None:
        loss = build_loss_function("focal")
        pred = torch.randn(2, 2, 8, 8, 4)
        target = torch.randint(0, 2, (2, 1, 8, 8, 4))
        result = loss(pred, target)
        assert result.ndim == 0

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown loss"):
            build_loss_function("unknown")

    def test_dice_ce_is_finite(self) -> None:
        loss = build_loss_function("dice_ce")
        pred = torch.randn(2, 2, 8, 8, 4)
        target = torch.randint(0, 2, (2, 1, 8, 8, 4))
        result = loss(pred, target)
        assert torch.isfinite(result)

    def test_loss_positive(self) -> None:
        """Loss values should be non-negative."""
        for name in ("dice_ce", "dice", "focal"):
            loss = build_loss_function(name)
            pred = torch.randn(2, 2, 8, 8, 4)
            target = torch.randint(0, 2, (2, 1, 8, 8, 4))
            result = loss(pred, target)
            assert result.item() >= 0.0, f"{name} loss was negative"

    def test_custom_num_classes(self) -> None:
        loss = build_loss_function("dice_ce", num_classes=4)
        pred = torch.randn(2, 4, 8, 8, 4)
        target = torch.randint(0, 4, (2, 1, 8, 8, 4))
        result = loss(pred, target)
        assert result.ndim == 0

    def test_perfect_prediction_low_dice_loss(self) -> None:
        """A perfect prediction should yield near-zero dice loss."""
        loss = build_loss_function("dice", softmax=False, to_onehot_y=False)
        # Create one-hot prediction that matches target perfectly
        target = torch.zeros(1, 2, 8, 8, 4)
        target[:, 1] = 1.0  # all foreground
        pred = target.clone()
        result = loss(pred, target)
        assert result.item() < 0.1


class TestSegmentationMetrics:
    """Test TorchMetrics wrapper."""

    def test_binary_metrics(self) -> None:
        metrics = SegmentationMetrics(num_classes=2)
        pred = torch.zeros(2, 2, 8, 8, 4)
        pred[:, 1] = 1.0  # all foreground
        target = torch.ones(2, 8, 8, 4, dtype=torch.long)
        metrics.update(pred, target)
        result = metrics.compute()
        assert isinstance(result, MetricResult)
        assert "dice" in result.values
        assert result.values["dice"] > 0.0

    def test_perfect_prediction_dice_one(self) -> None:
        """Perfect prediction should yield dice = 1.0."""
        metrics = SegmentationMetrics(num_classes=2)
        pred = torch.zeros(2, 2, 8, 8, 4)
        pred[:, 1] = 1.0  # all foreground
        target = torch.ones(2, 8, 8, 4, dtype=torch.long)
        metrics.update(pred, target)
        result = metrics.compute()
        assert result.values["dice"] == pytest.approx(1.0, abs=1e-5)

    def test_f1_foreground_present_for_binary(self) -> None:
        metrics = SegmentationMetrics(num_classes=2)
        pred = torch.zeros(2, 2, 8, 8, 4)
        pred[:, 1] = 1.0
        target = torch.ones(2, 8, 8, 4, dtype=torch.long)
        metrics.update(pred, target)
        result = metrics.compute()
        assert "f1_foreground" in result.values

    def test_reset(self) -> None:
        metrics = SegmentationMetrics(num_classes=2)
        pred = torch.zeros(2, 2, 8, 8, 4)
        pred[:, 1] = 1.0
        target = torch.ones(2, 8, 8, 4, dtype=torch.long)
        metrics.update(pred, target)
        metrics.reset()
        # After reset, computing without new data should raise
        with pytest.raises((ValueError, RuntimeError)):
            metrics.compute()

    def test_to_dict(self) -> None:
        result = MetricResult(values={"dice": 0.9, "f1": 0.85})
        d = result.to_dict()
        assert d == {"dice": 0.9, "f1": 0.85}

    def test_handles_5d_target(self) -> None:
        """Target with shape (B, 1, D, H, W) should be handled correctly."""
        metrics = SegmentationMetrics(num_classes=2)
        pred = torch.zeros(2, 2, 8, 8, 4)
        pred[:, 1] = 1.0
        target = torch.ones(2, 1, 8, 8, 4, dtype=torch.long)
        metrics.update(pred, target)
        result = metrics.compute()
        assert result.values["dice"] == pytest.approx(1.0, abs=1e-5)

    def test_multiple_updates(self) -> None:
        """Metrics should accumulate across multiple update calls."""
        metrics = SegmentationMetrics(num_classes=2)
        for _ in range(3):
            pred = torch.zeros(1, 2, 8, 8, 4)
            pred[:, 1] = 1.0
            target = torch.ones(1, 8, 8, 4, dtype=torch.long)
            metrics.update(pred, target)
        result = metrics.compute()
        assert result.values["dice"] == pytest.approx(1.0, abs=1e-5)


class TestEpochResult:
    """Test EpochResult dataclass."""

    def test_creation(self) -> None:
        result = EpochResult(loss=0.5, metrics={"dice": 0.85})
        assert result.loss == 0.5
        assert result.metrics["dice"] == 0.85

    def test_default_metrics(self) -> None:
        result = EpochResult(loss=1.0)
        assert result.metrics == {}

    def test_default_metrics_independent(self) -> None:
        """Default dicts should not be shared across instances."""
        r1 = EpochResult(loss=1.0)
        r2 = EpochResult(loss=2.0)
        r1.metrics["key"] = "value"
        assert "key" not in r2.metrics


class TestSegmentationTrainer:
    """Test training engine (CPU only, tiny model)."""

    @pytest.fixture
    def trainer(self) -> SegmentationTrainer:
        config = TrainingConfig(
            max_epochs=2,
            batch_size=1,
            learning_rate=1e-3,
            mixed_precision=False,  # CPU
            early_stopping_patience=5,
            warmup_epochs=1,
        )
        model_config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="test",
            in_channels=1,
            out_channels=2,
        )
        model = SegResNetAdapter(model_config)
        return SegmentationTrainer(model, config, device="cpu")

    def test_build_optimizer_adamw(self, trainer: SegmentationTrainer) -> None:
        assert trainer.optimizer.__class__.__name__ == "AdamW"

    def test_build_scheduler(self, trainer: SegmentationTrainer) -> None:
        assert trainer.scheduler is not None
        assert trainer.scheduler.__class__.__name__ == "SequentialLR"

    def test_build_optimizer_sgd(self) -> None:
        config = TrainingConfig(
            max_epochs=2,
            learning_rate=1e-3,
            mixed_precision=False,
            optimizer="sgd",
        )
        model_config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="test-sgd",
            in_channels=1,
            out_channels=2,
        )
        model = SegResNetAdapter(model_config)
        trainer = SegmentationTrainer(model, config, device="cpu")
        assert trainer.optimizer.__class__.__name__ == "SGD"

    def test_unknown_optimizer_raises(self) -> None:
        """Building a trainer with an invalid optimizer should raise."""
        config = TrainingConfig(
            max_epochs=2,
            learning_rate=1e-3,
            mixed_precision=False,
            optimizer="lamb",  # valid in config but not in trainer
        )
        model_config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="test-bad",
            in_channels=1,
            out_channels=2,
        )
        model = SegResNetAdapter(model_config)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            SegmentationTrainer(model, config, device="cpu")

    def test_criterion_is_dice_ce(self, trainer: SegmentationTrainer) -> None:
        assert trainer.criterion.__class__.__name__ == "DiceCELoss"

    def test_device_assignment(self, trainer: SegmentationTrainer) -> None:
        assert trainer.device == torch.device("cpu")

    def test_train_epoch_with_synthetic_data(
        self, trainer: SegmentationTrainer
    ) -> None:
        """Run a single training epoch with synthetic data."""
        # Create a minimal fake loader (list of dicts)
        batch = {
            "image": torch.randn(1, 1, 32, 32, 16),
            "label": torch.randint(0, 2, (1, 1, 32, 32, 16)),
        }
        loader = [batch]
        result = trainer.train_epoch(loader)
        assert isinstance(result, EpochResult)
        assert result.loss > 0.0
        assert torch.isfinite(torch.tensor(result.loss))

    def test_validate_epoch_with_synthetic_data(
        self, trainer: SegmentationTrainer
    ) -> None:
        """Run a single validation epoch with synthetic data."""
        batch = {
            "image": torch.randn(1, 1, 32, 32, 16),
            "label": torch.randint(0, 2, (1, 1, 32, 32, 16)),
        }
        loader = [batch]
        result = trainer.validate_epoch(loader)
        assert isinstance(result, EpochResult)
        assert result.loss > 0.0

    def test_fit_returns_summary(self, trainer: SegmentationTrainer) -> None:
        """Full fit loop should return a summary dict."""
        batch = {
            "image": torch.randn(1, 1, 32, 32, 16),
            "label": torch.randint(0, 2, (1, 1, 32, 32, 16)),
        }
        train_loader = [batch]
        val_loader = [batch]
        summary = trainer.fit(train_loader, val_loader)
        assert "best_val_loss" in summary
        assert "final_epoch" in summary
        assert "history" in summary
        assert summary["final_epoch"] == 2  # max_epochs=2
        assert len(summary["history"]["train_loss"]) == 2
        assert len(summary["history"]["val_loss"]) == 2
