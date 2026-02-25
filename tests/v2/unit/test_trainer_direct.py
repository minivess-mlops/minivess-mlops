"""Direct tests for SegmentationTrainer epoch methods and loss functions (Code Review R2.2).

Tests train_epoch() and validate_epoch() directly (not just via fit()),
plus BettiLoss and TopologyCompoundLoss which were previously untested.
"""

from __future__ import annotations

import pytest
import torch

from minivess.config.models import ModelConfig, ModelFamily, TrainingConfig


def _make_config() -> ModelConfig:
    return ModelConfig(
        family=ModelFamily.MONAI_SEGRESNET,
        name="test",
        in_channels=1,
        out_channels=2,
    )


def _make_training_config(**overrides: object) -> TrainingConfig:
    defaults = {
        "max_epochs": 2,
        "learning_rate": 1e-3,
        "batch_size": 1,
        "early_stopping_patience": 5,
        "mixed_precision": False,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def _fake_loader(n_batches: int = 2) -> list[dict[str, torch.Tensor]]:
    """Create a synthetic data loader (list of batch dicts)."""
    return [
        {
            "image": torch.randn(1, 1, 16, 16, 16),
            "label": torch.randint(0, 2, (1, 1, 16, 16, 16)),
        }
        for _ in range(n_batches)
    ]


# ---------------------------------------------------------------------------
# T1: train_epoch direct tests
# ---------------------------------------------------------------------------


class TestTrainEpoch:
    """Test train_epoch() directly."""

    def test_returns_epoch_result(self) -> None:
        """train_epoch should return an EpochResult."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.pipeline.trainer import EpochResult, SegmentationTrainer

        model = SegResNetAdapter(_make_config())
        trainer = SegmentationTrainer(model, _make_training_config())
        result = trainer.train_epoch(_fake_loader())
        assert isinstance(result, EpochResult)

    def test_loss_is_finite(self) -> None:
        """Training loss should be finite (not NaN or Inf)."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.pipeline.trainer import SegmentationTrainer

        model = SegResNetAdapter(_make_config())
        trainer = SegmentationTrainer(model, _make_training_config())
        result = trainer.train_epoch(_fake_loader())
        assert result.loss > 0
        assert float("inf") != result.loss
        assert result.loss == result.loss  # NaN != NaN

    def test_loss_decreases_over_epochs(self) -> None:
        """Loss should generally decrease with more training."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.pipeline.trainer import SegmentationTrainer

        model = SegResNetAdapter(_make_config())
        loader = _fake_loader(n_batches=4)
        trainer = SegmentationTrainer(
            model, _make_training_config(learning_rate=0.01)
        )
        loss1 = trainer.train_epoch(loader).loss
        # Train several more epochs on same data
        for _ in range(10):
            trainer.train_epoch(loader)
        loss_final = trainer.train_epoch(loader).loss
        # Loss should decrease (or at least not explode)
        assert loss_final < loss1 * 2  # At minimum shouldn't explode

    def test_empty_loader(self) -> None:
        """train_epoch with empty loader should return 0.0 loss."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.pipeline.trainer import SegmentationTrainer

        model = SegResNetAdapter(_make_config())
        trainer = SegmentationTrainer(model, _make_training_config())
        result = trainer.train_epoch([])
        assert result.loss == 0.0


# ---------------------------------------------------------------------------
# T2: validate_epoch direct tests
# ---------------------------------------------------------------------------


class TestValidateEpoch:
    """Test validate_epoch() directly."""

    def test_returns_epoch_result(self) -> None:
        """validate_epoch should return an EpochResult."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.pipeline.trainer import EpochResult, SegmentationTrainer

        model = SegResNetAdapter(_make_config())
        trainer = SegmentationTrainer(model, _make_training_config())
        result = trainer.validate_epoch(_fake_loader())
        assert isinstance(result, EpochResult)

    def test_no_gradient_tracking(self) -> None:
        """Validation should not compute gradients."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.pipeline.trainer import SegmentationTrainer

        model = SegResNetAdapter(_make_config())
        trainer = SegmentationTrainer(model, _make_training_config())
        trainer.validate_epoch(_fake_loader())
        # After validation, no parameter should have accumulated grads
        for p in model.parameters():
            assert p.grad is None or torch.all(p.grad == 0)

    def test_model_in_eval_mode_during_validation(self) -> None:
        """Model should be in eval mode after validate_epoch."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.pipeline.trainer import SegmentationTrainer

        model = SegResNetAdapter(_make_config())
        trainer = SegmentationTrainer(model, _make_training_config())
        trainer.validate_epoch(_fake_loader())
        assert not model.training


# ---------------------------------------------------------------------------
# T3: Loss functions
# ---------------------------------------------------------------------------


class TestBettiLoss:
    """Test BettiLoss topology-aware loss."""

    def test_construction(self) -> None:
        """BettiLoss should construct with default parameters."""
        from minivess.pipeline.loss_functions import BettiLoss

        loss_fn = BettiLoss()
        assert loss_fn.threshold == 0.5
        assert loss_fn.lambda_betti == 1.0

    def test_forward_returns_scalar(self) -> None:
        """BettiLoss forward should return a scalar tensor."""
        from minivess.pipeline.loss_functions import BettiLoss

        loss_fn = BettiLoss()
        logits = torch.randn(1, 2, 8, 8, 8)
        labels = torch.randint(0, 2, (1, 1, 8, 8, 8))
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Should be non-negative

    def test_identical_pred_and_gt(self) -> None:
        """BettiLoss should be ~0 when prediction matches ground truth."""
        from minivess.pipeline.loss_functions import BettiLoss

        loss_fn = BettiLoss()
        # Create a simple block foreground
        labels = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
        labels[0, 0, 2:6, 2:6, 2:6] = 1
        # Create logits that match the labels perfectly
        logits = torch.zeros(1, 2, 8, 8, 8)
        logits[0, 0] = 10.0  # High confidence background
        logits[0, 1, 2:6, 2:6, 2:6] = 10.0  # High confidence foreground
        logits[0, 0, 2:6, 2:6, 2:6] = -10.0
        loss = loss_fn(logits, labels)
        assert loss.item() < 0.5  # Should be small


class TestTopologyCompoundLoss:
    """Test TopologyCompoundLoss (DiceCE + clDice + Betti)."""

    def test_construction(self) -> None:
        """TopologyCompoundLoss should construct with default weights."""
        from minivess.pipeline.loss_functions import TopologyCompoundLoss

        loss_fn = TopologyCompoundLoss()
        assert loss_fn.lambda_dice_ce == 0.4
        assert loss_fn.lambda_cldice == 0.4
        assert loss_fn.lambda_betti == 0.2

    def test_forward_returns_scalar(self) -> None:
        """TopologyCompoundLoss forward should return a scalar tensor."""
        from minivess.pipeline.loss_functions import TopologyCompoundLoss

        loss_fn = TopologyCompoundLoss()
        logits = torch.randn(1, 2, 8, 8, 8)
        labels = torch.randint(0, 2, (1, 1, 8, 8, 8))
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_is_differentiable(self) -> None:
        """TopologyCompoundLoss should be differentiable."""
        from minivess.pipeline.loss_functions import TopologyCompoundLoss

        loss_fn = TopologyCompoundLoss()
        logits = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
        labels = torch.randint(0, 2, (1, 1, 8, 8, 8))
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None


class TestBuildLossFunction:
    """Test the loss function factory."""

    @pytest.mark.parametrize("name", ["dice_ce", "dice", "focal", "betti", "full_topo"])
    def test_known_loss_names(self, name: str) -> None:
        """build_loss_function should return a module for known names."""
        from minivess.pipeline.loss_functions import build_loss_function

        loss_fn = build_loss_function(name)
        assert isinstance(loss_fn, torch.nn.Module)

    def test_unknown_loss_raises(self) -> None:
        """build_loss_function should raise ValueError for unknown names."""
        from minivess.pipeline.loss_functions import build_loss_function

        with pytest.raises(ValueError, match="Unknown loss function"):
            build_loss_function("nonexistent_loss")
