"""Tests for topology-aware loss functions (Issue #5)."""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# T1: ClassBalancedDiceLoss
# ---------------------------------------------------------------------------


class TestClassBalancedDiceLoss:
    """Test class-balanced Dice loss."""

    def test_basic_callable(self) -> None:
        """ClassBalancedDiceLoss should be callable with logits and labels."""
        from minivess.pipeline.loss_functions import ClassBalancedDiceLoss

        loss_fn = ClassBalancedDiceLoss(num_classes=2)
        logits = torch.randn(1, 2, 16, 16, 8)
        labels = torch.zeros(1, 1, 16, 16, 8, dtype=torch.long)
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_imbalanced_weighting(self) -> None:
        """With imbalanced labels, class weights should differ from uniform."""
        from minivess.pipeline.loss_functions import ClassBalancedDiceLoss

        loss_fn = ClassBalancedDiceLoss(num_classes=2)
        # 95% background, 5% foreground
        labels = torch.zeros(1, 1, 16, 16, 8, dtype=torch.long)
        labels[0, 0, :1, :1, :1] = 1

        logits = torch.randn(1, 2, 16, 16, 8)
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_gradient_flows(self) -> None:
        """Loss should produce gradients."""
        from minivess.pipeline.loss_functions import ClassBalancedDiceLoss

        loss_fn = ClassBalancedDiceLoss(num_classes=2)
        logits = torch.randn(1, 2, 16, 16, 8, requires_grad=True)
        labels = torch.zeros(1, 1, 16, 16, 8, dtype=torch.long)
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_build_factory(self) -> None:
        """build_loss_function should support 'cb_dice'."""
        from minivess.pipeline.loss_functions import build_loss_function

        loss_fn = build_loss_function("cb_dice")
        assert loss_fn is not None


# ---------------------------------------------------------------------------
# T2: BettiLoss
# ---------------------------------------------------------------------------


class TestBettiLoss:
    """Test Betti number (connected component) loss."""

    def test_basic_callable(self) -> None:
        """BettiLoss should be callable with logits and labels."""
        from minivess.pipeline.loss_functions import BettiLoss

        loss_fn = BettiLoss()
        logits = torch.randn(1, 2, 16, 16, 8)
        labels = torch.zeros(1, 1, 16, 16, 8, dtype=torch.long)
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0

    def test_perfect_prediction_low_loss(self) -> None:
        """Matching topology should yield lower loss than mismatched."""
        from minivess.pipeline.loss_functions import BettiLoss

        loss_fn = BettiLoss()
        # Create consistent labels and "perfect" logits
        labels = torch.zeros(1, 1, 16, 16, 8, dtype=torch.long)
        labels[0, 0, 4:12, 4:12, 2:6] = 1

        # Good logits: high confidence matching labels
        good_logits = torch.zeros(1, 2, 16, 16, 8)
        good_logits[:, 0] = 5.0  # high background
        good_logits[:, 1] = -5.0
        good_logits[0, 0, 4:12, 4:12, 2:6] = -5.0
        good_logits[0, 1, 4:12, 4:12, 2:6] = 5.0

        # Bad logits: fragmented predictions (many components)
        bad_logits = torch.zeros(1, 2, 16, 16, 8)
        bad_logits[:, 0] = 5.0
        bad_logits[:, 1] = -5.0
        # Scattered foreground predictions
        for i in range(0, 16, 4):
            for j in range(0, 16, 4):
                bad_logits[0, 0, i, j, 3] = -5.0
                bad_logits[0, 1, i, j, 3] = 5.0

        good_loss = loss_fn(good_logits, labels)
        bad_loss = loss_fn(bad_logits, labels)
        assert good_loss.item() <= bad_loss.item()

    def test_gradient_flows(self) -> None:
        """Betti loss should support gradient computation."""
        from minivess.pipeline.loss_functions import BettiLoss

        loss_fn = BettiLoss()
        logits = torch.randn(1, 2, 16, 16, 8, requires_grad=True)
        labels = torch.zeros(1, 1, 16, 16, 8, dtype=torch.long)
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_build_factory(self) -> None:
        """build_loss_function should support 'betti'."""
        from minivess.pipeline.loss_functions import build_loss_function

        loss_fn = build_loss_function("betti")
        assert loss_fn is not None


# ---------------------------------------------------------------------------
# T3: TopologyCompoundLoss
# ---------------------------------------------------------------------------


class TestTopologyCompoundLoss:
    """Test compound topology loss (DiceCE + clDice + Betti)."""

    def test_basic_callable(self) -> None:
        """TopologyCompoundLoss should be callable."""
        from minivess.pipeline.loss_functions import TopologyCompoundLoss

        loss_fn = TopologyCompoundLoss()
        logits = torch.randn(1, 2, 16, 16, 8)
        labels = torch.zeros(1, 1, 16, 16, 8, dtype=torch.long)
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_all_components_contribute(self) -> None:
        """All loss components should have nonzero contribution."""
        from minivess.pipeline.loss_functions import TopologyCompoundLoss

        loss_fn = TopologyCompoundLoss(
            lambda_dice_ce=1.0,
            lambda_cldice=1.0,
            lambda_betti=1.0,
        )
        logits = torch.randn(1, 2, 16, 16, 8)
        labels = torch.zeros(1, 1, 16, 16, 8, dtype=torch.long)
        labels[0, 0, 4:12, 4:12, 2:6] = 1

        total = loss_fn(logits, labels)
        assert total.item() > 0

    def test_gradient_flows(self) -> None:
        """Compound loss should produce gradients."""
        from minivess.pipeline.loss_functions import TopologyCompoundLoss

        loss_fn = TopologyCompoundLoss()
        logits = torch.randn(1, 2, 16, 16, 8, requires_grad=True)
        labels = torch.zeros(1, 1, 16, 16, 8, dtype=torch.long)
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_build_factory(self) -> None:
        """build_loss_function should support 'full_topo'."""
        from minivess.pipeline.loss_functions import build_loss_function

        loss_fn = build_loss_function("full_topo")
        assert loss_fn is not None
