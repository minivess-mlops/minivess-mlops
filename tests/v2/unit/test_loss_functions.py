"""Tests for VesselCompoundLoss and loss function coverage (Issue #53 — R5.7).

Tests forward pass, gradient computation, numerical stability, and edge cases
for the compound loss functions used in vessel segmentation.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logits_labels(
    batch: int = 1,
    classes: int = 2,
    spatial: tuple[int, int, int] = (16, 16, 8),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic logits and labels for loss testing."""
    torch.manual_seed(42)
    logits = torch.randn(batch, classes, *spatial, requires_grad=True)
    labels = torch.zeros(batch, 1, *spatial, dtype=torch.long)
    # Mark a small region as foreground (class 1)
    labels[:, :, 4:12, 4:12, 2:6] = 1
    return logits, labels


# ---------------------------------------------------------------------------
# R5.7 T1: VesselCompoundLoss forward pass
# ---------------------------------------------------------------------------


class TestVesselCompoundLossForward:
    """Test VesselCompoundLoss produces valid loss values."""

    def test_forward_returns_scalar(self) -> None:
        """Loss forward should return a scalar tensor."""
        from minivess.pipeline.loss_functions import VesselCompoundLoss

        loss_fn = VesselCompoundLoss()
        logits, labels = _make_logits_labels()
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0

    def test_forward_positive(self) -> None:
        """Loss value should be positive for random inputs."""
        from minivess.pipeline.loss_functions import VesselCompoundLoss

        loss_fn = VesselCompoundLoss()
        logits, labels = _make_logits_labels()
        loss = loss_fn(logits, labels)
        assert loss.item() > 0.0

    def test_forward_finite(self) -> None:
        """Loss should not be NaN or Inf."""
        from minivess.pipeline.loss_functions import VesselCompoundLoss

        loss_fn = VesselCompoundLoss()
        logits, labels = _make_logits_labels()
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss).item()

    def test_custom_lambda_weights(self) -> None:
        """Custom lambda weights should change the loss value."""
        from minivess.pipeline.loss_functions import VesselCompoundLoss

        logits, labels = _make_logits_labels()
        loss_equal = VesselCompoundLoss(lambda_dice_ce=0.5, lambda_cldice=0.5)(
            logits, labels
        )
        loss_dice_heavy = VesselCompoundLoss(lambda_dice_ce=0.9, lambda_cldice=0.1)(
            logits, labels
        )
        # Different weights should give different loss (almost certainly)
        assert not torch.allclose(loss_equal, loss_dice_heavy)


# ---------------------------------------------------------------------------
# R5.7 T2: Gradient computation
# ---------------------------------------------------------------------------


class TestVesselCompoundLossGradients:
    """Test that VesselCompoundLoss produces valid gradients."""

    def test_backward_produces_gradients(self) -> None:
        """loss.backward() should produce non-zero gradients on logits."""
        from minivess.pipeline.loss_functions import VesselCompoundLoss

        loss_fn = VesselCompoundLoss()
        logits, labels = _make_logits_labels()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0.0

    def test_gradients_are_finite(self) -> None:
        """All gradients should be finite (no NaN/Inf)."""
        from minivess.pipeline.loss_functions import VesselCompoundLoss

        loss_fn = VesselCompoundLoss()
        logits, labels = _make_logits_labels()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert torch.isfinite(logits.grad).all().item()


# ---------------------------------------------------------------------------
# R5.7 T3: Numerical stability edge cases
# ---------------------------------------------------------------------------


class TestLossNumericalStability:
    """Test compound loss handles numerical edge cases."""

    def test_all_zeros_labels(self) -> None:
        """All-background labels (no foreground) should produce finite loss."""
        from minivess.pipeline.loss_functions import VesselCompoundLoss

        loss_fn = VesselCompoundLoss()
        torch.manual_seed(0)
        logits = torch.randn(1, 2, 16, 16, 8, requires_grad=True)
        labels = torch.zeros(1, 1, 16, 16, 8, dtype=torch.long)
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss).item()

    def test_all_ones_labels(self) -> None:
        """All-foreground labels should produce finite loss."""
        from minivess.pipeline.loss_functions import VesselCompoundLoss

        loss_fn = VesselCompoundLoss()
        torch.manual_seed(0)
        logits = torch.randn(1, 2, 16, 16, 8, requires_grad=True)
        labels = torch.ones(1, 1, 16, 16, 8, dtype=torch.long)
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss).item()

    def test_single_voxel_volume(self) -> None:
        """Single-voxel volume should produce finite loss (degenerate case)."""
        from minivess.pipeline.loss_functions import VesselCompoundLoss

        loss_fn = VesselCompoundLoss()
        logits = torch.randn(1, 2, 1, 1, 1, requires_grad=True)
        labels = torch.ones(1, 1, 1, 1, 1, dtype=torch.long)
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss).item()

    def test_build_loss_function_factory(self) -> None:
        """build_loss_function should return valid loss for all registered names."""
        from minivess.pipeline.loss_functions import build_loss_function

        for name in ("dice_ce", "dice", "focal", "cb_dice", "betti", "full_topo"):
            loss_fn = build_loss_function(name)
            logits, labels = _make_logits_labels()
            loss = loss_fn(logits, labels)
            assert torch.isfinite(loss).item(), f"{name} produced non-finite loss"
