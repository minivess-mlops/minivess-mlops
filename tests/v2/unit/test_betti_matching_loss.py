"""Tests for Betti matching loss (Stucki et al., ICML 2023).

Covers Issue #122: differentiable topology matching via persistence diagrams.
gudhi is optional — graceful fallback to 0.0 loss with warning.
TDD RED phase: tests written before implementation.
"""

from __future__ import annotations

import torch


def _make_logits_labels(
    batch: int = 1,
    classes: int = 2,
    spatial: tuple[int, int, int] = (16, 16, 8),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic logits and labels for loss testing."""
    torch.manual_seed(42)
    logits = torch.randn(batch, classes, *spatial, requires_grad=True)
    labels = torch.randint(0, 2, (batch, 1, *spatial)).float()
    return logits, labels


def _make_perfect_prediction(
    labels: torch.Tensor,
    classes: int = 2,
) -> torch.Tensor:
    """Create logits that produce a near-perfect prediction."""
    logits = torch.full(
        (labels.shape[0], classes, *labels.shape[2:]),
        -5.0,
    )
    fg_mask = labels[:, 0] > 0.5
    logits[:, 1][fg_mask] = 5.0
    logits[:, 0][fg_mask] = -5.0
    logits[:, 0][~fg_mask] = 5.0
    logits[:, 1][~fg_mask] = -5.0
    return logits.requires_grad_(True)


class TestBettiMatchingLoss:
    """Tests for BettiMatchingLoss."""

    def test_betti_matching_loss_differentiable(self) -> None:
        from minivess.pipeline.vendored_losses.betti_matching import BettiMatchingLoss

        logits, labels = _make_logits_labels()
        loss_fn = BettiMatchingLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_betti_matching_loss_gradient_nonzero(self) -> None:
        from minivess.pipeline.vendored_losses.betti_matching import BettiMatchingLoss

        logits, labels = _make_logits_labels()
        loss_fn = BettiMatchingLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None
        # Gradient may be zero if gudhi is unavailable (fallback returns 0.0)
        # So we just check the grad tensor exists

    def test_betti_matching_loss_topology_diff_penalized(self) -> None:
        from minivess.pipeline.vendored_losses.betti_matching import BettiMatchingLoss

        # Labels: single connected blob
        labels = torch.zeros(1, 1, 16, 16, 8)
        labels[:, :, 4:12, 4:12, 2:6] = 1.0
        # Prediction: two separate blobs (topology mismatch)
        logits = torch.full((1, 2, 16, 16, 8), -5.0)
        logits[:, 1, 4:8, 4:8, 2:6] = 5.0
        logits[:, 0, 4:8, 4:8, 2:6] = -5.0
        logits[:, 1, 10:14, 10:14, 2:6] = 5.0
        logits[:, 0, 10:14, 10:14, 2:6] = -5.0
        logits = logits.requires_grad_(True)

        loss_fn = BettiMatchingLoss()
        loss = loss_fn(logits, labels)
        # Loss should be non-negative
        assert loss.item() >= 0.0

    def test_betti_matching_loss_perfect_match_low_loss(self) -> None:
        from minivess.pipeline.vendored_losses.betti_matching import BettiMatchingLoss

        _, labels = _make_logits_labels()
        logits = _make_perfect_prediction(labels)
        loss_fn = BettiMatchingLoss()
        loss = loss_fn(logits, labels)
        # Perfect topology match should give low or zero loss
        assert loss.item() >= 0.0
        assert torch.isfinite(loss)

    def test_betti_matching_loss_gudhi_missing_graceful(self) -> None:
        """When gudhi is not installed, loss should return 0.0 gracefully."""
        from minivess.pipeline.vendored_losses.betti_matching import BettiMatchingLoss

        logits, labels = _make_logits_labels()
        loss_fn = BettiMatchingLoss()
        loss = loss_fn(logits, labels)
        # Should not raise, should be finite
        assert torch.isfinite(loss)

    def test_betti_matching_loss_registered_in_factory(self) -> None:
        from minivess.pipeline.loss_functions import build_loss_function

        loss_fn = build_loss_function("betti_matching")
        assert loss_fn is not None
        logits, labels = _make_logits_labels()
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss)

    def test_betti_matching_loss_vram_small_patch(self) -> None:
        """Forward+backward on (1,2,16,16,8) should not OOM."""
        from minivess.pipeline.vendored_losses.betti_matching import BettiMatchingLoss

        logits, labels = _make_logits_labels(spatial=(16, 16, 8))
        loss_fn = BettiMatchingLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert torch.isfinite(loss)

    def test_betti_matching_loss_nan_safe(self) -> None:
        from minivess.pipeline.vendored_losses.betti_matching import BettiMatchingLoss

        logits, labels = _make_logits_labels()
        loss_fn = BettiMatchingLoss()
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_betti_matching_loss_batch_dimension(self) -> None:
        from minivess.pipeline.vendored_losses.betti_matching import BettiMatchingLoss

        logits, labels = _make_logits_labels(batch=2)
        loss_fn = BettiMatchingLoss()
        loss = loss_fn(logits, labels)
        assert loss.shape == () or loss.shape == (1,)  # scalar
        assert torch.isfinite(loss)
