"""Tests for CAPE loss (Connectivity-Aware Path Enforcement).

Covers Issue #115: differentiable loss penalising broken paths between
endpoint pairs on the ground-truth skeleton.
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


def _make_connected_tube_labels(
    spatial: tuple[int, int, int] = (16, 16, 8),
) -> torch.Tensor:
    """Create labels with a connected cylindrical tube structure.

    Uses cylindrical geometry (not rectangular block) so that
    skimage.morphology.skeletonize (Lee94) produces a valid skeleton.
    """
    labels = torch.zeros(1, 1, *spatial)
    # Cylinder along z-axis, radius ~2 centered at (y=8, x=4)
    for z in range(2, 14):
        for y in range(5, 11):
            for x in range(2, 6):
                if (y - 8) ** 2 + (x - 4) ** 2 <= 4:
                    labels[0, 0, z, y, x] = 1.0
    return labels


def _make_perfect_prediction(
    labels: torch.Tensor,
    classes: int = 2,
) -> torch.Tensor:
    """Create logits that produce a near-perfect prediction for the given labels."""
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


class TestCAPELoss:
    """Tests for CAPELoss (Connectivity-Aware Path Enforcement)."""

    def test_cape_loss_differentiable(self) -> None:
        from minivess.pipeline.vendored_losses.cape import CAPELoss

        logits, labels = _make_logits_labels()
        loss_fn = CAPELoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_cape_loss_gradient_nonzero(self) -> None:
        from minivess.pipeline.vendored_losses.cape import CAPELoss

        logits, labels = _make_logits_labels()
        loss_fn = CAPELoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None
        assert torch.any(logits.grad != 0)

    def test_cape_loss_connected_prediction_low_loss(self) -> None:
        from minivess.pipeline.vendored_losses.cape import CAPELoss

        labels = _make_connected_tube_labels()
        logits = _make_perfect_prediction(labels)
        loss_fn = CAPELoss()
        loss = loss_fn(logits, labels)
        # Connected prediction matching GT should give low loss
        assert loss.item() < 0.5

    def test_cape_loss_severed_prediction_high_loss(self) -> None:
        from minivess.pipeline.vendored_losses.cape import CAPELoss

        labels = _make_connected_tube_labels()
        # Create prediction that severs the tube in the middle
        logits = _make_perfect_prediction(labels)
        logits_data = logits.detach().clone()
        # Cut the tube at z=7-9 (sever the path)
        logits_data[:, 1, 7:9, :, :] = -5.0
        logits_data[:, 0, 7:9, :, :] = 5.0
        logits = logits_data.requires_grad_(True)

        loss_fn = CAPELoss()
        loss_connected = loss_fn(_make_perfect_prediction(labels), labels)
        loss_severed = loss_fn(logits, labels)

        # Severed prediction should give higher loss
        assert loss_severed.item() > loss_connected.item()

    def test_cape_loss_softmax_configurable(self) -> None:
        from minivess.pipeline.vendored_losses.cape import CAPELoss

        logits, labels = _make_logits_labels()
        # Should work with both softmax modes
        for softmax in [True, False]:
            loss_fn = CAPELoss(softmax=softmax)
            loss = loss_fn(logits, labels)
            assert torch.isfinite(loss)

    def test_cape_loss_registered_in_factory(self) -> None:
        from minivess.pipeline.loss_functions import build_loss_function

        loss_fn = build_loss_function("cape")
        assert loss_fn is not None
        logits, labels = _make_logits_labels()
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss)

    def test_cape_loss_vram_small_patch(self) -> None:
        """Forward+backward on (1,2,16,16,8) should not OOM."""
        from minivess.pipeline.vendored_losses.cape import CAPELoss

        logits, labels = _make_logits_labels(spatial=(16, 16, 8))
        loss_fn = CAPELoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert torch.isfinite(loss)

    def test_cape_loss_nan_safe(self) -> None:
        from minivess.pipeline.vendored_losses.cape import CAPELoss

        logits, labels = _make_logits_labels()
        loss_fn = CAPELoss()
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_cape_loss_batch_dimension(self) -> None:
        from minivess.pipeline.vendored_losses.cape import CAPELoss

        logits, labels = _make_logits_labels(batch=2)
        loss_fn = CAPELoss()
        loss = loss_fn(logits, labels)
        assert loss.shape == () or loss.shape == (1,)  # scalar
        assert torch.isfinite(loss)
