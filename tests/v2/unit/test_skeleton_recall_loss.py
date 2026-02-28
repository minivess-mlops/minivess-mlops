"""Tests for skeleton recall loss (Kirchhoff et al., ECCV 2024).

Covers Issue #114: differentiable loss penalising missed skeleton voxels.
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
    """Create logits that produce a near-perfect prediction for the given labels."""
    # High confidence logits: class 0 gets -5, class 1 gets +5 where label=1
    logits = torch.full(
        (labels.shape[0], classes, *labels.shape[2:]),
        -5.0,
        requires_grad=True,
    )
    fg_mask = labels[:, 0] > 0.5
    logits_data = logits.detach().clone()
    logits_data[:, 1][fg_mask] = 5.0
    logits_data[:, 0][fg_mask] = -5.0
    logits_data[:, 0][~fg_mask] = 5.0
    logits_data[:, 1][~fg_mask] = -5.0
    return logits_data.requires_grad_(True)


class TestSkeletonRecallLoss:
    """Tests for SkeletonRecallLoss."""

    def test_skeleton_recall_loss_differentiable(self) -> None:
        from minivess.pipeline.vendored_losses.skeleton_recall import (
            SkeletonRecallLoss,
        )

        logits, labels = _make_logits_labels()
        loss_fn = SkeletonRecallLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_skeleton_recall_loss_gradient_nonzero(self) -> None:
        from minivess.pipeline.vendored_losses.skeleton_recall import (
            SkeletonRecallLoss,
        )

        logits, labels = _make_logits_labels()
        loss_fn = SkeletonRecallLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None
        assert torch.any(logits.grad != 0)

    def test_skeleton_recall_loss_perfect_prediction_near_zero(self) -> None:
        from minivess.pipeline.vendored_losses.skeleton_recall import (
            SkeletonRecallLoss,
        )

        _, labels = _make_logits_labels()
        logits = _make_perfect_prediction(labels)
        loss_fn = SkeletonRecallLoss()
        loss = loss_fn(logits, labels)
        # Perfect prediction should give low loss
        assert loss.item() < 0.5

    def test_skeleton_recall_loss_missed_skeleton_high_loss(self) -> None:
        from minivess.pipeline.vendored_losses.skeleton_recall import (
            SkeletonRecallLoss,
        )

        # Labels with foreground, prediction is all background
        _, labels = _make_logits_labels()
        # Make sure there's foreground in labels
        labels[:, :, 4:12, 4:12, 2:6] = 1.0
        # Logits strongly predict background everywhere
        logits = torch.full((1, 2, 16, 16, 8), 0.0, requires_grad=True)
        logits_data = logits.detach().clone()
        logits_data[:, 0] = 5.0  # strong background
        logits_data[:, 1] = -5.0  # no foreground
        logits = logits_data.requires_grad_(True)

        loss_fn = SkeletonRecallLoss()
        loss = loss_fn(logits, labels)
        # Missing all skeleton voxels should give high loss
        assert loss.item() > 0.3

    def test_skeleton_recall_loss_registered_in_factory(self) -> None:
        from minivess.pipeline.loss_functions import build_loss_function

        loss_fn = build_loss_function("skeleton_recall")
        assert loss_fn is not None
        logits, labels = _make_logits_labels()
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss)

    def test_skeleton_recall_loss_vram_small_patch(self) -> None:
        """Forward+backward on (1,1,16,16,8) should not OOM."""
        from minivess.pipeline.vendored_losses.skeleton_recall import (
            SkeletonRecallLoss,
        )

        logits, labels = _make_logits_labels(spatial=(16, 16, 8))
        loss_fn = SkeletonRecallLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert torch.isfinite(loss)

    def test_skeleton_recall_loss_nan_safe(self) -> None:
        from minivess.pipeline.vendored_losses.skeleton_recall import (
            SkeletonRecallLoss,
        )

        logits, labels = _make_logits_labels()
        loss_fn = SkeletonRecallLoss()
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_skeleton_recall_loss_batch_dimension(self) -> None:
        from minivess.pipeline.vendored_losses.skeleton_recall import (
            SkeletonRecallLoss,
        )

        logits, labels = _make_logits_labels(batch=2)
        loss_fn = SkeletonRecallLoss()
        loss = loss_fn(logits, labels)
        assert loss.shape == () or loss.shape == (1,)  # scalar
        assert torch.isfinite(loss)
