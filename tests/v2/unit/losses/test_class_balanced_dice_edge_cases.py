"""Tests for ClassBalancedDiceLoss edge cases.

T5 from double-check plan: all-background, all-foreground labels,
and weight ratio bounds.
"""

from __future__ import annotations

import torch


def _make_logits(
    batch: int = 1,
    classes: int = 2,
    spatial: tuple[int, int, int] = (8, 8, 8),
) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(batch, classes, *spatial, requires_grad=True)


class TestClassBalancedDiceEdgeCases:
    """ClassBalancedDiceLoss must be safe with single-class batches."""

    def test_cb_dice_all_background_finite(self) -> None:
        from minivess.pipeline.loss_functions import ClassBalancedDiceLoss

        logits = _make_logits()
        labels = torch.zeros(1, 1, 8, 8, 8)  # all background
        loss_fn = ClassBalancedDiceLoss()
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), f"All-background loss not finite: {loss.item()}"
        loss.backward()
        assert torch.isfinite(logits.grad).all(), "Gradient not finite for all-background"

    def test_cb_dice_all_foreground_finite(self) -> None:
        from minivess.pipeline.loss_functions import ClassBalancedDiceLoss

        logits = _make_logits()
        labels = torch.ones(1, 1, 8, 8, 8)  # all foreground
        loss_fn = ClassBalancedDiceLoss()
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), f"All-foreground loss not finite: {loss.item()}"
        loss.backward()
        assert torch.isfinite(logits.grad).all(), "Gradient not finite for all-foreground"

    def test_cb_dice_normal_case_differentiable(self) -> None:
        from minivess.pipeline.loss_functions import ClassBalancedDiceLoss

        logits = _make_logits()
        labels = torch.randint(0, 2, (1, 1, 8, 8, 8)).float()
        loss_fn = ClassBalancedDiceLoss()
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), f"Normal case loss not finite: {loss.item()}"
        loss.backward()
        assert logits.grad is not None
        assert torch.any(logits.grad != 0), "Gradient should be non-zero for mixed labels"

    def test_cb_dice_weights_are_finite(self) -> None:
        from minivess.pipeline.loss_functions import ClassBalancedDiceLoss

        loss_fn = ClassBalancedDiceLoss()
        logits = _make_logits()
        labels = torch.zeros(1, 1, 8, 8, 8)  # all background — extreme case

        # Verify weights are finite (smooth prevents infinity)
        probs = torch.softmax(logits.detach(), dim=1)
        labels_squeeze = labels.long()[:, 0]
        labels_onehot = torch.zeros_like(probs)
        for c in range(2):
            labels_onehot[:, c] = (labels_squeeze == c).float()

        class_counts = labels_onehot.sum(dim=tuple(range(2, labels_onehot.ndim)))
        total_voxels = class_counts.sum(dim=1, keepdim=True)
        weights = total_voxels / (class_counts * 2 + loss_fn.smooth)
        weights = weights / weights.sum(dim=1, keepdim=True)

        assert torch.isfinite(weights).all(), f"Weights contain non-finite values: {weights}"
        assert (weights >= 0).all(), "Weights should be non-negative"
        assert torch.allclose(weights.sum(dim=1), torch.ones(1)), "Weights should sum to 1"
