"""Tests for BettiLoss spatial gradient edge cases.

T11 from double-check plan: minimum spatial dims, all-zero predictions,
perfect match, gradient direction.
"""

from __future__ import annotations

import torch


class TestBettiLossEdgeCases:
    """BettiLoss must handle extreme inputs safely."""

    def test_minimum_spatial_dims(self) -> None:
        """D=H=W=2: torch.diff yields size 1 per dim, still finite."""
        from minivess.pipeline.loss_functions import BettiLoss

        logits = torch.randn(1, 2, 2, 2, 2, requires_grad=True)
        labels = torch.randint(0, 2, (1, 1, 2, 2, 2)).float()
        loss_fn = BettiLoss()
        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), f"Min spatial dims: {loss.item()}"
        loss.backward()
        assert logits.grad is not None

    def test_all_zero_predictions(self) -> None:
        """Near-zero foreground probs with all-foreground labels."""
        from minivess.pipeline.loss_functions import BettiLoss

        # Strong bias toward background
        logits = torch.full((1, 2, 8, 8, 8), -10.0, requires_grad=True)
        labels = torch.ones(1, 1, 8, 8, 8)  # all foreground
        loss = BettiLoss()(logits, labels)
        assert torch.isfinite(loss), f"All-zero pred: {loss.item()}"
        loss.backward()
        assert torch.isfinite(logits.grad).all()

    def test_perfect_match_near_zero(self) -> None:
        """When prediction matches GT, loss should be near zero."""
        from minivess.pipeline.loss_functions import BettiLoss

        # All-foreground labels + confident foreground prediction
        logits = torch.zeros(1, 2, 8, 8, 8, requires_grad=True)
        logits_data = logits.detach().clone()
        logits_data[:, 1, :, :, :] = 10.0  # high foreground confidence
        logits = logits_data.requires_grad_(True)
        labels = torch.ones(1, 1, 8, 8, 8)  # all foreground

        loss = BettiLoss()(logits, labels)
        # Both pred and GT have zero fragmentation → loss ≈ 0
        assert loss.item() < 0.1, f"Perfect match should give near-zero loss: {loss.item()}"

    def test_fragmented_pred_positive_loss(self) -> None:
        """Fragmented prediction with smooth GT produces positive loss."""
        from minivess.pipeline.loss_functions import BettiLoss

        torch.manual_seed(42)
        # Alternating pattern = fragmented
        logits = torch.zeros(1, 2, 8, 8, 8, requires_grad=True)
        logits_data = logits.detach().clone()
        logits_data[:, 1, ::2, :, :] = 10.0  # every other slice is foreground
        logits_data[:, 0, 1::2, :, :] = 10.0  # alternating
        logits = logits_data.requires_grad_(True)
        labels = torch.ones(1, 1, 8, 8, 8)  # smooth all-foreground GT

        loss = BettiLoss()(logits, labels)
        assert loss.item() > 0.0, "Fragmented pred should give positive loss"
