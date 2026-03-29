"""Edge case tests for topology losses and metrics.

Tests degenerate inputs that can produce silent zeros, NaN, or missing
gradients in topology-aware losses and metrics.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestGraphTopologyAllZeroWeights:
    """GraphTopologyLoss with all-zero weights should produce zero loss."""

    def test_all_zero_weights_is_zero(self) -> None:
        """Loss with w_cbdice_cldice=0, w_skeleton_recall=0, w_cape=0 → 0."""
        from minivess.pipeline.loss_functions import GraphTopologyLoss

        loss_fn = GraphTopologyLoss(
            w_cbdice_cldice=0.0,
            w_skeleton_recall=0.0,
            w_cape=0.0,
        )
        torch.manual_seed(42)
        logits = torch.randn(1, 2, 8, 16, 16)
        labels = torch.zeros(1, 1, 8, 16, 16)
        labels[:, :, 2:6, 4:12, 4:12] = 1.0

        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), f"All-zero weights produced non-finite loss: {loss}"
        assert loss.item() == pytest.approx(0.0, abs=1e-6), (
            f"Expected 0.0 for all-zero weights, got {loss.item()}"
        )


class TestCcDiceEdgeCases:
    """ccDice metric must handle degenerate connected components."""

    def test_no_overlap_returns_zero(self) -> None:
        """Pred with 3 components, GT with no overlap → ccDice = 0."""
        from minivess.pipeline.topology_metrics import compute_ccdice

        # Pred: 3 separate blobs
        pred = np.zeros((16, 16, 16), dtype=bool)
        pred[2:4, 2:4, 2:4] = True
        pred[8:10, 8:10, 8:10] = True
        pred[12:14, 12:14, 12:14] = True

        # GT: single blob in a completely different location
        target = np.zeros((16, 16, 16), dtype=bool)
        target[6:8, 6:8, 6:8] = True

        result = compute_ccdice(pred, target)
        assert np.isfinite(result), f"ccDice is not finite: {result}"
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_both_empty_returns_one(self) -> None:
        """Both pred and GT empty → ccDice = 1.0 (perfect agreement on nothing)."""
        from minivess.pipeline.topology_metrics import compute_ccdice

        pred = np.zeros((16, 16, 16), dtype=bool)
        target = np.zeros((16, 16, 16), dtype=bool)

        result = compute_ccdice(pred, target)
        assert result == pytest.approx(1.0)

    def test_pred_empty_gt_nonempty_returns_zero(self) -> None:
        """Empty prediction with non-empty GT → ccDice = 0."""
        from minivess.pipeline.topology_metrics import compute_ccdice

        pred = np.zeros((16, 16, 16), dtype=bool)
        target = np.zeros((16, 16, 16), dtype=bool)
        target[4:8, 4:8, 4:8] = True

        result = compute_ccdice(pred, target)
        assert np.isfinite(result)
        assert result == pytest.approx(0.0)


class TestBettiLossEdgeCases:
    """BettiLoss must handle extreme inputs without NaN or missing gradients."""

    def test_all_zero_predictions_finite(self) -> None:
        """Logits strongly favoring background → loss must be finite with gradient."""
        from minivess.pipeline.loss_functions import BettiLoss

        loss_fn = BettiLoss()
        # Strong background logits → softmax ≈ 0 foreground
        data = torch.full((1, 2, 8, 16, 16), -10.0)
        data[:, 0] = 10.0  # Background channel high
        logits = data.clone().detach().requires_grad_(True)
        labels = torch.zeros(1, 1, 8, 16, 16)
        labels[:, :, 2:6, 4:12, 4:12] = 1.0

        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), f"BettiLoss is not finite: {loss}"

        loss.backward()
        assert logits.grad is not None, "No gradient for BettiLoss"
        assert not torch.any(torch.isnan(logits.grad)), "NaN in BettiLoss gradient"

    def test_all_foreground_predictions_finite(self) -> None:
        """Logits strongly favoring foreground → loss finite."""
        from minivess.pipeline.loss_functions import BettiLoss

        loss_fn = BettiLoss()
        data = torch.full((1, 2, 8, 16, 16), -10.0)
        data[:, 1] = 10.0  # Foreground channel high
        logits = data.clone().detach().requires_grad_(True)
        labels = torch.zeros(1, 1, 8, 16, 16)
        labels[:, :, 2:6, 4:12, 4:12] = 1.0

        loss = loss_fn(logits, labels)
        assert torch.isfinite(loss), f"BettiLoss is not finite: {loss}"
