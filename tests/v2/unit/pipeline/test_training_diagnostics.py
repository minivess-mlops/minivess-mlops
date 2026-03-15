"""Tests for training diagnostics logging.

T1.3: Verify gradient norm and prediction stats are computed correctly.
"""

from __future__ import annotations

import torch
from torch import nn


class TestGradientDiagnostics:
    """Gradient diagnostics should compute norm and detect NaN."""

    def test_compute_gradient_norm(self) -> None:
        """compute_gradient_norm returns L2 norm of all gradients."""
        from minivess.pipeline.training_diagnostics import compute_gradient_norm

        model = nn.Linear(4, 2)
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()

        norm = compute_gradient_norm(model)
        assert norm > 0.0
        assert not torch.isnan(torch.tensor(norm))

    def test_gradient_norm_zero_when_no_grad(self) -> None:
        """Returns 0.0 when no parameters have gradients."""
        from minivess.pipeline.training_diagnostics import compute_gradient_norm

        model = nn.Linear(4, 2)
        # No backward pass → no gradients
        norm = compute_gradient_norm(model)
        assert norm == 0.0


class TestPredictionDiagnostics:
    """Prediction diagnostics should compute stats and detect NaN."""

    def test_compute_prediction_stats(self) -> None:
        """compute_prediction_stats returns dict with min, max, mean, std."""
        from minivess.pipeline.training_diagnostics import compute_prediction_stats

        preds = torch.randn(2, 2, 8, 8)
        stats = compute_prediction_stats(preds)
        assert "pred_min" in stats
        assert "pred_max" in stats
        assert "pred_mean" in stats
        assert "pred_std" in stats
        assert stats["pred_std"] > 0.0

    def test_detect_nan_predictions(self) -> None:
        """has_nan_or_inf returns True for NaN predictions."""
        from minivess.pipeline.training_diagnostics import has_nan_or_inf

        preds = torch.tensor([1.0, float("nan"), 3.0])
        assert has_nan_or_inf(preds) is True

    def test_detect_inf_predictions(self) -> None:
        """has_nan_or_inf returns True for Inf predictions."""
        from minivess.pipeline.training_diagnostics import has_nan_or_inf

        preds = torch.tensor([1.0, float("inf"), 3.0])
        assert has_nan_or_inf(preds) is True

    def test_normal_predictions(self) -> None:
        """has_nan_or_inf returns False for normal predictions."""
        from minivess.pipeline.training_diagnostics import has_nan_or_inf

        preds = torch.randn(2, 2, 8, 8)
        assert has_nan_or_inf(preds) is False
