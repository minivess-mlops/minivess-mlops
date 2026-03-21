"""Tests for predict_with_uncertainty Prefect task (#886).

Validates the analysis flow task that wraps DeepEnsemblePredictor
for entropy/MI uncertainty decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class _MockOutput:
    prediction: Tensor


class _MockModel(nn.Module):
    def __init__(self, prob: float = 0.8) -> None:
        super().__init__()
        self._prob = prob

    def forward(self, x: Tensor) -> _MockOutput:
        b, _c, d, h, w = x.shape
        fg = torch.full((b, 1, d, h, w), self._prob, dtype=x.dtype)
        bg = 1.0 - fg
        return _MockOutput(prediction=torch.cat([bg, fg], dim=1))


class TestPredictWithUncertainty:
    """predict_with_uncertainty task wraps DeepEnsemblePredictor."""

    def test_returns_required_keys(self) -> None:
        """Result dict must contain all UQ keys."""
        from minivess.orchestration.flows.analysis_flow import (
            predict_with_uncertainty,
        )

        models = [_MockModel(p) for p in [0.3, 0.7, 0.9]]
        images = torch.randn(1, 1, 4, 4, 4)

        result = predict_with_uncertainty(models=models, images=images)

        required = {
            "prediction",
            "uncertainty_map",
            "total_uncertainty",
            "aleatoric_uncertainty",
            "epistemic_uncertainty",
            "n_members",
        }
        assert required.issubset(result.keys())

    def test_prediction_shape(self) -> None:
        """Mean prediction must be (B, C, D, H, W)."""
        from minivess.orchestration.flows.analysis_flow import (
            predict_with_uncertainty,
        )

        models = [_MockModel(0.6), _MockModel(0.8)]
        images = torch.randn(2, 1, 4, 4, 4)

        result = predict_with_uncertainty(models=models, images=images)
        assert result["prediction"].shape == (2, 2, 4, 4, 4)

    def test_uncertainty_map_shape(self) -> None:
        """Uncertainty map must be (B, 1, D, H, W)."""
        from minivess.orchestration.flows.analysis_flow import (
            predict_with_uncertainty,
        )

        models = [_MockModel(0.5)]
        images = torch.randn(1, 1, 4, 4, 4)

        result = predict_with_uncertainty(models=models, images=images)
        assert result["uncertainty_map"].shape == (1, 1, 4, 4, 4)

    def test_n_members_matches_input(self) -> None:
        """n_members must equal number of input models."""
        from minivess.orchestration.flows.analysis_flow import (
            predict_with_uncertainty,
        )

        models = [_MockModel(p) for p in [0.3, 0.5, 0.7, 0.9]]
        images = torch.randn(1, 1, 4, 4, 4)

        result = predict_with_uncertainty(models=models, images=images)
        assert result["n_members"] == 4

    def test_epistemic_equals_total_minus_aleatoric(self) -> None:
        """MI identity: epistemic = total - aleatoric."""
        from minivess.orchestration.flows.analysis_flow import (
            predict_with_uncertainty,
        )

        models = [_MockModel(p) for p in [0.2, 0.5, 0.9]]
        images = torch.randn(1, 1, 4, 4, 4)

        result = predict_with_uncertainty(models=models, images=images)
        expected = result["total_uncertainty"] - result["aleatoric_uncertainty"]
        assert torch.allclose(result["epistemic_uncertainty"], expected, atol=1e-6)

    def test_is_prefect_task(self) -> None:
        """predict_with_uncertainty must be callable (Prefect task)."""
        from minivess.orchestration.flows.analysis_flow import (
            predict_with_uncertainty,
        )

        assert callable(predict_with_uncertainty)
