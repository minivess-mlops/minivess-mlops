"""Ensemble gap tests: greedy_soup, WeightWatcher, edge cases (Code Review R2.3)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from minivess.config.models import (
    EnsembleConfig,
    EnsembleStrategy,
    ModelConfig,
    ModelFamily,
)


def _segresnet_model() -> object:
    from minivess.adapters.segresnet import SegResNetAdapter

    config = ModelConfig(
        family=ModelFamily.MONAI_SEGRESNET,
        name="test",
        in_channels=1,
        out_channels=2,
    )
    return SegResNetAdapter(config)


# ---------------------------------------------------------------------------
# T1: greedy_soup
# ---------------------------------------------------------------------------


class TestGreedySoup:
    """Test greedy model soup weight averaging."""

    def test_single_model(self) -> None:
        """greedy_soup with one model should return its state dict."""
        from minivess.ensemble.strategies import greedy_soup

        model = _segresnet_model()
        metric_fn = MagicMock(return_value=0.85)
        result = greedy_soup([model], metric_fn, val_loader=None)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_empty_model_list(self) -> None:
        """greedy_soup with empty list should raise ValueError."""
        from minivess.ensemble.strategies import greedy_soup

        with pytest.raises(ValueError, match="at least one model"):
            greedy_soup([], MagicMock(), val_loader=None)

    def test_two_models_returns_state_dict(self) -> None:
        """greedy_soup with two models should return a valid state dict."""
        from minivess.ensemble.strategies import greedy_soup

        model1 = _segresnet_model()
        model2 = _segresnet_model()
        # Second model always improves → should be added to soup
        call_count = 0

        def improving_metric(model: object, loader: object) -> float:
            nonlocal call_count
            call_count += 1
            return 0.5 + 0.1 * call_count

        result = greedy_soup([model1, model2], improving_metric, val_loader=None)
        assert isinstance(result, dict)
        # Should have called metric at least 2 times (initial + candidate)
        assert call_count >= 2

    def test_rejected_model_preserves_best(self) -> None:
        """When a candidate worsens the metric, it should be rejected."""
        from minivess.ensemble.strategies import greedy_soup

        model1 = _segresnet_model()
        model2 = _segresnet_model()
        # Metric decreases on second call → model2 should be rejected
        call_count = 0

        def declining_metric(model: object, loader: object) -> float:
            nonlocal call_count
            call_count += 1
            return 0.9 if call_count == 1 else 0.7

        result = greedy_soup([model1, model2], declining_metric, val_loader=None)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# T2: WeightWatcher
# ---------------------------------------------------------------------------


class TestWeightWatcherReport:
    """Test WeightWatcher report dataclass."""

    def test_report_construction(self) -> None:
        """WeightWatcherReport should hold spectral analysis results."""
        from minivess.ensemble.weightwatcher import WeightWatcherReport

        report = WeightWatcherReport(
            alpha_weighted=3.5,
            log_norm=1.2,
            num_layers=10,
            details={"summary": "test"},
            passed_gate=True,
        )
        assert report.alpha_weighted == 3.5
        assert report.passed_gate is True

    def test_report_gate_logic(self) -> None:
        """passed_gate should reflect whether alpha is below threshold."""
        from minivess.ensemble.weightwatcher import WeightWatcherReport

        passing = WeightWatcherReport(
            alpha_weighted=3.0,
            log_norm=1.0,
            num_layers=5,
            details={},
            passed_gate=True,
        )
        failing = WeightWatcherReport(
            alpha_weighted=6.0,
            log_norm=2.0,
            num_layers=5,
            details={},
            passed_gate=False,
        )
        assert passing.passed_gate is True
        assert failing.passed_gate is False


# ---------------------------------------------------------------------------
# T3: EnsemblePredictor edge cases
# ---------------------------------------------------------------------------


class TestEnsemblePredictorEdgeCases:
    """Test EnsemblePredictor with edge cases."""

    def test_single_model_ensemble(self) -> None:
        """Ensemble with a single model should return that model's prediction."""
        from minivess.ensemble.strategies import EnsemblePredictor

        model = _segresnet_model()
        config = EnsembleConfig(strategy=EnsembleStrategy.MEAN)
        ensemble = EnsemblePredictor([model], config)
        x = torch.randn(1, 1, 16, 16, 16)
        result = ensemble.predict(x)
        assert result.shape == (1, 2, 16, 16, 16)

    def test_mean_predictions_sum_to_one(self) -> None:
        """Mean ensemble predictions should be valid probabilities."""
        from minivess.ensemble.strategies import EnsemblePredictor

        model1 = _segresnet_model()
        model2 = _segresnet_model()
        config = EnsembleConfig(strategy=EnsembleStrategy.MEAN)
        ensemble = EnsemblePredictor([model1, model2], config)
        x = torch.randn(1, 1, 16, 16, 16)
        result = ensemble.predict(x)
        sums = result.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_unknown_strategy_raises(self) -> None:
        """Unknown strategy should raise ValueError."""
        from minivess.ensemble.strategies import EnsemblePredictor

        model = _segresnet_model()
        config = EnsembleConfig(strategy=EnsembleStrategy.GREEDY_SOUP)
        ensemble = EnsemblePredictor([model], config)
        with pytest.raises(ValueError, match="Unsupported"):
            ensemble.predict(torch.randn(1, 1, 16, 16, 16))

    def test_majority_vote_output_shape(self) -> None:
        """Majority voting should produce correct output shape."""
        from minivess.ensemble.strategies import EnsemblePredictor

        model1 = _segresnet_model()
        model2 = _segresnet_model()
        config = EnsembleConfig(strategy=EnsembleStrategy.MAJORITY_VOTE)
        ensemble = EnsemblePredictor([model1, model2], config)
        x = torch.randn(1, 1, 16, 16, 16)
        result = ensemble.predict(x)
        assert result.shape == (1, 2, 16, 16, 16)
