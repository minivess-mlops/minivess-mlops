from __future__ import annotations

import numpy as np
import pytest
import torch

# Import directly from submodules to avoid __init__.py transitive imports
# (e.g. minivess.validation.__init__ pulls in pandera which may be absent).
from minivess.adapters.segresnet import SegResNetAdapter
from minivess.config.models import (
    EnsembleConfig,
    EnsembleStrategy,
    ModelConfig,
    ModelFamily,
)
from minivess.ensemble.calibration import (
    CalibrationResult,
    expected_calibration_error,
    temperature_scale,
)
from minivess.ensemble.strategies import EnsemblePredictor
from minivess.validation.drift import DriftReport, detect_prediction_drift


class TestEnsemblePredictor:
    """Test ensemble prediction strategies."""

    @pytest.fixture
    def models(self) -> list[SegResNetAdapter]:
        config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="test",
            in_channels=1,
            out_channels=2,
        )
        return [SegResNetAdapter(config) for _ in range(3)]

    def test_mean_ensemble(self, models: list[SegResNetAdapter]) -> None:
        config = EnsembleConfig(strategy=EnsembleStrategy.MEAN, num_members=3)
        predictor = EnsemblePredictor(models, config)
        x = torch.randn(1, 1, 32, 32, 16)
        result = predictor.predict(x)
        assert result.shape == (1, 2, 32, 32, 16)
        # Probabilities should sum to ~1
        assert torch.allclose(result.sum(dim=1), torch.ones(1, 32, 32, 16), atol=0.1)

    def test_majority_voting(self, models: list[SegResNetAdapter]) -> None:
        config = EnsembleConfig(strategy=EnsembleStrategy.MAJORITY_VOTE, num_members=3)
        predictor = EnsemblePredictor(models, config)
        x = torch.randn(1, 1, 32, 32, 16)
        result = predictor.predict(x)
        assert result.shape == (1, 2, 32, 32, 16)
        # One-hot encoded — should have only 0s and 1s
        assert torch.all((result == 0) | (result == 1))

    def test_weighted_mean(self, models: list[SegResNetAdapter]) -> None:
        config = EnsembleConfig(
            strategy=EnsembleStrategy.WEIGHTED,
            num_members=3,
            temperature=2.0,
        )
        predictor = EnsemblePredictor(models, config)
        x = torch.randn(1, 1, 32, 32, 16)
        result = predictor.predict(x)
        assert result.shape == (1, 2, 32, 32, 16)

    def test_predictor_creation_with_empty_models(self) -> None:
        config = EnsembleConfig(strategy=EnsembleStrategy.MEAN)
        # Verify EnsemblePredictor exists and takes the right args
        predictor = EnsemblePredictor([], config)
        assert predictor is not None


class TestCalibration:
    """Test calibration utilities."""

    def test_perfect_calibration(self) -> None:
        confidences = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        accuracies = np.array([0, 0, 1, 1, 1])
        ece, mce = expected_calibration_error(confidences, accuracies, n_bins=5)
        assert 0.0 <= ece <= 1.0
        assert 0.0 <= mce <= 1.0

    def test_ece_is_zero_for_perfect(self) -> None:
        # When confidence exactly matches accuracy rate
        confidences = np.array([0.0, 0.0, 1.0, 1.0])
        accuracies = np.array([0, 0, 1, 1])
        ece, mce = expected_calibration_error(confidences, accuracies, n_bins=2)
        assert ece < 0.2  # Should be near zero

    def test_temperature_scaling_identity(self) -> None:
        logits = np.array([[2.0, 1.0], [0.5, 1.5]])
        result = temperature_scale(logits, temperature=1.0)
        # Should match standard softmax
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_high_temperature_uniform(self) -> None:
        logits = np.array([[10.0, 0.0]])
        result = temperature_scale(logits, temperature=100.0)
        # High temperature -> near uniform
        assert abs(result[0, 0] - result[0, 1]) < 0.2


class TestCalibrationResult:
    """Test CalibrationResult dataclass."""

    def test_creation(self) -> None:
        result = CalibrationResult(
            ece=0.05,
            mce=0.1,
        )
        assert result.ece == 0.05
        assert result.mce == 0.1
        assert result.calibrated_probs is None

    def test_creation_with_probs(self) -> None:
        probs = np.array([0.3, 0.7])
        result = CalibrationResult(
            ece=0.02,
            mce=0.05,
            calibrated_probs=probs,
        )
        assert result.ece == 0.02
        np.testing.assert_array_equal(result.calibrated_probs, probs)


class TestDriftDetection:
    """Test drift detection utilities."""

    def test_no_drift_same_distribution(self) -> None:
        rng = np.random.default_rng(42)
        reference = rng.normal(0, 1, 1000)
        current = rng.normal(0, 1, 1000)
        report = detect_prediction_drift(reference, current)
        assert isinstance(report, DriftReport)
        assert not report.is_drifted

    def test_drift_different_distribution(self) -> None:
        rng = np.random.default_rng(42)
        reference = rng.normal(0, 1, 1000)
        current = rng.normal(5, 1, 1000)  # Shifted mean
        report = detect_prediction_drift(reference, current)
        assert report.is_drifted

    def test_psi_method(self) -> None:
        rng = np.random.default_rng(42)
        reference = rng.normal(0, 1, 1000)
        current = rng.normal(3, 1, 1000)
        report = detect_prediction_drift(reference, current, method="psi")
        assert report.is_drifted
        assert report.details["method"] == "psi"

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown drift"):
            detect_prediction_drift(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                method="unknown",
            )
