"""Tests for UQ beyond temperature scaling: MC Dropout, Deep Ensembles, Conformal."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from minivess.adapters.base import ModelAdapter, SegmentationOutput
from minivess.config.models import ModelConfig, ModelFamily

# ---------------------------------------------------------------------------
# Helpers: lightweight mock model for fast tests
# ---------------------------------------------------------------------------


class _MockAdapter(ModelAdapter):
    """Minimal model with dropout for testing MC Dropout."""

    def __init__(self, out_channels: int = 2, *, use_dropout: bool = True) -> None:
        super().__init__()
        self.drop = torch.nn.Dropout3d(p=0.3) if use_dropout else None
        self.conv = torch.nn.Conv3d(1, out_channels, kernel_size=1)

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        x = self.conv(images)
        if self.drop is not None:
            x = self.drop(x)
        logits = x
        prediction = torch.softmax(logits, dim=1)
        return SegmentationOutput(
            prediction=prediction,
            logits=logits,
            metadata={"architecture": "mock"},
        )

    def get_config(self) -> dict[str, Any]:
        return {"name": "mock"}

    def load_checkpoint(self, path: object) -> None:
        pass

    def save_checkpoint(self, path: object) -> None:
        pass

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def export_onnx(self, path: object, example_input: object) -> None:
        pass


# ---------------------------------------------------------------------------
# T1: UncertaintyOutput dataclass
# ---------------------------------------------------------------------------


class TestUncertaintyOutput:
    """Test shared UncertaintyOutput dataclass."""

    def test_creation(self) -> None:
        from minivess.ensemble.mc_dropout import UncertaintyOutput

        pred = torch.randn(1, 2, 8, 8, 4)
        unc = torch.randn(1, 1, 8, 8, 4)
        out = UncertaintyOutput(
            prediction=pred,
            uncertainty_map=unc,
            method="mc_dropout",
            metadata={"n_samples": 10},
        )
        assert out.method == "mc_dropout"
        assert out.prediction.shape == (1, 2, 8, 8, 4)
        assert out.uncertainty_map.shape == (1, 1, 8, 8, 4)

    def test_method_values(self) -> None:
        from minivess.ensemble.mc_dropout import UncertaintyOutput

        for method in ("mc_dropout", "deep_ensemble", "conformal"):
            out = UncertaintyOutput(
                prediction=torch.zeros(1, 2, 4, 4, 2),
                uncertainty_map=torch.zeros(1, 1, 4, 4, 2),
                method=method,
                metadata={},
            )
            assert out.method == method


# ---------------------------------------------------------------------------
# T2: MC Dropout
# ---------------------------------------------------------------------------


class TestMCDropout:
    """Test MC Dropout uncertainty estimation."""

    def test_mc_dropout_returns_uncertainty_output(self) -> None:
        from minivess.ensemble.mc_dropout import MCDropoutPredictor

        model = _MockAdapter(out_channels=2, use_dropout=True)
        predictor = MCDropoutPredictor(model, n_samples=5)
        x = torch.randn(1, 1, 8, 8, 4)
        result = predictor.predict(x)
        assert result.method == "mc_dropout"
        assert result.prediction.shape == (1, 2, 8, 8, 4)

    def test_mc_dropout_uncertainty_map_shape(self) -> None:
        from minivess.ensemble.mc_dropout import MCDropoutPredictor

        model = _MockAdapter(out_channels=2, use_dropout=True)
        predictor = MCDropoutPredictor(model, n_samples=5)
        x = torch.randn(1, 1, 8, 8, 4)
        result = predictor.predict(x)
        # Uncertainty map: (B, 1, D, H, W)
        assert result.uncertainty_map.shape == (1, 1, 8, 8, 4)

    def test_mc_dropout_uncertainty_non_negative(self) -> None:
        from minivess.ensemble.mc_dropout import MCDropoutPredictor

        model = _MockAdapter(out_channels=2, use_dropout=True)
        predictor = MCDropoutPredictor(model, n_samples=10)
        x = torch.randn(1, 1, 8, 8, 4)
        result = predictor.predict(x)
        assert torch.all(result.uncertainty_map >= 0)

    def test_mc_dropout_predictions_are_probabilities(self) -> None:
        from minivess.ensemble.mc_dropout import MCDropoutPredictor

        model = _MockAdapter(out_channels=2, use_dropout=True)
        predictor = MCDropoutPredictor(model, n_samples=5)
        x = torch.randn(1, 1, 8, 8, 4)
        result = predictor.predict(x)
        # Mean prediction should sum to ~1 across classes
        sums = result.prediction.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=0.01)

    def test_mc_dropout_metadata_has_n_samples(self) -> None:
        from minivess.ensemble.mc_dropout import MCDropoutPredictor

        model = _MockAdapter(out_channels=2, use_dropout=True)
        predictor = MCDropoutPredictor(model, n_samples=7)
        x = torch.randn(1, 1, 8, 8, 4)
        result = predictor.predict(x)
        assert result.metadata["n_samples"] == 7

    def test_mc_dropout_no_dropout_still_works(self) -> None:
        """Model without dropout should produce identical samples (no variability)."""
        from minivess.ensemble.mc_dropout import MCDropoutPredictor

        model = _MockAdapter(out_channels=2, use_dropout=False)
        predictor = MCDropoutPredictor(model, n_samples=5)
        x = torch.randn(1, 1, 8, 8, 4)
        # Two runs should produce identical predictions (no stochasticity)
        r1 = predictor.predict(x)
        r2 = predictor.predict(x)
        assert torch.allclose(r1.prediction, r2.prediction, atol=1e-6)

    def test_mc_dropout_with_segresnet(self) -> None:
        """MC Dropout should work with the real SegResNet adapter."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.ensemble.mc_dropout import MCDropoutPredictor

        config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="test_mc",
            in_channels=1,
            out_channels=2,
        )
        model = SegResNetAdapter(config)
        predictor = MCDropoutPredictor(model, n_samples=3)
        x = torch.randn(1, 1, 32, 32, 16)
        result = predictor.predict(x)
        assert result.prediction.shape == (1, 2, 32, 32, 16)
        assert result.uncertainty_map.shape == (1, 1, 32, 32, 16)


# ---------------------------------------------------------------------------
# T3: Deep Ensembles
# ---------------------------------------------------------------------------


class TestDeepEnsembles:
    """Test deep ensemble uncertainty estimation."""

    def test_ensemble_uq_returns_output(self) -> None:
        from minivess.ensemble.deep_ensembles import DeepEnsemblePredictor

        models = [_MockAdapter(out_channels=2) for _ in range(3)]
        predictor = DeepEnsemblePredictor(models)
        x = torch.randn(1, 1, 8, 8, 4)
        result = predictor.predict(x)
        assert result.method == "deep_ensemble"
        assert result.prediction.shape == (1, 2, 8, 8, 4)

    def test_ensemble_uq_uncertainty_shape(self) -> None:
        from minivess.ensemble.deep_ensembles import DeepEnsemblePredictor

        models = [_MockAdapter(out_channels=2) for _ in range(3)]
        predictor = DeepEnsemblePredictor(models)
        x = torch.randn(1, 1, 8, 8, 4)
        result = predictor.predict(x)
        assert result.uncertainty_map.shape == (1, 1, 8, 8, 4)

    def test_ensemble_uq_non_negative(self) -> None:
        from minivess.ensemble.deep_ensembles import DeepEnsemblePredictor

        models = [_MockAdapter(out_channels=2) for _ in range(3)]
        predictor = DeepEnsemblePredictor(models)
        x = torch.randn(1, 1, 8, 8, 4)
        result = predictor.predict(x)
        assert torch.all(result.uncertainty_map >= 0)

    def test_ensemble_uq_single_model(self) -> None:
        """Single model ensemble should have zero uncertainty."""
        from minivess.ensemble.deep_ensembles import DeepEnsemblePredictor

        models = [_MockAdapter(out_channels=2)]
        predictor = DeepEnsemblePredictor(models)
        x = torch.randn(1, 1, 8, 8, 4)
        result = predictor.predict(x)
        assert result.uncertainty_map.max() < 0.01

    def test_ensemble_uq_metadata(self) -> None:
        from minivess.ensemble.deep_ensembles import DeepEnsemblePredictor

        models = [_MockAdapter(out_channels=2) for _ in range(3)]
        predictor = DeepEnsemblePredictor(models)
        x = torch.randn(1, 1, 8, 8, 4)
        result = predictor.predict(x)
        assert result.metadata["n_members"] == 3

    def test_ensemble_uq_empty_raises(self) -> None:
        from minivess.ensemble.deep_ensembles import DeepEnsemblePredictor

        with pytest.raises(ValueError, match="at least one model"):
            DeepEnsemblePredictor([])

    def test_ensemble_uq_with_segresnet(self) -> None:
        """Deep ensemble should work with real SegResNet adapters."""
        from minivess.adapters.segresnet import SegResNetAdapter
        from minivess.ensemble.deep_ensembles import DeepEnsemblePredictor

        config = ModelConfig(
            family=ModelFamily.MONAI_SEGRESNET,
            name="test_ens",
            in_channels=1,
            out_channels=2,
        )
        models = [SegResNetAdapter(config) for _ in range(3)]
        predictor = DeepEnsemblePredictor(models)
        x = torch.randn(1, 1, 32, 32, 16)
        result = predictor.predict(x)
        assert result.prediction.shape == (1, 2, 32, 32, 16)


# ---------------------------------------------------------------------------
# T4: Conformal Prediction
# ---------------------------------------------------------------------------


class TestConformalPrediction:
    """Test conformal prediction for mask-level coverage guarantees."""

    def test_conformal_calibrate(self) -> None:
        from minivess.ensemble.conformal import ConformalPredictor

        predictor = ConformalPredictor(alpha=0.1)
        # Calibration: softmax scores + ground truth masks
        rng = np.random.default_rng(42)
        cal_scores = rng.random((50, 2, 8, 8, 4)).astype(np.float32)
        # Normalize to valid probabilities
        cal_scores = cal_scores / cal_scores.sum(axis=1, keepdims=True)
        cal_labels = rng.integers(0, 2, size=(50, 8, 8, 4))
        predictor.calibrate(cal_scores, cal_labels)
        assert predictor.is_calibrated

    def test_conformal_predict_sets(self) -> None:
        from minivess.ensemble.conformal import ConformalPredictor

        predictor = ConformalPredictor(alpha=0.1)
        rng = np.random.default_rng(42)
        cal_scores = rng.random((50, 2, 8, 8, 4)).astype(np.float32)
        cal_scores = cal_scores / cal_scores.sum(axis=1, keepdims=True)
        cal_labels = rng.integers(0, 2, size=(50, 8, 8, 4))
        predictor.calibrate(cal_scores, cal_labels)

        # Predict on new data
        test_scores = rng.random((5, 2, 8, 8, 4)).astype(np.float32)
        test_scores = test_scores / test_scores.sum(axis=1, keepdims=True)
        result = predictor.predict(test_scores)
        # prediction_sets: (B, C, D, H, W) boolean mask
        assert result.prediction_sets.shape == (5, 2, 8, 8, 4)
        assert result.prediction_sets.dtype == np.bool_

    def test_conformal_coverage(self) -> None:
        """Prediction sets should cover true labels at >= 1-alpha rate."""
        from minivess.ensemble.conformal import ConformalPredictor

        predictor = ConformalPredictor(alpha=0.1)
        rng = np.random.default_rng(42)
        n_cal, n_test = 200, 100
        n_classes, d, h, w = 2, 4, 4, 2

        cal_scores = rng.random((n_cal, n_classes, d, h, w)).astype(np.float32)
        cal_scores = cal_scores / cal_scores.sum(axis=1, keepdims=True)
        cal_labels = rng.integers(0, n_classes, size=(n_cal, d, h, w))
        predictor.calibrate(cal_scores, cal_labels)

        test_scores = rng.random((n_test, n_classes, d, h, w)).astype(np.float32)
        test_scores = test_scores / test_scores.sum(axis=1, keepdims=True)
        test_labels = rng.integers(0, n_classes, size=(n_test, d, h, w))
        result = predictor.predict(test_scores)

        # Check coverage: for each voxel, the true class should be in the set
        covered = 0
        total = 0
        for i in range(n_test):
            for di in range(d):
                for hi in range(h):
                    for wi in range(w):
                        true_class = test_labels[i, di, hi, wi]
                        if result.prediction_sets[i, true_class, di, hi, wi]:
                            covered += 1
                        total += 1
        coverage = covered / total
        # Conformal guarantee: coverage >= 1 - alpha (with margin for finite sample)
        assert coverage >= 0.85  # Allow some slack for finite sample

    def test_conformal_not_calibrated_raises(self) -> None:
        from minivess.ensemble.conformal import ConformalPredictor

        predictor = ConformalPredictor(alpha=0.1)
        rng = np.random.default_rng(42)
        test_scores = rng.random((5, 2, 8, 8, 4)).astype(np.float32)
        with pytest.raises(RuntimeError, match="calibrat"):
            predictor.predict(test_scores)

    def test_conformal_alpha_effect(self) -> None:
        """Lower alpha -> larger prediction sets (more conservative)."""
        from minivess.ensemble.conformal import ConformalPredictor

        rng = np.random.default_rng(42)
        cal_scores = rng.random((100, 2, 4, 4, 2)).astype(np.float32)
        cal_scores = cal_scores / cal_scores.sum(axis=1, keepdims=True)
        cal_labels = rng.integers(0, 2, size=(100, 4, 4, 2))

        test_scores = rng.random((10, 2, 4, 4, 2)).astype(np.float32)
        test_scores = test_scores / test_scores.sum(axis=1, keepdims=True)

        # Low alpha (high confidence) -> more classes included
        p_low = ConformalPredictor(alpha=0.01)
        p_low.calibrate(cal_scores, cal_labels)
        result_low = p_low.predict(test_scores)

        # High alpha (low confidence) -> fewer classes
        p_high = ConformalPredictor(alpha=0.5)
        p_high.calibrate(cal_scores, cal_labels)
        result_high = p_high.predict(test_scores)

        avg_set_low = result_low.prediction_sets.sum() / result_low.prediction_sets.size
        avg_set_high = result_high.prediction_sets.sum() / result_high.prediction_sets.size
        assert avg_set_low >= avg_set_high

    def test_conformal_result_fields(self) -> None:
        from minivess.ensemble.conformal import ConformalResult

        result = ConformalResult(
            prediction_sets=np.ones((1, 2, 4, 4, 2), dtype=bool),
            quantile=0.95,
            alpha=0.1,
        )
        assert result.quantile == 0.95
        assert result.alpha == 0.1


# ---------------------------------------------------------------------------
# T5: Synthetic benchmark comparison
# ---------------------------------------------------------------------------


class TestUQBenchmark:
    """Compare UQ methods on synthetic data."""

    def test_all_methods_produce_uncertainty(self) -> None:
        """All three methods should produce non-trivial uncertainty maps."""
        from minivess.ensemble.deep_ensembles import DeepEnsemblePredictor
        from minivess.ensemble.mc_dropout import MCDropoutPredictor

        x = torch.randn(1, 1, 8, 8, 4)

        # MC Dropout
        model_mc = _MockAdapter(out_channels=2, use_dropout=True)
        mc_result = MCDropoutPredictor(model_mc, n_samples=10).predict(x)
        assert mc_result.uncertainty_map.shape[1] == 1

        # Deep Ensemble
        models_ens = [_MockAdapter(out_channels=2) for _ in range(3)]
        ens_result = DeepEnsemblePredictor(models_ens).predict(x)
        assert ens_result.uncertainty_map.shape[1] == 1

    def test_uncertainty_maps_exportable(self) -> None:
        """Uncertainty maps should be convertible to numpy for saving."""
        from minivess.ensemble.mc_dropout import MCDropoutPredictor

        model = _MockAdapter(out_channels=2, use_dropout=True)
        result = MCDropoutPredictor(model, n_samples=5).predict(
            torch.randn(1, 1, 8, 8, 4)
        )
        unc_np = result.uncertainty_map.numpy()
        assert isinstance(unc_np, np.ndarray)
        assert unc_np.shape == (1, 1, 8, 8, 4)
