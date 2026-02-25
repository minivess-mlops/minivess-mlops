from __future__ import annotations

import numpy as np
import pytest

from minivess.pipeline.evaluation import EvaluationRunner, FoldResult


def _make_binary_volume(shape: tuple[int, ...] = (8, 8, 8)) -> np.ndarray:
    """Create a random binary volume."""
    return np.random.default_rng(42).integers(0, 2, size=shape).astype(int)


class TestEvaluateVolume:
    def test_perfect_prediction_dsc_one(self) -> None:
        label = _make_binary_volume()
        runner = EvaluationRunner()
        result = runner.evaluate_volume(label, label)
        assert result["dsc"] == pytest.approx(1.0)

    def test_zero_overlap_dsc_zero(self) -> None:
        pred = np.ones((8, 8, 8), dtype=int)
        label = np.zeros((8, 8, 8), dtype=int)
        runner = EvaluationRunner()
        result = runner.evaluate_volume(pred, label)
        assert result["dsc"] == pytest.approx(0.0)

    def test_all_primary_metrics_present(self) -> None:
        pred = _make_binary_volume()
        label = _make_binary_volume((8, 8, 8))
        runner = EvaluationRunner()
        result = runner.evaluate_volume(pred, label)
        for name in EvaluationRunner.PRIMARY_METRICS:
            assert name in result

    def test_include_expensive_metrics(self) -> None:
        pred = _make_binary_volume()
        label = _make_binary_volume((8, 8, 8))
        runner = EvaluationRunner(include_expensive=True)
        result = runner.evaluate_volume(pred, label)
        for name in EvaluationRunner.EXPENSIVE_METRICS:
            assert name in result

    def test_no_nan_for_valid_input(self) -> None:
        rng = np.random.default_rng(123)
        pred = rng.integers(0, 2, size=(8, 8, 8))
        label = rng.integers(0, 2, size=(8, 8, 8))
        runner = EvaluationRunner()
        result = runner.evaluate_volume(pred, label)
        assert not np.isnan(result["dsc"])


class TestEvaluateFold:
    def test_fold_returns_fold_result(self) -> None:
        rng = np.random.default_rng(42)
        preds = [rng.integers(0, 2, size=(8, 8, 8)) for _ in range(5)]
        labels = [rng.integers(0, 2, size=(8, 8, 8)) for _ in range(5)]
        runner = EvaluationRunner()
        result = runner.evaluate_fold(preds, labels, n_resamples=100, seed=42)

        assert isinstance(result, FoldResult)
        assert "dsc" in result.per_volume_metrics
        assert len(result.per_volume_metrics["dsc"]) == 5

    def test_fold_aggregated_has_ci(self) -> None:
        rng = np.random.default_rng(42)
        preds = [rng.integers(0, 2, size=(8, 8, 8)) for _ in range(5)]
        labels = [rng.integers(0, 2, size=(8, 8, 8)) for _ in range(5)]
        runner = EvaluationRunner()
        result = runner.evaluate_fold(preds, labels, n_resamples=100, seed=42)

        assert "dsc" in result.aggregated
        ci = result.aggregated["dsc"]
        assert ci.lower <= ci.point_estimate <= ci.upper

    def test_fold_length_mismatch_raises(self) -> None:
        runner = EvaluationRunner()
        with pytest.raises(ValueError, match="length mismatch"):
            runner.evaluate_fold(
                [np.zeros((4, 4, 4))],
                [np.zeros((4, 4, 4)), np.zeros((4, 4, 4))],
            )

    def test_perfect_fold_dsc_near_one(self) -> None:
        rng = np.random.default_rng(42)
        labels = [rng.integers(0, 2, size=(8, 8, 8)) for _ in range(5)]
        runner = EvaluationRunner()
        result = runner.evaluate_fold(labels, labels, n_resamples=100, seed=42)
        assert result.aggregated["dsc"].point_estimate == pytest.approx(1.0)
