"""Tests for deploy verification — training vs serving metric matching.

Covers:
- DeployVerificationResult structure
- verify_deploy_metrics with matching metrics
- verify_deploy_metrics with tolerance
- verify_deploy_metrics with mismatch
- verify_onnx_vs_pytorch output comparison
- Edge cases: empty arrays, single metric

Closes #184, #188.
"""

from __future__ import annotations

import numpy as np
import pytest

from minivess.pipeline.ci import ConfidenceInterval
from minivess.pipeline.deploy_verification import (
    DeployVerificationResult,
    verify_deploy_metrics,
    verify_onnx_vs_pytorch,
)
from minivess.pipeline.evaluation import FoldResult


def _make_fold_result(dsc: float = 0.85) -> FoldResult:
    return FoldResult(
        volume_ids=["mv01", "mv02"],
        per_volume_metrics={"dsc": [dsc, dsc - 0.05]},
        aggregated={
            "dsc": ConfidenceInterval(
                point_estimate=dsc,
                lower=dsc - 0.05,
                upper=dsc + 0.05,
                confidence_level=0.95,
                method="percentile",
            ),
        },
    )


class TestDeployVerificationResult:
    """Test DeployVerificationResult dataclass."""

    def test_construction(self) -> None:
        result = DeployVerificationResult(
            training_metrics={"dsc": 0.85},
            serving_metrics={"dsc": 0.85},
            metric_diffs={"dsc": 0.0},
            all_match=True,
            tolerance=1e-5,
        )
        assert result.all_match is True
        assert result.metric_diffs["dsc"] == pytest.approx(0.0)

    def test_mismatch_detected(self) -> None:
        result = DeployVerificationResult(
            training_metrics={"dsc": 0.85},
            serving_metrics={"dsc": 0.80},
            metric_diffs={"dsc": 0.05},
            all_match=False,
            tolerance=1e-5,
        )
        assert result.all_match is False


class TestVerifyDeployMetrics:
    """Test verify_deploy_metrics function."""

    def test_matching_metrics(self) -> None:
        fr = _make_fold_result(0.85)
        preds = [np.ones((10, 10, 10), dtype=np.int64)] * 2
        labels = [np.ones((10, 10, 10), dtype=np.int64)] * 2
        result = verify_deploy_metrics(fr, preds, labels)
        assert isinstance(result, DeployVerificationResult)

    def test_tolerance_default(self) -> None:
        fr = _make_fold_result()
        preds = [np.ones((10, 10, 10), dtype=np.int64)] * 2
        labels = [np.ones((10, 10, 10), dtype=np.int64)] * 2
        result = verify_deploy_metrics(fr, preds, labels)
        assert result.tolerance == pytest.approx(1e-5)

    def test_custom_tolerance(self) -> None:
        fr = _make_fold_result()
        preds = [np.ones((10, 10, 10), dtype=np.int64)] * 2
        labels = [np.ones((10, 10, 10), dtype=np.int64)] * 2
        result = verify_deploy_metrics(fr, preds, labels, tolerance=0.01)
        assert result.tolerance == pytest.approx(0.01)

    def test_empty_predictions_raises(self) -> None:
        fr = _make_fold_result()
        with pytest.raises(ValueError, match="predictions"):
            verify_deploy_metrics(fr, [], [])

    def test_training_metrics_populated(self) -> None:
        fr = _make_fold_result(0.85)
        preds = [np.ones((10, 10, 10), dtype=np.int64)] * 2
        labels = [np.ones((10, 10, 10), dtype=np.int64)] * 2
        result = verify_deploy_metrics(fr, preds, labels)
        assert "dsc" in result.training_metrics

    def test_perfect_match_all_ones(self) -> None:
        """Perfect predictions should give dsc=1.0 for serving metrics."""
        fr = FoldResult(
            volume_ids=["mv01"],
            per_volume_metrics={"dsc": [1.0]},
            aggregated={
                "dsc": ConfidenceInterval(
                    point_estimate=1.0,
                    lower=1.0,
                    upper=1.0,
                    confidence_level=0.95,
                    method="percentile",
                ),
            },
        )
        preds = [np.ones((5, 5, 5), dtype=np.int64)]
        labels = [np.ones((5, 5, 5), dtype=np.int64)]
        result = verify_deploy_metrics(fr, preds, labels)
        assert result.serving_metrics["dsc"] == pytest.approx(1.0)
        assert result.all_match is True


class TestVerifyOnnxVsPytorch:
    """Test ONNX vs PyTorch output comparison."""

    def test_matching_outputs(self) -> None:
        """Two identical arrays should match."""
        output_a = np.random.default_rng(42).random((1, 2, 8, 8, 8)).astype(np.float32)
        result = verify_onnx_vs_pytorch(output_a, output_a)
        assert result is True

    def test_mismatched_outputs(self) -> None:
        """Very different arrays should not match."""
        output_a = np.zeros((1, 2, 8, 8, 8), dtype=np.float32)
        output_b = np.ones((1, 2, 8, 8, 8), dtype=np.float32)
        result = verify_onnx_vs_pytorch(output_a, output_b)
        assert result is False

    def test_close_outputs_within_tolerance(self) -> None:
        """Slightly different arrays should match with large tolerance."""
        rng = np.random.default_rng(42)
        output_a = rng.random((1, 2, 8, 8, 8)).astype(np.float32)
        output_b = output_a + 1e-4
        result = verify_onnx_vs_pytorch(output_a, output_b, tolerance=1e-3)
        assert result is True

    def test_close_outputs_outside_tolerance(self) -> None:
        """Slightly different arrays should fail with tight tolerance."""
        rng = np.random.default_rng(42)
        output_a = rng.random((1, 2, 8, 8, 8)).astype(np.float32)
        output_b = output_a + 0.01
        result = verify_onnx_vs_pytorch(output_a, output_b, tolerance=1e-4)
        assert result is False
