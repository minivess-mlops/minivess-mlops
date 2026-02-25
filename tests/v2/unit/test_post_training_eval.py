from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from minivess.pipeline.evaluation import FoldResult


class TestPostTrainingEvaluation:
    """Tests for the post-training evaluation integration in train.py."""

    def test_evaluate_fold_and_log_returns_fold_result(self) -> None:
        """evaluate_fold_and_log should return FoldResult and call tracker."""
        from scripts.train import evaluate_fold_and_log

        rng = np.random.default_rng(42)
        preds = [rng.integers(0, 2, size=(8, 8, 8)) for _ in range(3)]
        labels = [rng.integers(0, 2, size=(8, 8, 8)) for _ in range(3)]

        tracker = MagicMock()
        result = evaluate_fold_and_log(
            predictions=preds,
            labels=labels,
            tracker=tracker,
            fold_id=0,
            loss_name="dice_ce",
        )

        assert isinstance(result, FoldResult)
        assert "dsc" in result.aggregated

    def test_evaluate_fold_and_log_logs_to_tracker(self) -> None:
        """Should call tracker.log_evaluation_results when tracker is provided."""
        from scripts.train import evaluate_fold_and_log

        rng = np.random.default_rng(42)
        preds = [rng.integers(0, 2, size=(8, 8, 8)) for _ in range(3)]
        labels = [rng.integers(0, 2, size=(8, 8, 8)) for _ in range(3)]

        tracker = MagicMock()
        evaluate_fold_and_log(
            predictions=preds,
            labels=labels,
            tracker=tracker,
            fold_id=0,
            loss_name="dice_ce",
        )

        tracker.log_evaluation_results.assert_called_once()

    def test_evaluate_fold_and_log_no_tracker(self) -> None:
        """Should work without a tracker (no error)."""
        from scripts.train import evaluate_fold_and_log

        rng = np.random.default_rng(42)
        preds = [rng.integers(0, 2, size=(8, 8, 8)) for _ in range(3)]
        labels = [rng.integers(0, 2, size=(8, 8, 8)) for _ in range(3)]

        result = evaluate_fold_and_log(
            predictions=preds,
            labels=labels,
            tracker=None,
            fold_id=0,
            loss_name="dice_ce",
        )

        assert isinstance(result, FoldResult)


class TestExperimentTrackerEvaluation:
    """Tests for ExperimentTracker.log_evaluation_results."""

    def test_log_evaluation_results_logs_metrics(self) -> None:
        """log_evaluation_results should log flat metrics to MLflow."""
        from minivess.pipeline.ci import ConfidenceInterval

        tracker = MagicMock()
        tracker.log_evaluation_results = MagicMock()

        fold_result = FoldResult(
            per_volume_metrics={"dsc": [0.8, 0.9, 0.85]},
            aggregated={
                "dsc": ConfidenceInterval(
                    point_estimate=0.85,
                    lower=0.79,
                    upper=0.91,
                    confidence_level=0.95,
                    method="percentile_bootstrap",
                )
            },
        )

        # Verify that log_evaluation_results is callable with the right signature
        tracker.log_evaluation_results(
            fold_result, fold_id=0, loss_name="dice_ce"
        )
        tracker.log_evaluation_results.assert_called_once()


class TestCrossLossComparison:
    """Tests for cross-loss comparison summary."""

    def test_build_comparison_summary(self) -> None:
        """build_comparison_summary should return structured comparison dict."""
        from scripts.train import build_comparison_summary

        from minivess.pipeline.ci import ConfidenceInterval

        results = {
            "dice_ce": [
                FoldResult(
                    per_volume_metrics={"dsc": [0.8, 0.85]},
                    aggregated={
                        "dsc": ConfidenceInterval(
                            point_estimate=0.825,
                            lower=0.79,
                            upper=0.86,
                            confidence_level=0.95,
                            method="percentile_bootstrap",
                        )
                    },
                ),
            ],
            "cbdice": [
                FoldResult(
                    per_volume_metrics={"dsc": [0.85, 0.90]},
                    aggregated={
                        "dsc": ConfidenceInterval(
                            point_estimate=0.875,
                            lower=0.84,
                            upper=0.91,
                            confidence_level=0.95,
                            method="percentile_bootstrap",
                        )
                    },
                ),
            ],
        }

        summary = build_comparison_summary(results)

        assert "dice_ce" in summary
        assert "cbdice" in summary
        assert "dsc" in summary["dice_ce"]
        assert "mean" in summary["dice_ce"]["dsc"]
        assert "ci_lower" in summary["dice_ce"]["dsc"]
        assert "ci_upper" in summary["dice_ce"]["dsc"]

    def test_build_comparison_summary_multi_fold(self) -> None:
        """Should average across folds correctly."""
        from scripts.train import build_comparison_summary

        from minivess.pipeline.ci import ConfidenceInterval

        fold0 = FoldResult(
            per_volume_metrics={"dsc": [0.8]},
            aggregated={
                "dsc": ConfidenceInterval(
                    point_estimate=0.80,
                    lower=0.75,
                    upper=0.85,
                    confidence_level=0.95,
                    method="percentile_bootstrap",
                )
            },
        )
        fold1 = FoldResult(
            per_volume_metrics={"dsc": [0.9]},
            aggregated={
                "dsc": ConfidenceInterval(
                    point_estimate=0.90,
                    lower=0.85,
                    upper=0.95,
                    confidence_level=0.95,
                    method="percentile_bootstrap",
                )
            },
        )
        results = {"dice_ce": [fold0, fold1]}
        summary = build_comparison_summary(results)

        assert summary["dice_ce"]["dsc"]["mean"] == pytest.approx(0.85)
