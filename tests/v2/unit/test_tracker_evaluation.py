from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from minivess.pipeline.ci import ConfidenceInterval
from minivess.pipeline.evaluation import FoldResult


class TestLogEvaluationResults:
    """Tests for ExperimentTracker.log_evaluation_results."""

    def _make_tracker(self):
        """Create an ExperimentTracker with mocked MLflow dependencies."""
        from minivess.config.models import (
            DataConfig,
            ExperimentConfig,
            ModelConfig,
            ModelFamily,
            TrainingConfig,
        )
        from minivess.observability.tracking import ExperimentTracker

        config = ExperimentConfig(
            experiment_name="test",
            data=DataConfig(dataset_name="test"),
            model=ModelConfig(family=ModelFamily.MONAI_DYNUNET, name="test"),
            training=TrainingConfig(),
        )
        with (
            patch("minivess.observability.tracking.mlflow"),
            patch("minivess.observability.tracking.MlflowClient"),
        ):
            tracker = ExperimentTracker(config, tracking_uri="file:///tmp/test")
        return tracker

    def test_log_evaluation_results_exists(self) -> None:
        """ExperimentTracker should have log_evaluation_results method."""
        tracker = self._make_tracker()
        assert hasattr(tracker, "log_evaluation_results")
        assert callable(tracker.log_evaluation_results)

    @patch("mlflow.log_metrics")
    def test_log_evaluation_results_calls_mlflow(
        self, mock_log_metrics: MagicMock
    ) -> None:
        """Should call mlflow.log_metrics with flattened CI data."""
        tracker = self._make_tracker()

        fold_result = FoldResult(
            per_volume_metrics={"dsc": [0.8, 0.9], "centreline_dsc": [0.7, 0.8]},
            aggregated={
                "dsc": ConfidenceInterval(
                    point_estimate=0.85,
                    lower=0.79,
                    upper=0.91,
                    confidence_level=0.95,
                    method="percentile_bootstrap",
                ),
                "centreline_dsc": ConfidenceInterval(
                    point_estimate=0.75,
                    lower=0.69,
                    upper=0.81,
                    confidence_level=0.95,
                    method="percentile_bootstrap",
                ),
            },
        )

        tracker.log_evaluation_results(fold_result, fold_id=0, loss_name="dice_ce")

        mock_log_metrics.assert_called_once()
        logged = mock_log_metrics.call_args[0][0]
        assert "eval_fold0_dsc" in logged
        assert "eval_fold0_dsc_ci_lower" in logged
        assert "eval_fold0_centreline_dsc" in logged

    @patch("mlflow.log_metrics")
    def test_log_evaluation_results_metric_values(
        self, mock_log_metrics: MagicMock
    ) -> None:
        """Logged metric values should match the fold result."""
        tracker = self._make_tracker()

        fold_result = FoldResult(
            per_volume_metrics={"dsc": [0.85]},
            aggregated={
                "dsc": ConfidenceInterval(
                    point_estimate=0.85,
                    lower=0.79,
                    upper=0.91,
                    confidence_level=0.95,
                    method="percentile_bootstrap",
                ),
            },
        )

        tracker.log_evaluation_results(fold_result, fold_id=1, loss_name="cbdice")

        logged = mock_log_metrics.call_args[0][0]
        assert logged["eval_fold1_dsc"] == pytest.approx(0.85)
        assert logged["eval_fold1_dsc_ci_lower"] == pytest.approx(0.79)
        assert logged["eval_fold1_dsc_ci_upper"] == pytest.approx(0.91)
