"""Tests for _discover_runs empty experiment and zero matching runs.

T7 from double-check plan: verify warnings on missing experiments and zero runs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    import pytest


class TestDiscoverRunsMissingExperiment:
    """EnsembleBuilder.discover_training_runs_raw must warn on missing experiment."""

    def test_missing_experiment_returns_empty_with_warning(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Any
    ) -> None:
        from minivess.ensemble.builder import EnsembleBuilder

        mock_config = MagicMock()
        mock_config.mlflow_training_experiment = "nonexistent_experiment"
        mock_config.require_eval_metrics = True

        builder = EnsembleBuilder(
            eval_config=mock_config,
            model_config={},
            tracking_uri=str(tmp_path / "mlruns"),
        )

        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None

        with (
            caplog.at_level(logging.WARNING),
            patch(
                "mlflow.tracking.MlflowClient",
                return_value=mock_client,
            ),
        ):
            result = builder.discover_training_runs_raw()

        assert result == []
        assert any("not found" in r.message.lower() for r in caplog.records), (
            f"Expected 'not found' warning, got: {[r.message for r in caplog.records]}"
        )

    def test_no_matching_runs_returns_empty(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Any
    ) -> None:
        from minivess.ensemble.builder import EnsembleBuilder

        mock_config = MagicMock()
        mock_config.mlflow_training_experiment = "test_experiment"
        mock_config.require_eval_metrics = True

        builder = EnsembleBuilder(
            eval_config=mock_config,
            model_config={},
            tracking_uri=str(tmp_path / "mlruns"),
        )

        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "1"

        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = []  # No matching runs

        with (
            caplog.at_level(logging.DEBUG),
            patch(
                "mlflow.tracking.MlflowClient",
                return_value=mock_client,
            ),
        ):
            result = builder.discover_training_runs_raw()

        assert result == []


class TestDiscoverRunsWithResults:
    """Builder returns run info when runs exist."""

    def test_matching_runs_returned(self, tmp_path: Any) -> None:
        from minivess.ensemble.builder import EnsembleBuilder

        mock_config = MagicMock()
        mock_config.mlflow_training_experiment = "test_experiment"
        mock_config.require_eval_metrics = False  # Don't require fold2 metrics

        builder = EnsembleBuilder(
            eval_config=mock_config,
            model_config={},
            tracking_uri=str(tmp_path / "mlruns"),
        )

        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "1"

        mock_run = MagicMock()
        mock_run.info.run_id = "run_abc123"
        mock_run.info.artifact_uri = str(tmp_path / "artifacts")
        mock_run.data.tags = {"loss_function": "dice_ce"}
        mock_run.data.metrics = {"val/dice": 0.85}

        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = [mock_run]

        with patch(
            "mlflow.tracking.MlflowClient",
            return_value=mock_client,
        ):
            result = builder.discover_training_runs_raw(require_eval_metrics=False)

        assert len(result) >= 1
        assert result[0]["loss_type"] == "dice_ce"
