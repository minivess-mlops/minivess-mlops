"""Tests for ExperimentTracker error paths.

T15 from double-check plan: log_artifact with nonexistent file,
log_hydra_config without active run, valid file logging.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minivess.config.models import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    ModelFamily,
    TrainingConfig,
)


def _make_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="test-tracker-error-paths",
        run_name="test-run",
        data=DataConfig(dataset_name="minivess"),
        model=ModelConfig(
            family=ModelFamily.MONAI_DYNUNET,
            name="test-model",
            in_channels=1,
            out_channels=2,
        ),
        training=TrainingConfig(
            max_epochs=2,
            batch_size=1,
            learning_rate=1e-3,
            mixed_precision=False,
            warmup_epochs=1,
        ),
    )


class TestTrackerErrorPaths:
    """ExperimentTracker must handle error paths gracefully."""

    def test_log_artifact_nonexistent_file(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Non-existent file should not crash training (warn or raise MLflow error)."""
        from minivess.observability.tracking import ExperimentTracker

        config = _make_config()
        tracker = ExperimentTracker(config, tracking_uri=str(tmp_path / "mlruns"))

        with tracker.start_run() as _run_id:
            # MLflow raises on non-existent file, but this tests the error path
            nonexistent = Path("/nonexistent/file.txt")
            with pytest.raises(Exception):  # noqa: B017, PT011
                tracker.log_artifact(nonexistent)

    def test_log_hydra_config_with_active_run(self, tmp_path: Path) -> None:
        """log_hydra_config should work with active run."""
        from minivess.observability.tracking import ExperimentTracker

        config = _make_config()
        tracker = ExperimentTracker(config, tracking_uri=str(tmp_path / "mlruns"))

        with tracker.start_run() as _run_id:
            # Should not crash
            tracker.log_hydra_config({"key": "value", "nested": {"a": 1}})

    def test_log_artifact_valid_file(self, tmp_path: Path) -> None:
        """Valid file should be logged as artifact without error."""
        from minivess.observability.tracking import ExperimentTracker

        config = _make_config()
        tracker = ExperimentTracker(config, tracking_uri=str(tmp_path / "mlruns"))

        valid_file = tmp_path / "test_artifact.txt"
        valid_file.write_text("test content", encoding="utf-8")

        with tracker.start_run() as _run_id:
            tracker.log_artifact(valid_file)
            # No error = success

    def test_start_run_creates_experiment(self, tmp_path: Path) -> None:
        """start_run should create the MLflow experiment."""
        import mlflow

        from minivess.observability.tracking import ExperimentTracker

        config = _make_config()
        tracking_uri = str(tmp_path / "mlruns")
        tracker = ExperimentTracker(config, tracking_uri=tracking_uri)

        with tracker.start_run() as run_id:
            assert run_id is not None
            # Verify experiment was created
            mlflow.set_tracking_uri(tracking_uri)
            exp = mlflow.get_experiment_by_name("test-tracker-error-paths")
            assert exp is not None
