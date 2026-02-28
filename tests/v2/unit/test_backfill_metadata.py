"""Tests for retroactive MLflow metadata backfill.

Uses isolated tmp_path MLflow backends, never touches real mlruns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow
import pytest

if TYPE_CHECKING:
    from pathlib import Path
from mlflow.tracking import MlflowClient


@pytest.fixture()
def isolated_mlflow(tmp_path: Path) -> str:
    """Create isolated MLflow backend with a test run."""
    uri = str(tmp_path / "mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test-backfill")
    return uri


def _create_test_run(
    uri: str,
    *,
    params: dict[str, str] | None = None,
    tags: dict[str, str] | None = None,
    status: str = "FINISHED",
) -> str:
    """Create a test run and return its run_id."""
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test-backfill")
    with mlflow.start_run(tags=tags or {}) as run:
        if params:
            mlflow.log_params(params)
        run_id = run.info.run_id

    if status != "FINISHED":
        client = MlflowClient(tracking_uri=uri)
        client.set_terminated(run_id, status=status)

    return run_id


class TestBackfillAddNewParams:
    """Backfill should add new params without touching existing ones."""

    def test_adds_new_params(self, isolated_mlflow: str) -> None:
        from minivess.pipeline.backfill_metadata import backfill_run

        run_id = _create_test_run(isolated_mlflow, params={"learning_rate": "0.001"})
        backfill_run(
            run_id,
            new_params={"sys_python_version": "3.13.2"},
            tracking_uri=isolated_mlflow,
        )

        client = MlflowClient(tracking_uri=isolated_mlflow)
        run = client.get_run(run_id)
        assert run.data.params["sys_python_version"] == "3.13.2"
        assert run.data.params["learning_rate"] == "0.001"

    def test_skips_existing_params(self, isolated_mlflow: str) -> None:
        """If a param already exists, skip it (don't throw)."""
        from minivess.pipeline.backfill_metadata import backfill_run

        run_id = _create_test_run(isolated_mlflow, params={"learning_rate": "0.001"})
        # Trying to set learning_rate to a different value should be skipped
        result = backfill_run(
            run_id,
            new_params={"learning_rate": "0.01", "sys_python_version": "3.13.2"},
            tracking_uri=isolated_mlflow,
        )

        client = MlflowClient(tracking_uri=isolated_mlflow)
        run = client.get_run(run_id)
        # Original value preserved
        assert run.data.params["learning_rate"] == "0.001"
        # New param added
        assert run.data.params["sys_python_version"] == "3.13.2"
        assert int(result["skipped"]) >= 1

    def test_idempotent_same_value(self, isolated_mlflow: str) -> None:
        """Running backfill twice with same values should not throw."""
        from minivess.pipeline.backfill_metadata import backfill_run

        run_id = _create_test_run(isolated_mlflow)
        backfill_run(
            run_id,
            new_params={"sys_python_version": "3.13.2"},
            tracking_uri=isolated_mlflow,
        )
        # Second call should succeed silently
        backfill_run(
            run_id,
            new_params={"sys_python_version": "3.13.2"},
            tracking_uri=isolated_mlflow,
        )


class TestBackfillPreservesRunStatus:
    """Backfill must not change run status."""

    def test_preserves_finished_status(self, isolated_mlflow: str) -> None:
        from minivess.pipeline.backfill_metadata import backfill_run

        run_id = _create_test_run(isolated_mlflow, status="FINISHED")
        backfill_run(
            run_id,
            new_params={"sys_python_version": "3.13.2"},
            tracking_uri=isolated_mlflow,
        )

        client = MlflowClient(tracking_uri=isolated_mlflow)
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"

    def test_preserves_failed_status(self, isolated_mlflow: str) -> None:
        from minivess.pipeline.backfill_metadata import backfill_run

        run_id = _create_test_run(isolated_mlflow, status="FAILED")
        backfill_run(
            run_id,
            new_params={"sys_python_version": "3.13.2"},
            tracking_uri=isolated_mlflow,
        )

        client = MlflowClient(tracking_uri=isolated_mlflow)
        run = client.get_run(run_id)
        assert run.info.status == "FAILED"

    def test_skips_running_status(self, isolated_mlflow: str) -> None:
        """RUNNING runs should be skipped (likely crashed)."""
        from minivess.pipeline.backfill_metadata import backfill_run

        run_id = _create_test_run(isolated_mlflow, status="RUNNING")
        result = backfill_run(
            run_id,
            new_params={"sys_python_version": "3.13.2"},
            tracking_uri=isolated_mlflow,
        )
        assert result["skipped_reason"] == "RUNNING"


class TestBackfillNote:
    """Backfill should add provenance note."""

    def test_adds_backfill_note(self, isolated_mlflow: str) -> None:
        from minivess.pipeline.backfill_metadata import backfill_run

        run_id = _create_test_run(isolated_mlflow)
        backfill_run(
            run_id,
            new_params={"sys_python_version": "3.13.2"},
            tracking_uri=isolated_mlflow,
        )

        client = MlflowClient(tracking_uri=isolated_mlflow)
        run = client.get_run(run_id)
        assert "sys_backfill_note" in run.data.params


class TestBackfillExperiment:
    """Test backfill_experiment that processes all runs."""

    def test_backfills_multiple_runs(self, isolated_mlflow: str) -> None:
        from minivess.pipeline.backfill_metadata import backfill_experiment

        _create_test_run(isolated_mlflow, params={"loss_name": "dice_ce"})
        _create_test_run(isolated_mlflow, params={"loss_name": "cbdice"})

        results = backfill_experiment(
            experiment_name="test-backfill",
            new_params={"sys_python_version": "3.13.2"},
            tracking_uri=isolated_mlflow,
        )
        assert results["total"] == 2
        assert results["updated"] == 2
