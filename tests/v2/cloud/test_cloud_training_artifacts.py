"""Cloud tests: verify smoke test training artifacts on MLflow (#639, T4.1).

Post-run verification that smoke test runs produced expected artifacts
and metrics on the cloud MLflow server. Auto-skips without credentials.

Run: uv run pytest tests/v2/cloud/test_cloud_training_artifacts.py -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from mlflow import MlflowClient


@pytest.mark.cloud_mlflow
class TestCloudTrainingArtifacts:
    """Verify smoke test run artifacts on cloud MLflow."""

    def _find_latest_smoke_run(
        self,
        client: MlflowClient,
        model_family: str = "sam3_vanilla",
    ) -> object | None:
        """Find the latest FINISHED smoke test run for a model."""
        experiments = client.search_experiments()
        smoke_exps = [
            e for e in experiments if "smoke_test" in e.name and model_family in e.name
        ]
        if not smoke_exps:
            return None
        runs = client.search_runs(
            experiment_ids=[smoke_exps[0].experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            max_results=1,
            order_by=["start_time DESC"],
        )
        return runs[0] if runs else None

    def test_smoke_run_exists(
        self,
        cloud_mlflow_client: MlflowClient,
    ) -> None:
        """At least one FINISHED smoke test run exists."""
        run = self._find_latest_smoke_run(cloud_mlflow_client)
        if run is None:
            pytest.skip("No smoke test runs found — run make smoke-test-gpu first")
        assert run.info.status == "FINISHED"

    def test_smoke_run_has_metrics(
        self,
        cloud_mlflow_client: MlflowClient,
    ) -> None:
        """Smoke test run has logged metrics (at least loss)."""
        run = self._find_latest_smoke_run(cloud_mlflow_client)
        if run is None:
            pytest.skip("No smoke test runs found")
        assert run.data.metrics, "No metrics logged in smoke test run"

    def test_smoke_run_has_model_family_param(
        self,
        cloud_mlflow_client: MlflowClient,
    ) -> None:
        """Smoke test run logs model_family as param."""
        run = self._find_latest_smoke_run(cloud_mlflow_client)
        if run is None:
            pytest.skip("No smoke test runs found")
        assert (
            "model_family" in run.data.params or "arch_model_family" in run.data.params
        )
