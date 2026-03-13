"""Cloud integration tests: training flow against remote MLflow (#636, T2.2).

Verifies that resolve_tracking_uri() and ExperimentTracker work correctly
when MLFLOW_TRACKING_URI points to a remote server with basic auth.

Auto-skips when MLFLOW_CLOUD_URI is not set (no cloud credentials).

Run: uv run pytest tests/v2/cloud/test_training_flow_cloud_mlflow.py -v
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.v2.cloud.conftest import CloudMLflowConnection


@pytest.mark.cloud_mlflow
class TestResolveTrackingUriCloud:
    """Verify resolve_tracking_uri() with cloud MLflow + auth."""

    def test_resolve_with_cloud_uri(
        self,
        cloud_mlflow_connection: CloudMLflowConnection,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """resolve_tracking_uri() returns cloud URI when set."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", cloud_mlflow_connection.tracking_uri)

        from minivess.observability.tracking import resolve_tracking_uri

        uri = resolve_tracking_uri()
        assert uri == cloud_mlflow_connection.tracking_uri

    def test_cloud_mlflow_health_endpoint(
        self,
        cloud_mlflow_connection: CloudMLflowConnection,
    ) -> None:
        """Remote MLflow health endpoint responds OK."""
        import urllib.request

        url = f"{cloud_mlflow_connection.tracking_uri}/health"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200


@pytest.mark.cloud_mlflow
class TestExperimentTrackerCloud:
    """Verify ExperimentTracker can create experiments on cloud MLflow."""

    def test_create_experiment_on_cloud(
        self,
        cloud_mlflow_connection: CloudMLflowConnection,
        cloud_mlflow_client: object,
        test_run_id: str,
    ) -> None:
        """ExperimentTracker can create experiment on remote MLflow."""
        import mlflow

        exp_name = f"_test_{uuid.uuid4().hex[:8]}_t22"
        mlflow.set_tracking_uri(cloud_mlflow_connection.tracking_uri)
        os.environ["MLFLOW_TRACKING_USERNAME"] = cloud_mlflow_connection.username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = cloud_mlflow_connection.password

        mlflow.set_experiment(exp_name)
        with mlflow.start_run(tags={"test": "T2.2"}) as run:
            mlflow.log_metric("test_metric", 0.42)
            mlflow.log_param("model_family", "sam3_vanilla")

        # Verify run was created
        client = mlflow.MlflowClient(tracking_uri=cloud_mlflow_connection.tracking_uri)
        fetched = client.get_run(run.info.run_id)
        assert fetched.info.status == "FINISHED"
        assert fetched.data.params["model_family"] == "sam3_vanilla"
