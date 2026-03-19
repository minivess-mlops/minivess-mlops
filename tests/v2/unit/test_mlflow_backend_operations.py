"""L1 generic MLflow backend operations (#621).

Backend-agnostic tests that must work on any MLflow backend
(filesystem and server). Uses the parametrized ``mlflow_backend``
fixture from ``tests/v2/fixtures/mlflow_backends.py``.

These tests verify core MLflow operations without cloud credentials,
ensuring the tracking API works identically regardless of backend type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow

if TYPE_CHECKING:
    from pathlib import Path

# Register the mlflow_backend fixture from the fixtures module.
pytest_plugins = ["tests.v2.fixtures.mlflow_backends"]


class TestBackendOperations:
    """Backend-agnostic MLflow operations that must work on any backend."""

    def test_create_experiment(self, mlflow_backend: str) -> None:
        """Create an experiment on any backend."""
        mlflow.set_tracking_uri(mlflow_backend)
        exp_id = mlflow.create_experiment("test_create_exp")
        assert exp_id is not None
        exp = mlflow.get_experiment(exp_id)
        assert exp.name == "test_create_exp"

    def test_create_run_log_params_metrics(self, mlflow_backend: str) -> None:
        """Full run lifecycle: create -> log params/metrics -> end."""
        mlflow.set_tracking_uri(mlflow_backend)
        mlflow.create_experiment("test_run_lifecycle")
        mlflow.set_experiment("test_run_lifecycle")
        with mlflow.start_run() as run:
            mlflow.log_param("learning_rate", 0.001)
            mlflow.log_param("batch_size", 16)
            mlflow.log_metric("loss", 0.5)
            mlflow.log_metric("loss", 0.3, step=1)
            mlflow.log_metric("accuracy", 0.85)
        # Verify data persisted
        client = mlflow.MlflowClient(tracking_uri=mlflow_backend)
        fetched = client.get_run(run.info.run_id)
        assert fetched.data.params["learning_rate"] == "0.001"
        assert fetched.data.metrics["loss"] == 0.3  # latest step
        assert fetched.data.metrics["accuracy"] == 0.85

    def test_run_lifecycle_finished(self, mlflow_backend: str) -> None:
        """Run ends with FINISHED status by default."""
        mlflow.set_tracking_uri(mlflow_backend)
        mlflow.set_experiment("test_finished_status")
        with mlflow.start_run() as run:
            pass
        client = mlflow.MlflowClient(tracking_uri=mlflow_backend)
        fetched = client.get_run(run.info.run_id)
        assert fetched.info.status == "FINISHED"

    def test_run_lifecycle_failed(self, mlflow_backend: str) -> None:
        """Run with exception ends with FAILED status."""
        mlflow.set_tracking_uri(mlflow_backend)
        client = mlflow.MlflowClient(tracking_uri=mlflow_backend)
        exp = mlflow.get_experiment_by_name("test_failed_status")
        exp_id = (
            exp.experiment_id if exp else client.create_experiment("test_failed_status")
        )
        run = client.create_run(experiment_id=exp_id)
        client.set_terminated(run.info.run_id, status="FAILED")
        fetched = client.get_run(run.info.run_id)
        assert fetched.info.status == "FAILED"

    def test_search_runs_filter(self, mlflow_backend: str) -> None:
        """Search runs with filter_string works on any backend."""
        mlflow.set_tracking_uri(mlflow_backend)
        exp_id = mlflow.create_experiment("test_search_filter")
        mlflow.set_experiment("test_search_filter")

        # Create two runs with different params
        with mlflow.start_run():
            mlflow.log_param("model", "dynunet")
            mlflow.log_metric("dsc", 0.82)
        with mlflow.start_run():
            mlflow.log_param("model", "vesselfm")
            mlflow.log_metric("dsc", 0.78)

        # Search for dynunet runs only
        client = mlflow.MlflowClient(tracking_uri=mlflow_backend)
        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string="params.model = 'dynunet'",
        )
        assert len(runs) == 1
        assert runs[0].data.params["model"] == "dynunet"

    def test_log_artifact_roundtrip(self, mlflow_backend: str, tmp_path: Path) -> None:
        """Upload and download a file artifact."""
        mlflow.set_tracking_uri(mlflow_backend)
        mlflow.set_experiment("test_artifact_roundtrip")

        artifact = tmp_path / "test_data.txt"
        artifact.write_text("hello from test", encoding="utf-8")

        with mlflow.start_run() as run:
            mlflow.log_artifact(str(artifact))

        client = mlflow.MlflowClient(tracking_uri=mlflow_backend)
        artifacts = client.list_artifacts(run.info.run_id)
        artifact_names = [a.path for a in artifacts]
        assert "test_data.txt" in artifact_names

    def test_set_tag_after_run_end(self, mlflow_backend: str) -> None:
        """Tags can be set on a completed run (champion tagging pattern)."""
        mlflow.set_tracking_uri(mlflow_backend)
        mlflow.set_experiment("test_tag_after_end")

        with mlflow.start_run() as run:
            mlflow.log_metric("loss", 0.1)

        client = mlflow.MlflowClient(tracking_uri=mlflow_backend)
        client.set_tag(run.info.run_id, "champion", "true")

        fetched = client.get_run(run.info.run_id)
        assert fetched.data.tags["champion"] == "true"

    def test_log_batch_metrics(self, mlflow_backend: str) -> None:
        """Log a batch of metrics in one call."""
        from mlflow.entities import Metric

        mlflow.set_tracking_uri(mlflow_backend)
        exp_id = mlflow.create_experiment("test_batch_metrics")

        client = mlflow.MlflowClient(tracking_uri=mlflow_backend)
        run = client.create_run(exp_id)

        metrics = [
            Metric(key=f"metric_{i}", value=float(i), timestamp=0, step=i)
            for i in range(50)
        ]
        client.log_batch(run.info.run_id, metrics=metrics)
        client.set_terminated(run.info.run_id, status="FINISHED")

        fetched = client.get_run(run.info.run_id)
        assert len(fetched.data.metrics) == 50
        assert fetched.data.metrics["metric_0"] == 0.0
        assert fetched.data.metrics["metric_49"] == 49.0
