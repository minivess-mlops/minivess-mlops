"""DagsHub-specific MLflow tests.

Verify that MLflow tracking works correctly with DagsHub as the backend:
token authentication, experiment creation, metric logging, artifact upload.

Runs only when DAGSHUB_TOKEN is set. Part of `make test-cloud-mlflow`.
NOT run in CI (GitHub Actions disabled, Rule #21).

Usage:
    DAGSHUB_TOKEN=<token> make test-cloud-mlflow
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from mlflow import MlflowClient

# Skip entire module if DagsHub credentials not available
pytestmark = pytest.mark.cloud_mlflow

_DAGSHUB_TOKEN = os.environ.get("DAGSHUB_TOKEN", "")
_MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
_IS_DAGSHUB = "dagshub.com" in _MLFLOW_URI


@pytest.fixture()
def dagshub_client() -> MlflowClient:
    """Create an authenticated MlflowClient for DagsHub."""
    if not _IS_DAGSHUB:
        pytest.skip("MLFLOW_TRACKING_URI is not a DagsHub URL")
    if not _DAGSHUB_TOKEN:
        pytest.skip("DAGSHUB_TOKEN not set")

    import mlflow

    mlflow.set_tracking_uri(_MLFLOW_URI)
    return mlflow.MlflowClient()


class TestDagsHubAuthentication:
    """Verify DagsHub token-based authentication works."""

    def test_token_authentication_succeeds(self, dagshub_client: MlflowClient) -> None:
        """Authenticated request to DagsHub MLflow should list experiments."""
        experiments = dagshub_client.search_experiments()
        assert len(experiments) >= 1, "DagsHub should have at least 1 experiment"

    def test_experiment_names_visible(self, dagshub_client: MlflowClient) -> None:
        """All experiment names should be non-empty strings."""
        experiments = dagshub_client.search_experiments()
        for exp in experiments:
            assert exp.name, f"Experiment {exp.experiment_id} has empty name"


class TestDagsHubMetricLogging:
    """Verify metrics logged from local training appear on DagsHub."""

    def test_create_experiment_on_dagshub(self, dagshub_client: MlflowClient) -> None:
        """Can create a test experiment on DagsHub."""
        exp_name = f"_test_dagshub_verify_{int(time.time())}"
        exp_id = dagshub_client.create_experiment(exp_name)
        assert exp_id is not None

        # Clean up
        dagshub_client.delete_experiment(exp_id)

    def test_log_metric_roundtrip(self, dagshub_client: MlflowClient) -> None:
        """Log a metric and read it back."""
        import mlflow

        exp_name = f"_test_dagshub_metric_{int(time.time())}"
        mlflow.set_experiment(exp_name)

        with mlflow.start_run(run_name="test_metric_roundtrip") as run:
            mlflow.log_metric("test_val_dice", 0.8765)
            mlflow.log_param("test_model", "dynunet")
            mlflow.set_tag("test_tag", "dagshub_verification")

        # Read back
        fetched = dagshub_client.get_run(run.info.run_id)
        assert fetched.data.metrics["test_val_dice"] == pytest.approx(0.8765)
        assert fetched.data.params["test_model"] == "dynunet"
        assert fetched.data.tags["test_tag"] == "dagshub_verification"

        # Clean up
        dagshub_client.delete_run(run.info.run_id)
        exp = dagshub_client.get_experiment_by_name(exp_name)
        if exp:
            dagshub_client.delete_experiment(exp.experiment_id)

    def test_training_experiment_has_runs(self, dagshub_client: MlflowClient) -> None:
        """The local_dynunet_mechanics_training experiment should have runs."""
        exp = dagshub_client.get_experiment_by_name("local_dynunet_mechanics_training")
        if exp is None:
            pytest.skip("local_dynunet_mechanics_training experiment not created yet")

        runs = dagshub_client.search_runs([exp.experiment_id])
        assert len(runs) >= 1, (
            "local_dynunet_mechanics_training should have at least 1 run "
            "(training should have started)"
        )


class TestDagsHubArtifactStorage:
    """Verify artifact upload to DagsHub's S3-compatible storage."""

    def test_log_text_artifact(self, dagshub_client: MlflowClient, tmp_path: object) -> None:
        """Log a small text artifact and verify it exists."""
        import mlflow
        from pathlib import Path

        exp_name = f"_test_dagshub_artifact_{int(time.time())}"
        mlflow.set_experiment(exp_name)

        artifact_path = Path(str(tmp_path)) / "test_artifact.txt"
        artifact_path.write_text("DagsHub artifact test", encoding="utf-8")

        with mlflow.start_run(run_name="test_artifact_roundtrip") as run:
            mlflow.log_artifact(str(artifact_path))

        # Verify artifact exists
        artifacts = dagshub_client.list_artifacts(run.info.run_id)
        artifact_names = [a.path for a in artifacts]
        assert "test_artifact.txt" in artifact_names, (
            f"Artifact not found. Available: {artifact_names}"
        )

        # Clean up
        dagshub_client.delete_run(run.info.run_id)
        exp = dagshub_client.get_experiment_by_name(exp_name)
        if exp:
            dagshub_client.delete_experiment(exp.experiment_id)
