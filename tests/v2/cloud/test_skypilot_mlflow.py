"""L4 SkyPilot -> Cloud MLflow tests (#624).

Unit tests for SkyPilot YAML configuration and tracking URI resolution.
Cloud tests simulate SkyPilot VM logging to remote MLflow.

Unit tests (no creds): run in staging tier.
Cloud tests (@pytest.mark.skypilot_cloud): require MLFLOW_CLOUD_* env vars.

Note: train_generic.yaml and train_hpo_sweep.yaml were deleted (bare-VM, Docker mandate).
All SkyPilot YAMLs now use Docker image_id pattern (see smoke_test_gpu.yaml).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from mlflow import MlflowClient


class TestSkyPilotTrackingUriResolution:
    """Verify tracking URI assembly from SkyPilot-style env vars.

    No cloud credentials needed — pure unit tests.
    """

    def test_resolve_from_skypilot_host_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MLFLOW_TRACKING_URI env var resolves correctly."""
        from minivess.observability.tracking import resolve_tracking_uri

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://my-mlflow-server:5000")
        monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
        monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
        uri = resolve_tracking_uri(tracking_uri=None, use_dynaconf=False)
        assert uri == "http://my-mlflow-server:5000"

    def test_auth_credentials_embedded_in_uri(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MLFLOW_TRACKING_USERNAME/PASSWORD -> embedded in URI."""
        from minivess.observability.tracking import resolve_tracking_uri

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "admin")
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "secret")
        uri = resolve_tracking_uri(tracking_uri=None, use_dynaconf=False)
        assert "admin:secret@" in uri


class TestSmokeTestYamlConfiguration:
    """Validate smoke_test_gpu.yaml references correct env vars."""

    def test_smoke_test_yaml_has_mlflow_tracking(self) -> None:
        """smoke_test_gpu.yaml has MLFLOW_TRACKING_URI in envs."""
        config = yaml.safe_load(
            Path("deployment/skypilot/smoke_test_gpu.yaml").read_text(encoding="utf-8")
        )
        envs = config.get("envs", {})
        assert "MLFLOW_TRACKING_URI" in envs

    def test_smoke_test_yaml_has_required_fields(self) -> None:
        """smoke_test_gpu.yaml has resources, envs, and run sections."""
        config = yaml.safe_load(
            Path("deployment/skypilot/smoke_test_gpu.yaml").read_text(encoding="utf-8")
        )
        assert "resources" in config
        assert "envs" in config
        assert "run" in config or "setup" in config

    def test_smoke_test_yaml_requests_gpu(self) -> None:
        """smoke_test_gpu.yaml requests GPU resources."""
        config = yaml.safe_load(
            Path("deployment/skypilot/smoke_test_gpu.yaml").read_text(encoding="utf-8")
        )
        resources = config.get("resources", {})
        assert "accelerators" in resources, "SkyPilot YAML must request GPU"

    def test_smoke_test_yaml_uses_docker_image(self) -> None:
        """smoke_test_gpu.yaml uses Docker image_id (not bare-VM)."""
        config = yaml.safe_load(
            Path("deployment/skypilot/smoke_test_gpu.yaml").read_text(encoding="utf-8")
        )
        resources = config.get("resources", {})
        image_id = resources.get("image_id", "")
        assert str(image_id).startswith("docker:"), (
            f"Must use Docker image_id, got: {image_id}"
        )


@pytest.mark.skypilot_cloud
class TestSkyPilotRemoteLogging:
    """Simulate SkyPilot environment logging to remote MLflow.

    Requires MLFLOW_CLOUD_* credentials.
    """

    def test_simulated_spot_vm_logs_run(
        self,
        cloud_mlflow_client: MlflowClient,
        test_run_id: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Set env vars as SkyPilot would, create run, log metrics."""
        monkeypatch.setenv(
            "MLFLOW_TRACKING_URI",
            cloud_mlflow_client.tracking_uri,
        )
        exp_name = f"{test_run_id}_skypilot_sim"
        exp_id = cloud_mlflow_client.create_experiment(exp_name)
        run = cloud_mlflow_client.create_run(exp_id)
        cloud_mlflow_client.log_metric(run.info.run_id, "loss", 0.42)
        cloud_mlflow_client.log_param(run.info.run_id, "model", "dynunet")
        cloud_mlflow_client.set_terminated(run.info.run_id, "FINISHED")

        fetched = cloud_mlflow_client.get_run(run.info.run_id)
        assert fetched.data.metrics["loss"] == 0.42

    def test_simulated_spot_vm_uploads_artifact(
        self,
        cloud_mlflow_client: MlflowClient,
        test_run_id: str,
        tmp_path: Path,
    ) -> None:
        """Simulate artifact upload from ephemeral VM to S3."""
        artifact = tmp_path / "model_weights.bin"
        artifact.write_bytes(b"\x00" * 1024)  # 1 KB dummy
        exp_name = f"{test_run_id}_skypilot_artifact"
        exp_id = cloud_mlflow_client.create_experiment(exp_name)
        run = cloud_mlflow_client.create_run(exp_id)
        cloud_mlflow_client.log_artifact(run.info.run_id, str(artifact))
        artifacts = cloud_mlflow_client.list_artifacts(run.info.run_id)
        assert any(a.path == "model_weights.bin" for a in artifacts)

    def test_simulated_preemption_recovery(
        self,
        cloud_mlflow_client: MlflowClient,
        test_run_id: str,
    ) -> None:
        """Start run, simulate preemption (KILLED), resume with new run.

        Verify both runs are queryable and data intact.
        """
        exp_name = f"{test_run_id}_preemption"
        exp_id = cloud_mlflow_client.create_experiment(exp_name)

        # Phase 1: start run, log some metrics, get preempted
        run1 = cloud_mlflow_client.create_run(exp_id)
        cloud_mlflow_client.log_metric(run1.info.run_id, "epoch", 5)
        cloud_mlflow_client.set_terminated(run1.info.run_id, "KILLED")

        # Phase 2: resume on new spot VM — new run, same experiment
        run2 = cloud_mlflow_client.create_run(exp_id)
        cloud_mlflow_client.log_metric(run2.info.run_id, "epoch", 10)
        cloud_mlflow_client.log_param(
            run2.info.run_id, "resumed_from", run1.info.run_id
        )
        cloud_mlflow_client.set_terminated(run2.info.run_id, "FINISHED")

        # Verify both runs exist and data intact
        runs = cloud_mlflow_client.search_runs([exp_id])
        assert len(runs) == 2
        killed = [r for r in runs if r.info.status == "KILLED"]
        finished = [r for r in runs if r.info.status == "FINISHED"]
        assert len(killed) == 1
        assert len(finished) == 1
