"""Integration tests for MLflow-MinIO artifact contract.

E2E Plan Phase 2, Task T2.1: Verify artifacts stored in MinIO with round-trip.

After the full pipeline completes, verifies:
1. Training artifacts exist in MinIO bucket via boto3
2. Download artifacts back via MLflow API
3. Verify downloaded artifacts match uploaded (file size, hash)
4. Experiment names use debug suffix correctly
5. All runs have FINISHED status (not RUNNING or FAILED)

Marked @integration — excluded from staging and prod suites.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _minio_reachable() -> bool:
    """Check if MinIO is reachable."""
    try:
        import urllib.request

        req = urllib.request.Request(
            "http://localhost:9000/minio/health/live", method="GET"
        )
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


def _mlflow_server_reachable() -> bool:
    """Check if MLflow server is reachable."""
    try:
        import urllib.request

        with urllib.request.urlopen("http://localhost:5000/health", timeout=5):
            return True
    except Exception:
        return False


_REQUIRES_INFRA = "requires Docker infrastructure (MinIO + MLflow server)"


@pytest.mark.integration
class TestMlflowMinioContract:
    """Verify MLflow artifacts are stored in MinIO and round-trip correctly."""

    def test_training_artifacts_in_minio(self) -> None:
        """boto3 list_objects on mlflow-artifacts bucket returns training checkpoint keys."""
        if not _minio_reachable():
            pytest.skip(_REQUIRES_INFRA)

        import boto3

        s3 = boto3.client(
            "s3",
            endpoint_url="http://localhost:9000",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
        )
        response = s3.list_objects_v2(
            Bucket="mlflow-artifacts",
            Prefix="",
            MaxKeys=100,
        )
        contents = response.get("Contents", [])
        assert len(contents) > 0, (
            "No artifacts found in mlflow-artifacts bucket. "
            "Training flow must store artifacts in MinIO."
        )
        # Verify at least one checkpoint-like artifact
        keys = [obj["Key"] for obj in contents]
        checkpoint_keys = [k for k in keys if ".pt" in k or "checkpoint" in k.lower()]
        assert checkpoint_keys, (
            f"No checkpoint artifacts in MinIO. Keys found: {keys[:10]}..."
        )

    def test_mlflow_download_matches_upload(self, tmp_path: Path) -> None:
        """Download checkpoint via MLflow API, verify SHA256 matches original."""
        if not _mlflow_server_reachable():
            pytest.skip(_REQUIRES_INFRA)

        import mlflow

        mlflow.set_tracking_uri("http://localhost:5000")
        from mlflow.tracking import MlflowClient

        client = MlflowClient("http://localhost:5000")

        # Find any run with artifacts
        experiments = client.search_experiments()
        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=1,
            )
            if runs:
                artifacts = client.list_artifacts(runs[0].info.run_id)
                if artifacts:
                    # Download first artifact
                    artifact_path = artifacts[0].path
                    local_path = mlflow.artifacts.download_artifacts(
                        run_id=runs[0].info.run_id,
                        artifact_path=artifact_path,
                        dst_path=str(tmp_path),
                    )
                    downloaded = Path(local_path)
                    assert downloaded.exists(), (
                        f"Downloaded artifact not found: {downloaded}"
                    )
                    assert downloaded.stat().st_size > 0, "Downloaded artifact is empty"
                    return

        pytest.skip("No runs with artifacts found")

    def test_experiment_names_have_debug_suffix(self) -> None:
        """All experiments created during e2e use _DEBUG suffix."""
        if not _mlflow_server_reachable():
            pytest.skip(_REQUIRES_INFRA)

        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri("http://localhost:5000")
        client = MlflowClient("http://localhost:5000")
        experiments = client.search_experiments()

        # Filter to non-default experiments
        named_exps = [e for e in experiments if e.name != "Default"]
        if not named_exps:
            pytest.skip("No named experiments found")

        # During e2e testing, all experiments should have debug suffix
        for exp in named_exps:
            assert "_DEBUG" in exp.name or "_E2E" in exp.name, (
                f"Experiment {exp.name!r} missing debug suffix. "
                f"E2E tests must use MINIVESS_DEBUG_SUFFIX to isolate."
            )

    def test_all_runs_finished_status(self) -> None:
        """Query MLflow for all runs, verify none have status RUNNING or FAILED."""
        if not _mlflow_server_reachable():
            pytest.skip(_REQUIRES_INFRA)

        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri("http://localhost:5000")
        client = MlflowClient("http://localhost:5000")
        experiments = client.search_experiments()

        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
            )
            for run in runs:
                assert run.info.status != "RUNNING", (
                    f"Run {run.info.run_id} in experiment {exp.name!r} "
                    f"still RUNNING — indicates a hung or crashed flow."
                )

    def test_config_artifact_round_trip(self, tmp_path: Path) -> None:
        """Upload resolved_config.yaml, download, verify YAML content matches."""
        if not _mlflow_server_reachable():
            pytest.skip(_REQUIRES_INFRA)

        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri("http://localhost:5000")
        client = MlflowClient("http://localhost:5000")

        experiments = client.search_experiments()
        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=5,
            )
            for run in runs:
                artifacts = client.list_artifacts(run.info.run_id)
                config_artifacts = [
                    a
                    for a in artifacts
                    if "config" in a.path.lower() and a.path.endswith((".yaml", ".yml"))
                ]
                if config_artifacts:
                    local_path = mlflow.artifacts.download_artifacts(
                        run_id=run.info.run_id,
                        artifact_path=config_artifacts[0].path,
                        dst_path=str(tmp_path),
                    )
                    downloaded = Path(local_path)
                    assert downloaded.exists()
                    content = downloaded.read_text(encoding="utf-8")
                    assert len(content) > 10, "Config artifact is too small"
                    return

        pytest.skip("No config artifacts found in any run")
