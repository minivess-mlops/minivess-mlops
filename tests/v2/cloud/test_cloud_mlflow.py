"""L2 cloud MLflow tests (#622).

Verify MLflow works against a live cloud deployment (GCP Cloud Run, etc.).
Requires MLFLOW_TRACKING_URI set to a remote URL + MLFLOW_TRACKING_PASSWORD.
Skipped automatically when MLFLOW_TRACKING_URI is localhost or file-based.

Run with: ``make test-cloud-mlflow``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from mlflow import MlflowClient

    from tests.v2.cloud.conftest import CloudMLflowConnection


@pytest.mark.cloud_mlflow
class TestCloudMLflowHealth:
    """Verify the remote MLflow deployment is healthy."""

    def test_health_endpoint_public(
        self, cloud_mlflow_connection: CloudMLflowConnection
    ) -> None:
        """GET /health returns 200 without auth."""
        import urllib.request

        url = f"{cloud_mlflow_connection.tracking_uri}/health"
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            assert resp.status == 200

    def test_unauthenticated_api_returns_401(
        self, cloud_mlflow_connection: CloudMLflowConnection
    ) -> None:
        """GET /api/2.0/mlflow/experiments/search without auth -> 401."""
        import urllib.error
        import urllib.request

        url = (
            f"{cloud_mlflow_connection.tracking_uri}/api/2.0/mlflow/experiments/search"
        )
        req = urllib.request.Request(url, method="GET")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)  # noqa: S310
        assert exc_info.value.code == 401

    def test_authenticated_api_returns_200(
        self, cloud_mlflow_client: MlflowClient
    ) -> None:
        """Authenticated experiment search returns results."""
        experiments = cloud_mlflow_client.search_experiments(max_results=1)
        # At minimum, the Default experiment exists
        assert len(experiments) >= 1

    def test_wrong_password_returns_401(
        self, cloud_mlflow_connection: CloudMLflowConnection
    ) -> None:
        """Auth with wrong password -> 401 (not 500)."""
        import base64
        import urllib.error
        import urllib.request

        url = (
            f"{cloud_mlflow_connection.tracking_uri}/api/2.0/mlflow/experiments/search"
        )
        creds = base64.b64encode(b"admin:WRONG_PASSWORD").decode()
        req = urllib.request.Request(
            url, headers={"Authorization": f"Basic {creds}"}, method="GET"
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)  # noqa: S310
        assert exc_info.value.code == 401


@pytest.mark.cloud_mlflow
class TestCloudMLflowTracking:
    """Verify experiment tracking against remote server."""

    def test_create_experiment(
        self, cloud_mlflow_client: MlflowClient, test_run_id: str
    ) -> None:
        """Create experiment on remote server, verify it exists."""
        exp_name = f"{test_run_id}_create_exp"
        exp_id = cloud_mlflow_client.create_experiment(exp_name)
        exp = cloud_mlflow_client.get_experiment(exp_id)
        assert exp.name == exp_name

    def test_create_run_log_params_metrics(
        self, cloud_mlflow_client: MlflowClient, test_run_id: str
    ) -> None:
        """Full run lifecycle: create -> log params/metrics -> end."""
        exp_name = f"{test_run_id}_run_lifecycle"
        exp_id = cloud_mlflow_client.create_experiment(exp_name)
        run = cloud_mlflow_client.create_run(exp_id)
        run_id = run.info.run_id

        cloud_mlflow_client.log_param(run_id, "model", "dynunet")
        cloud_mlflow_client.log_param(run_id, "learning_rate", "0.001")
        cloud_mlflow_client.log_metric(run_id, "loss", 0.5, step=0)
        cloud_mlflow_client.log_metric(run_id, "loss", 0.3, step=1)
        cloud_mlflow_client.log_metric(run_id, "dsc", 0.82)
        cloud_mlflow_client.set_terminated(run_id, status="FINISHED")

        fetched = cloud_mlflow_client.get_run(run_id)
        assert fetched.data.params["model"] == "dynunet"
        assert fetched.data.metrics["loss"] == 0.3
        assert fetched.data.metrics["dsc"] == 0.82
        assert fetched.info.status == "FINISHED"

    def test_search_runs_filter(
        self, cloud_mlflow_client: MlflowClient, test_run_id: str
    ) -> None:
        """Search runs with filter_string on PostgreSQL backend."""
        exp_name = f"{test_run_id}_search_filter"
        exp_id = cloud_mlflow_client.create_experiment(exp_name)

        # Create two runs with different params
        for model, dsc in [("dynunet", 0.82), ("vesselfm", 0.78)]:
            run = cloud_mlflow_client.create_run(exp_id)
            cloud_mlflow_client.log_param(run.info.run_id, "model", model)
            cloud_mlflow_client.log_metric(run.info.run_id, "dsc", dsc)
            cloud_mlflow_client.set_terminated(run.info.run_id, status="FINISHED")

        runs = cloud_mlflow_client.search_runs(
            experiment_ids=[exp_id],
            filter_string="params.model = 'dynunet'",
        )
        assert len(runs) == 1
        assert runs[0].data.params["model"] == "dynunet"

    def test_log_batch_metrics(
        self, cloud_mlflow_client: MlflowClient, test_run_id: str
    ) -> None:
        """Log batch of 100 metrics in one call — PostgreSQL perf check."""
        from mlflow.entities import Metric

        exp_name = f"{test_run_id}_batch_metrics"
        exp_id = cloud_mlflow_client.create_experiment(exp_name)
        run = cloud_mlflow_client.create_run(exp_id)

        metrics = [
            Metric(key=f"m_{i}", value=float(i), timestamp=0, step=i)
            for i in range(100)
        ]
        cloud_mlflow_client.log_batch(run.info.run_id, metrics=metrics)
        cloud_mlflow_client.set_terminated(run.info.run_id, status="FINISHED")

        fetched = cloud_mlflow_client.get_run(run.info.run_id)
        assert len(fetched.data.metrics) == 100

    def test_tag_run_after_completion(
        self, cloud_mlflow_client: MlflowClient, test_run_id: str
    ) -> None:
        """Set tag on completed run (champion tagging pattern)."""
        exp_name = f"{test_run_id}_tag_after"
        exp_id = cloud_mlflow_client.create_experiment(exp_name)
        run = cloud_mlflow_client.create_run(exp_id)
        cloud_mlflow_client.set_terminated(run.info.run_id, status="FINISHED")

        cloud_mlflow_client.set_tag(run.info.run_id, "champion", "true")

        fetched = cloud_mlflow_client.get_run(run.info.run_id)
        assert fetched.data.tags["champion"] == "true"

    def test_postgresql_soft_delete(
        self, cloud_mlflow_client: MlflowClient, test_run_id: str
    ) -> None:
        """Delete experiment -> lifecycle_stage=deleted, not truly removed."""
        from mlflow.entities import ViewType

        exp_name = f"{test_run_id}_soft_delete"
        exp_id = cloud_mlflow_client.create_experiment(exp_name)
        cloud_mlflow_client.delete_experiment(exp_id)

        # Should be findable with DELETED_ONLY view
        deleted = cloud_mlflow_client.search_experiments(
            view_type=ViewType.DELETED_ONLY,
            filter_string=f"name = '{exp_name}'",
        )
        assert len(deleted) >= 1


@pytest.mark.cloud_mlflow
class TestCloudMLflowArtifacts:
    """Verify S3-compatible artifact storage."""

    def test_log_artifact_roundtrip(
        self,
        cloud_mlflow_client: MlflowClient,
        test_run_id: str,
        tmp_path: Path,
    ) -> None:
        """Upload file artifact via MLflow, download, verify content."""
        exp_name = f"{test_run_id}_artifact_rt"
        exp_id = cloud_mlflow_client.create_experiment(exp_name)
        run = cloud_mlflow_client.create_run(exp_id)

        artifact = tmp_path / "test_data.txt"
        artifact.write_text("cloud artifact test", encoding="utf-8")
        cloud_mlflow_client.log_artifact(run.info.run_id, str(artifact))

        artifacts = cloud_mlflow_client.list_artifacts(run.info.run_id)
        assert any(a.path == "test_data.txt" for a in artifacts)

    def test_log_large_artifact(
        self,
        cloud_mlflow_client: MlflowClient,
        test_run_id: str,
        tmp_path: Path,
    ) -> None:
        """Upload 10 MB artifact — verifies S3 multipart works."""
        exp_name = f"{test_run_id}_large_artifact"
        exp_id = cloud_mlflow_client.create_experiment(exp_name)
        run = cloud_mlflow_client.create_run(exp_id)

        large_file = tmp_path / "large_model.bin"
        large_file.write_bytes(b"\x00" * (10 * 1024 * 1024))  # 10 MB
        cloud_mlflow_client.log_artifact(run.info.run_id, str(large_file))

        artifacts = cloud_mlflow_client.list_artifacts(run.info.run_id)
        assert any(a.path == "large_model.bin" for a in artifacts)

    def test_list_artifacts(
        self, cloud_mlflow_client: MlflowClient, test_run_id: str, tmp_path: Path
    ) -> None:
        """List artifacts for a run, verify expected structure."""
        exp_name = f"{test_run_id}_list_artifacts"
        exp_id = cloud_mlflow_client.create_experiment(exp_name)
        run = cloud_mlflow_client.create_run(exp_id)

        for name in ["a.txt", "b.txt", "c.txt"]:
            f = tmp_path / name
            f.write_text(f"content of {name}", encoding="utf-8")
            cloud_mlflow_client.log_artifact(run.info.run_id, str(f))

        artifacts = cloud_mlflow_client.list_artifacts(run.info.run_id)
        artifact_names = {a.path for a in artifacts}
        assert {"a.txt", "b.txt", "c.txt"} <= artifact_names

    def test_direct_s3_connectivity(self, cloud_s3_client) -> None:
        """Direct boto3 ListBuckets against S3 endpoint (path-style)."""
        response = cloud_s3_client.list_buckets()
        assert "Buckets" in response

    def test_s3_bucket_exists(
        self, cloud_s3_client, cloud_mlflow_connection: CloudMLflowConnection
    ) -> None:
        """Verify the configured artifact bucket exists."""
        response = cloud_s3_client.list_buckets()
        bucket_names = [b["Name"] for b in response["Buckets"]]
        assert cloud_mlflow_connection.s3_bucket in bucket_names


@pytest.mark.cloud_mlflow
class TestCloudMLflowConnectionFailures:
    """Verify graceful handling of connection failures."""

    def test_connection_refused_on_wrong_port(
        self, cloud_mlflow_connection: CloudMLflowConnection
    ) -> None:
        """Connection to port+1 raises error, not hangs."""
        import urllib.error
        import urllib.parse
        import urllib.request

        parsed = urllib.parse.urlparse(cloud_mlflow_connection.tracking_uri)
        wrong_port = (parsed.port or 5000) + 1
        bad_url = f"{parsed.scheme}://{parsed.hostname}:{wrong_port}/health"

        with pytest.raises((urllib.error.URLError, TimeoutError, OSError)):
            urllib.request.urlopen(bad_url, timeout=5)  # noqa: S310
