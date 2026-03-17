"""Tests for MLflow GCP Cloud Run configuration (T1 — #755).

Verifies that MLflow Cloud Run uses --no-serve-artifacts
so that clients upload directly to GCS, bypassing Cloud Run's
32 MB request body limit.
"""

from __future__ import annotations

from pathlib import Path


class TestMlflowGcpDockerfile:
    """Verify MLflow GCP Dockerfile uses --no-serve-artifacts."""

    def test_dockerfile_no_serve_artifacts(self) -> None:
        """Dockerfile.mlflow-gcp must use --no-serve-artifacts."""
        dockerfile = Path("deployment/docker/Dockerfile.mlflow-gcp")
        content = dockerfile.read_text(encoding="utf-8")
        assert "--no-serve-artifacts" in content, (
            "Dockerfile.mlflow-gcp MUST use --no-serve-artifacts to bypass "
            "Cloud Run's 32 MB request body limit. Large checkpoints (~900 MB) "
            "must be uploaded directly from client to GCS."
        )

    def test_dockerfile_not_serve_artifacts(self) -> None:
        """Dockerfile.mlflow-gcp must NOT use --serve-artifacts (without --no prefix)."""
        dockerfile = Path("deployment/docker/Dockerfile.mlflow-gcp")
        content = dockerfile.read_text(encoding="utf-8")
        # Replace --no-serve-artifacts temporarily to check for bare --serve-artifacts
        content_without_no = content.replace("--no-serve-artifacts", "")
        assert "--serve-artifacts" not in content_without_no, (
            "Dockerfile.mlflow-gcp has --serve-artifacts which proxies ALL "
            "artifact uploads through Cloud Run (32 MB limit). Use "
            "--no-serve-artifacts instead."
        )

    def test_dockerfile_has_default_artifact_root(self) -> None:
        """Dockerfile should reference GCS artifact root."""
        dockerfile = Path("deployment/docker/Dockerfile.mlflow-gcp")
        content = dockerfile.read_text(encoding="utf-8")
        assert (
            "default-artifact-root" in content
            or "MLFLOW_ARTIFACTS_DESTINATION" in content
        ), (
            "Dockerfile.mlflow-gcp must set --default-artifact-root or "
            "reference MLFLOW_ARTIFACTS_DESTINATION for GCS artifact storage."
        )

    def test_dockerfile_has_gcs_storage_package(self) -> None:
        """Dockerfile must install google-cloud-storage for GCS backend."""
        dockerfile = Path("deployment/docker/Dockerfile.mlflow-gcp")
        content = dockerfile.read_text(encoding="utf-8")
        assert "google-cloud-storage" in content


class TestPulumiMlflowConfig:
    """Verify Pulumi MLflow Cloud Run config is correct."""

    def test_pulumi_no_proxy_multipart(self) -> None:
        """Pulumi should NOT set MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD.

        With --no-serve-artifacts, proxy multipart is irrelevant —
        clients upload directly to GCS.
        """
        pulumi_main = Path("deployment/pulumi/gcp/__main__.py")
        content = pulumi_main.read_text(encoding="utf-8")
        assert "MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD" not in content, (
            "MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD is unnecessary with "
            "--no-serve-artifacts. Clients upload directly to GCS."
        )


class TestSkyPilotMlflowConfig:
    """Verify SkyPilot YAML doesn't set proxy multipart upload."""

    def test_smoke_test_gcp_no_proxy_multipart(self) -> None:
        """smoke_test_gcp.yaml should not set proxy multipart upload."""
        yaml_path = Path("deployment/skypilot/smoke_test_gcp.yaml")
        if not yaml_path.exists():
            return  # Skip if file doesn't exist
        content = yaml_path.read_text(encoding="utf-8")
        assert "MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD" not in content, (
            "MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD is unnecessary with "
            "--no-serve-artifacts. Remove from SkyPilot env vars."
        )
