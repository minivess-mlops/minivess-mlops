"""Cloud architecture enforcement tests.

Verify that SkyPilot YAML files enforce the two-provider architecture:
  - GCP: GCS data source (ADC auth), no UpCloud/S3 credentials
  - RunPod: Network Volume data source, file-based MLflow, no S3 fallback

These tests run in CI without cloud credentials — they parse YAML only.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

SKYPILOT_DIR = pathlib.Path("deployment/skypilot")


@pytest.fixture
def gcp_yaml() -> dict:
    """Load smoke_test_gcp.yaml."""
    path = SKYPILOT_DIR / "smoke_test_gcp.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture
def runpod_yaml() -> dict:
    """Load smoke_test_gpu.yaml."""
    path = SKYPILOT_DIR / "smoke_test_gpu.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestGcpYamlArchitecture:
    """GCP YAML must use GCS data source and Cloud Run MLflow."""

    def test_gcp_yaml_uses_gcs_remote(self, gcp_yaml: dict) -> None:
        """DVC_REMOTE must be 'gcs' (not upcloud, minio, or s3)."""
        envs = gcp_yaml.get("envs", {})
        assert envs.get("DVC_REMOTE") == "gcs", (
            "GCP YAML must use DVC_REMOTE=gcs (GCS via ADC auth)"
        )

    def test_gcp_yaml_contains_no_upcloud_credentials(self, gcp_yaml: dict) -> None:
        """No UpCloud S3 credential env vars in GCP YAML envs block."""
        envs = gcp_yaml.get("envs", {})
        banned = [
            "DVC_S3_ENDPOINT_URL",
            "DVC_S3_ACCESS_KEY",
            "DVC_S3_SECRET_KEY",
            "DVC_S3_BUCKET",
            "MLFLOW_CLOUD_URI",
            "MLFLOW_CLOUD_USERNAME",
            "MLFLOW_CLOUD_PASSWORD",
        ]
        found = [b for b in banned if b in envs]
        assert not found, f"Banned UpCloud env vars in GCP YAML envs: {found}"

    def test_gcp_yaml_uses_l4_not_t4(self, gcp_yaml: dict) -> None:
        """Accelerator must include L4 and NOT T4 (Turing = no BF16 = NaN)."""
        accs = gcp_yaml.get("resources", {}).get("accelerators", {})
        assert "L4" in accs, "GCP YAML must use L4 GPU (Ada Lovelace, BF16-capable)"
        assert "T4" not in accs, "T4 BANNED: Turing architecture, no BF16 → NaN in SAM3"

    def test_mlflow_uri_direct_not_aliased(self, gcp_yaml: dict) -> None:
        """MLFLOW_TRACKING_URI must use ${MLFLOW_TRACKING_URI}, not ${MLFLOW_GCP_URI}."""
        envs = gcp_yaml.get("envs", {})
        uri = envs.get("MLFLOW_TRACKING_URI", "")
        assert "MLFLOW_GCP_URI" not in str(uri), (
            "MLFLOW_TRACKING_URI must reference ${MLFLOW_TRACKING_URI} directly, "
            "not ${MLFLOW_GCP_URI} (SkyPilot cannot chain env var refs)"
        )

    def test_gcp_yaml_uses_spot(self, gcp_yaml: dict) -> None:
        """GCP jobs should use spot instances for cost efficiency."""
        resources = gcp_yaml.get("resources", {})
        assert resources.get("use_spot") is True, "GCP YAML should use spot instances"


class TestRunPodYamlArchitecture:
    """RunPod YAML must use Network Volume and file-based MLflow."""

    def test_runpod_yaml_uses_network_volume(self, runpod_yaml: dict) -> None:
        """RunPod YAML must have volumes: section mounting minivess-dev."""
        volumes = runpod_yaml.get("volumes", {})
        assert "/opt/vol" in volumes, (
            "RunPod YAML must mount Network Volume at /opt/vol"
        )
        assert volumes["/opt/vol"] == "minivess-dev", (
            "RunPod YAML must mount minivess-dev volume"
        )

    def test_mlflow_uri_is_file_based(self, runpod_yaml: dict) -> None:
        """RunPod MLflow uses file-based tracking on Network Volume."""
        envs = runpod_yaml.get("envs", {})
        uri = envs.get("MLFLOW_TRACKING_URI", "")
        assert uri.startswith("/opt/vol"), (
            f"RunPod MLFLOW_TRACKING_URI must be file-based on Network Volume, got: {uri}"
        )

    def test_no_s3_reference_in_runpod_envs(self, runpod_yaml: dict) -> None:
        """No S3 or UpCloud references in RunPod YAML envs."""
        envs = runpod_yaml.get("envs", {})
        banned_prefixes = ["DVC_S3_", "MLFLOW_CLOUD_"]
        found = [k for k in envs if any(k.startswith(bp) for bp in banned_prefixes)]
        assert not found, f"Banned S3/UpCloud env vars in RunPod YAML: {found}"

    def test_no_s3_url_in_runpod_yaml_text(self) -> None:
        """No s3:// URLs anywhere in RunPod YAML (not even in comments)."""
        path = SKYPILOT_DIR / "smoke_test_gpu.yaml"
        content = path.read_text(encoding="utf-8")
        # Check non-comment lines only
        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            assert "s3://" not in line, (
                f"s3:// URL found in RunPod YAML line {i}: {line.strip()}"
            )

    def test_setup_fails_fast_when_data_missing(self) -> None:
        """RunPod setup block must exit 1 when data is missing, with upload instructions."""
        path = SKYPILOT_DIR / "smoke_test_gpu.yaml"
        content = path.read_text(encoding="utf-8")
        assert "exit 1" in content, "Setup must exit 1 when data is missing"
        assert "dev-gpu-upload-data" in content, (
            "Setup must mention 'dev-gpu-upload-data' in error message"
        )
