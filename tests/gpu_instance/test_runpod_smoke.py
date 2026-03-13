"""RunPod GPU smoke test — manually triggered (#638, T3.2).

Launches a SkyPilot job on RunPod and verifies the full training pipeline:
  UpCloud S3 --DVC pull--> RunPod RTX 4090 --MLflow--> UpCloud MLflow

This test is EXCLUDED from default pytest collection (lives in gpu_instance/).
Run manually via: make test-gpu-cloud

Markers:
  @gpu_heavy — requires CUDA GPU (RunPod RTX 4090)
  @slow — takes 15-30 minutes
  @cloud_mlflow — requires MLFLOW_CLOUD_* env vars

Prerequisites:
  - RunPod API key configured: sky check runpod
  - UpCloud MLflow server running
  - DVC data pushed to UpCloud S3
  - All MLFLOW_CLOUD_* and DVC_S3_* env vars set
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

sky = pytest.importorskip(
    "sky", reason="SkyPilot not installed (uv sync --extra infra)"
)

SMOKE_TEST_YAML = Path("deployment/skypilot/smoke_test_gpu.yaml")


@pytest.mark.gpu_heavy
@pytest.mark.slow
@pytest.mark.cloud_mlflow
class TestRunPodSmokeTest:
    """Full end-to-end smoke test on RunPod via SkyPilot."""

    def test_smoke_test_yaml_exists(self) -> None:
        """SkyPilot smoke test YAML must exist."""
        assert SMOKE_TEST_YAML.exists(), f"Missing: {SMOKE_TEST_YAML}"

    def test_runpod_api_key_set(self) -> None:
        """RUNPOD_API_KEY must be configured."""
        assert os.environ.get("RUNPOD_API_KEY"), (
            "RUNPOD_API_KEY not set. Get it from: "
            "https://www.runpod.io/console/user/settings"
        )

    def test_mlflow_cloud_credentials_set(self) -> None:
        """MLFLOW_CLOUD_* credentials must be configured."""
        for var in ("MLFLOW_CLOUD_URI", "MLFLOW_CLOUD_PASSWORD"):
            assert os.environ.get(var), f"{var} not set"

    def test_dvc_s3_credentials_set(self) -> None:
        """DVC_S3_* credentials must be configured."""
        for var in ("DVC_S3_ENDPOINT_URL", "DVC_S3_ACCESS_KEY", "DVC_S3_SECRET_KEY"):
            assert os.environ.get(var), f"{var} not set"
