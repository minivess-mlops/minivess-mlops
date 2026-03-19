"""RunPod GPU smoke test — manually triggered (#638, T3.2).

Launches a SkyPilot job on RunPod and verifies the full training pipeline:
  Network Volume (cache-first) / AWS S3 fallback --DVC pull--> RunPod RTX 4090
  --MLflow (file-based)--> /opt/vol/mlruns --rsync--> local mlruns/

UpCloud archived 2026-03-16. File-based MLflow on Network Volume replaces remote server.
Post-run sync: make dev-gpu-sync (sky rsync down minivess-dev:/opt/vol/mlruns/ mlruns/)

This test is EXCLUDED from default pytest collection (lives in gpu_instance/).
Run manually via: make test-gpu-cloud

Markers:
  @gpu_heavy — requires CUDA GPU (RunPod RTX 4090)
  @slow — takes 15-30 minutes
  @cloud_mlflow — skippable (file-based MLflow needs no server)

Prerequisites:
  - RunPod API key configured: sky check runpod
  - Network Volume: sky storage ls | grep minivess-dev
  - Data on volume: make dev-gpu-upload-data (first time)
  - No remote MLflow env vars needed (file-based MLflow on Network Volume)
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

    def test_mlflow_tracking_uri_set(self) -> None:
        """MLFLOW_TRACKING_URI must be configured (file-based or remote)."""
        assert os.environ.get("MLFLOW_TRACKING_URI"), "MLFLOW_TRACKING_URI not set"

    def test_dvc_s3_credentials_set(self) -> None:
        """DVC_S3_* credentials must be configured."""
        for var in ("DVC_S3_ENDPOINT_URL", "DVC_S3_ACCESS_KEY", "DVC_S3_SECRET_KEY"):
            assert os.environ.get(var), f"{var} not set"
