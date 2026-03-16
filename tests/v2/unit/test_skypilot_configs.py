"""Tests for SkyPilot YAML configs.

Docker mandate: All SkyPilot YAMLs use image_id: docker:... for deps.
Setup sections are DATA ONLY — no apt-get, uv sync, git clone, pip install.

See: .claude/metalearning/2026-03-14-skypilot-bare-vm-docker-violation.md
"""

from __future__ import annotations

from pathlib import Path

import yaml

_SMOKE_TEST_GPU = Path("deployment/skypilot/smoke_test_gpu.yaml")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# smoke_test_gpu.yaml (#633, T1.1)
# ---------------------------------------------------------------------------


class TestSmokeTestGpuYaml:
    """Validate smoke_test_gpu.yaml for RunPod 4090 GPU testing."""

    def test_smoke_test_yaml_is_valid(self) -> None:
        """smoke_test_gpu.yaml must parse without error."""
        config = _load(_SMOKE_TEST_GPU)
        assert isinstance(config, dict)

    def test_smoke_test_resources_include_gpu(self) -> None:
        """Resources must include at least RTX 4090 in GPU fallback list."""
        config = _load(_SMOKE_TEST_GPU)
        resources = config.get("resources", {})
        accel = resources.get("accelerators", "")
        accel_str = str(accel)
        assert "RTX4090" in accel_str or "4090" in accel_str, (
            f"smoke_test_gpu.yaml must include RTX4090 in accelerators, got: {accel}"
        )

    def test_smoke_test_resources_runpod_cloud(self) -> None:
        """Resources must target RunPod cloud."""
        config = _load(_SMOKE_TEST_GPU)
        resources = config.get("resources", {})
        cloud = resources.get("cloud", "")
        assert cloud == "runpod", (
            f"smoke_test_gpu.yaml must target runpod, got: {cloud}"
        )

    def test_smoke_test_envs_has_dvc_remote(self) -> None:
        """Envs must include DVC_REMOTE pointing to remote_storage (AWS S3 public).

        UpCloud S3 dropped 2026-03-16. No DVC_S3_* credentials needed for public bucket.
        """
        config = _load(_SMOKE_TEST_GPU)
        envs = config.get("envs", {})
        assert "DVC_REMOTE" in envs, "smoke_test_gpu.yaml missing DVC_REMOTE in envs"
        assert envs["DVC_REMOTE"] == "remote_storage", (
            f"DVC_REMOTE should be 'remote_storage' (AWS S3 public), got: {envs['DVC_REMOTE']}"
        )
        # UpCloud credentials must NOT be present (provider archived)
        for stale_var in (
            "DVC_S3_ENDPOINT_URL",
            "DVC_S3_ACCESS_KEY",
            "DVC_S3_SECRET_KEY",
        ):
            assert stale_var not in envs, (
                f"smoke_test_gpu.yaml has stale UpCloud credential {stale_var} — "
                "UpCloud archived 2026-03-16, remove it"
            )

    def test_smoke_test_envs_has_mlflow_cloud_uri(self) -> None:
        """Envs must include MLFLOW_TRACKING_URI for remote tracking."""
        config = _load(_SMOKE_TEST_GPU)
        envs = config.get("envs", {})
        assert "MLFLOW_TRACKING_URI" in envs, (
            "smoke_test_gpu.yaml missing MLFLOW_TRACKING_URI in envs"
        )

    def test_smoke_test_envs_has_host_escape_hatch(self) -> None:
        """Envs must have MINIVESS_ALLOW_HOST=1 (cloud VM escape hatch)."""
        config = _load(_SMOKE_TEST_GPU)
        envs = config.get("envs", {})
        assert envs.get("MINIVESS_ALLOW_HOST") == "1", (
            "smoke_test_gpu.yaml must set MINIVESS_ALLOW_HOST=1"
        )

    def test_smoke_test_uses_docker_image(self) -> None:
        """smoke_test_gpu.yaml must use Docker image_id (not bare-VM uv install)."""
        config = _load(_SMOKE_TEST_GPU)
        resources = config.get("resources", {})
        image_id = resources.get("image_id", "")
        assert str(image_id).startswith("docker:"), (
            f"smoke_test_gpu.yaml must use Docker image_id, got: {image_id}"
        )

    def test_smoke_test_setup_has_dvc_pull(self) -> None:
        """Setup must include DVC pull step."""
        config = _load(_SMOKE_TEST_GPU)
        setup = config.get("setup", "")
        assert "dvc pull" in setup, (
            "smoke_test_gpu.yaml setup must include 'dvc pull -r remote_storage' "
            "(AWS S3 fallback — UpCloud dropped 2026-03-16)"
        )
