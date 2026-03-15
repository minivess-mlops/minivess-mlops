"""SkyPilot + RunPod integration tests.

Validates that SkyPilot recognises RunPod as an enabled cloud backend,
that the RunPod GPU catalog contains expected accelerators, and that the
smoke-test YAML is structurally valid.

All tests auto-skip when RUNPOD_API_KEY is not set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

_RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
_skip_no_runpod = pytest.mark.skipif(
    not _RUNPOD_API_KEY,
    reason="RUNPOD_API_KEY not set -- skipping RunPod tests",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_TEST_YAML = REPO_ROOT / "deployment" / "skypilot" / "smoke_test_gpu.yaml"


@_skip_no_runpod
@pytest.mark.skypilot_cloud
class TestSkyPilotRunPodBackend:
    """Verify SkyPilot can reach RunPod and enumerate its GPU catalog."""

    def test_sky_check_runpod_enabled(self) -> None:
        """RunPod must appear as an enabled cloud in SkyPilot."""
        import sky

        # SkyPilot nightly (1.0.0.dev) exposes the cloud registry.
        # Try the check() API first; fall back to registry inspection.
        try:
            # sky.check.check(clouds=['runpod']) raises if RunPod is not configured.
            sky.check.check(clouds=["runpod"])
        except AttributeError:
            # Older / different SkyPilot build: verify via cloud registry.
            registry = getattr(sky.clouds, "CLOUD_REGISTRY", None)
            if registry is not None:
                cloud_names = {name.lower() for name in registry}
                assert "runpod" in cloud_names, (
                    f"RunPod not in CLOUD_REGISTRY: {sorted(cloud_names)}"
                )
            else:
                # Last resort: confirm the config file exists.
                config_path = Path.home() / ".runpod" / "config.toml"
                assert config_path.exists(), (
                    "Neither sky.check.check(clouds=['runpod']) nor "
                    "CLOUD_REGISTRY available, and ~/.runpod/config.toml "
                    "does not exist. Is RunPod configured?"
                )

    def test_runpod_gpu_catalog_has_rtx4090(self) -> None:
        """RunPod GPU catalog must list the RTX 4090 accelerator."""
        import sky

        # Attempt the Resources validation path: if SkyPilot accepts
        # a Resources spec with RTX4090 on RunPod, the accelerator is
        # catalogued.
        try:
            resources = sky.Resources(
                cloud=sky.clouds.RunPod(),
                accelerators="RTX4090:1",
            )
            # Resources() does not raise -> RTX4090 is known.
            assert resources is not None
        except (ValueError, sky.exceptions.ResourcesUnavailableError) as exc:
            pytest.fail(f"RTX4090 not available on RunPod via SkyPilot: {exc}")

    def test_smoke_test_yaml_valid(self) -> None:
        """smoke_test_gpu.yaml must parse and contain required sections."""
        assert SMOKE_TEST_YAML.exists(), f"Smoke-test YAML not found: {SMOKE_TEST_YAML}"

        config = yaml.safe_load(SMOKE_TEST_YAML.read_text(encoding="utf-8"))

        assert "resources" in config, "Missing 'resources' section"
        assert "envs" in config, "Missing 'envs' section"
        assert "run" in config, "Missing 'run' section"

        # Verify RunPod-specific resource constraints.
        resources = config["resources"]
        assert resources.get("cloud") == "runpod", (
            f"Expected cloud: runpod, got: {resources.get('cloud')}"
        )
        assert "accelerators" in resources, "Missing 'accelerators' in resources"
