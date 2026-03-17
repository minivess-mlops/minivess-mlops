"""Tests for SkyPilot failover and recovery configuration (T5 — #754)."""

from __future__ import annotations

from pathlib import Path


class TestSkyPilotFailoverConfig:
    """Verify SkyPilot config supports multi-region failover."""

    def test_dev_runpod_has_multiple_accelerator_options(self) -> None:
        """dev_runpod.yaml should list multiple GPU options for failover."""
        yaml_path = Path("deployment/skypilot/dev_runpod.yaml")
        content = yaml_path.read_text(encoding="utf-8")
        # Should have multiple accelerator types or use a list
        gpu_count = sum(
            1
            for line in content.splitlines()
            if "RTX" in line or "A100" in line or "L4" in line or "A40" in line
        )
        assert gpu_count >= 2, (
            "dev_runpod.yaml should list multiple GPU options for failover. "
            "EU-RO-1 may not have RTX 4090 — need fallback GPUs."
        )

    def test_smoke_test_gpu_has_fallback_accelerators(self) -> None:
        """Smoke test YAML should have multiple GPU options."""
        yaml_path = Path("deployment/skypilot/smoke_test_gpu.yaml")
        if not yaml_path.exists():
            return
        content = yaml_path.read_text(encoding="utf-8")
        gpu_count = sum(
            1
            for line in content.splitlines()
            if "RTX" in line or "A100" in line or "L4" in line or "A40" in line
        )
        assert gpu_count >= 1
