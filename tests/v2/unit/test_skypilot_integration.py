"""Tests for SkyPilot compute integration (#282).

Covers:
- SkyPilot task YAML existence and structure
- SkyPilotLauncher class
- Prefect-SkyPilot bridge tasks
- Justfile targets

Note: train_generic.yaml and train_hpo_sweep.yaml were deleted (bare-VM, Docker mandate).
All SkyPilot YAMLs now use Docker image_id pattern.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestSkyPilotYAMLs:
    """Test SkyPilot task YAML files."""

    def test_smoke_test_yaml_exists(self) -> None:
        path = PROJECT_ROOT / "deployment" / "skypilot" / "smoke_test_gpu.yaml"
        assert path.exists()

    def test_smoke_test_yaml_valid(self) -> None:
        import yaml

        path = PROJECT_ROOT / "deployment" / "skypilot" / "smoke_test_gpu.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "resources" in data
        assert "setup" in data or "run" in data

    def test_smoke_test_yaml_has_gpu(self) -> None:
        import yaml

        path = PROJECT_ROOT / "deployment" / "skypilot" / "smoke_test_gpu.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        resources = data.get("resources", {})
        assert "accelerators" in resources


class TestSkyPilotLauncher:
    """Test SkyPilotLauncher class."""

    def test_launcher_instantiation(self) -> None:
        from minivess.compute.skypilot_launcher import SkyPilotLauncher

        launcher = SkyPilotLauncher()
        assert launcher is not None

    def test_launcher_has_launch_method(self) -> None:
        from minivess.compute.skypilot_launcher import SkyPilotLauncher

        launcher = SkyPilotLauncher()
        assert hasattr(launcher, "launch_training_job")

    def test_launch_training_job_dry_run(self) -> None:
        from minivess.compute.skypilot_launcher import SkyPilotLauncher

        launcher = SkyPilotLauncher()
        result = launcher.launch_training_job(
            config={"loss_name": "dice_ce", "model_family": "dynunet"},
            dry_run=True,
        )
        assert result["status"] == "dry_run"

    def test_launcher_has_hpo_method(self) -> None:
        from minivess.compute.skypilot_launcher import SkyPilotLauncher

        launcher = SkyPilotLauncher()
        assert hasattr(launcher, "launch_hpo_sweep")


class TestPrefectSkyBridge:
    """Test Prefect-SkyPilot bridge tasks."""

    def test_bridge_module_exists(self) -> None:
        from minivess.compute import prefect_sky_tasks

        assert hasattr(prefect_sky_tasks, "launch_sky_training")

    def test_launch_sky_training_is_callable(self) -> None:
        from minivess.compute.prefect_sky_tasks import launch_sky_training

        assert callable(launch_sky_training)

    def test_wait_sky_job_is_callable(self) -> None:
        from minivess.compute.prefect_sky_tasks import wait_sky_job

        assert callable(wait_sky_job)


class TestJustfileTargets:
    """Test justfile has SkyPilot targets."""

    def test_justfile_has_sky_train(self) -> None:
        path = PROJECT_ROOT / "justfile"
        content = path.read_text(encoding="utf-8")
        assert "sky-train" in content

    def test_justfile_has_sky_status(self) -> None:
        path = PROJECT_ROOT / "justfile"
        content = path.read_text(encoding="utf-8")
        assert "sky-status" in content
