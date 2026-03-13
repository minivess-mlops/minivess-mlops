"""Tests for SkyPilot YAML configs.

T-07: Configs must use prefect deployment run (not scripts).
T1.1 (#633): Smoke test GPU YAML validates resources, envs, setup.

Uses yaml.safe_load() for all YAML parsing — NO regex (CLAUDE.md Rule #16).
"""

from __future__ import annotations

from pathlib import Path

import yaml

_TRAIN_GENERIC = Path("deployment/skypilot/train_generic.yaml")
_TRAIN_HPO = Path("deployment/skypilot/train_hpo_sweep.yaml")
_SMOKE_TEST_GPU = Path("deployment/skypilot/smoke_test_gpu.yaml")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# train_generic.yaml
# ---------------------------------------------------------------------------


class TestTrainGenericYaml:
    def test_train_generic_yaml_is_valid(self) -> None:
        """train_generic.yaml must parse without error."""
        config = _load(_TRAIN_GENERIC)
        assert isinstance(config, dict)

    def test_train_generic_no_python_script_invocation(self) -> None:
        """train_generic.yaml run section must not invoke train_monitored.py directly."""
        config = _load(_TRAIN_GENERIC)
        run_section: str = config.get("run", "")
        assert "scripts/train_monitored.py" not in run_section, (
            "train_generic.yaml still invokes scripts/train_monitored.py directly. "
            "Replace with: prefect deployment run 'training-flow/default' --params ..."
        )

    def test_train_generic_uses_prefect_run(self) -> None:
        """train_generic.yaml run section must use prefect deployment run."""
        config = _load(_TRAIN_GENERIC)
        run_section: str = config.get("run", "")
        assert "prefect deployment run" in run_section, (
            "train_generic.yaml does not use 'prefect deployment run'. "
            "Add: prefect deployment run 'training-flow/default' --params ..."
        )

    def test_train_generic_has_prefect_api_url_env(self) -> None:
        """train_generic.yaml must declare PREFECT_API_URL in envs."""
        config = _load(_TRAIN_GENERIC)
        envs: dict = config.get("envs", {})
        assert "PREFECT_API_URL" in envs, (
            "train_generic.yaml missing PREFECT_API_URL in envs section. "
            "Add PREFECT_API_URL: ${PREFECT_API_URL} so the spot instance can reach "
            "the Prefect server."
        )

    def test_train_generic_has_experiment_name_env(self) -> None:
        """train_generic.yaml must have EXPERIMENT_NAME env var for the flow."""
        config = _load(_TRAIN_GENERIC)
        envs: dict = config.get("envs", {})
        assert "EXPERIMENT_NAME" in envs, (
            "train_generic.yaml missing EXPERIMENT_NAME in envs section."
        )


# ---------------------------------------------------------------------------
# train_hpo_sweep.yaml
# ---------------------------------------------------------------------------


class TestTrainHpoSweepYaml:
    def test_train_hpo_yaml_is_valid(self) -> None:
        """train_hpo_sweep.yaml must parse without error."""
        config = _load(_TRAIN_HPO)
        assert isinstance(config, dict)

    def test_train_hpo_no_python_script(self) -> None:
        """train_hpo_sweep.yaml run section must not invoke run_hpo.py directly."""
        config = _load(_TRAIN_HPO)
        run_section: str = config.get("run", "")
        assert "scripts/run_hpo.py" not in run_section, (
            "train_hpo_sweep.yaml still invokes scripts/run_hpo.py directly. "
            "Replace with: prefect deployment run 'hpo-flow/default' --params ..."
        )


# ---------------------------------------------------------------------------
# smoke_test_gpu.yaml (#633, T1.1)
# ---------------------------------------------------------------------------


class TestSmokeTestGpuYaml:
    """Validate smoke_test_gpu.yaml for RunPod 4090 GPU testing."""

    def test_smoke_test_yaml_is_valid(self) -> None:
        """smoke_test_gpu.yaml must parse without error."""
        config = _load(_SMOKE_TEST_GPU)
        assert isinstance(config, dict)

    def test_smoke_test_resources_specify_4090(self) -> None:
        """Resources must request RTX 4090 GPU."""
        config = _load(_SMOKE_TEST_GPU)
        resources = config.get("resources", {})
        accel = resources.get("accelerators", "")
        assert "4090" in str(accel), (
            f"smoke_test_gpu.yaml must request RTX 4090, got: {accel}"
        )

    def test_smoke_test_resources_runpod_cloud(self) -> None:
        """Resources must target RunPod cloud."""
        config = _load(_SMOKE_TEST_GPU)
        resources = config.get("resources", {})
        cloud = resources.get("cloud", "")
        assert cloud == "runpod", (
            f"smoke_test_gpu.yaml must target runpod, got: {cloud}"
        )

    def test_smoke_test_envs_has_dvc_vars(self) -> None:
        """Envs must include DVC_S3_* variables for data pull."""
        config = _load(_SMOKE_TEST_GPU)
        envs = config.get("envs", {})
        for var in ("DVC_S3_ENDPOINT_URL", "DVC_S3_ACCESS_KEY", "DVC_S3_SECRET_KEY"):
            assert var in envs, f"smoke_test_gpu.yaml missing {var} in envs"

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

    def test_smoke_test_setup_installs_uv(self) -> None:
        """Setup must install uv (not available on RunPod base images, RC8)."""
        config = _load(_SMOKE_TEST_GPU)
        setup = config.get("setup", "")
        assert "uv" in setup, "smoke_test_gpu.yaml setup must install uv"

    def test_smoke_test_setup_has_dvc_pull(self) -> None:
        """Setup must include DVC pull step."""
        config = _load(_SMOKE_TEST_GPU)
        setup = config.get("setup", "")
        assert "dvc pull" in setup, (
            "smoke_test_gpu.yaml setup must include 'dvc pull -r upcloud'"
        )
