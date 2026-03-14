"""Verify smoke_test_gpu.yaml supports VesselFM + branch checkout (T2.2).

Validates the SkyPilot YAML has all env vars and setup steps needed
for VesselFM training on RunPod via SkyPilot.
"""

from __future__ import annotations

from pathlib import Path

import yaml

SMOKE_YAML = Path("deployment/skypilot/smoke_test_gpu.yaml")


class TestSmokeTestYamlVesselFM:
    """Verify smoke_test_gpu.yaml is ready for VesselFM."""

    def _load_yaml(self) -> dict:
        return yaml.safe_load(SMOKE_YAML.read_text(encoding="utf-8"))

    def test_yaml_has_hf_token_env(self) -> None:
        """HF_TOKEN must be passed through to the RunPod VM."""
        config = self._load_yaml()
        envs = config.get("envs", {})
        assert "HF_TOKEN" in envs, "HF_TOKEN not in smoke_test_gpu.yaml envs"

    def test_yaml_supports_vesselfm_experiment(self) -> None:
        """EXPERIMENT env var references MODEL_FAMILY for VesselFM dispatch."""
        config = self._load_yaml()
        envs = config.get("envs", {})
        experiment = envs.get("EXPERIMENT", "")
        assert "MODEL_FAMILY" in experiment, (
            f"EXPERIMENT does not reference MODEL_FAMILY: {experiment}"
        )

    def test_yaml_has_branch_checkout(self) -> None:
        """Setup section must support git branch checkout."""
        config = self._load_yaml()
        setup = config.get("setup", "")
        assert "git checkout" in setup, "No git checkout in setup section"
        envs = config.get("envs", {})
        assert "GIT_BRANCH" in envs, "GIT_BRANCH not in envs"

    def test_yaml_has_dvc_credentials(self) -> None:
        """DVC S3 credentials must be passed for data pull."""
        config = self._load_yaml()
        envs = config.get("envs", {})
        for var in ["DVC_S3_ENDPOINT_URL", "DVC_S3_ACCESS_KEY", "DVC_S3_SECRET_KEY"]:
            assert var in envs, f"{var} not in smoke_test_gpu.yaml envs"

    def test_yaml_has_mlflow_credentials(self) -> None:
        """MLflow tracking credentials must be passed for experiment logging."""
        config = self._load_yaml()
        envs = config.get("envs", {})
        for var in [
            "MLFLOW_TRACKING_URI",
            "MLFLOW_TRACKING_USERNAME",
            "MLFLOW_TRACKING_PASSWORD",
        ]:
            assert var in envs, f"{var} not in smoke_test_gpu.yaml envs"

    def test_yaml_has_host_escape_hatches(self) -> None:
        """MINIVESS_ALLOW_HOST and PREFECT_DISABLED must be set for cloud VMs."""
        config = self._load_yaml()
        envs = config.get("envs", {})
        assert envs.get("MINIVESS_ALLOW_HOST") == "1", "MINIVESS_ALLOW_HOST not set"
        assert envs.get("PREFECT_DISABLED") == "1", "PREFECT_DISABLED not set"
