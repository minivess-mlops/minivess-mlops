"""Verify smoke_test_gpu.yaml supports VesselFM + branch checkout (T2.2).

Validates the SkyPilot YAML has all env vars and setup steps needed
for VesselFM training on RunPod via SkyPilot.

Failure hypotheses addressed:
  H13: splits_file ignored → setup copies smoke split to splits.json
  H14: Python 3.13 not on RunPod → setup runs uv python install 3.13
  H15: SPLITS_DIR/CHECKPOINT_DIR missing → envs section sets them
  H17: boto3 CRC32C on RunPod → AWS_REQUEST_CHECKSUM_CALCULATION set
  H23: UUID EXPERIMENT breaks Hydra → separated via HYDRA_OVERRIDES
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


class TestSmokeTestYamlCriticalFixes:
    """Verify critical failure hypothesis mitigations are in the YAML."""

    def _load_yaml(self) -> dict:
        return yaml.safe_load(SMOKE_YAML.read_text(encoding="utf-8"))

    def test_h14_python_install(self) -> None:
        """H14: Setup must install Python 3.13 (RunPod has 3.10/3.11)."""
        config = self._load_yaml()
        setup = config.get("setup", "")
        assert "uv python install" in setup, "Missing uv python install for H14"

    def test_h15_train_flow_env_vars(self) -> None:
        """H15: SPLITS_DIR and CHECKPOINT_DIR must be set for train_flow.py."""
        config = self._load_yaml()
        envs = config.get("envs", {})
        assert "SPLITS_DIR" in envs, "SPLITS_DIR not in envs (H15)"
        assert "CHECKPOINT_DIR" in envs, "CHECKPOINT_DIR not in envs (H15)"

    def test_h17_boto3_checksum_workaround(self) -> None:
        """H17: AWS_REQUEST_CHECKSUM_CALCULATION must be set for UpCloud S3."""
        config = self._load_yaml()
        envs = config.get("envs", {})
        assert envs.get("AWS_REQUEST_CHECKSUM_CALCULATION") == "WHEN_REQUIRED", (
            "Missing AWS_REQUEST_CHECKSUM_CALCULATION=WHEN_REQUIRED (H17)"
        )

    def test_h13_smoke_splits_copy(self) -> None:
        """H13: Setup must copy smoke split file to splits.json."""
        config = self._load_yaml()
        setup = config.get("setup", "")
        assert "smoke_test_1fold_4vol.json" in setup, (
            "Setup must copy smoke splits file (H13)"
        )
        assert "splits.json" in setup, (
            "Setup must create splits.json from smoke splits (H13)"
        )

    def test_h23_experiment_not_overwritten_in_run(self) -> None:
        """H23: Run block must NOT override EXPERIMENT (Hydra config name)."""
        config = self._load_yaml()
        run = config.get("run", "")
        assert "export EXPERIMENT=" not in run, (
            "Run block must NOT override EXPERIMENT — use HYDRA_OVERRIDES (H23)"
        )
        assert "HYDRA_OVERRIDES" in run, (
            "Run block must use HYDRA_OVERRIDES for experiment name isolation (H23)"
        )

    def test_setup_has_fail_fast(self) -> None:
        """Setup must use set -ex for fail-fast + trace."""
        config = self._load_yaml()
        setup = config.get("setup", "")
        assert "set -ex" in setup, "Setup must use set -ex for fail-fast"

    def test_setup_creates_required_dirs(self) -> None:
        """Setup must create checkpoints and logs directories."""
        config = self._load_yaml()
        setup = config.get("setup", "")
        assert "mkdir" in setup, "Setup must create required directories"

    def test_setup_verifies_gpu(self) -> None:
        """Setup must verify GPU availability before training."""
        config = self._load_yaml()
        setup = config.get("setup", "")
        assert "nvidia-smi" in setup, "Setup must verify GPU with nvidia-smi"
