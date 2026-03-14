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

    def test_experiment_not_in_envs(self) -> None:
        """EXPERIMENT must NOT be in envs — SkyPilot can't resolve intra-envs ${}."""
        config = self._load_yaml()
        envs = config.get("envs", {})
        assert "EXPERIMENT" not in envs, (
            "EXPERIMENT must NOT be in envs section — move to run: block. "
            "SkyPilot can't resolve ${MODEL_FAMILY} from intra-envs references, "
            "and OmegaConf interprets ${} as config interpolation."
        )

    def test_experiment_computed_in_run(self) -> None:
        """EXPERIMENT must be computed in run: block from MODEL_FAMILY."""
        config = self._load_yaml()
        run = config.get("run", "")
        assert 'export EXPERIMENT="smoke_${MODEL_FAMILY}"' in run, (
            "run: block must compute EXPERIMENT from MODEL_FAMILY via shell expansion"
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

    def test_h23_experiment_and_hydra_overrides_in_run(self) -> None:
        """H23: EXPERIMENT computed in run block; UUID isolation via HYDRA_OVERRIDES."""
        config = self._load_yaml()
        run = config.get("run", "")
        assert "export EXPERIMENT=" in run, (
            "Run block must compute EXPERIMENT from MODEL_FAMILY (H23)"
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


class TestSmokeTestRootCauseRegressions:
    """Regression tests for 6 systemic issues (debug-plan.xml 2026-03-14).

    Each test prevents recurrence of a specific root-cause failure discovered
    during iterative RunPod launch debugging.
    """

    def _load_yaml(self) -> dict:
        return yaml.safe_load(SMOKE_YAML.read_text(encoding="utf-8"))

    def test_issue5_experiment_not_in_envs(self) -> None:
        """Issue 5 (P0): EXPERIMENT with ${} in envs causes OmegaConf crash."""
        config = self._load_yaml()
        envs = config.get("envs", {})
        assert "EXPERIMENT" not in envs, (
            "EXPERIMENT must NOT be in envs — SkyPilot can't resolve intra-envs "
            "${MODEL_FAMILY} and OmegaConf interprets it as config interpolation"
        )

    def test_issue5_experiment_computed_in_run(self) -> None:
        """Issue 5 (P0): EXPERIMENT must be computed in run block via shell."""
        config = self._load_yaml()
        run = config.get("run", "")
        assert "export EXPERIMENT=" in run, (
            "EXPERIMENT must be computed in run: block, not envs:"
        )
        assert "smoke_${MODEL_FAMILY}" in run, (
            "EXPERIMENT must derive from MODEL_FAMILY"
        )

    def test_issue4_run_guards_splits(self) -> None:
        """Issue 4 (P1): Run block must guard against missing splits.json."""
        config = self._load_yaml()
        run = config.get("run", "")
        assert "splits.json" in run, (
            "Run block must check for splits.json (catches sky exec without setup)"
        )

    def test_issue4_run_guards_data(self) -> None:
        """Issue 4 (P1): Run block must guard against missing training data."""
        config = self._load_yaml()
        run = config.get("run", "")
        assert "data/raw/minivess/imagesTr" in run, (
            "Run block must check for training data directory"
        )

    def test_issue6_header_says_jobs_launch(self) -> None:
        """Issue 6 (P2): YAML header must recommend sky jobs launch for spot recovery."""
        text = SMOKE_YAML.read_text(encoding="utf-8")
        assert "sky jobs launch" in text, (
            "YAML must recommend 'sky jobs launch' (not 'sky launch') for spot recovery"
        )

    def test_issue2_makefile_env_file(self) -> None:
        """Issue 2 (P0): Makefile must use --env-file .env for SkyPilot."""
        makefile = Path("Makefile").read_text(encoding="utf-8")
        assert "--env-file .env" in makefile, (
            "Makefile smoke-test-gpu must use --env-file .env "
            "to pass DVC/MLflow credentials to SkyPilot API server"
        )

    def test_issue2_makefile_uses_double_dash_env(self) -> None:
        """Issue 2 (P0): Makefile must use --env (not -e) for SkyPilot vars."""
        makefile = Path("Makefile").read_text(encoding="utf-8")
        # Find the smoke-test-gpu section
        lines = makefile.split("\n")
        in_target = False
        for line in lines:
            if line.startswith("smoke-test-gpu:"):
                in_target = True
                continue
            if in_target:
                if line and not line.startswith("\t") and not line.startswith(" "):
                    break  # Next target
                if "-e MODEL_FAMILY" in line or "-e GIT_BRANCH" in line:
                    msg = "Makefile uses '-e' (Docker flag) instead of '--env' (SkyPilot flag)"
                    raise AssertionError(msg)

    def test_issue3_dvc_preprocess_frozen(self) -> None:
        """Issue 3 (P1): DVC preprocess stage must be frozen."""
        dvc_yaml = Path("dvc.yaml").read_text(encoding="utf-8")
        dvc_config = yaml.safe_load(dvc_yaml)
        preprocess = dvc_config.get("stages", {}).get("preprocess", {})
        assert preprocess.get("frozen") is True, (
            "DVC preprocess stage must be frozen — its outputs have no cache on "
            "UpCloud S3, causing 'dvc pull' to fail"
        )

    def test_issue1_sky_config_exists(self) -> None:
        """Issue 1 (P0): ~/.sky/config.yaml must cap controller disk for RunPod."""
        sky_config = Path.home() / ".sky" / "config.yaml"
        assert sky_config.exists(), (
            "~/.sky/config.yaml must exist to cap jobs controller disk_size ≤ 40 GB"
        )
        config = yaml.safe_load(sky_config.read_text(encoding="utf-8"))
        controller_disk = (
            config.get("jobs", {})
            .get("controller", {})
            .get("resources", {})
            .get("disk_size", 999)
        )
        assert controller_disk <= 40, (
            f"Controller disk_size must be ≤ 40 (RunPod max), got {controller_disk}"
        )
