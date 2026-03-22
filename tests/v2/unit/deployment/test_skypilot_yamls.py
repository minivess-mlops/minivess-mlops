"""SkyPilot YAML validation tests — catch config errors before burning cloud credits.

Uses sky.Task.from_yaml() for structural validation + custom assertions for
project-specific invariants (T4 ban, env vars, Docker image, banned commands).

Issue #908. See: docs/planning/skypilot-fake-mock-ssh-test-suite-plan.md
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

SKYPILOT_DIR = Path("deployment/skypilot")
FACTORIAL_YAML = SKYPILOT_DIR / "train_factorial.yaml"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Structural validation via sky.Task.from_yaml()
# ---------------------------------------------------------------------------


class TestSkyPilotYamlParsing:
    """Every SkyPilot YAML must parse without errors."""

    def test_factorial_yaml_parses(self) -> None:
        """train_factorial.yaml must be valid SkyPilot YAML."""
        try:
            import sky

            task = sky.Task.from_yaml(str(FACTORIAL_YAML))
            assert task is not None
        except ImportError:
            pytest.skip("skypilot not installed")

    def test_factorial_yaml_has_resources(self) -> None:
        """Must declare resources (accelerators, cloud)."""
        config = _load_yaml(FACTORIAL_YAML)
        assert "resources" in config, "Missing resources section"

    def test_factorial_yaml_has_setup(self) -> None:
        """Must have a setup section."""
        config = _load_yaml(FACTORIAL_YAML)
        assert "setup" in config, "Missing setup section"
        assert len(config["setup"].strip()) > 0, "Empty setup section"

    def test_factorial_yaml_has_run(self) -> None:
        """Must have a run section."""
        config = _load_yaml(FACTORIAL_YAML)
        assert "run" in config, "Missing run section"
        assert len(config["run"].strip()) > 0, "Empty run section"


# ---------------------------------------------------------------------------
# T4 ban (Turing, no BF16 → FP16 overflow → NaN in SAM3)
# ---------------------------------------------------------------------------


class TestNoT4:
    """T4 is banned — Turing architecture has no BF16 support."""

    def test_no_t4_in_accelerators(self) -> None:
        """accelerators must NOT include T4."""
        config = _load_yaml(FACTORIAL_YAML)
        accels = config.get("resources", {}).get("accelerators", {})
        if isinstance(accels, dict):
            assert "T4" not in accels, f"T4 banned (no BF16). Accelerators: {accels}"
        elif isinstance(accels, str):
            assert "T4" not in accels


# ---------------------------------------------------------------------------
# Spot instances
# ---------------------------------------------------------------------------


class TestBannedFields:
    """Fields removed or unsupported in SkyPilot v1.0+."""

    def test_no_job_recovery_field(self) -> None:
        """job_recovery was removed in SkyPilot v1.0 (caused YAML parse failure in 4th pass)."""
        config = _load_yaml(FACTORIAL_YAML)
        assert "job_recovery" not in config, (
            "job_recovery field is banned — removed in SkyPilot v1.0. "
            "See: run-debug-factorial-experiment-report-4th-pass-failure.md"
        )


class TestSpotEnabled:
    """Production tasks must use spot instances for cost savings."""

    def test_spot_enabled(self) -> None:
        config = _load_yaml(FACTORIAL_YAML)
        use_spot = config.get("resources", {}).get("use_spot", False)
        assert use_spot is True, "use_spot must be true for cost savings"


# ---------------------------------------------------------------------------
# Docker image (no bare VM)
# ---------------------------------------------------------------------------


class TestDockerImage:
    """SkyPilot YAML must use Docker image, not bare VM."""

    def test_image_id_set(self) -> None:
        config = _load_yaml(FACTORIAL_YAML)
        image_id = config.get("resources", {}).get("image_id", "")
        assert image_id, "image_id not set — bare VM is banned"
        assert image_id.startswith("docker:"), (
            f"image_id must start with 'docker:': {image_id}"
        )

    def test_image_id_points_to_gar(self) -> None:
        """Docker image must be from GAR (same region as training VMs)."""
        config = _load_yaml(FACTORIAL_YAML)
        image_id = config.get("resources", {}).get("image_id", "")
        assert "europe-north1-docker.pkg.dev" in image_id, (
            f"Image must be from GAR europe-north1: {image_id}"
        )


# ---------------------------------------------------------------------------
# Required env vars
# ---------------------------------------------------------------------------


class TestRequiredEnvVars:
    """All env vars needed by train_flow.py must be declared in the YAML."""

    def test_required_envs_declared(self) -> None:
        config = _load_yaml(FACTORIAL_YAML)
        envs = config.get("envs", {})
        required = {
            "MODEL_FAMILY",
            "LOSS_NAME",
            "FOLD_ID",
            "WITH_AUX_CALIB",
            "MAX_EPOCHS",
            "EXPERIMENT_NAME",
            "POST_TRAINING_METHODS",
            "MLFLOW_TRACKING_URI",  # DC-2: silent MLflow failure without this
        }
        declared = set(envs.keys())
        missing = required - declared
        assert not missing, f"Missing env vars in SkyPilot YAML: {missing}"

    def test_hf_token_declared(self) -> None:
        """HF_TOKEN must be in envs (needed for SAM3/VesselFM weight download)."""
        config = _load_yaml(FACTORIAL_YAML)
        envs = config.get("envs", {})
        assert "HF_TOKEN" in envs, "HF_TOKEN not in envs — SAM3 will fail"


# ---------------------------------------------------------------------------
# Banned commands in setup
# ---------------------------------------------------------------------------


class TestSetupBannedCommands:
    """Setup must NOT install packages — everything is in the Docker image."""

    def test_no_apt_get(self) -> None:
        config = _load_yaml(FACTORIAL_YAML)
        setup = config.get("setup", "")
        assert "apt-get" not in setup, "apt-get banned in setup (use Docker image)"

    def test_no_uv_sync(self) -> None:
        config = _load_yaml(FACTORIAL_YAML)
        setup = config.get("setup", "")
        assert "uv sync" not in setup, "uv sync banned in setup (use Docker image)"

    def test_no_pip_install(self) -> None:
        config = _load_yaml(FACTORIAL_YAML)
        setup = config.get("setup", "")
        assert "pip install" not in setup, (
            "pip install banned in setup (use Docker image)"
        )

    def test_no_git_clone(self) -> None:
        config = _load_yaml(FACTORIAL_YAML)
        setup = config.get("setup", "")
        assert "git clone" not in setup, "git clone banned in setup (use Docker image)"


# ---------------------------------------------------------------------------
# DVC pull path validation
# ---------------------------------------------------------------------------


class TestSetupDvcPull:
    """Setup DVC pull must use path-specific pull, not bare dvc pull."""

    def test_no_bare_dvc_pull(self) -> None:
        """Must NOT use bare 'dvc pull -r gcs' — will fail on unpushed tracked files."""
        config = _load_yaml(FACTORIAL_YAML)
        setup = config.get("setup", "")
        for line in setup.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "dvc pull" in stripped and "-r gcs" in stripped:
                # Must have a path argument between "pull" and "-r"
                parts = stripped.split()
                pull_idx = parts.index("pull") if "pull" in parts else -1
                if pull_idx >= 0:
                    next_parts = parts[pull_idx + 1 :]
                    # Filter out flags
                    path_args = [p for p in next_parts if not p.startswith("-")]
                    assert len(path_args) > 0, (
                        f"Bare 'dvc pull -r gcs' found (no path filter). "
                        f"Use: dvc pull data/raw/minivess -r gcs. Line: {stripped}"
                    )

    def test_setup_has_error_handling_for_dvc(self) -> None:
        """DVC pull must have error handling (|| { exit 1; })."""
        config = _load_yaml(FACTORIAL_YAML)
        setup = config.get("setup", "")
        assert "exit 1" in setup, (
            "Setup must exit on DVC pull failure (no silent continuation)"
        )


# ---------------------------------------------------------------------------
# Run section guards
# ---------------------------------------------------------------------------


class TestRunSectionGuards:
    """Run section must verify prerequisites before training."""

    def test_run_checks_splits_file(self) -> None:
        """Run must check splits.json exists before proceeding."""
        config = _load_yaml(FACTORIAL_YAML)
        run = config.get("run", "")
        assert "splits.json" in run, "Run section must check for splits.json existence"


# ---------------------------------------------------------------------------
# CLI arg name consistency (CRITICAL — 4th pass root cause)
# ---------------------------------------------------------------------------


class TestCliArgConsistency:
    """YAML run section CLI args MUST match train_flow.py argparse exactly."""

    def test_uses_model_family_not_model(self) -> None:
        """Must use --model-family, not --model (4th pass root cause)."""
        config = _load_yaml(FACTORIAL_YAML)
        run = config.get("run", "")
        assert "--model-family" in run, (
            "YAML uses --model but argparse expects --model-family. "
            "This caused EVERY job to crash in 4th pass."
        )

    def test_uses_loss_name_not_loss(self) -> None:
        """Must use --loss-name, not --loss."""
        config = _load_yaml(FACTORIAL_YAML)
        run = config.get("run", "")
        assert "--loss-name" in run, (
            "YAML uses --loss but argparse expects --loss-name."
        )

    def test_uses_experiment_name_not_experiment(self) -> None:
        """Must use --experiment-name, not --experiment."""
        config = _load_yaml(FACTORIAL_YAML)
        run = config.get("run", "")
        assert "--experiment-name" in run, (
            "YAML uses --experiment but argparse expects --experiment-name."
        )

    def test_all_cli_args_match_argparse(self) -> None:
        """Every CLI arg in the YAML run section must exist in train_flow.py argparse."""
        import ast

        config = _load_yaml(FACTORIAL_YAML)
        run = config.get("run", "")

        # Extract CLI args from YAML run section
        yaml_args = set()
        for line in run.splitlines():
            stripped = line.strip()
            if stripped.startswith("--") and not stripped.startswith("# "):
                arg_name = stripped.split()[0].rstrip('"')
                yaml_args.add(arg_name)

        # Extract argparse args from train_flow.py
        train_flow_path = Path("src/minivess/orchestration/flows/train_flow.py")
        source = train_flow_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        argparse_args = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
            ):
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(
                    first_arg.value, str
                ):
                    argparse_args.add(first_arg.value)

        # Every YAML arg must be a valid argparse arg
        missing = yaml_args - argparse_args
        assert not missing, (
            f"YAML run section uses CLI args not in train_flow.py argparse: {missing}. "
            f"Valid args: {sorted(argparse_args)}"
        )


# ---------------------------------------------------------------------------
# MLflow credentials in envs (CRITICAL — silent data loss)
# ---------------------------------------------------------------------------


class TestMlflowCredentials:
    """MLflow credentials must be declared in envs to prevent silent data loss."""

    def test_mlflow_username_in_envs(self) -> None:
        """MLFLOW_TRACKING_USERNAME must be in envs for Cloud Run auth."""
        config = _load_yaml(FACTORIAL_YAML)
        envs = config.get("envs", {})
        assert "MLFLOW_TRACKING_USERNAME" in envs, (
            "MLFLOW_TRACKING_USERNAME missing — Cloud Run MLflow will return 401. "
            "Training 'succeeds' but ALL metrics are silently lost."
        )

    def test_mlflow_password_in_envs(self) -> None:
        """MLFLOW_TRACKING_PASSWORD must be in envs for Cloud Run auth."""
        config = _load_yaml(FACTORIAL_YAML)
        envs = config.get("envs", {})
        assert "MLFLOW_TRACKING_PASSWORD" in envs, (
            "MLFLOW_TRACKING_PASSWORD missing — Cloud Run MLflow will return 401."
        )

    def test_mlflow_retry_configured(self) -> None:
        """MLflow HTTP retries must be configured for cloud resilience."""
        config = _load_yaml(FACTORIAL_YAML)
        envs = config.get("envs", {})
        assert "MLFLOW_HTTP_REQUEST_MAX_RETRIES" in envs, (
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES missing — transient failures will crash training."
        )


# ---------------------------------------------------------------------------
# MLflow health check in setup (prevents silent data loss)
# ---------------------------------------------------------------------------


class TestMlflowHealthCheck:
    """Setup must verify MLflow connectivity before training."""

    def test_setup_has_mlflow_health_check(self) -> None:
        """Setup must check MLflow server is reachable."""
        config = _load_yaml(FACTORIAL_YAML)
        setup = config.get("setup", "")
        assert "MlflowClient" in setup or "mlflow" in setup.lower(), (
            "Setup must verify MLflow connectivity — without this, "
            "training runs for hours then silently loses all metrics."
        )


# ---------------------------------------------------------------------------
# DeepVess data pull for zero-shot (prevents FAILED_SETUP)
# ---------------------------------------------------------------------------


class TestDeepVessDataPull:
    """Setup must pull DeepVess data for zero-shot evaluation conditions."""

    def test_setup_handles_deepvess(self) -> None:
        """Setup must conditionally pull DeepVess data when EVAL_DATASET=deepvess."""
        config = _load_yaml(FACTORIAL_YAML)
        setup = config.get("setup", "")
        assert "deepvess" in setup, (
            "Setup must pull DeepVess data for zero-shot conditions. "
            "Without this, VesselFM zero-shot evaluation crashes."
        )


# ---------------------------------------------------------------------------
# File mounts validation (P1-G1 from double-check audit)
# ---------------------------------------------------------------------------


class TestFileMounts:
    """file_mounts must reference valid GCS paths for checkpoint persistence."""

    def test_file_mounts_exists(self) -> None:
        """Must have file_mounts for checkpoint persistence on spot VMs."""
        config = _load_yaml(FACTORIAL_YAML)
        assert "file_mounts" in config, (
            "Missing file_mounts — checkpoints lost on preemption"
        )

    def test_file_mounts_gcs_bucket(self) -> None:
        """file_mounts must reference a gs:// bucket."""
        config = _load_yaml(FACTORIAL_YAML)
        file_mounts = config.get("file_mounts", {})
        gcs_found = False
        for _mount_path, mount_config in file_mounts.items():
            source = ""
            if isinstance(mount_config, dict):
                source = mount_config.get("source", "")
            elif isinstance(mount_config, str):
                source = mount_config
            if source.startswith("gs://"):
                gcs_found = True
        assert gcs_found, "file_mounts must reference a GCS bucket (gs://)"

    def test_file_mounts_uses_mount_cached(self) -> None:
        """file_mounts should use MOUNT_CACHED for async checkpoint upload."""
        config = _load_yaml(FACTORIAL_YAML)
        file_mounts = config.get("file_mounts", {})
        for mount_path, mount_config in file_mounts.items():
            if isinstance(mount_config, dict) and mount_config.get(
                "source", ""
            ).startswith("gs://"):
                mode = mount_config.get("mode", "")
                assert mode == "MOUNT_CACHED", (
                    f"file_mount {mount_path} should use MOUNT_CACHED (async upload), not {mode}"
                )


# ---------------------------------------------------------------------------
# Disk size validation (P1-G7 from double-check audit)
# ---------------------------------------------------------------------------


class TestDiskSize:
    """Disk must be large enough for Docker image + DVC data + checkpoints."""

    def test_disk_size_at_least_100gb(self) -> None:
        """disk_size must be >= 100 GB to avoid DISK_FULL mid-training."""
        config = _load_yaml(FACTORIAL_YAML)
        disk_size = config.get("resources", {}).get("disk_size", 0)
        assert disk_size >= 100, (
            f"disk_size={disk_size} GB is too small. Need >= 100 GB "
            "(Docker image ~9 GB, DVC data ~2.7 GB, checkpoints ~5 GB/model)"
        )


# ---------------------------------------------------------------------------
# Setup script DVC init validation (P1-G8 from double-check audit)
# ---------------------------------------------------------------------------


class TestSetupDvcInit:
    """Setup must initialize DVC without git (Docker container has no repo)."""

    def test_dvc_init_no_scm(self) -> None:
        """Setup must run 'dvc init --no-scm' for Docker containers without git."""
        config = _load_yaml(FACTORIAL_YAML)
        setup = config.get("setup", "")
        assert "dvc init" in setup, "Setup must run dvc init"
        assert "--no-scm" in setup, (
            "DVC init must use --no-scm (Docker container has no git repo)"
        )


# ---------------------------------------------------------------------------
# Escape hatch env vars (P1-G6 from double-check audit)
# ---------------------------------------------------------------------------


class TestEscapeHatchEnvVars:
    """Cloud VMs need MINIVESS_ALLOW_HOST and PREFECT_DISABLED escape hatches."""

    def test_minivess_allow_host_set(self) -> None:
        """MINIVESS_ALLOW_HOST=1 must be set (bypass Docker context gate on cloud VM)."""
        config = _load_yaml(FACTORIAL_YAML)
        envs = config.get("envs", {})
        assert "MINIVESS_ALLOW_HOST" in envs, (
            "MINIVESS_ALLOW_HOST must be in envs — train_flow.py Docker gate blocks without it"
        )

    def test_prefect_disabled_set(self) -> None:
        """PREFECT_DISABLED=1 must be set (no Prefect server on cloud VM)."""
        config = _load_yaml(FACTORIAL_YAML)
        envs = config.get("envs", {})
        assert "PREFECT_DISABLED" in envs, (
            "PREFECT_DISABLED must be in envs — no Prefect server available on cloud VM"
        )


# ---------------------------------------------------------------------------
# Setup GPU verification (P1-G9 from double-check audit)
# ---------------------------------------------------------------------------


class TestSetupGpuVerification:
    """Setup should verify GPU availability as a preflight check."""

    def test_nvidia_smi_in_setup(self) -> None:
        """Setup should run nvidia-smi to verify GPU is available."""
        config = _load_yaml(FACTORIAL_YAML)
        setup = config.get("setup", "")
        assert "nvidia-smi" in setup, (
            "Setup should run nvidia-smi as GPU preflight check"
        )


# ---------------------------------------------------------------------------
# Parametrized validation for ALL SkyPilot YAMLs (P2-G5 from audit)
# ---------------------------------------------------------------------------


def _discover_skypilot_yamls() -> list[Path]:
    """Find all SkyPilot YAML files in deployment/skypilot/."""
    skypilot_dir = Path("deployment/skypilot")
    if not skypilot_dir.exists():
        return []
    return sorted(skypilot_dir.glob("*.yaml"))


class TestAllSkyPilotYamls:
    """Every YAML in deployment/skypilot/ must pass basic structural checks."""

    @pytest.mark.parametrize(
        "yaml_path",
        _discover_skypilot_yamls(),
        ids=[p.name for p in _discover_skypilot_yamls()],
    )
    def test_yaml_parseable(self, yaml_path: Path) -> None:
        """Every SkyPilot YAML must parse without YAML errors."""
        config = _load_yaml(yaml_path)
        assert isinstance(config, dict), f"{yaml_path.name} did not parse to a dict"

    @pytest.mark.parametrize(
        "yaml_path",
        _discover_skypilot_yamls(),
        ids=[p.name for p in _discover_skypilot_yamls()],
    )
    def test_no_t4_anywhere(self, yaml_path: Path) -> None:
        """T4 must NOT appear in any SkyPilot YAML accelerators."""
        config = _load_yaml(yaml_path)
        accels = config.get("resources", {}).get("accelerators", {})
        if isinstance(accels, dict | str):
            assert "T4" not in accels, f"T4 banned in {yaml_path.name}"

    @pytest.mark.parametrize(
        "yaml_path",
        _discover_skypilot_yamls(),
        ids=[p.name for p in _discover_skypilot_yamls()],
    )
    def test_no_job_recovery_anywhere(self, yaml_path: Path) -> None:
        """job_recovery must NOT appear in any SkyPilot YAML (removed in v1.0)."""
        config = _load_yaml(yaml_path)
        assert "job_recovery" not in config, (
            f"job_recovery banned in {yaml_path.name} — removed in SkyPilot v1.0"
        )
