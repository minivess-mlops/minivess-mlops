"""Tests for train_factorial.yaml SkyPilot config and run_factorial.sh script.

Validates:
- SkyPilot YAML structure, Docker mandate, T4 ban, required env vars
- run_factorial.sh syntax, dry-run parsing, condition count generation
- Factorial config parsing for both debug (34 conditions) and production (102 conditions)

Source of truth: knowledge-graph/decisions/L3-technology/paper_model_comparison.yaml
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

_TRAIN_FACTORIAL = Path("deployment/skypilot/train_factorial.yaml")
_RUN_FACTORIAL_SH = Path("scripts/run_factorial.sh")
_DEBUG_CONFIG = Path("configs/experiment/debug_factorial.yaml")
_PAPER_CONFIG = Path("configs/hpo/paper_factorial.yaml")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# train_factorial.yaml — SkyPilot YAML validation
# ---------------------------------------------------------------------------


class TestTrainFactorialYaml:
    """Validate train_factorial.yaml for GCP factorial experiment."""

    def test_yaml_is_valid(self) -> None:
        """train_factorial.yaml must parse without error."""
        config = _load(_TRAIN_FACTORIAL)
        assert isinstance(config, dict)

    def test_uses_docker_image_id(self) -> None:
        """Must use image_id: docker:... (bare VM BANNED)."""
        config = _load(_TRAIN_FACTORIAL)
        resources = config.get("resources", {})
        image_id = resources.get("image_id", "")
        assert image_id.startswith("docker:"), (
            f"train_factorial.yaml must use Docker image_id, got: {image_id}"
        )

    def test_no_t4_gpu(self) -> None:
        """T4 BANNED — Turing architecture, no BF16 (CLAUDE.md)."""
        config = _load(_TRAIN_FACTORIAL)
        resources = config.get("resources", {})
        accel = resources.get("accelerators", {})
        accel_str = str(accel).upper()
        assert "T4" not in accel_str, (
            f"train_factorial.yaml contains T4 GPU (BANNED): {accel}"
        )

    def test_uses_l4_gpu(self) -> None:
        """Must include L4 (Ada Lovelace, BF16-capable)."""
        config = _load(_TRAIN_FACTORIAL)
        resources = config.get("resources", {})
        accel = resources.get("accelerators", {})
        assert "L4" in accel, f"train_factorial.yaml must include L4 GPU, got: {accel}"

    def test_uses_spot_instances(self) -> None:
        """Must use spot instances (60-91% cheaper)."""
        config = _load(_TRAIN_FACTORIAL)
        resources = config.get("resources", {})
        assert resources.get("use_spot") is True

    def test_targets_gcp(self) -> None:
        """Must target GCP cloud."""
        config = _load(_TRAIN_FACTORIAL)
        resources = config.get("resources", {})
        assert resources.get("cloud") == "gcp"

    def test_no_hardcoded_region(self) -> None:
        """Region must NOT be hardcoded — L4 not in europe-north1.

        SkyPilot auto-selects cheapest region with L4 availability.
        GAR image (europe-north1) is pullable from any GCP region.
        """
        config = _load(_TRAIN_FACTORIAL)
        resources = config.get("resources", {})
        assert "region" not in resources, (
            "region must not be hardcoded — L4 not available in europe-north1"
        )

    def test_has_required_env_vars(self) -> None:
        """Must have all per-condition env vars."""
        config = _load(_TRAIN_FACTORIAL)
        envs = config.get("envs", {})
        required = [
            "MODEL_FAMILY",
            "LOSS_NAME",
            "FOLD_ID",
            "WITH_AUX_CALIB",
            "MAX_EPOCHS",
            "MAX_TRAIN_VOLUMES",
            "MAX_VAL_VOLUMES",
            "EXPERIMENT_NAME",
            "MLFLOW_TRACKING_URI",
            "HF_TOKEN",
            "SPLITS_DIR",
            "CHECKPOINT_DIR",
            "LOGS_DIR",
        ]
        for var in required:
            assert var in envs, f"train_factorial.yaml missing env var: {var}"

    def test_has_cloud_vm_escape_hatches(self) -> None:
        """Must set MINIVESS_ALLOW_HOST=1 and PREFECT_DISABLED=1 for cloud VM."""
        config = _load(_TRAIN_FACTORIAL)
        envs = config.get("envs", {})
        assert envs.get("MINIVESS_ALLOW_HOST") == "1"
        assert envs.get("PREFECT_DISABLED") == "1"

    def test_setup_has_dvc_pull(self) -> None:
        """Setup must pull data from GCS via DVC."""
        config = _load(_TRAIN_FACTORIAL)
        setup = config.get("setup", "")
        assert "dvc pull" in setup, "Setup must include DVC pull from GCS"

    def test_setup_has_no_banned_commands(self) -> None:
        """Setup must not contain bare-VM commands."""
        config = _load(_TRAIN_FACTORIAL)
        setup = config.get("setup", "")
        banned = ["apt-get", "uv sync", "git clone", "pip install", "conda"]
        for cmd in banned:
            assert cmd not in setup, f"Setup contains banned command '{cmd}'"

    def test_run_uses_module_invocation(self) -> None:
        """Run must use python -m (module invocation), not standalone script."""
        config = _load(_TRAIN_FACTORIAL)
        run_section = config.get("run", "")
        assert "python -m" in run_section, (
            "Run must use 'python -m minivess...' (Rule #17)"
        )
        assert "python scripts/" not in run_section, (
            "Run must NOT use standalone scripts (Rule #17)"
        )


# ---------------------------------------------------------------------------
# run_factorial.sh — script validation
# ---------------------------------------------------------------------------


class TestRunFactorialSh:
    """Validate run_factorial.sh shell script."""

    def test_script_is_syntactically_valid(self) -> None:
        """run_factorial.sh must pass bash -n syntax check."""
        result = subprocess.run(
            ["bash", "-n", str(_RUN_FACTORIAL_SH)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"bash -n failed:\n{result.stderr}"

    def test_script_is_executable(self) -> None:
        """run_factorial.sh must have execute permission."""
        import os

        assert os.access(_RUN_FACTORIAL_SH, os.X_OK), (
            "run_factorial.sh must be executable (chmod +x)"
        )

    def test_script_references_correct_skypilot_yaml(self) -> None:
        """Must reference deployment/skypilot/train_factorial.yaml."""
        content = _RUN_FACTORIAL_SH.read_text(encoding="utf-8")
        assert "train_factorial.yaml" in content

    def test_script_has_dry_run_flag(self) -> None:
        """Must support --dry-run flag."""
        content = _RUN_FACTORIAL_SH.read_text(encoding="utf-8")
        assert "--dry-run" in content

    def test_script_has_rate_limiting(self) -> None:
        """Must have sleep between launches for API quota."""
        content = _RUN_FACTORIAL_SH.read_text(encoding="utf-8")
        assert "sleep" in content

    def test_script_captures_job_ids(self) -> None:
        """Must write job tracking info to outputs/ directory."""
        content = _RUN_FACTORIAL_SH.read_text(encoding="utf-8")
        assert "job_ids.txt" in content or "JOB_LOG" in content

    def test_script_has_named_jobs(self) -> None:
        """Must use --name for per-condition job identification."""
        content = _RUN_FACTORIAL_SH.read_text(encoding="utf-8")
        assert "--name" in content

    def test_script_handles_launch_failures(self) -> None:
        """Must not abort all conditions on single launch failure."""
        content = _RUN_FACTORIAL_SH.read_text(encoding="utf-8")
        # Script should handle failures per-condition, not use set -e for sky jobs
        assert "LAUNCH_FAILED" in content or "FAILED" in content


# ---------------------------------------------------------------------------
# Factorial config condition counting
# ---------------------------------------------------------------------------


class TestFactorialConfigParsing:
    """Verify factorial configs produce correct condition counts."""

    def test_debug_factorial_has_32_trainable(self) -> None:
        """Debug: 4 models x 4 losses x 2 aux_calib = 32 trainable on fold-0."""
        config = _load(_DEBUG_CONFIG)
        factors = config.get("factors", {})
        models = factors.get("model_family", [])
        losses = factors.get("loss_name", [])
        aux_calibs = factors.get("aux_calibration", [False])
        fixed = config.get("fixed", {})
        num_folds = fixed.get("num_folds", 1)

        total = len(models) * len(losses) * len(aux_calibs) * num_folds
        assert total == 32, (
            f"Debug factorial must have 32 trainable conditions, got {total} "
            f"({len(models)} models x {len(losses)} losses x "
            f"{len(aux_calibs)} aux_calibs x {num_folds} folds)"
        )

    def test_debug_factorial_has_2_zero_shot(self) -> None:
        """Debug: SAM3 Vanilla + VesselFM = 2 zero-shot baselines."""
        config = _load(_DEBUG_CONFIG)
        baselines = config.get("zero_shot_baselines", [])
        total_zs = sum(b.get("folds", 1) for b in baselines)
        assert total_zs == 2, (
            f"Debug factorial must have 2 zero-shot baselines, got {total_zs}"
        )

    def test_debug_factorial_total_34_conditions(self) -> None:
        """Debug: 32 trainable + 2 zero-shot = 34 total conditions."""
        config = _load(_DEBUG_CONFIG)
        factors = config.get("factors", {})
        models = factors.get("model_family", [])
        losses = factors.get("loss_name", [])
        aux_calibs = factors.get("aux_calibration", [False])
        fixed = config.get("fixed", {})
        num_folds = fixed.get("num_folds", 1)
        trainable = len(models) * len(losses) * len(aux_calibs) * num_folds
        baselines = config.get("zero_shot_baselines", [])
        zero_shot = sum(b.get("folds", 1) for b in baselines)
        assert trainable + zero_shot == 34

    def test_debug_uses_2_epochs(self) -> None:
        """Debug run uses 2 epochs (not 50)."""
        config = _load(_DEBUG_CONFIG)
        fixed = config.get("fixed", {})
        assert fixed.get("max_epochs") == 2

    def test_debug_uses_1_fold(self) -> None:
        """Debug run uses fold-0 only."""
        config = _load(_DEBUG_CONFIG)
        fixed = config.get("fixed", {})
        assert fixed.get("num_folds") == 1

    def test_debug_uses_half_data(self) -> None:
        """Debug run uses ~23 train / ~12 val (half dataset)."""
        config = _load(_DEBUG_CONFIG)
        fixed = config.get("fixed", {})
        assert fixed.get("max_train_volumes") == 23
        assert fixed.get("max_val_volumes") == 12

    def test_production_factorial_has_96_trainable(self) -> None:
        """Production: 4 models x 4 losses x 2 aux_calib x 3 folds = 96."""
        if not _PAPER_CONFIG.exists():
            import pytest

            pytest.skip("paper_factorial.yaml not present")
        config = _load(_PAPER_CONFIG)
        factors = config.get("factors", {})
        models = factors.get("model_family", [])
        losses = factors.get("loss_name", [])
        aux_calibs = factors.get("aux_calibration", [False])
        fixed = config.get("fixed", {})
        num_folds = fixed.get("num_folds", 3)
        total = len(models) * len(losses) * len(aux_calibs) * num_folds
        assert total == 96, f"Production factorial must have 96 trainable, got {total}"

    def test_debug_models_match_kg(self) -> None:
        """Debug factorial models must match KG paper_model_comparison.yaml.

        Trainable models: dynunet, mambavesselnet, sam3_topolora, sam3_hybrid.
        Source: knowledge-graph/decisions/L3-technology/paper_model_comparison.yaml
        """
        config = _load(_DEBUG_CONFIG)
        factors = config.get("factors", {})
        models = set(factors.get("model_family", []))
        expected_trainable = {
            "dynunet",
            "mambavesselnet",
            "sam3_topolora",
            "sam3_hybrid",
        }
        assert models == expected_trainable, (
            f"Trainable models mismatch. Got {models}, expected {expected_trainable}"
        )

    def test_debug_zero_shot_models_correct(self) -> None:
        """Zero-shot baselines: sam3_vanilla (minivess) + vesselfm (deepvess)."""
        config = _load(_DEBUG_CONFIG)
        baselines = config.get("zero_shot_baselines", [])
        zs_models = {b["model"] for b in baselines}
        assert zs_models == {"sam3_vanilla", "vesselfm"}

    def test_vesselfm_evaluates_deepvess_only(self) -> None:
        """VesselFM must evaluate on DeepVess ONLY (data leakage on MiniVess)."""
        config = _load(_DEBUG_CONFIG)
        baselines = config.get("zero_shot_baselines", [])
        for b in baselines:
            if b["model"] == "vesselfm":
                assert b.get("dataset") == "deepvess", (
                    f"VesselFM must evaluate on deepvess only, got: {b.get('dataset')}"
                )
                assert b.get("strategy") == "zero_shot_only", (
                    f"VesselFM must be zero_shot_only, got: {b.get('strategy')}"
                )
