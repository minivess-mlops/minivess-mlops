"""Test that run_factorial.sh reads infrastructure params from cloud config.

Phase 2 of 6th pass post-run fix plan — config-to-code wiring.
Verifies the script uses config values instead of hardcoded constants.

Plan: run-debug-factorial-experiment-report-6th-pass-post-run-fix.xml v2
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_FACTORIAL = REPO_ROOT / "scripts" / "run_factorial.sh"
SYNC_SKY_CONFIG = REPO_ROOT / "scripts" / "sync_sky_config.py"
CLOUD_GCP = REPO_ROOT / "configs" / "cloud" / "gcp_spot.yaml"
CLOUD_RUNPOD = REPO_ROOT / "configs" / "cloud" / "runpod_dev.yaml"
FACTORIAL_DEBUG = REPO_ROOT / "configs" / "factorial" / "debug.yaml"


# ---------------------------------------------------------------------------
# Task 2.1a: run_factorial.sh reads cloud_config from factorial YAML
# ---------------------------------------------------------------------------


class TestRunFactorialReadsCloudConfig:
    """run_factorial.sh must extract infrastructure params from config chain."""

    def test_script_extracts_cloud_config_name(self) -> None:
        """Script must read infrastructure.cloud_config from factorial YAML."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        # The script must parse cloud_config from the factorial YAML
        assert "cloud_config" in source, (
            "run_factorial.sh must extract cloud_config from factorial YAML "
            "to load infrastructure parameters (parallel_submissions, rate_limit)"
        )

    def test_script_reads_parallel_submissions(self) -> None:
        """Script must extract parallel_submissions from cloud config."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        assert "PARALLEL_SUBMISSIONS" in source or "parallel_submissions" in source, (
            "run_factorial.sh must read parallel_submissions from cloud config "
            "instead of launching sequentially"
        )

    def test_script_reads_rate_limit(self) -> None:
        """Script must use rate_limit_seconds from config, not hardcoded sleep 5."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        assert "RATE_LIMIT" in source or "rate_limit" in source, (
            "run_factorial.sh must read rate_limit_seconds from cloud config "
            "instead of hardcoded 'sleep 5'"
        )

    def test_no_hardcoded_sleep_5(self) -> None:
        """Script must NOT have hardcoded 'sleep 5' — must use config value."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        lines = source.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#"):
                continue
            # Allow "sleep 5" only if it's inside the Python zero-shot block
            # (which we'll also fix to use config)
            if stripped == "sleep 5":
                raise AssertionError(
                    f"run_factorial.sh line {i}: hardcoded 'sleep 5'. "
                    "Must use config value: sleep ${{RATE_LIMIT_SECONDS}}"
                )

    def test_script_calls_sync_sky_config(self) -> None:
        """Script must call sync_sky_config.py before launching jobs."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        assert "sync_sky_config" in source, (
            "run_factorial.sh must call sync_sky_config.py to ensure "
            "SkyPilot controller matches job cloud"
        )


# ---------------------------------------------------------------------------
# Task 2.1b: Cloud configs have infrastructure params
# ---------------------------------------------------------------------------


class TestCloudConfigInfrastructureParams:
    """Cloud configs must define parallel_submissions and rate_limit_seconds."""

    def test_gcp_spot_has_infrastructure(self) -> None:
        """gcp_spot.yaml must have infrastructure block."""
        cfg = yaml.safe_load(CLOUD_GCP.read_text(encoding="utf-8"))
        infra = cfg.get("infrastructure")
        assert infra is not None, "gcp_spot.yaml missing infrastructure block"
        assert "parallel_submissions" in infra
        assert "rate_limit_seconds" in infra

    def test_runpod_dev_has_infrastructure(self) -> None:
        """runpod_dev.yaml must have infrastructure block."""
        cfg = yaml.safe_load(CLOUD_RUNPOD.read_text(encoding="utf-8"))
        infra = cfg.get("infrastructure")
        assert infra is not None, "runpod_dev.yaml missing infrastructure block"
        assert "parallel_submissions" in infra
        assert "rate_limit_seconds" in infra

    def test_factorial_debug_references_cloud_config(self) -> None:
        """Factorial debug YAML must reference a cloud config."""
        cfg = yaml.safe_load(FACTORIAL_DEBUG.read_text(encoding="utf-8"))
        infra = cfg.get("infrastructure")
        assert infra is not None, "debug.yaml missing infrastructure block"
        cloud_config = infra.get("cloud_config")
        assert cloud_config is not None, (
            "debug.yaml missing infrastructure.cloud_config"
        )
        # Verify the referenced cloud config file exists
        cloud_file = REPO_ROOT / "configs" / "cloud" / f"{cloud_config}.yaml"
        assert cloud_file.exists(), f"Referenced cloud config not found: {cloud_file}"


# ---------------------------------------------------------------------------
# Task 2.1c: sync_sky_config.py callable with cloud config path
# ---------------------------------------------------------------------------


class TestSyncSkyConfigCallable:
    """sync_sky_config.py must be importable and have the right interface."""

    def test_sync_sky_config_function_exists(self) -> None:
        """sync_sky_config module must export sync_sky_config function."""
        source = SYNC_SKY_CONFIG.read_text(encoding="utf-8")
        tree = ast.parse(source)
        func_names = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        assert "sync_sky_config" in func_names

    def test_sync_sky_config_accepts_path(self) -> None:
        """sync_sky_config function must accept a Path argument."""
        source = SYNC_SKY_CONFIG.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "sync_sky_config":
                assert len(node.args.args) >= 1, (
                    "sync_sky_config must accept at least one argument (cloud_config_path)"
                )
                break


# ---------------------------------------------------------------------------
# Task 2.1d: Parallel launch support
# ---------------------------------------------------------------------------


class TestParallelLaunchSupport:
    """run_factorial.sh must support parallel job launches."""

    def test_script_has_parallel_launch_logic(self) -> None:
        """Script must implement background job pool with wait."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        # Parallel launch requires background jobs (&) and wait
        has_background = "& " in source or "&\n" in source or "& #" in source
        has_wait = "wait" in source
        assert has_background and has_wait, (
            "run_factorial.sh must implement parallel launch with background "
            "jobs (&) and wait. Currently launches sequentially."
        )
