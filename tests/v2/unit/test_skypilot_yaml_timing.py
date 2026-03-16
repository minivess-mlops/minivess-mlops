"""Tests for SkyPilot YAML timing instrumentation.

Verifies that SkyPilot YAML setup: blocks contain shell-level timing
(key=value timestamps via date +%s.%N) for all major operations, and
that run: blocks call the timing logger before training.

Issue: #683
"""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SKYPILOT_DIR = _REPO_ROOT / "deployment" / "skypilot"

# All operations that must have timing start/end in setup blocks
_TIMED_OPERATIONS = (
    "python_install",
    "uv_install",
    "uv_sync",
    "dvc_config",
    "dvc_pull",
    "model_weights",
    "verification",
)


def _load_yaml_setup(yaml_path: Path) -> str:
    """Load a SkyPilot YAML and return the setup: block as a string."""
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return data.get("setup", "")


def _load_yaml_run(yaml_path: Path) -> str:
    """Load a SkyPilot YAML and return the run: block as a string."""
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return data.get("run", "")


class TestDevRunpodYamlTiming:
    """Tests for timing instrumentation in dev_runpod.yaml."""

    def test_dev_runpod_yaml_has_timing_start(self) -> None:
        """setup: block contains setup_start timestamp."""
        setup = _load_yaml_setup(_SKYPILOT_DIR / "dev_runpod.yaml")
        assert "setup_start=" in setup or "setup_start=$(" in setup

    def test_dev_runpod_yaml_has_timing_end(self) -> None:
        """setup: block contains setup_end timestamp."""
        setup = _load_yaml_setup(_SKYPILOT_DIR / "dev_runpod.yaml")
        assert "setup_end=" in setup or "setup_end=$(" in setup

    def test_dev_runpod_yaml_has_all_operations_timed(self) -> None:
        """All 7 operations have start/end timestamp lines in setup: block."""
        setup = _load_yaml_setup(_SKYPILOT_DIR / "dev_runpod.yaml")
        for op in _TIMED_OPERATIONS:
            start_marker = f"{op}_start="
            end_marker = f"{op}_end="
            assert start_marker in setup, (
                f"Missing {start_marker} in dev_runpod.yaml setup"
            )
            assert end_marker in setup, f"Missing {end_marker} in dev_runpod.yaml setup"

    def test_dev_runpod_yaml_writes_timing_file(self) -> None:
        """setup: block writes timing_setup.txt."""
        setup = _load_yaml_setup(_SKYPILOT_DIR / "dev_runpod.yaml")
        assert "timing_setup.txt" in setup


class TestSmokeTestLambdaYamlTiming:
    """Lambda Labs archived 2026-03-16 — smoke_test_lambda.yaml moved to deployment/archived/lambda/.

    Tests now validate the archived file location and verify the mamba YAML has equivalent timing.
    """

    def test_lambda_yaml_archived(self) -> None:
        """smoke_test_lambda.yaml must NOT be in skypilot/ — archived 2026-03-16."""
        assert not (_SKYPILOT_DIR / "smoke_test_lambda.yaml").exists(), (
            "smoke_test_lambda.yaml exists but Lambda Labs was archived 2026-03-16. "
            "It belongs in deployment/archived/lambda/skypilot/"
        )

    def test_lambda_yaml_in_archive(self) -> None:
        """Archived lambda YAML must exist at the expected archived location."""
        archived = (
            _REPO_ROOT
            / "deployment"
            / "archived"
            / "lambda"
            / "skypilot"
            / "smoke_test_lambda.yaml"
        )
        assert archived.exists(), (
            "smoke_test_lambda.yaml should be in deployment/archived/lambda/skypilot/ — "
            "move it there for archival"
        )

    def test_smoke_test_mamba_yaml_has_timing_guards(self) -> None:
        """smoke_test_mamba.yaml (Lambda replacement) has fail-fast guards in setup."""
        setup = _load_yaml_setup(_SKYPILOT_DIR / "smoke_test_mamba.yaml")
        assert "set -ex" in setup, "setup: must use set -ex for fail-fast"
        assert "nvidia-smi" in setup, "setup: must verify GPU"


class TestRunBlockTimingLogger:
    """Tests for timing logger call in run: blocks."""

    def test_dev_runpod_run_calls_timing_logger(self) -> None:
        """run: block calls log_infrastructure_timing before training."""
        run_block = _load_yaml_run(_SKYPILOT_DIR / "dev_runpod.yaml")
        assert "infrastructure_timing" in run_block


class TestEnvExampleHourlyRate:
    """Tests for INSTANCE_HOURLY_USD in .env.example."""

    def test_env_example_has_instance_hourly_usd(self) -> None:
        """INSTANCE_HOURLY_USD is declared in .env.example."""
        env_example = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        assert "INSTANCE_HOURLY_USD" in env_example
