"""Tests for SkyPilot YAML exit code consistency — T5 regression tests.

Bug: train_factorial.yaml declares recover_on_exit_codes: [33, 34] but the
setup script uses exit 1 everywhere. The EAGER_NEXT_REGION strategy is dead.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

YAML_FILE = (
    Path(__file__).resolve().parents[4]
    / "deployment"
    / "skypilot"
    / "train_factorial.yaml"
)


@pytest.fixture()
def yaml_content() -> dict:
    """Load and parse train_factorial.yaml."""
    return yaml.safe_load(YAML_FILE.read_text(encoding="utf-8"))


@pytest.fixture()
def setup_script() -> str:
    """Extract the setup script from train_factorial.yaml."""
    content = YAML_FILE.read_text(encoding="utf-8")
    # The setup script is after "setup: |" or "setup: |-"
    in_setup = False
    lines: list[str] = []
    for line in content.split("\n"):
        if line.strip().startswith("setup:"):
            in_setup = True
            continue
        if in_setup:
            if line and not line.startswith(" ") and not line.startswith("\t"):
                break
            lines.append(line)
    return "\n".join(lines)


class TestSetupScriptExitCodes:
    """T5: Setup script exit codes must match recover_on_exit_codes."""

    def test_recover_on_exit_codes_declared(self, yaml_content):
        """YAML must declare recover_on_exit_codes in job_recovery."""
        # job_recovery is nested under resources in SkyPilot YAML
        resources = yaml_content.get("resources", {})
        jr = resources.get("job_recovery", yaml_content.get("job_recovery", {}))
        codes = jr.get("recover_on_exit_codes", [])
        assert 33 in codes, "Exit code 33 (DVC network error) must be in recover_on_exit_codes"
        assert 34 in codes, "Exit code 34 (HF download timeout) must be in recover_on_exit_codes"

    def test_setup_script_dvc_exit_code_matches_recovery(self, setup_script):
        """DVC pull failure should exit with code 33 (matching recovery config)."""
        assert "exit 33" in setup_script, (
            "DVC pull failure should 'exit 33' to trigger EAGER_NEXT_REGION recovery. "
            "Currently uses 'exit 1' which does NOT trigger region failover."
        )

    def test_setup_script_hf_exit_code_matches_recovery(self, setup_script):
        """HF download failure should exit with code 34 (matching recovery config)."""
        # HF weight download uses timeout + retry loop. After all attempts fail,
        # it should exit 34 to trigger region failover.
        assert "exit 34" in setup_script, (
            "HF weight download failure should 'exit 34' to trigger region failover. "
            "Currently uses 'exit 1' which does NOT trigger region failover."
        )
