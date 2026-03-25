"""Tests for model_overrides wiring in run_factorial.sh.

Verifies that the dry-run output shows correct BATCH_SIZE and GRAD_ACCUM_STEPS
for SAM3 models (BS=1, accum=4) vs DynUNet (BS=2, accum=1).

Issue: #940 (SAM3 batch_size=1 fix)
Plan: docs/planning/sam3-batch-size-1-and-robustification-plan.xml Task 1.2
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[4]
RUN_FACTORIAL = REPO_ROOT / "scripts" / "run_factorial.sh"
DEBUG_CONFIG = "configs/factorial/debug.yaml"


@pytest.mark.slow
class TestRunFactorialDryRunModelOverrides:
    """Verify dry-run output shows per-model BATCH_SIZE and GRAD_ACCUM_STEPS."""

    @pytest.fixture(scope="class")
    def dry_run_output(self) -> str:
        """Run the factorial script in dry-run mode and capture output."""
        result = subprocess.run(
            [str(RUN_FACTORIAL), "--dry-run", DEBUG_CONFIG],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=30,
            env={
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "HOME": str(pathlib.Path.home()),
                "SKIP_PREFLIGHT": "1",
            },
        )
        return result.stdout + result.stderr

    def test_sam3_topolora_batch_size_1(self, dry_run_output: str) -> None:
        """SAM3 TopoLoRA conditions should show BATCH_SIZE=1."""
        lines = [
            line for line in dry_run_output.splitlines()
            if "sam3_topolora" in line and "bs=" in line
        ]
        assert len(lines) > 0, "No sam3_topolora conditions found in dry-run output"
        for line in lines:
            assert "bs=1" in line, f"Expected bs=1 for sam3_topolora: {line}"

    def test_sam3_topolora_grad_accum_4(self, dry_run_output: str) -> None:
        """SAM3 TopoLoRA conditions should show accum=4."""
        lines = [
            line for line in dry_run_output.splitlines()
            if "sam3_topolora" in line and "accum=" in line
        ]
        assert len(lines) > 0, "No sam3_topolora conditions found in dry-run output"
        for line in lines:
            assert "accum=4" in line, f"Expected accum=4 for sam3_topolora: {line}"

    def test_dynunet_batch_size_2(self, dry_run_output: str) -> None:
        """DynUNet conditions should show BATCH_SIZE=2 (global default)."""
        lines = [
            line for line in dry_run_output.splitlines()
            if "dynunet" in line and "bs=" in line
        ]
        assert len(lines) > 0, "No dynunet conditions found in dry-run output"
        for line in lines:
            assert "bs=2" in line, f"Expected bs=2 for dynunet: {line}"

    def test_dynunet_grad_accum_1(self, dry_run_output: str) -> None:
        """DynUNet conditions should show accum=1 (default, no accumulation)."""
        lines = [
            line for line in dry_run_output.splitlines()
            if "dynunet" in line and "accum=" in line
        ]
        assert len(lines) > 0, "No dynunet conditions found in dry-run output"
        for line in lines:
            assert "accum=1" in line, f"Expected accum=1 for dynunet: {line}"

    def test_dry_run_shows_batch_size_in_launch_cmd(self, dry_run_output: str) -> None:
        """Dry-run launch commands should include BATCH_SIZE env var."""
        dry_lines = [
            line for line in dry_run_output.splitlines()
            if "[DRY RUN]" in line
        ]
        assert len(dry_lines) > 0, "No DRY RUN lines found"
        for line in dry_lines:
            assert "BATCH_SIZE=" in line, f"Missing BATCH_SIZE in: {line}"
            assert "GRAD_ACCUM_STEPS=" in line, f"Missing GRAD_ACCUM_STEPS in: {line}"

    def test_sam3_effective_batch_size_4(self, dry_run_output: str) -> None:
        """SAM3 TopoLoRA should have effective_bs=4 (1*4)."""
        lines = [
            line for line in dry_run_output.splitlines()
            if "sam3_topolora" in line and "eff_bs=" in line
        ]
        assert len(lines) > 0, "No sam3_topolora conditions found"
        for line in lines:
            assert "eff_bs=4" in line, f"Expected eff_bs=4 for sam3_topolora: {line}"
