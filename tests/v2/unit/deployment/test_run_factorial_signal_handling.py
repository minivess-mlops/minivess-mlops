"""Tests for run_factorial.sh signal handling — T16 regression test.

Bug: cleanup() kills background shell subshells but NOT the sky jobs launch
child processes within them. Ctrl+C leaks submissions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

RUN_FACTORIAL = (
    Path(__file__).resolve().parents[4] / "scripts" / "run_factorial.sh"
)


class TestSignalHandling:
    """T16: Signal handling must properly clean up child processes."""

    @pytest.fixture()
    def _source(self) -> str:
        return RUN_FACTORIAL.read_text(encoding="utf-8")

    def test_cleanup_function_exists(self, _source):
        """run_factorial.sh must have a cleanup() function."""
        assert "cleanup()" in _source or "cleanup ()" in _source, (
            "run_factorial.sh must define a cleanup() function"
        )

    def test_subshell_has_trap_or_process_group_kill(self, _source):
        """Subshells must propagate signals to sky jobs launch children."""
        # Either approach is acceptable:
        # 1. trap 'kill 0' in subshells
        # 2. kill -- -$$ in cleanup (process group kill)
        has_subshell_trap = "trap" in _source and "kill 0" in _source
        has_process_group_kill = "kill --" in _source or "kill -- -$$" in _source
        has_jobs_kill = "jobs -p" in _source
        assert has_subshell_trap or has_process_group_kill or has_jobs_kill, (
            "run_factorial.sh must propagate signals to child processes via "
            "trap 'kill 0', process group kill, or jobs -p"
        )
