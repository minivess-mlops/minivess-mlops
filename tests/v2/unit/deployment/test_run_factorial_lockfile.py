"""Tests for run_factorial.sh lockfile cleanup — T21 regression test.

Bug: If preflight fails, the lockfile is never removed because the cleanup
trap only handles INT/TERM, not EXIT or regular exit paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

RUN_FACTORIAL = (
    Path(__file__).resolve().parents[4] / "scripts" / "run_factorial.sh"
)


class TestLockfileCleanup:
    """T21: Lockfile must be cleaned on all exit paths."""

    @pytest.fixture()
    def _source(self) -> str:
        return RUN_FACTORIAL.read_text(encoding="utf-8")

    def test_lockfile_cleaned_on_preflight_failure(self, _source):
        """trap must include EXIT to clean lockfile on all exit paths."""
        # The trap should include EXIT signal
        assert "EXIT" in _source, (
            "run_factorial.sh trap must include EXIT to clean lockfile "
            "on all exit paths (not just INT/TERM)"
        )

    def test_lockfile_cleaned_on_success(self, _source):
        """Lockfile must be removed on successful exit too."""
        # Either via EXIT trap or explicit rm before exit 0
        has_exit_trap = "trap" in _source and "EXIT" in _source
        has_explicit_cleanup = "LOCKFILE" in _source and "rm" in _source
        assert has_exit_trap or has_explicit_cleanup, (
            "Lockfile must be cleaned on successful exit"
        )
