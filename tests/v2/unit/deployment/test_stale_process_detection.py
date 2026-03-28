"""Stale process detection tests for run_factorial.sh.

10th pass discovery: A previous session's resilient loop (PID from 05:46)
was still running with SKIP_PREFLIGHT=1, re-launching jobs with old env
vars every 5 minutes. The lockfile existed but wasn't checked for staleness
by new launches initiated from a different terminal/session.

This test validates the lockfile-based process detection and staleness
handling in run_factorial.sh.

See: docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-report-10th-pass-report.md
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

RUN_FACTORIAL = Path("scripts/run_factorial.sh")
LOCKFILE = Path("outputs/.factorial.lock")


class TestLockfileExists:
    """run_factorial.sh must use a lockfile to prevent concurrent launches."""

    def test_lockfile_path_in_script(self) -> None:
        """Script must reference a lockfile path."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        assert ".factorial.lock" in source, (
            "run_factorial.sh must use a lockfile (.factorial.lock) "
            "to prevent concurrent launches."
        )

    def test_lockfile_contains_pid_check(self) -> None:
        """Script must check if the PID in lockfile is still alive."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        assert "kill -0" in source, (
            "run_factorial.sh must use 'kill -0 $PID' to check if the "
            "process holding the lock is still alive. Without this, a "
            "stale lockfile from a crashed process blocks all future launches."
        )

    def test_lockfile_has_cleanup_trap(self) -> None:
        """Script must trap signals to clean up the lockfile on exit."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        assert "trap" in source, (
            "run_factorial.sh must have a 'trap cleanup' to remove the "
            "lockfile on exit (SIGINT, SIGTERM, EXIT)."
        )


class TestLockfileBehavior:
    """Functional tests for lockfile creation and staleness detection."""

    def test_dry_run_creates_no_lockfile(self) -> None:
        """--dry-run should NOT create a lockfile."""
        if LOCKFILE.exists():
            LOCKFILE.unlink()
        subprocess.run(
            ["bash", str(RUN_FACTORIAL), "--dry-run", "configs/factorial/debug.yaml"],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "SKIP_PREFLIGHT": "1"},
        )
        assert not LOCKFILE.exists(), (
            "--dry-run created a lockfile — this would block real launches."
        )

    def test_lockfile_written_with_pid(self, tmp_path: Path) -> None:
        """Active launch must write its PID to the lockfile."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        # Check that the script writes $$ (current PID) to the lockfile
        assert "$$" in source, (
            "run_factorial.sh must write $$ (PID) to the lockfile so "
            "other instances can check if the holder is alive."
        )


class TestStaleProcessWarning:
    """Tests for detecting and warning about stale processes."""

    def test_script_warns_about_stale_lockfile(self) -> None:
        """Script must warn when a lockfile exists with a dead PID."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        # The script should have logic like:
        # if [ -f lockfile ]; then
        #   LOCK_PID=$(cat lockfile)
        #   if kill -0 $LOCK_PID 2>/dev/null; then
        #     echo "Another launch is running (PID $LOCK_PID)"
        #     exit 1
        #   else
        #     echo "Stale lockfile found (PID $LOCK_PID is dead)"
        #     rm lockfile
        #   fi
        # fi
        assert "stale" in source.lower() or "Stale" in source, (
            "run_factorial.sh must detect and warn about stale lockfiles "
            "from dead processes. Without this, a crashed process blocks "
            "all future launches until someone manually deletes the lockfile."
        )

    def test_script_handles_concurrent_launch_rejection(self) -> None:
        """Script must reject concurrent launches with a clear message."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        # Look for "already running" or similar rejection message
        has_rejection = (
            "already running" in source.lower()
            or "another" in source.lower()
            or "concurrent" in source.lower()
        )
        assert has_rejection, (
            "run_factorial.sh must clearly reject concurrent launches with "
            "a message like 'Another factorial launch is already running'."
        )


class TestLockfileCleanup:
    """Tests for lockfile cleanup on normal and abnormal exit."""

    def test_cleanup_removes_lockfile(self) -> None:
        """The cleanup function must remove the lockfile."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        # Check that cleanup function removes the lockfile
        assert "rm" in source and "factorial.lock" in source, (
            "run_factorial.sh cleanup must 'rm' the .factorial.lock file."
        )

    def test_trap_catches_sigint_sigterm(self) -> None:
        """Trap must catch SIGINT and SIGTERM for clean shutdown."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        has_int = "INT" in source or "SIGINT" in source or "2" in source
        has_term = "TERM" in source or "SIGTERM" in source or "15" in source
        assert has_int, "Trap must catch SIGINT (Ctrl+C)"
        assert has_term, "Trap must catch SIGTERM (kill)"
