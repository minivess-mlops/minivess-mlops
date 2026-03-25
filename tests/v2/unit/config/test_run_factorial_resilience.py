"""Resilience tests for run_factorial.sh — 10-gap fix plan.

Verifies retry logic, signal handling, idempotency, exit codes,
and error handling in the factorial launch script.

Source: cold-start-prompt-skypilot-resilience-fixes.md
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_FACTORIAL = REPO_ROOT / "scripts" / "run_factorial.sh"


# ---------------------------------------------------------------------------
# Gap 1: Retry on launch failure (exponential backoff)
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Launch failures must be retried with exponential backoff."""

    def test_has_retry_logic(self) -> None:
        """Script must implement retry loop for failed launches."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        assert "MAX_RETRIES" in source or "max_retries" in source, (
            "run_factorial.sh must implement retry logic with MAX_RETRIES "
            "for failed sky jobs launch calls (Gap #1)"
        )

    def test_has_exponential_backoff(self) -> None:
        """Retry must use exponential backoff, not fixed sleep."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        # Exponential backoff: some form of doubling or multiplying delay
        has_backoff = (
            "BACKOFF" in source
            or "backoff" in source
            or "retry_delay" in source
            or "RETRY_DELAY" in source
        )
        assert has_backoff, (
            "run_factorial.sh must use exponential backoff between retries "
            "(e.g., 2s, 4s, 8s) — not fixed delay (Gap #1)"
        )


# ---------------------------------------------------------------------------
# Gap 3: Resume on crash (--resume flag)
# ---------------------------------------------------------------------------


class TestResumeSupport:
    """Script must support resuming from partial launches."""

    def test_has_resume_flag(self) -> None:
        """Script must accept --resume flag."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        assert "--resume" in source, (
            "run_factorial.sh must accept --resume flag to skip "
            "already-submitted conditions (Gap #3)"
        )

    def test_resume_checks_existing_jobs(self) -> None:
        """--resume mode must check sky jobs queue for existing conditions."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        has_queue_check = (
            "jobs queue" in source
            or "jobs_queue" in source
            or "EXISTING_JOBS" in source
        )
        assert has_queue_check, (
            "run_factorial.sh --resume must check sky jobs queue for "
            "already-submitted condition names before re-launching (Gap #3)"
        )


# ---------------------------------------------------------------------------
# Gap 6: Signal handling (trap cleanup)
# ---------------------------------------------------------------------------


class TestSignalHandling:
    """Script must handle INT/TERM signals gracefully."""

    def test_has_trap(self) -> None:
        """Script must have trap for EXIT/INT/TERM signals."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        assert "trap " in source, (
            "run_factorial.sh must trap EXIT/INT/TERM signals to clean up "
            "background jobs on Ctrl+C or kill (Gap #6)"
        )

    def test_trap_kills_background_jobs(self) -> None:
        """Trap handler must kill background launch subshells."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        # The trap handler should reference killing jobs or children
        has_cleanup = "cleanup" in source or "kill " in source or "jobs -p" in source
        assert has_cleanup, (
            "run_factorial.sh trap handler must kill background jobs on signal (Gap #6)"
        )


# ---------------------------------------------------------------------------
# Gap 7: Permissive Python parsing — check exit codes
# ---------------------------------------------------------------------------


class TestPythonParsingSafety:
    """Python config parsing must fail fast on errors."""

    def test_python_parsing_checks_exit_code(self) -> None:
        """Python config extraction must have error handling."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        # The Python config parsing blocks should either:
        # 1. Use set -e (already set), or
        # 2. Check $? explicitly, or
        # 3. Use || exit patterns
        # Since set -euo pipefail is at line 29, Python failures will exit.
        # But we should also have try/except in the Python blocks
        has_python_error_handling = (
            "sys.exit(1)" in source or "except" in source or "traceback" in source
        )
        assert has_python_error_handling, (
            "run_factorial.sh Python parsing blocks must have error handling "
            "(try/except with traceback) to diagnose config parse failures (Gap #7)"
        )


# ---------------------------------------------------------------------------
# Gap 10: Ambiguous exit codes — 0/1/2
# ---------------------------------------------------------------------------


class TestExitCodes:
    """Script must return distinct exit codes for different outcomes."""

    def test_has_partial_failure_exit_code(self) -> None:
        """Script must exit 2 for partial failure (some launched, some failed)."""
        source = RUN_FACTORIAL.read_text(encoding="utf-8")
        assert "exit 2" in source, (
            "run_factorial.sh must exit 2 for partial failure "
            "(some conditions launched, some failed) — not just 0 or 1 (Gap #10)"
        )
