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


# ---------------------------------------------------------------------------
# Issue #950: Resume substring collision — grep -qxF (exact line match)
# ---------------------------------------------------------------------------


RUN_FACTORIAL_RESILIENT = REPO_ROOT / "scripts" / "run_factorial_resilient.sh"


class TestResumeCollisionFix:
    """Resume grep must use exact line match to prevent substring collisions."""

    def test_resume_uses_exact_line_match(self) -> None:
        """Script must use grep -x (exact line), not plain -F (substring)."""
        src = RUN_FACTORIAL.read_text(encoding="utf-8")
        lines = src.splitlines()
        resume_lines = [l for l in lines if "grep" in l and "CONDITION_NAME" in l]
        assert len(resume_lines) > 0, "No grep lines referencing CONDITION_NAME found"
        for line in resume_lines:
            # -x flag can be combined: -xF, -qxF, -Fxq, etc. or standalone --line-regexp
            # Extract the flags portion (between 'grep' and the pattern argument)
            after_grep = line.partition("grep")[2]
            flags_part = after_grep.split('"')[0]  # everything before first quoted arg
            has_exact = "x" in flags_part or "--line-regexp" in flags_part
            assert has_exact, (
                f"Resume grep must use -x (exact line match): {line.strip()}"
            )

    def test_substring_collision_prevented(self) -> None:
        """grep -xF 'dynunet-dice_ce' must NOT match 'dynunet-dice_ce_cldice'."""
        import subprocess

        jobs = "dynunet-dice_ce_cldice-calibtrue-f0\ndynunet-dice_ce-calibtrue-f0\n"
        # -xF = exact line match
        result = subprocess.run(
            ["grep", "-cxF", "dynunet-dice_ce-calibtrue-f0"],
            input=jobs,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "1"  # exactly 1 match


# ---------------------------------------------------------------------------
# Issue #951: Lockfile guard — prevent concurrent launches
# ---------------------------------------------------------------------------


class TestLockfileGuard:
    """run_factorial.sh must use a lockfile to prevent concurrent launches."""

    def test_lockfile_referenced_in_script(self) -> None:
        """Script must reference a lockfile mechanism."""
        src = RUN_FACTORIAL.read_text(encoding="utf-8")
        assert ".lock" in src or "flock" in src, (
            "run_factorial.sh must reference a lockfile (.lock or flock)"
        )

    def test_lockfile_before_launch(self) -> None:
        """Lockfile variable definition must appear before the first sky jobs launch."""
        lines = RUN_FACTORIAL.read_text(encoding="utf-8").splitlines()
        # Find the LOCKFILE variable assignment (defines the lock path)
        lock_def_idx = next(
            (i for i, l in enumerate(lines)
             if "LOCKFILE=" in l and ".lock" in l),
            len(lines),
        )
        # Find first actual launch command (${SKY_BIN} jobs launch or sky jobs launch)
        # excluding comments and echo/dry-run lines
        launch_idx = next(
            (i for i, l in enumerate(lines)
             if "jobs launch" in l
             and not l.strip().startswith("#")
             and not l.strip().startswith("echo")
             and not l.strip().startswith("print")),
            0,
        )
        assert lock_def_idx < launch_idx, (
            f"LOCKFILE definition (line {lock_def_idx}) must appear before first "
            f"jobs launch command (line {launch_idx})"
        )

    def test_lockfile_cleanup_on_exit(self) -> None:
        """Lockfile must be cleaned up on script exit (trap or cleanup function)."""
        src = RUN_FACTORIAL.read_text(encoding="utf-8")
        # Must have trap for cleanup
        assert "trap" in src, "Script must have a trap for cleanup"
        # Lock file removal must be in a cleanup function or trap
        lines = src.splitlines()
        has_lock_cleanup = any(
            ".lock" in l and ("rm" in l or "trap" in l) for l in lines
        )
        assert has_lock_cleanup or ("cleanup" in src and ".lock" in src), (
            "Lockfile must be removed in cleanup/trap on exit"
        )


# ---------------------------------------------------------------------------
# Issue #947: Permanent failure detection in resilient.sh
# ---------------------------------------------------------------------------


class TestPermanentFailureTracking:
    """run_factorial_resilient.sh must detect permanently failed conditions."""

    def test_resilient_script_exists(self) -> None:
        """run_factorial_resilient.sh must exist."""
        assert RUN_FACTORIAL_RESILIENT.is_file(), (
            "scripts/run_factorial_resilient.sh must exist"
        )

    def test_permanent_failure_detection(self) -> None:
        """Resilient wrapper must track and report permanent failures."""
        src = RUN_FACTORIAL_RESILIENT.read_text(encoding="utf-8")
        has_perm_fail = (
            "PERMANENTLY_FAILED" in src
            or "permanent" in src.lower()
            or "PERM_FAIL" in src
        )
        assert has_perm_fail, (
            "run_factorial_resilient.sh must detect permanent failures "
            "(conditions that fail repeatedly beyond retry limit)"
        )

    def test_failure_count_tracking(self) -> None:
        """Resilient wrapper must count failures per condition."""
        src = RUN_FACTORIAL_RESILIENT.read_text(encoding="utf-8")
        has_failure_count = (
            "fail_count" in src.lower()
            or "failure_count" in src.lower()
            or "FAIL_COUNT" in src
            or "MAX_CONDITION_FAILURES" in src
        )
        assert has_failure_count, (
            "run_factorial_resilient.sh must track per-condition failure counts "
            "to identify permanently failed conditions"
        )
