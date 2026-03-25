"""Fire-and-forget factorial launch — resilience verification.

Tests that run_factorial.sh can survive interruptions and resume,
and that run_factorial_resilient.sh provides fire-and-forget execution.

Plan: retries-for-skypilot-spots-and-autoresume-plan.md
"""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path("scripts/run_factorial.sh")
WRAPPER = Path("scripts/run_factorial_resilient.sh")
FACTORIAL_CONFIG = Path("configs/factorial/debug.yaml")


class TestRunFactorialBugFixes:
    """Verify bash script bugs are fixed."""

    def test_launched_failed_counts_are_clean_integers(self) -> None:
        """LAUNCHED and FAILED variables must be clean integers (no newlines)."""
        source = SCRIPT.read_text(encoding="utf-8")
        # The bug: grep -c can return "0\n0" in some cases.
        # Fix: pipe through tr -d or use $(( )) arithmetic.
        # Check that LAUNCHED and FAILED are sanitized before integer comparison.
        assert (
            "tr -d" in source
            or "xargs" in source
            or "$((" in source
            or "| head -1" in source
            or ".strip()" in source
        ), (
            "LAUNCHED/FAILED counts from grep -c must be sanitized to clean integers. "
            "Bug: multiline output causes 'integer expression expected' at exit code check."
        )


class TestFireAndForgetWrapper:
    """run_factorial_resilient.sh must exist for fire-and-forget execution."""

    def test_resilient_wrapper_exists(self) -> None:
        """scripts/run_factorial_resilient.sh must exist."""
        assert WRAPPER.exists(), (
            "Fire-and-forget wrapper missing. User runs one command, comes back "
            "in 1 week — results are there. run_factorial.sh alone is not enough "
            "(it exits after one pass, no outer retry loop)."
        )

    def test_resilient_wrapper_has_max_wait(self) -> None:
        """Wrapper must have configurable max wait time."""
        if not WRAPPER.exists():
            return
        source = WRAPPER.read_text(encoding="utf-8")
        assert "MAX_WAIT" in source or "max_wait" in source, (
            "Wrapper must have a configurable maximum wait time (default ~1 week). "
            "Without it, the wrapper runs forever if spots never become available."
        )

    def test_resilient_wrapper_calls_resume(self) -> None:
        """Wrapper must call run_factorial.sh --resume in a loop."""
        if not WRAPPER.exists():
            return
        source = WRAPPER.read_text(encoding="utf-8")
        assert "--resume" in source, (
            "Wrapper must call run_factorial.sh --resume (not bare run). "
            "Without --resume, each iteration resubmits ALL conditions."
        )

    def test_resilient_wrapper_has_sleep_interval(self) -> None:
        """Wrapper must sleep between retry attempts."""
        if not WRAPPER.exists():
            return
        source = WRAPPER.read_text(encoding="utf-8")
        assert "sleep" in source or "RETRY_INTERVAL" in source, (
            "Wrapper must sleep between retry attempts to avoid hammering SkyPilot API."
        )

    def test_resilient_wrapper_is_executable(self) -> None:
        """Wrapper must have execute permission."""
        if not WRAPPER.exists():
            return
        import os

        assert os.access(WRAPPER, os.X_OK), (
            "run_factorial_resilient.sh must be executable (chmod +x)"
        )


class TestTimeoutConfig:
    """Factorial config must have resilience timeout settings."""

    def test_factorial_config_has_resilience_section(self) -> None:
        """debug.yaml must have resilience settings for fire-and-forget."""
        import yaml

        cfg = yaml.safe_load(FACTORIAL_CONFIG.read_text(encoding="utf-8"))
        infra = cfg.get("infrastructure", {})
        assert "max_wait_hours" in infra or "resilience" in cfg, (
            "Factorial config must have max_wait_hours or resilience section "
            "for fire-and-forget execution. Default: 168 hours (1 week)."
        )
