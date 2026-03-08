"""Tests for the pre-push hook infrastructure (#469).

Verifies that:
- scripts/pr_readiness_check.sh exists and is executable
- The script contains all required validation commands
- scripts/install-hooks.sh exists and wires the pre-push hook
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PR_READINESS = REPO_ROOT / "scripts" / "pr_readiness_check.sh"
INSTALL_HOOKS = REPO_ROOT / "scripts" / "install-hooks.sh"


class TestPrReadinessScript:
    def test_pr_readiness_script_exists(self) -> None:
        assert PR_READINESS.exists(), (
            f"scripts/pr_readiness_check.sh not found at {PR_READINESS}. "
            "Create it before wiring as a pre-push hook."
        )

    def test_pr_readiness_script_executable(self) -> None:
        assert PR_READINESS.exists(), "pr_readiness_check.sh does not exist"
        mode = os.stat(PR_READINESS).st_mode
        assert mode & stat.S_IXUSR, (
            "scripts/pr_readiness_check.sh is not user-executable. "
            "Run: chmod +x scripts/pr_readiness_check.sh"
        )

    def test_pr_readiness_script_runs_pytest(self) -> None:
        content = PR_READINESS.read_text(encoding="utf-8")
        assert "pytest" in content, (
            "scripts/pr_readiness_check.sh must invoke pytest. "
            "Add: uv run pytest tests/unit/ -x -q --tb=short"
        )

    def test_pr_readiness_script_runs_ruff(self) -> None:
        content = PR_READINESS.read_text(encoding="utf-8")
        assert "ruff" in content, (
            "scripts/pr_readiness_check.sh must invoke ruff. "
            "Add: uv run ruff check src/minivess/ tests/"
        )

    def test_pr_readiness_script_runs_mypy(self) -> None:
        content = PR_READINESS.read_text(encoding="utf-8")
        assert "mypy" in content, (
            "scripts/pr_readiness_check.sh must invoke mypy. "
            "Add: uv run mypy src/minivess/"
        )


class TestInstallHooksScript:
    def test_pre_push_hook_template_exists(self) -> None:
        assert INSTALL_HOOKS.exists(), (
            f"scripts/install-hooks.sh not found at {INSTALL_HOOKS}. "
            "Create it to wire the pre-push hook for new contributors."
        )

    def test_install_hooks_is_executable(self) -> None:
        assert INSTALL_HOOKS.exists(), "install-hooks.sh does not exist"
        mode = os.stat(INSTALL_HOOKS).st_mode
        assert mode & stat.S_IXUSR, (
            "scripts/install-hooks.sh is not user-executable. "
            "Run: chmod +x scripts/install-hooks.sh"
        )

    def test_install_hooks_references_pre_push(self) -> None:
        assert INSTALL_HOOKS.exists(), "install-hooks.sh does not exist"
        content = INSTALL_HOOKS.read_text(encoding="utf-8")
        assert "pre-push" in content, (
            "scripts/install-hooks.sh must create a pre-push git hook. "
            "See .git/hooks/pre-push"
        )

    def test_install_hooks_references_pr_readiness(self) -> None:
        assert INSTALL_HOOKS.exists(), "install-hooks.sh does not exist"
        content = INSTALL_HOOKS.read_text(encoding="utf-8")
        assert "pr_readiness_check.sh" in content, (
            "scripts/install-hooks.sh must call scripts/pr_readiness_check.sh "
            "from the pre-push hook body."
        )
