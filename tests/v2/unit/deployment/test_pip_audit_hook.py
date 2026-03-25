"""Tests for pip-audit pre-commit hook (CVE scanning for Python dependencies).

pip-audit scans installed Python packages against the OSV vulnerability database.
Integrated as a local pre-commit hook via ``uvx pip-audit`` (uv tool runner),
which runs pip-audit in an isolated venv to avoid dependency conflicts with
whylogs (platformdirs<4) and cyclonedx-bom (cyclonedx-python-lib>=8).

Rule #16: No regex. Use yaml.safe_load(), tomllib, str methods, pathlib.Path.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent.parent.parent
PRE_COMMIT_CONFIG = ROOT / ".pre-commit-config.yaml"
MAKEFILE = ROOT / "Makefile"


def _load_pre_commit_hooks() -> list[dict[str, object]]:
    """Load all hooks from .pre-commit-config.yaml."""
    cfg = yaml.safe_load(PRE_COMMIT_CONFIG.read_text(encoding="utf-8"))
    hooks: list[dict[str, object]] = []
    for repo in cfg.get("repos", []):
        for hook in repo.get("hooks", []):
            hooks.append(hook)
    return hooks


def _get_pip_audit_hook() -> dict[str, object] | None:
    """Return the pip-audit hook dict, or None if not found."""
    for hook in _load_pre_commit_hooks():
        if hook.get("id") == "pip-audit":
            return hook
    return None


class TestPipAuditHook:
    """pip-audit pre-commit hook configuration tests."""

    def test_pre_commit_has_pip_audit_hook(self) -> None:
        """pip-audit hook must exist in .pre-commit-config.yaml."""
        hook_ids = [hook.get("id") for hook in _load_pre_commit_hooks()]
        assert "pip-audit" in hook_ids, (
            "pip-audit hook not found in .pre-commit-config.yaml. "
            "Add a local hook with id: pip-audit for CVE scanning."
        )

    def test_pip_audit_hook_is_local_system(self) -> None:
        """Hook must use language: system (runs via uvx)."""
        hook = _get_pip_audit_hook()
        assert hook is not None, "pip-audit hook not found"
        assert hook.get("language") == "system", (
            f"pip-audit hook language={hook.get('language')!r}, expected 'system'. "
            "Must use language: system to run via uvx (uv tool runner)."
        )

    def test_pip_audit_hook_uses_uvx(self) -> None:
        """Entry must use uvx or 'uv tool run' to invoke pip-audit."""
        hook = _get_pip_audit_hook()
        assert hook is not None, "pip-audit hook not found"
        entry = str(hook.get("entry", ""))
        uses_uvx = entry.startswith("uvx pip-audit") or entry.startswith(
            "uv tool run pip-audit"
        )
        assert uses_uvx, (
            f"pip-audit hook entry={entry!r}, expected to start with "
            "'uvx pip-audit' or 'uv tool run pip-audit'. "
            "uvx runs pip-audit in an isolated venv to avoid dependency conflicts."
        )

    def test_pip_audit_hook_uses_strict(self) -> None:
        """Must use --strict flag."""
        hook = _get_pip_audit_hook()
        assert hook is not None, "pip-audit hook not found"
        entry = str(hook.get("entry", ""))
        assert "--strict" in entry, (
            f"pip-audit hook entry={entry!r}, missing --strict flag. "
            "--strict treats warnings as errors for a stricter audit."
        )

    def test_pip_audit_hook_gated_on_dep_files(self) -> None:
        """files: pattern must include pyproject.toml."""
        hook = _get_pip_audit_hook()
        assert hook is not None, "pip-audit hook not found"
        files_pattern = str(hook.get("files", ""))
        assert "pyproject" in files_pattern, (
            f"pip-audit hook files={files_pattern!r}, must include pyproject.toml. "
            "Hook should only run when dependency files change."
        )

    def test_pip_audit_available_via_uvx(self) -> None:
        """pip-audit must be invocable via uvx (uv tool runner)."""
        import subprocess

        result = subprocess.run(
            ["uvx", "pip-audit", "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"uvx pip-audit --version failed (rc={result.returncode}): "
            f"{result.stderr.strip()}. "
            "pip-audit must be available via uvx for the pre-commit hook."
        )

    def test_makefile_has_audit_target(self) -> None:
        """Makefile must have audit: target."""
        content = MAKEFILE.read_text(encoding="utf-8")
        assert "audit:" in content, (
            "Makefile missing 'audit:' target. "
            "Add target to run pip-audit for Python dependency CVE scanning."
        )
