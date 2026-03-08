"""Tests that .sh training scripts use Docker, not bare Python (T-01, T-06).

Verifies:
- No bare `uv run python` in training scripts
- No `PREFECT_DISABLED=1` in scripts
- Training scripts invoke `docker compose` or `docker run`

References:
  - docs/planning/minivess-vision-enforcement-plan-execution.xml (T-01, T-06)
  - CLAUDE.md Rules #17, #18, #19
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"

# Training-related keywords — if a .sh script contains any of these,
# it MUST use Docker (not bare Python).
_TRAINING_KEYWORDS = frozenset(
    {
        "training_flow",
        "train_flow",
        "model_family",
        "train-flow",
    }
)


def _read_sh_scripts() -> list[tuple[str, str]]:
    """Return list of (filename, content) for all .sh scripts."""
    if not SCRIPTS_DIR.is_dir():
        return []
    return [
        (f.name, f.read_text(encoding="utf-8"))
        for f in sorted(SCRIPTS_DIR.glob("*.sh"))
    ]


class TestNoBarePythonInTrainingScripts:
    """Verify training scripts don't use bare `uv run python`."""

    def test_no_bare_python_in_training_scripts(self) -> None:
        """No .sh script with training keywords should use `uv run python`."""
        violations: list[str] = []
        for name, content in _read_sh_scripts():
            is_training = any(kw in content for kw in _TRAINING_KEYWORDS)
            if not is_training:
                continue
            # Skip comment lines (lines starting with #) — comments may document
            # what NOT to do without actually doing it.
            non_comment_lines = [
                line
                for line in content.splitlines()
                if not line.lstrip().startswith("#")
            ]
            non_comment_content = " ".join(non_comment_lines)
            if "uv run python" in non_comment_content:
                violations.append(name)
        assert not violations, (
            f"Training scripts using bare 'uv run python' (must use Docker): {violations}"
        )


class TestNoPrefectDisabledInScripts:
    """Verify PREFECT_DISABLED=1 is not in any .sh script."""

    def test_no_prefect_disabled_in_scripts(self) -> None:
        """PREFECT_DISABLED=1 is banned in scripts/ (test-only escape hatch)."""
        violations: list[str] = []
        for name, content in _read_sh_scripts():
            if "PREFECT_DISABLED=1" in content:
                violations.append(name)
        assert not violations, (
            f"Scripts with PREFECT_DISABLED=1 (banned — test only): {violations}"
        )


class TestTrainingScriptsInvokeDocker:
    """Verify training scripts invoke Docker."""

    def test_training_scripts_invoke_docker(self) -> None:
        """Any .sh script with training keywords must use docker or prefect deployment."""
        violations: list[str] = []
        for name, content in _read_sh_scripts():
            is_training = any(kw in content for kw in _TRAINING_KEYWORDS)
            if not is_training:
                continue
            has_docker = "docker compose" in content or "docker run" in content
            has_prefect = "prefect deployment run" in content
            if not has_docker and not has_prefect:
                violations.append(name)
        assert not violations, (
            f"Training scripts without Docker or Prefect invocation: {violations}"
        )

    @pytest.mark.parametrize(
        "script_name",
        [f.name for f in sorted(SCRIPTS_DIR.glob("train*.sh"))]
        if SCRIPTS_DIR.is_dir()
        else [],
    )
    def test_train_scripts_have_docker(self, script_name: str) -> None:
        """Each train*.sh script must use docker compose."""
        content = (SCRIPTS_DIR / script_name).read_text(encoding="utf-8")
        assert "docker compose" in content or "docker run" in content, (
            f"{script_name} must invoke Docker, not bare Python"
        )
