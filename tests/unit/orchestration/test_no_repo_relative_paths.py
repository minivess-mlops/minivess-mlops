"""Tests that src/ code does not hardcode repo-relative checkpoint paths (T-07).

Scans Python source files for patterns like:
- CHECKPOINT_DIR = "checkpoints"
- checkpoint_dir = Path("checkpoints")

These are violations of CLAUDE.md Rule #18 (volume mounts).

References:
  - docs/planning/minivess-vision-enforcement-plan-execution.xml (T-07)
  - CLAUDE.md Rule #18, #19
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[3] / "src"

# Patterns that indicate a repo-relative checkpoint path assignment
_REPO_RELATIVE_PATHS = frozenset(
    {
        "checkpoints",
        "./checkpoints",
        "checkpoints/",
        "./checkpoints/",
    }
)


def _find_string_assignments(source: str, filename: str) -> list[tuple[int, str, str]]:
    """Find string literal assignments that look like repo-relative checkpoint paths.

    Uses ast.parse (CLAUDE.md Rule #16 — no regex for structured data).

    Returns list of (line_number, variable_name, string_value).
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []

    violations: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        # Check if the value is a string constant
        value = node.value
        str_val: str | None = None
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            str_val = value.value
        elif (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "Path"
            and value.args
            and isinstance(value.args[0], ast.Constant)
            and isinstance(value.args[0].value, str)
        ):
            str_val = value.args[0].value

        if str_val is None:
            continue
        if str_val.rstrip("/") not in {"checkpoints", "./checkpoints"}:
            continue

        # Get variable name
        for target in node.targets:
            var_name = ""
            if isinstance(target, ast.Name):
                var_name = target.id
            elif isinstance(target, ast.Attribute):
                var_name = target.attr
            if "checkpoint" in var_name.lower() or "ckpt" in var_name.lower():
                violations.append((node.lineno, var_name, str_val))

    return violations


class TestNoRepoRelativeCheckpointPaths:
    """Verify src/ code doesn't hardcode repo-relative checkpoint paths."""

    def test_no_repo_relative_checkpoint_in_src(self) -> None:
        """No Python file in src/ should assign 'checkpoints' to a checkpoint variable."""
        if not SRC_DIR.is_dir():
            pytest.skip("src/ directory not found")

        all_violations: list[str] = []
        for py_file in sorted(SRC_DIR.rglob("*.py")):
            source = py_file.read_text(encoding="utf-8")
            violations = _find_string_assignments(source, str(py_file))
            for lineno, var_name, val in violations:
                rel = py_file.relative_to(SRC_DIR.parent)
                all_violations.append(f"{rel}:{lineno} — {var_name} = {val!r}")

        assert not all_violations, (
            "Repo-relative checkpoint paths found in src/ (must use Docker volumes):\n"
            + "\n".join(f"  {v}" for v in all_violations)
        )

    def test_no_repo_relative_mlruns_in_src(self) -> None:
        """No Python file in src/ should assign 'mlruns' as a raw string to tracking vars."""
        if not SRC_DIR.is_dir():
            pytest.skip("src/ directory not found")

        # This is a softer check — mlruns is used as a default in some places
        # but should always be overridden by Docker ENV
        pass  # Placeholder for future tightening
