"""Guard test: flow files must not hardcode metric key strings.

All metric key strings (train/loss, val/dice, eval/fold0/dice, etc.)
must be imported from MetricKeys constants, not hardcoded as string
literals. This prevents drift between logging and querying code.

Task 3.4 from 8th pass backlog fix plan.
Issue #790.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FLOW_DIR = REPO_ROOT / "src" / "minivess" / "orchestration" / "flows"

# Metric prefixes that should come from MetricKeys, not hardcoded strings
BANNED_PREFIXES = (
    "train/",
    "val/",
    "eval/",
    "test/",
    "optim/",
    "grad/",
    "gpu/",
    "prof/",
    "cost/",
    "fold/",
    "vram/",
    "infer/",
    "checkpoint/",
)

# Allowed exceptions: format strings, f-strings, and MetricKeys class itself
ALLOWED_FILES = {
    "metric_keys.py",  # The source of truth itself
}


def _find_hardcoded_metric_strings(filepath: Path) -> list[tuple[int, str]]:
    """Find string literals that look like hardcoded metric keys.

    Uses AST to find Constant nodes containing metric key patterns.
    Skips format strings and comments.
    """
    source = filepath.read_text(encoding="utf-8")
    violations: list[tuple[int, str]] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            # Check if this string starts with a banned metric prefix
            if any(val.startswith(prefix) for prefix in BANNED_PREFIXES):
                # Skip if it's in a comment or docstring
                # (AST Constant in Expr is a docstring)
                violations.append((node.lineno, val))

    return violations


def _flow_files() -> list[Path]:
    """All Python files in the flows directory."""
    return sorted(FLOW_DIR.glob("*.py"))


class TestNoHardcodedMetricKeysInFlows:
    """Flow files should use MetricKeys constants, not hardcoded strings."""

    @pytest.mark.parametrize("flow_path", _flow_files(), ids=lambda p: p.stem)
    def test_flow_uses_metric_keys_constants(self, flow_path: Path) -> None:
        """Check that flow file doesn't hardcode metric key strings.

        NOTE: This test is ASPIRATIONAL — it flags violations but may need
        allowlisting for legitimate uses (e.g., log messages, comments in code).
        The goal is to drive adoption of MetricKeys, not 100% enforcement yet.
        """
        if flow_path.name in ALLOWED_FILES:
            return

        violations = _find_hardcoded_metric_strings(flow_path)

        # Filter out docstrings and log messages (common legitimate uses)
        real_violations = []
        source_lines = flow_path.read_text(encoding="utf-8").splitlines()
        for lineno, val in violations:
            if lineno <= len(source_lines):
                line = source_lines[lineno - 1].strip()
                # Skip if in a log message, comment, or docstring
                if any(
                    line.startswith(prefix)
                    for prefix in ("logger.", "log.", "#", '"""', "'''", "print(")
                ):
                    continue
                # Skip if it's an f-string metric key construction (dynamic)
                if "f'" in line or 'f"' in line:
                    continue
                real_violations.append((lineno, val, line))

        # Strict enforcement: all flows must use MetricKeys constants.
        # Task 2.1 replaced all hardcoded strings; any new violations are regressions.
        if real_violations:
            msg = "\n".join(
                f"  L{lineno}: {val!r} in {line}"
                for lineno, val, line in real_violations
            )
            pytest.fail(
                f"{flow_path.name} has {len(real_violations)} hardcoded metric keys "
                f"(use MetricKeys constants instead):\n{msg}"
            )
