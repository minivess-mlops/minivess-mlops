"""Guard test: No hardcoded significance levels in biostatistics code.

Ensures alpha values always come from BiostatisticsConfig, never hardcoded.
Issue #881: Hardcoded alpha=0.05 violates single-source-of-truth.

Metalearning: .claude/metalearning/2026-03-20-hardcoded-significance-level-antipattern.md
"""

from __future__ import annotations

import ast
from pathlib import Path

# Directories to scan for hardcoded alpha
_BIOSTAT_TEST_DIRS = [
    Path("tests/v2/unit/biostatistics"),
    Path("tests/v2/unit/test_biostatistics_statistics.py"),
    Path("tests/v2/unit/test_factorial_anova.py"),
    Path("tests/v2/integration/test_biostatistics_factorial_integration.py"),
]

# Allowlist: files that legitimately contain 0.05 in non-alpha contexts
_ALLOWLIST = {
    # This file itself uses 0.05 in docstrings/comments
    "test_no_hardcoded_alpha.py",
}


class TestNoHardcodedAlpha:
    """No hardcoded 0.05 significance level in biostatistics tests."""

    def test_no_hardcoded_alpha_in_biostatistics_tests(self) -> None:
        """Scan biostatistics test files for hardcoded alpha=0.05.

        Checks for patterns like:
        - `< 0.05` or `> 0.05` in assertions
        - `alpha=0.05` in function calls (should use cfg.alpha)

        Allowed: `_CFG.alpha` or `BiostatisticsConfig().alpha` references.
        """
        violations: list[str] = []

        for path_spec in _BIOSTAT_TEST_DIRS:
            if path_spec.is_file():
                files = [path_spec]
            elif path_spec.is_dir():
                files = list(path_spec.glob("*.py"))
            else:
                continue

            for py_file in files:
                if py_file.name in _ALLOWLIST:
                    continue

                source = py_file.read_text(encoding="utf-8")

                # Parse AST to find numeric literals == 0.05
                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    continue

                for node in ast.walk(tree):
                    # Look for keyword arguments named 'alpha' with value 0.05
                    if (
                        isinstance(node, ast.keyword)
                        and node.arg == "alpha"
                        and isinstance(node.value, ast.Constant)
                        and node.value.value == 0.05
                    ):
                        violations.append(
                            f"{py_file.name}:{node.value.lineno} — "
                            f"hardcoded alpha=0.05. "
                            f"Use BiostatisticsConfig().alpha"
                        )
                    # Look for comparisons: p < 0.05 or p > 0.05
                    if isinstance(node, ast.Compare):
                        for comparator in node.comparators:
                            if (
                                isinstance(comparator, ast.Constant)
                                and comparator.value == 0.05
                            ):
                                violations.append(
                                    f"{py_file.name}:{comparator.lineno} — "
                                    f"hardcoded 0.05 in comparison. "
                                    f"Use BiostatisticsConfig().alpha"
                                )

        assert not violations, (
            f"Hardcoded alpha=0.05 found in {len(violations)} location(s):\n"
            + "\n".join(f"  - {v}" for v in violations)
            + "\n\nFix: Use BiostatisticsConfig().alpha instead. "
            "See Issue #881."
        )
