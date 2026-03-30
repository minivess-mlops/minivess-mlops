"""Tests for FlowContract import deduplication (T-03, closes #405).

TDD RED phase: FlowContract must not be imported inside try-except blocks
in flow files — it should be a top-level module import.
"""

from __future__ import annotations

import ast
from pathlib import Path

FLOW_FILES = [
    Path("src/minivess/orchestration/flows/train_flow.py"),
    Path("src/minivess/orchestration/flows/analysis_flow.py"),
    Path("src/minivess/orchestration/flows/deploy_flow.py"),
    Path("src/minivess/orchestration/flows/data_flow.py"),
]


def _find_try_body_imports(source_path: Path) -> list[str]:
    """Find all imports of FlowContract inside try-except blocks using AST."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        # Check body AND handlers for FlowContract imports
        all_stmts = list(node.body) + [s for h in node.handlers for s in h.body]
        for stmt in all_stmts:
            if not isinstance(stmt, ast.ImportFrom):
                continue
            mod = stmt.module or ""
            names = [alias.name for alias in stmt.names]
            if "FlowContract" in names or "flow_contract" in mod:
                found.append(f"{source_path.name}: import inside try-except")
    return found


class TestFlowContractImportLocation:
    def test_flow_contract_not_imported_in_try_except(self) -> None:
        """FlowContract must be imported at module level, not inside try-except blocks."""
        violations: list[str] = []
        for fpath in FLOW_FILES:
            if not fpath.exists():
                continue
            violations.extend(_find_try_body_imports(fpath))
        assert not violations, (
            "FlowContract imported inside try-except (move to module top-level):\n"
            + "\n".join(f"  {v}" for v in violations)
        )
