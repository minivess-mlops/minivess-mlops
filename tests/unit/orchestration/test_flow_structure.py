"""T-7.1: Structural enforcement tests for Prefect flow architecture.

Uses ast.parse() for source inspection — NO regex (CLAUDE.md Rule #16).
All tests run in < 2 seconds (AST parsing is fast).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_FLOWS_DIR = Path("src/minivess/orchestration/flows")
_ORCH_DIR = Path("src/minivess/orchestration")


def _get_flow_files() -> list[Path]:
    """Return all *_flow.py files in the flows directory."""
    if not _FLOWS_DIR.is_dir():
        pytest.skip("flows/ directory not found")
    return sorted(_FLOWS_DIR.glob("*_flow.py"))


class TestFlowStructure:
    """Structural invariants for all Prefect flow files."""

    def test_all_flows_in_flows_directory(self) -> None:
        """No *_flow.py files should exist in orchestration/ root."""
        root_flows = list(_ORCH_DIR.glob("*_flow.py"))
        assert not root_flows, (
            f"Flow files must be in orchestration/flows/, not root: "
            f"{[f.name for f in root_flows]}"
        )

    def test_all_flows_have_main_block(self) -> None:
        """Every flow file must have an `if __name__ == '__main__':` block."""
        missing = []
        for flow_file in _get_flow_files():
            source = flow_file.read_text(encoding="utf-8")
            if "__name__" not in source or "__main__" not in source:
                missing.append(flow_file.name)
        assert not missing, (
            f"Flow files missing __main__ block: {missing}. "
            "Dockerfiles use CMD ['python', '-m', ...] which requires __main__."
        )

    def test_all_flows_have_docker_gate(self) -> None:
        """Every flow file must check Docker context at flow entry."""
        missing = []
        for flow_file in _get_flow_files():
            source = flow_file.read_text(encoding="utf-8")
            # Check for shared require_docker_context or inline gate pattern
            has_gate = (
                "require_docker_context" in source or "MINIVESS_ALLOW_HOST" in source
            )
            if not has_gate:
                missing.append(flow_file.name)
        assert not missing, (
            f"Flow files missing Docker context gate: {missing}. "
            "All flows must validate Docker context at entry."
        )

    # qa_flow is a legacy module — QA merged into dashboard health adapter (#342, PR #567).
    _LEGACY_FLOW_MODULES = {"qa_flow.py"}

    def test_all_flows_use_constants(self) -> None:
        """Every flow file must import flow name from constants module."""
        missing = []
        for flow_file in _get_flow_files():
            if flow_file.name in self._LEGACY_FLOW_MODULES:
                continue
            source = flow_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
            uses_constant = False
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and "constants" in node.module
                    and node.names
                ):
                    for alias in node.names:
                        if alias.name.startswith("FLOW_NAME_"):
                            uses_constant = True
                            break
            if not uses_constant:
                missing.append(flow_file.name)
        assert not missing, (
            f"Flow files not importing FLOW_NAME_* from constants: {missing}. "
            "All @flow(name=...) must use constants, not hardcoded strings."
        )

    def test_no_compat_layer_imports(self) -> None:
        """No source file should import from _prefect_compat (deleted)."""
        violations = []
        for py_file in Path("src").rglob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            if "_prefect_compat" in source:
                violations.append(str(py_file))
        assert not violations, (
            f"Files still importing _prefect_compat (deleted): {violations}"
        )

    def test_no_prefect_disabled_in_src(self) -> None:
        """PREFECT_DISABLED should only appear in test files, not src/."""
        violations = []
        for py_file in Path("src").rglob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            if "PREFECT_DISABLED" in source:
                violations.append(str(py_file))
        assert not violations, (
            f"PREFECT_DISABLED found in source files (should be tests only): "
            f"{violations}"
        )

    def test_all_flows_import_from_prefect(self) -> None:
        """Every flow file must import directly from prefect, not compat."""
        missing = []
        for flow_file in _get_flow_files():
            source = flow_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
            imports_prefect = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module == "prefect":
                        imports_prefect = True
                        break
                    if node.module and node.module.startswith("prefect."):
                        imports_prefect = True
                        break
            if not imports_prefect:
                missing.append(flow_file.name)
        assert not missing, (
            f"Flow files not importing from prefect: {missing}. "
            "All flows must import directly from prefect."
        )
