"""T-1.2: Enforce no cross-flow imports between flow files.

Each Prefect flow runs in its own Docker container. Importing one flow's
module from another creates a hidden coupling that breaks container isolation.
Flows must communicate ONLY via MLflow artifacts and Prefect run_deployment().

Allowed imports within the flows/ package:
  - dashboard_flow.py → dashboard_sections.py (helper module, not a flow)

Uses ast.parse() for source inspection — NO regex (CLAUDE.md Rule #16).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_FLOWS_DIR = Path("src/minivess/orchestration/flows")

# Helper modules that are NOT independent flows (OK to import)
_HELPER_MODULES = {"dashboard_sections"}

# Known violations with tracking issues (skip until fixed)
# T-1.1 FIXED: hpo_flow no longer imports train_flow (uses run_deployment)
_KNOWN_VIOLATIONS: set[tuple[str, str]] = set()

# All flow module names (files that define @flow functions)
_FLOW_MODULES = {
    "acquisition_flow",
    "analysis_flow",
    "annotation_flow",
    "dashboard_flow",
    "data_flow",
    "deploy_flow",
    "hpo_flow",
    "post_training_flow",
    "train_flow",
}


def _get_flow_imports(flow_file: Path) -> list[tuple[str, str]]:
    """Return list of (importer_module, imported_flow_module) for cross-flow imports."""
    source = flow_file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    importer = flow_file.stem
    violations = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is None:
            continue
        # Check if importing from another flow module
        parts = node.module.split(".")
        if "orchestration" not in parts or "flows" not in parts:
            continue
        # Extract the target module name (last part after "flows")
        flows_idx = parts.index("flows")
        if flows_idx + 1 >= len(parts):
            continue
        target = parts[flows_idx + 1]
        if target == importer:
            continue  # Self-import is fine
        if target in _HELPER_MODULES:
            continue  # Helper modules are OK
        violations.append((importer, target))

    return violations


class TestNoCrossFlowImports:
    """Verify that flow files do not import from other flow files."""

    def test_no_cross_flow_imports(self) -> None:
        """Flow files must not import from other flow modules."""
        if not _FLOWS_DIR.is_dir():
            pytest.skip("flows/ directory not found")

        all_violations = []
        for flow_file in sorted(_FLOWS_DIR.glob("*_flow.py")):
            violations = _get_flow_imports(flow_file)
            for pair in violations:
                if pair not in _KNOWN_VIOLATIONS:
                    all_violations.append(pair)

        assert not all_violations, (
            f"Cross-flow imports violate Docker isolation: {all_violations}. "
            "Flows must communicate via MLflow artifacts and run_deployment()."
        )

    def test_no_known_violations_remain(self) -> None:
        """All known violations have been fixed — list must be empty."""
        assert not _KNOWN_VIOLATIONS, (
            f"Known violations should be empty after T-1.1 fix: {_KNOWN_VIOLATIONS}"
        )
