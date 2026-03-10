"""Tests for @flow decorator name standardization (T-06, closes #416).

All Prefect @flow(name=...) values must be lowercase-hyphen and match
the FLOW_NAME_* constants from orchestration.constants.
"""

from __future__ import annotations

import ast
from pathlib import Path

FLOWS_DIR = Path("src/minivess/orchestration/flows")
# All flow files are in flows/ directory (deploy_flow moved from root in T-0.2)
FLOW_FILES = list(FLOWS_DIR.glob("*.py"))

# deployments.py at the orchestration level
DEPLOYMENTS_FILE = Path("src/minivess/orchestration/deployments.py")


def _get_flow_decorator_names(source_path: Path) -> list[str]:
    """Return all @flow(name=...) string values in a Python file using AST."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        # Look for decorated function definitions
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            # @flow(name="...") or @flow(name=CONSTANT)
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            # Accept both @flow(...) and @something.flow(...)
            func_name = (
                func.id
                if isinstance(func, ast.Name)
                else (func.attr if isinstance(func, ast.Attribute) else "")
            )
            if func_name != "flow":
                continue
            for kw in decorator.keywords:
                if kw.arg != "name":
                    continue
                if isinstance(kw.value, ast.Constant):
                    val = kw.value.value
                    if isinstance(val, str):
                        names.append(val)
                elif isinstance(kw.value, ast.Name):
                    # name=FLOW_NAME_TRAIN — resolve from constants
                    from minivess.orchestration import constants as _c

                    val = getattr(_c, kw.value.id, None)
                    if isinstance(val, str):
                        names.append(val)
    return names


class TestFlowDecoratorNames:
    def test_all_flow_names_are_lowercase_hyphen(self) -> None:
        """Every @flow(name=...) literal must be lowercase-hyphen (no spaces, no underscores)."""
        violations: list[str] = []
        for fpath in FLOW_FILES:
            if fpath.name.startswith("_"):
                continue
            names = _get_flow_decorator_names(fpath)
            for name in names:
                is_lower = name == name.lower()
                has_no_space = " " not in name
                has_no_underscore = "_" not in name
                if not (is_lower and has_no_space and has_no_underscore):
                    violations.append(f"{fpath.name}: {name!r}")
        assert not violations, (
            "Found @flow names that are not lowercase-hyphen:\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_constants_define_all_core_flow_names(self) -> None:
        """FLOW_NAME_* constants must cover the core flow names."""
        from minivess.orchestration.constants import (
            FLOW_NAME_ANALYSIS,
            FLOW_NAME_DASHBOARD,
            FLOW_NAME_DATA,
            FLOW_NAME_DEPLOY,
            FLOW_NAME_HPO,
            FLOW_NAME_POST_TRAINING,
            FLOW_NAME_TRAIN,
        )

        all_names_in_files: set[str] = set()
        for fpath in FLOW_FILES:
            if fpath.name.startswith("_"):
                continue
            all_names_in_files.update(_get_flow_decorator_names(fpath))

        # Core flow names must be present somewhere in the flow files
        core_constants = {
            FLOW_NAME_TRAIN,
            FLOW_NAME_DATA,
            FLOW_NAME_POST_TRAINING,
            FLOW_NAME_ANALYSIS,
            FLOW_NAME_DEPLOY,
            FLOW_NAME_DASHBOARD,
            FLOW_NAME_HPO,
        }
        missing = core_constants - all_names_in_files
        assert not missing, (
            f"Core FLOW_NAME_* constants not found as @flow names: {missing}"
        )
