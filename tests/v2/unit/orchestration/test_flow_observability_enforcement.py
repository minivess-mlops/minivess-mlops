"""AST enforcement: every @flow body MUST use observability context manager.

This test catches the "import but don't use" anti-pattern from passes 1 and 2.
It verifies that every @flow-decorated function contains a `with` statement
calling either flow_observability_context or gpu_flow_observability_context.

Per CLAUDE.md Rule #16: uses ast.parse(), never regex.
Per CLAUDE.md Rule #34: "Import != Done" — code must be CALLED.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_FLOWS_DIR = Path("src/minivess/orchestration/flows")

# GPU flows must use gpu_flow_observability_context
_GPU_FLOWS = {"train_flow.py", "hpo_flow.py", "post_training_flow.py", "analysis_flow.py"}

# Context manager function names we accept
_ACCEPTED_CONTEXT_MANAGERS = {
    "flow_observability_context",
    "gpu_flow_observability_context",
}


def _get_flow_functions(filepath: Path) -> list[tuple[str, ast.FunctionDef]]:
    """Find all @flow-decorated functions in a file."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source)
    results = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            is_flow = False
            if isinstance(dec, ast.Name) and dec.id == "flow":
                is_flow = True
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "flow":
                is_flow = True
            if is_flow:
                results.append((filepath.name, node))
    return results


def _has_observability_with(func_node: ast.FunctionDef) -> bool:
    """Check if a function body contains a `with` statement using an observability context manager."""
    for node in ast.walk(func_node):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            ctx = item.context_expr
            # Direct call: with flow_observability_context(...)
            if isinstance(ctx, ast.Call):
                if isinstance(ctx.func, ast.Name) and ctx.func.id in _ACCEPTED_CONTEXT_MANAGERS:
                    return True
                if isinstance(ctx.func, ast.Attribute) and ctx.func.attr in _ACCEPTED_CONTEXT_MANAGERS:
                    return True
    return False


def _collect_all_flow_functions() -> list[tuple[str, str]]:
    """Collect (filename, function_name) for all @flow functions across all flow files."""
    results = []
    for filepath in sorted(_FLOWS_DIR.glob("*_flow.py")):
        for filename, func_node in _get_flow_functions(filepath):
            results.append((filename, func_node.name))
    return results


# Subflows that are internal and don't need the top-level observability wrapper
# (they run INSIDE a flow that already has the wrapper)
_SUBFLOW_ALLOWLIST = {
    "training_subflow",
    "post_training_subflow",
}

_ALL_FLOWS = [
    (f, fn) for f, fn in _collect_all_flow_functions()
    if fn not in _SUBFLOW_ALLOWLIST
]


class TestEveryFlowUsesObservabilityContext:
    """EVERY @flow function (except subflows) must wrap its body in an observability context manager."""

    @pytest.mark.parametrize("flow_file,func_name", _ALL_FLOWS, ids=[f"{f}::{fn}" for f, fn in _ALL_FLOWS])
    def test_flow_body_has_observability_with(self, flow_file: str, func_name: str) -> None:
        filepath = _FLOWS_DIR / flow_file
        for _, func_node in _get_flow_functions(filepath):
            if func_node.name == func_name:
                has_obs = _has_observability_with(func_node)
                expected_cm = "gpu_flow_observability_context" if flow_file in _GPU_FLOWS else "flow_observability_context"
                assert has_obs, (
                    f"{flow_file}::{func_name} is decorated with @flow but does NOT wrap its body in "
                    f"`with {expected_cm}(...)`. This is dead code — the import exists but the "
                    f"context manager is never invoked. (Rule #34: Import != Done)"
                )
