"""Meta-test: Verify ALL flows have observability wired in.

Phase 11: AST-enforced contract that prevents regression.
Scans all flow files for required observability patterns.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_FLOWS_DIR = Path("src/minivess/orchestration/flows")

# GPU flows MUST have require_cuda_context()
_GPU_FLOWS = [
    "train_flow.py",
    "hpo_flow.py",
    "post_training_flow.py",
    "analysis_flow.py",
]


def _has_flow_decorator(path: Path) -> bool:
    """Check if a Python file contains @flow decorator."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name) and dec.id == "flow":
                        return True
                    if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "flow":
                        return True
        return False
    except (SyntaxError, OSError):
        return False


def _file_imports(path: Path, name: str) -> bool:
    """Check if a file imports a given name."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.names and any(alias.name == name for alias in node.names):
                return True
    return False


def _file_calls(path: Path, name: str) -> bool:
    """Check if a file calls a given function name."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == name:
                return True
    return False


def _file_contains_string(path: Path, text: str) -> bool:
    """Check if file contains a text string (not AST, for context manager usage)."""
    return text in path.read_text(encoding="utf-8")


class TestGpuFlowsCudaGuard:
    """ALL 4 GPU flows must import and call require_cuda_context()."""

    @pytest.mark.parametrize("flow_file", _GPU_FLOWS)
    def test_imports_cuda_guard(self, flow_file: str) -> None:
        path = _FLOWS_DIR / flow_file
        assert path.exists(), f"GPU flow file not found: {flow_file}"
        has_import = (
            _file_imports(path, "require_cuda_context")
            or _file_contains_string(path, "gpu_flow_observability_context")
        )
        assert has_import, (
            f"{flow_file} MUST import require_cuda_context or "
            f"gpu_flow_observability_context"
        )

    @pytest.mark.parametrize("flow_file", _GPU_FLOWS)
    def test_calls_cuda_guard(self, flow_file: str) -> None:
        path = _FLOWS_DIR / flow_file
        has_call = (
            _file_calls(path, "require_cuda_context")
            or _file_contains_string(path, "gpu_flow_observability_context")
        )
        assert has_call, (
            f"{flow_file} MUST call require_cuda_context() or use "
            f"gpu_flow_observability_context()"
        )
