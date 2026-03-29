"""Test that train_flow.py imports and calls require_cuda_context.

Phase 1, Task 1.2: Verify the CUDA guard is wired into the training flow.
Uses AST inspection to verify the import and call exist without running the flow.
"""

from __future__ import annotations

import ast
from pathlib import Path


_TRAIN_FLOW = Path("src/minivess/orchestration/flows/train_flow.py")


class TestTrainFlowImportsCudaGuard:
    """train_flow.py must import require_cuda_context."""

    def test_imports_require_cuda_context(self) -> None:
        source = _TRAIN_FLOW.read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "cuda_guard" in node.module:
                    for alias in node.names:
                        if alias.name == "require_cuda_context":
                            imported = True
        assert imported, "train_flow.py must import require_cuda_context from cuda_guard"


class TestTrainFlowCallsCudaGuard:
    """train_flow.py must call require_cuda_context() in the flow body."""

    def test_calls_require_cuda_context(self) -> None:
        source = _TRAIN_FLOW.read_text(encoding="utf-8")
        tree = ast.parse(source)

        called = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "require_cuda_context":
                    called = True
                elif isinstance(node.func, ast.Attribute) and node.func.attr == "require_cuda_context":
                    called = True
        assert called, "train_flow.py must call require_cuda_context()"
