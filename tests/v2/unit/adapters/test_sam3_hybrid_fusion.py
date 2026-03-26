"""Tests for Sam3HybridAdapter fusion — T18 regression test.

Bug: Sam3HybridAdapter.forward() manually reimplemented the fusion logic
instead of calling self.fusion(logits, sam_features). The class's forward()
method was dead code.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC_FILE = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "minivess"
    / "adapters"
    / "sam3_hybrid.py"
)


class TestGatedFusionForwardCalled:
    """T18: Sam3HybridAdapter.forward() must call self.fusion()."""

    def test_gated_fusion_forward_called(self):
        """forward() must call self.fusion(...) not inline reimplementation."""
        source = SRC_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name != "Sam3HybridAdapter":
                continue
            for method in node.body:
                if not isinstance(method, ast.FunctionDef):
                    continue
                if method.name != "forward":
                    continue
                # Check for self.fusion(...) call
                fusion_called = False
                for child in ast.walk(method):
                    if not isinstance(child, ast.Call):
                        continue
                    if not isinstance(child.func, ast.Attribute):
                        continue
                    if child.func.attr == "fusion" and (
                        (
                            isinstance(child.func.value, ast.Attribute)
                            and child.func.value.attr == "fusion"
                        )
                        or (
                            isinstance(child.func.value, ast.Name)
                            and child.func.value.id == "self"
                        )
                    ):
                        fusion_called = True
                assert fusion_called, (
                    "Sam3HybridAdapter.forward() must call self.fusion() "
                    "instead of inline reimplementation"
                )
                return
        pytest.fail("Sam3HybridAdapter.forward not found")

    def test_fusion_output_shape_correct(self):
        """GatedFeatureFusion should not appear as inline code in forward."""
        source = SRC_FILE.read_text(encoding="utf-8")
        # After fix, there should be no inline `self.fusion.proj` call in forward
        # The fusion module handles projection internally
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name != "Sam3HybridAdapter":
                continue
            for method in node.body:
                if not isinstance(method, ast.FunctionDef):
                    continue
                if method.name != "forward":
                    continue
                method_source = ast.get_source_segment(source, method) or ""
                assert "self.fusion.proj" not in method_source, (
                    "forward() should not directly call self.fusion.proj — "
                    "use self.fusion(logits, sam_features) instead"
                )
                return
