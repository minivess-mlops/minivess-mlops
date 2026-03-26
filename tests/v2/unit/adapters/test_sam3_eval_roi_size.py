"""Tests for SAM3 adapter get_eval_roi_size — T8 regression tests.

Bug: Sam3TopoLoraAdapter and Sam3HybridAdapter do NOT override get_eval_roi_size(),
falling back to base class default (128, 128, 16). This creates ~3300 validation
windows instead of ~27, making validation take ~6 hours instead of ~4 minutes.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[4] / "src" / "minivess" / "adapters"


class TestSam3EvalRoiSize:
    """T8: All SAM3 variants must override get_eval_roi_size → (512, 512, 3)."""

    def test_topolora_eval_roi_size_is_512_512_3(self):
        """Sam3TopoLoraAdapter must have get_eval_roi_size returning (512, 512, 3)."""
        source = (SRC_DIR / "sam3_topolora.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Sam3TopoLoraAdapter":
                method_names = [
                    n.name for n in node.body if isinstance(n, ast.FunctionDef)
                ]
                assert "get_eval_roi_size" in method_names, (
                    "Sam3TopoLoraAdapter missing get_eval_roi_size override — "
                    "validation will use default ROI creating ~3300 windows"
                )
                return
        pytest.fail("Sam3TopoLoraAdapter class not found")

    def test_hybrid_eval_roi_size_is_512_512_3(self):
        """Sam3HybridAdapter must have get_eval_roi_size returning (512, 512, 3)."""
        source = (SRC_DIR / "sam3_hybrid.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Sam3HybridAdapter":
                method_names = [
                    n.name for n in node.body if isinstance(n, ast.FunctionDef)
                ]
                assert "get_eval_roi_size" in method_names, (
                    "Sam3HybridAdapter missing get_eval_roi_size override — "
                    "validation will use default ROI creating ~3300 windows"
                )
                return
        pytest.fail("Sam3HybridAdapter class not found")

    def test_all_sam3_variants_have_eval_roi_override(self):
        """All SAM3 adapter files must have get_eval_roi_size in their adapter class."""
        sam3_files = {
            "sam3_vanilla.py": "Sam3VanillaAdapter",
            "sam3_topolora.py": "Sam3TopoLoraAdapter",
            "sam3_hybrid.py": "Sam3HybridAdapter",
        }
        for filename, class_name in sam3_files.items():
            source = (SRC_DIR / filename).read_text(encoding="utf-8")
            tree = ast.parse(source)
            found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    method_names = [
                        n.name for n in node.body if isinstance(n, ast.FunctionDef)
                    ]
                    assert "get_eval_roi_size" in method_names, (
                        f"{class_name} in {filename} missing get_eval_roi_size override"
                    )
                    found = True
                    break
            assert found, f"{class_name} not found in {filename}"

    def test_eval_roi_returns_512_512_3_in_vanilla(self):
        """Reference: sam3_vanilla returns (512, 512, 3)."""
        source = (SRC_DIR / "sam3_vanilla.py").read_text(encoding="utf-8")
        assert "(512, 512, 3)" in source, (
            "sam3_vanilla.py should return (512, 512, 3) from get_eval_roi_size"
        )
