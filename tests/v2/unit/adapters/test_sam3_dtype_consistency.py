"""Tests for SAM3 FP16→FP32 dtype consistency (D1, issue #680).

Verifies:
1. All SAM3 adapters cast encoder FP16 output to FP32 before trainable modules
2. No duplicate code blocks in sam3_backbone.py
3. Backbone extract_features NaN guard is not duplicated
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


class TestAdapterFloatCasts:
    """All SAM3 adapters must cast FP16 encoder output to FP32."""

    @pytest.mark.parametrize(
        "adapter_file",
        ["sam3_vanilla.py", "sam3_hybrid.py", "sam3_topolora.py"],
    )
    def test_adapter_has_float_cast(self, adapter_file: str) -> None:
        """Each adapter must call .float() on encoder features."""
        src = Path(f"src/minivess/adapters/{adapter_file}")
        content = src.read_text(encoding="utf-8")
        assert ".float()" in content, (
            f"{adapter_file} must cast FP16 encoder output to FP32 via .float()"
        )

    def test_backbone_loads_fp16(self) -> None:
        """sam3_backbone.py must load encoder with torch_dtype=torch.float16."""
        src = Path("src/minivess/adapters/sam3_backbone.py")
        content = src.read_text(encoding="utf-8")
        assert "torch.float16" in content or "float16" in content, (
            "sam3_backbone.py must load encoder in FP16 for VRAM efficiency"
        )

    def test_backbone_has_half_cast(self) -> None:
        """sam3_backbone.py must cast input to half() for frozen encoder."""
        src = Path("src/minivess/adapters/sam3_backbone.py")
        content = src.read_text(encoding="utf-8")
        assert ".half()" in content, (
            "sam3_backbone.py must cast input to FP16 to match frozen encoder weights"
        )


class TestNaNGuardNotDuplicated:
    """NaN guard in extract_features must appear exactly once."""

    def test_single_nan_guard_block(self) -> None:
        """extract_features should have exactly ONE nan_to_num call."""
        src = Path("src/minivess/adapters/sam3_backbone.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "extract_features":
                # Count nan_to_num calls in the function body
                nan_to_num_count = 0
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "nan_to_num"
                    ):
                        nan_to_num_count += 1
                assert nan_to_num_count == 1, (
                    f"extract_features should have exactly 1 nan_to_num call, "
                    f"found {nan_to_num_count} (duplicate detected)"
                )
                return
        pytest.fail("extract_features not found in sam3_backbone.py")

    def test_single_isfinite_check(self) -> None:
        """extract_features should have exactly ONE isfinite check."""
        src = Path("src/minivess/adapters/sam3_backbone.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "extract_features":
                isfinite_count = 0
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "isfinite"
                    ):
                        isfinite_count += 1
                assert isfinite_count == 1, (
                    f"extract_features should have exactly 1 isfinite check, "
                    f"found {isfinite_count}"
                )
                return
        pytest.fail("extract_features not found in sam3_backbone.py")
