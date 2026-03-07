"""Anti-stub enforcement tests for SAM3 adapters.

These tests verify that _StubSam3Encoder and all use_stub paths have been
permanently removed. They use AST analysis (no imports of stub classes) and
inspect.signature to verify the public API.

If any of these tests fail, it means stub code crept back in. The stub caused
the 2026-03-02 SAM3 fuckup (training ran to completion on random weights with
no warning). See .claude/metalearning/2026-03-02-sam3-implementation-fuckup.md
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Paths to source files — tested via AST to avoid importing stub classes
_SRC = Path(__file__).parent.parent.parent.parent / "src" / "minivess"
_BACKBONE_SRC = _SRC / "adapters" / "sam3_backbone.py"
_DECODER_SRC = _SRC / "adapters" / "sam3_decoder.py"
_VANILLA_SRC = _SRC / "adapters" / "sam3_vanilla.py"
_TOPOLORA_SRC = _SRC / "adapters" / "sam3_topolora.py"
_HYBRID_SRC = _SRC / "adapters" / "sam3_hybrid.py"
_BUILDER_SRC = _SRC / "adapters" / "model_builder.py"


def _class_names_in_file(path: Path) -> set[str]:
    """Return all class names defined at module level in a Python source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def _function_names_in_file(path: Path) -> set[str]:
    """Return all function/method names defined at module level in a Python source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _param_names_of_init(path: Path, class_name: str) -> set[str]:
    """Return parameter names of __init__ for a given class, via AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return {arg.arg for arg in item.args.args}
    return set()


class TestNoStubClasses:
    """Verify that all stub classes have been permanently removed."""

    def test_stub_encoder_class_does_not_exist(self) -> None:
        """_StubSam3Encoder must not exist in sam3_backbone.py."""
        classes = _class_names_in_file(_BACKBONE_SRC)
        assert "_StubSam3Encoder" not in classes, (
            "_StubSam3Encoder still exists in sam3_backbone.py — "
            "remove it to prevent silent random-weight training"
        )

    def test_stub_mlp_class_does_not_exist(self) -> None:
        """_StubMLP must not exist in sam3_backbone.py."""
        classes = _class_names_in_file(_BACKBONE_SRC)
        assert "_StubMLP" not in classes, "_StubMLP still exists in sam3_backbone.py"

    def test_stub_fpn_neck_class_does_not_exist(self) -> None:
        """_StubFPNNeck must not exist in sam3_backbone.py."""
        classes = _class_names_in_file(_BACKBONE_SRC)
        assert "_StubFPNNeck" not in classes, (
            "_StubFPNNeck still exists in sam3_backbone.py"
        )

    def test_stub_decoder_class_does_not_exist(self) -> None:
        """_StubSam3Decoder must not exist in sam3_decoder.py."""
        classes = _class_names_in_file(_DECODER_SRC)
        assert "_StubSam3Decoder" not in classes, (
            "_StubSam3Decoder still exists in sam3_decoder.py — "
            "remove it to prevent silent random-weight training"
        )

    def test_auto_stub_function_does_not_exist(self) -> None:
        """_auto_stub_sam3() must not exist in model_builder.py."""
        funcs = _function_names_in_file(_BUILDER_SRC)
        assert "_auto_stub_sam3" not in funcs, (
            "_auto_stub_sam3 still exists in model_builder.py — "
            "remove the pretrained:false escape hatch"
        )


class TestNoUseStubParam:
    """Verify that use_stub parameter has been removed from all adapter __init__ methods."""

    def test_sam3_backbone_has_no_use_stub_param(self) -> None:
        """Sam3Backbone.__init__ must not have a use_stub parameter."""
        params = _param_names_of_init(_BACKBONE_SRC, "Sam3Backbone")
        assert "use_stub" not in params, (
            "Sam3Backbone.__init__ still has use_stub param — remove it"
        )

    def test_sam3_vanilla_has_no_use_stub_param(self) -> None:
        """Sam3VanillaAdapter.__init__ must not have a use_stub parameter."""
        params = _param_names_of_init(_VANILLA_SRC, "Sam3VanillaAdapter")
        assert "use_stub" not in params, (
            "Sam3VanillaAdapter.__init__ still has use_stub param — remove it"
        )

    def test_sam3_topolora_has_no_use_stub_param(self) -> None:
        """Sam3TopoLoraAdapter.__init__ must not have a use_stub parameter."""
        params = _param_names_of_init(_TOPOLORA_SRC, "Sam3TopoLoraAdapter")
        assert "use_stub" not in params, (
            "Sam3TopoLoraAdapter.__init__ still has use_stub param — remove it"
        )

    def test_sam3_hybrid_has_no_use_stub_param(self) -> None:
        """Sam3HybridAdapter.__init__ must not have a use_stub parameter."""
        params = _param_names_of_init(_HYBRID_SRC, "Sam3HybridAdapter")
        assert "use_stub" not in params, (
            "Sam3HybridAdapter.__init__ still has use_stub param — remove it"
        )


class TestBuildAdapterRaisesWithoutSam3:
    """Verify build_adapter() raises RuntimeError when SAM3 not installed."""

    def test_build_adapter_raises_without_sam3_installed(self) -> None:
        """build_adapter() must raise RuntimeError when SAM3 is not installed.

        Mocks _sam3_package_available() to return False, then attempts to
        build a SAM3 adapter. Must raise RuntimeError with install instructions.
        """
        from unittest.mock import patch

        from minivess.adapters.model_builder import build_adapter
        from minivess.config.models import ModelConfig, ModelFamily

        config = ModelConfig(
            family=ModelFamily.SAM3_VANILLA,
            name="test",
            in_channels=1,
            out_channels=2,
        )

        with (
            patch(
                "minivess.adapters.model_builder._sam3_package_available",
                return_value=False,
            ),
            pytest.raises(RuntimeError, match="SAM3"),
        ):
            build_adapter(config)
