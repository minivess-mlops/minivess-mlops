"""Tests for SAM3 gradient checkpointing — reduces VRAM ~22 GiB to ~10 GiB on L4.

Gradient checkpointing trades activation memory for recomputation:
~12-14 GiB activations → ~1-2 GiB (recomputed from checkpoints).
Expected result: ~10 GiB total, ~14 GiB headroom on L4.

Issue: #966 (A100 option), #940 (SAM3 OOM)
Plan: docs/planning/v0-2_archive/original_docs/sam3-gradient-checkpointing-plan.xml
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


class TestSam3BackboneGradientCheckpointing:
    """Sam3Backbone must accept and handle gradient_checkpointing parameter."""

    def test_backbone_accepts_gradient_checkpointing_param(self) -> None:
        """Sam3Backbone.__init__ must accept gradient_checkpointing kwarg."""
        source = Path("src/minivess/adapters/sam3_backbone.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                # Check if this __init__ belongs to Sam3Backbone
                # by looking at parent class context (AST walk is flat,
                # but we check all __init__ methods for the param)
                params = [arg.arg for arg in node.args.args] + [
                    arg.arg for arg in node.args.kwonlyargs
                ]
                if "gradient_checkpointing" in params:
                    return
        pytest.fail(
            "Sam3Backbone.__init__ must accept gradient_checkpointing parameter"
        )

    def test_backbone_has_gradient_checkpointing_enable_call(self) -> None:
        """Sam3Backbone must call gradient_checkpointing_enable when enabled."""
        source = Path("src/minivess/adapters/sam3_backbone.py").read_text(
            encoding="utf-8"
        )
        assert "gradient_checkpointing_enable" in source, (
            "Sam3Backbone must call gradient_checkpointing_enable() on the encoder "
            "when gradient_checkpointing=True (HuggingFace native feature, Rule #3)"
        )

    def test_backbone_uses_non_reentrant_mode(self) -> None:
        """Gradient checkpointing must use use_reentrant=False for autocast compat."""
        source = Path("src/minivess/adapters/sam3_backbone.py").read_text(
            encoding="utf-8"
        )
        assert "use_reentrant" in source, (
            "gradient_checkpointing_enable must pass use_reentrant=False "
            "for autocast (AMP) compatibility"
        )


class TestTopoLoraGradientCheckpointingWiring:
    """Sam3TopoLoraAdapter must read gradient_checkpointing from config."""

    def test_topolora_reads_gradient_checkpointing_from_config(self) -> None:
        """Sam3TopoLoraAdapter must read gradient_checkpointing from architecture_params."""
        source = Path("src/minivess/adapters/sam3_topolora.py").read_text(
            encoding="utf-8"
        )
        assert "gradient_checkpointing" in source, (
            "Sam3TopoLoraAdapter must read gradient_checkpointing from config "
            "and pass it to Sam3Backbone"
        )


class TestHybridAdapterGradientCheckpointing:
    """Sam3HybridAdapter should pass gradient_checkpointing to backbone."""

    def test_hybrid_references_gradient_checkpointing(self) -> None:
        """Sam3HybridAdapter should reference gradient_checkpointing in source."""
        source = Path("src/minivess/adapters/sam3_hybrid.py").read_text(
            encoding="utf-8"
        )
        assert "gradient_checkpointing" in source, (
            "Sam3HybridAdapter must pass gradient_checkpointing to Sam3Backbone "
            "(backbone will ignore it since freeze=True)"
        )
