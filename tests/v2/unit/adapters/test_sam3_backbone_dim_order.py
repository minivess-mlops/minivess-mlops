"""Tests for Sam3Backbone.get_volume_embeddings dimension order — T4 regression test.

Bug: get_volume_embeddings() unpacks as (B, C, D, H, W) but MONAI convention
is (B, C, H, W, D) — depth LAST. The wrong order causes incorrect spatial slicing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[4] / "src"


class TestGetVolumeEmbeddingsDimOrder:
    """T4: get_volume_embeddings must use MONAI dimension order (B, C, H, W, D)."""

    def test_get_volume_embeddings_monai_dim_order(self):
        """The unpacking must be b, c, h, w, d (depth last), not b, c, d, h, w."""
        backbone_file = SRC_DIR / "minivess" / "adapters" / "sam3_backbone.py"
        source = backbone_file.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "get_volume_embeddings":
                continue
            # Find the tuple unpack assignment: b, c, h, w, d = volume.shape
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Tuple):
                            names = [
                                elt.id
                                for elt in target.elts
                                if isinstance(elt, ast.Name)
                            ]
                            if len(names) == 5 and "volume" in ast.dump(child.value):
                                assert names == ["b", "c", "h", "w", "d"], (
                                    f"get_volume_embeddings unpacks as {names}, "
                                    f"expected ['b', 'c', 'h', 'w', 'd'] (MONAI depth-last)"
                                )
                                return
            pytest.fail("No 5-element tuple unpack found in get_volume_embeddings")

    def test_get_volume_embeddings_output_shape(self):
        """Docstring should document output as (B, embed_dim, D, H_feat, W_feat)."""
        backbone_file = SRC_DIR / "minivess" / "adapters" / "sam3_backbone.py"
        source = backbone_file.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_volume_embeddings":
                docstring = ast.get_docstring(node) or ""
                # Input should be documented as (B, C, H, W, D) not (B, C, D, H, W)
                assert "(B, C, H, W, D)" in docstring, (
                    "get_volume_embeddings docstring should document input as "
                    "(B, C, H, W, D) — MONAI depth-last convention"
                )
                return
        pytest.fail("get_volume_embeddings function not found")

    def test_slice_iteration_uses_depth_dim(self):
        """for z_idx in range(d) must slice volume[:, :, :, :, z_idx] (depth last)."""
        backbone_file = SRC_DIR / "minivess" / "adapters" / "sam3_backbone.py"
        source = backbone_file.read_text(encoding="utf-8")

        # After fix: slicing should use z_idx as the 5th index (depth last)
        # volume[:, :, :, :, z_idx] = correct MONAI order
        assert "volume[:, :, :, :, z_idx]" in source, (
            "get_volume_embeddings should slice as volume[:, :, :, :, z_idx] "
            "(depth as last dimension, MONAI convention)"
        )
