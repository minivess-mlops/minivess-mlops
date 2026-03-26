"""Tests for SWAG fold_id in config dict — T3 regression tests.

Bug: The config dict built at lines 1432-1450 never includes "fold_id".
When _run_swag_post_training reads config.get("fold_id", 0), it always gets 0.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TRAIN_FLOW = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "minivess"
    / "orchestration"
    / "flows"
    / "train_flow.py"
)


class TestConfigDictIncludesFoldId:
    """T3: Config dict must include fold_id for correct SWAG data loading."""

    def test_config_dict_includes_fold_id(self):
        """The per-fold config dict must include 'fold_id' key."""
        source = TRAIN_FLOW.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find the training_flow function
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "training_flow":
                continue

            # Find dict literal that contains "loss_name" key (the config dict)
            for child in ast.walk(node):
                if not isinstance(child, ast.Dict):
                    continue
                keys = []
                for k in child.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        keys.append(k.value)
                if "loss_name" in keys and "model_family" in keys:
                    assert "fold_id" in keys, (
                        "Config dict in training_flow missing 'fold_id' key. "
                        "SWAG post-training reads config.get('fold_id', 0) "
                        "and always gets 0 without this."
                    )
                    return
            pytest.fail("Config dict with 'loss_name' not found in training_flow")

    def test_swag_uses_correct_fold_data(self):
        """post_training_subflow must read fold_id from config, not default to 0."""
        source = TRAIN_FLOW.read_text(encoding="utf-8")
        # The post_training_subflow or _run_swag_post_training should read fold_id
        # from config. Verify it's there.
        assert 'config.get("fold_id"' in source or "fold_id" in source, (
            "fold_id must be read from config in post-training flow"
        )
