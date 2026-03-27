"""Tests for POST_TRAINING_METHODS canonical env var name (Rule 26).

Verifies that the legacy singular POST_TRAINING_METHOD env var is
removed and only the canonical plural POST_TRAINING_METHODS exists.
"""

from __future__ import annotations

import ast
from pathlib import Path


TRAIN_FLOW_PATH = Path("src/minivess/orchestration/flows/train_flow.py")


class TestPostTrainingMethodCanonical:
    """Only POST_TRAINING_METHODS (plural) should exist — no legacy fallback."""

    def test_no_legacy_post_training_method_singular(self) -> None:
        """POST_TRAINING_METHOD (singular) must NOT appear as env var key."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find all os.environ.get() calls and check for singular form
        singular_refs = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "environ"
            ):
                # First argument is the env var name
                if node.args and isinstance(node.args[0], ast.Constant):
                    var_name = node.args[0].value
                    if var_name == "POST_TRAINING_METHOD":
                        singular_refs.append(var_name)

        assert len(singular_refs) == 0, (
            f"Found {len(singular_refs)} reference(s) to legacy "
            f"'POST_TRAINING_METHOD' (singular). Rule 26: greenfield, "
            f"delete old conventions. Only 'POST_TRAINING_METHODS' (plural) "
            f"should exist."
        )

    def test_post_training_methods_plural_exists(self) -> None:
        """POST_TRAINING_METHODS (plural) must be the canonical env var."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        assert "POST_TRAINING_METHODS" in source
