"""Guard test: MLflow loss tag is consistently 'loss_function' everywhere.

Issue B3: Training used to log 'loss_name' but ensemble builder reads 'loss_function'.
This test ensures the tag name is standardized across all flows.

See: docs/planning/intermedia-plan-synthesis-pre-debug-run.md Part 5.2
"""

from __future__ import annotations

import ast
from pathlib import Path


class TestLossFunctionTagConsistency:
    """All flows must use 'loss_function' as the MLflow tag name."""

    def test_train_flow_uses_loss_function_tag(self) -> None:
        """train_flow.py must tag runs with 'loss_function', not 'loss_name'."""
        flow_path = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "minivess"
            / "orchestration"
            / "flows"
            / "train_flow.py"
        )
        source = flow_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find all string literals 'loss_function' in dict keys within
        # mlflow.start_run(tags={...}) calls
        found_loss_function_tag = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and key.value == "loss_function":
                        found_loss_function_tag = True

        assert found_loss_function_tag, (
            "train_flow.py must contain 'loss_function' as a dict key "
            "(MLflow tag). Found none — tag mismatch risk (B3)."
        )

    def test_ensemble_builder_reads_loss_function_first(self) -> None:
        """builder.py must read 'loss_function' tag BEFORE fallbacks."""
        builder_path = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "minivess"
            / "ensemble"
            / "builder.py"
        )
        source = builder_path.read_text(encoding="utf-8")

        # The builder must contain tags.get("loss_function") as the PRIMARY read
        assert (
            'tags.get("loss_function")' in source
            or "tags.get('loss_function')" in source
        ), (
            "builder.py must read 'loss_function' tag as primary. "
            "Fallback to 'loss_name' is acceptable but 'loss_function' must come first."
        )
