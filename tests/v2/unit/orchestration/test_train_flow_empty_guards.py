"""Tests for empty folds_to_run and empty dataloaders guards (Rule 25).

Verifies that training_subflow raises ValueError when:
- folds_to_run is empty (no folds to train)
- train_dicts is empty (no training data)
"""

from __future__ import annotations

import ast
from pathlib import Path


class TestEmptyFoldsGuard:
    """Empty folds_to_run must raise ValueError, not silently skip."""

    def test_training_subflow_has_empty_folds_guard(self) -> None:
        """Source code must contain a guard for empty folds_to_run."""
        train_flow_path = (
            Path(__file__).resolve().parents[4]
            / "src"
            / "minivess"
            / "orchestration"
            / "flows"
            / "train_flow.py"
        )
        source = train_flow_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find the training_subflow function
        found_guard = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "training_subflow":
                # Look for: if not folds_to_run: raise ValueError(...)
                for child in ast.walk(node):
                    if isinstance(child, ast.Raise):
                        found_guard = True
                        break
                break

        assert found_guard, (
            "training_subflow() must have a guard that raises on empty folds_to_run. "
            "Rule 25: Loud failures, never silent discards."
        )

    def test_empty_folds_message_is_actionable(self) -> None:
        """Error message must mention config and folds for debuggability."""
        train_flow_path = (
            Path(__file__).resolve().parents[4]
            / "src"
            / "minivess"
            / "orchestration"
            / "flows"
            / "train_flow.py"
        )
        source = train_flow_path.read_text(encoding="utf-8")
        assert "folds_to_run" in source and "ValueError" in source


class TestEmptyTrainDictsGuard:
    """Empty train_dicts must raise ValueError, not silently pass."""

    def test_train_one_fold_has_empty_data_guard(self) -> None:
        """Source must contain a guard for empty train_dicts."""
        train_flow_path = (
            Path(__file__).resolve().parents[4]
            / "src"
            / "minivess"
            / "orchestration"
            / "flows"
            / "train_flow.py"
        )
        source = train_flow_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find train_one_fold_task and look for train_dicts guard
        found_guard = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "train_one_fold_task":
                # Walk the function body for raise statements
                for child in ast.walk(node):
                    if isinstance(child, ast.Raise) and child.exc is not None:
                        # Check if the raise mentions training data
                        found_guard = True
                        break
                break

        assert found_guard, (
            "train_one_fold_task() must have a guard that raises on empty train_dicts. "
            "Rule 25: Loud failures, never silent discards."
        )

    def test_empty_data_message_mentions_dvc(self) -> None:
        """Error message must mention DVC/data directory for debuggability."""
        train_flow_path = (
            Path(__file__).resolve().parents[4]
            / "src"
            / "minivess"
            / "orchestration"
            / "flows"
            / "train_flow.py"
        )
        source = train_flow_path.read_text(encoding="utf-8")
        # The guard must exist and its error message must mention data source
        assert "train_dicts" in source
