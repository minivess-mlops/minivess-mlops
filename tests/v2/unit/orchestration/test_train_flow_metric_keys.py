"""Tests for train_flow.py slash-prefix metric key migration (T2).

Verifies that log_fold_results_task() and training_flow() use
fold/{id}/ prefix for fold-level metrics and slash-prefix for
cross-fold metrics.

Issue: #790
"""

from __future__ import annotations

import ast
from pathlib import Path


def test_log_fold_results_uses_slash_prefix() -> None:
    """log_fold_results_task source must use fold/{id}/ prefix, not fold_{id}_."""
    src_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "minivess"
        / "orchestration"
        / "flows"
        / "train_flow.py"
    )
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Find the log_fold_results_task function
    fold_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "log_fold_results_task":
            fold_func = node
            break

    assert fold_func is not None, "log_fold_results_task not found"

    # Check all string literals in the function for old fold_{id}_ pattern
    for node in ast.walk(fold_func):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            # The old pattern: "fold_{fold_id}_best_val_loss"
            # uses f-string so we check for the prefix pattern
            if value.startswith("fold_") and "_best_val_loss" in value:
                raise AssertionError(
                    f"Old underscore fold key found: {value!r}. "
                    "Should use fold/{{id}}/best_val_loss"
                )

    # Check that fold metric f-strings use fold/{id}/ pattern
    # (checkpoint_dir_fold_ is a tag key, not a metric, so we exclude it)
    for node in ast.walk(fold_func):
        if isinstance(node, ast.JoinedStr):
            parts = []
            for val in node.values:
                if isinstance(val, ast.Constant):
                    parts.append(str(val.value))
            joined = "".join(parts)
            # Only flag f-strings that look like metric keys (fold_X_best_val_loss)
            if joined.startswith("fold_") and any(
                m in joined for m in ("best_val_loss", "final_epoch", "val_loss")
            ):
                raise AssertionError(
                    f"Old fold_ metric key pattern in f-string: {joined!r}. "
                    "Should use fold/{{id}}/... prefix"
                )


def test_training_flow_uses_fold_n_completed() -> None:
    """training_flow() must log fold/n_completed, not n_folds_completed."""
    src_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "minivess"
        / "orchestration"
        / "flows"
        / "train_flow.py"
    )
    source = src_path.read_text(encoding="utf-8")

    # The new key should be present
    assert '"fold/n_completed"' in source, (
        "training_flow should use fold/n_completed metric key"
    )
    # The old key should NOT be present as a metric key
    assert '"n_folds_completed"' not in source, (
        "Old n_folds_completed key should be migrated to fold/n_completed"
    )


def test_fold_results_uses_vram_slash_prefix() -> None:
    """log_fold_results_task must use vram/peak_mb, not vram_peak_mb."""
    src_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "minivess"
        / "orchestration"
        / "flows"
        / "train_flow.py"
    )
    source = src_path.read_text(encoding="utf-8")

    assert '"vram/peak_mb"' in source, (
        "log_fold_results_task should use vram/peak_mb metric key"
    )


def test_fold_results_uses_val_slash_prefix_for_cross_fold() -> None:
    """Cross-fold val_loss logging should use val/loss."""
    src_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "minivess"
        / "orchestration"
        / "flows"
        / "train_flow.py"
    )
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Find the log_fold_results_task function
    fold_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "log_fold_results_task":
            fold_func = node
            break

    assert fold_func is not None

    # Check for "val/loss" string literal in the function
    has_val_slash_loss = False
    for node in ast.walk(fold_func):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value == "val/loss"
        ):
            has_val_slash_loss = True
            break

    assert has_val_slash_loss, "Cross-fold val_loss should be logged as val/loss"
