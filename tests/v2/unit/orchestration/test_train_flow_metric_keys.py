"""Tests for train_flow.py slash-prefix metric key migration (T2).

Verifies that log_fold_results_task() and training_flow() use
MetricKeys constants for all metric key references.

Issue: #790
"""

from __future__ import annotations

import ast
from pathlib import Path


def _get_train_flow_source() -> tuple[str, ast.Module]:
    """Return source text and AST of train_flow.py."""
    src_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "minivess"
        / "orchestration"
        / "flows"
        / "train_flow.py"
    )
    source = src_path.read_text(encoding="utf-8")
    return source, ast.parse(source)


def test_log_fold_results_uses_slash_prefix() -> None:
    """log_fold_results_task source must use fold/{id}/ prefix, not fold_{id}_."""
    _source, tree = _get_train_flow_source()

    fold_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "log_fold_results_task":
            fold_func = node
            break

    assert fold_func is not None, "log_fold_results_task not found"

    for node in ast.walk(fold_func):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if value.startswith("fold_") and "_best_val_loss" in value:
                raise AssertionError(
                    f"Old underscore fold key found: {value!r}. "
                    "Should use fold/{{id}}/best_val_loss"
                )

    for node in ast.walk(fold_func):
        if isinstance(node, ast.JoinedStr):
            parts = []
            for val in node.values:
                if isinstance(val, ast.Constant):
                    parts.append(str(val.value))
            joined = "".join(parts)
            if joined.startswith("fold_") and any(
                m in joined for m in ("best_val_loss", "final_epoch", "val_loss")
            ):
                raise AssertionError(
                    f"Old fold_ metric key pattern in f-string: {joined!r}. "
                    "Should use fold/{{id}}/... prefix"
                )


def test_training_flow_uses_metric_keys_for_fold_n_completed() -> None:
    """training_flow() must use MetricKeys.FOLD_N_COMPLETED, not hardcoded string."""
    source, _tree = _get_train_flow_source()

    # Must import MetricKeys
    assert "MetricKeys" in source, (
        "train_flow.py must import MetricKeys from metric_keys"
    )
    # Must use the constant (not the hardcoded string directly)
    assert "MetricKeys.FOLD_N_COMPLETED" in source or "MetricKeys." in source, (
        "train_flow.py should use MetricKeys constants for fold/n_completed"
    )
    # Old key should NOT be present
    assert '"n_folds_completed"' not in source, (
        "Old n_folds_completed key should be migrated to MetricKeys.FOLD_N_COMPLETED"
    )


def test_fold_results_uses_metric_keys_for_vram() -> None:
    """log_fold_results_task must use MetricKeys.VRAM_PEAK_MB."""
    source, _tree = _get_train_flow_source()

    assert "MetricKeys.VRAM_PEAK_MB" in source, (
        "log_fold_results_task should use MetricKeys.VRAM_PEAK_MB"
    )


def test_fold_results_uses_metric_keys_for_val_loss() -> None:
    """Cross-fold val_loss logging should use MetricKeys.VAL_LOSS."""
    source, tree = _get_train_flow_source()

    # Find the log_fold_results_task function
    fold_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "log_fold_results_task":
            fold_func = node
            break

    assert fold_func is not None

    # Check for MetricKeys.VAL_LOSS attribute access in the function
    has_metric_keys_val_loss = False
    for node in ast.walk(fold_func):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "MetricKeys"
            and node.attr == "VAL_LOSS"
        ):
            has_metric_keys_val_loss = True
            break

    assert has_metric_keys_val_loss, (
        "Cross-fold val_loss should be logged as MetricKeys.VAL_LOSS"
    )
