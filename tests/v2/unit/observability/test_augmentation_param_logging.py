"""Tests for data augmentation config logging (T5).

Verifies that tracking.py can log data/augmentation_pipeline param
from Hydra config.

Issue: #790
"""

from __future__ import annotations

import ast
from pathlib import Path


def test_log_config_has_augmentation_pipeline_key() -> None:
    """_log_config() should log data/augmentation_pipeline from config."""
    src_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "minivess"
        / "observability"
        / "tracking.py"
    )
    source = src_path.read_text(encoding="utf-8")
    assert '"data/augmentation_pipeline"' in source, (
        "tracking.py should log data/augmentation_pipeline param"
    )


def test_tracking_uses_slash_prefix_for_model_params() -> None:
    """_log_config() should use model/ prefix for model params."""
    src_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "minivess"
        / "observability"
        / "tracking.py"
    )
    source = src_path.read_text(encoding="utf-8")
    assert '"model/family"' in source, "Should use model/family, not model_family"
    assert '"model/name"' in source, "Should use model/name, not model_name"


def test_tracking_uses_slash_prefix_for_train_params() -> None:
    """_log_config() should use train/ prefix for training params."""
    src_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "minivess"
        / "observability"
        / "tracking.py"
    )
    source = src_path.read_text(encoding="utf-8")
    assert '"train/batch_size"' in source, "Should use train/batch_size"
    assert '"train/max_epochs"' in source, "Should use train/max_epochs"
    assert '"train/learning_rate"' in source, "Should use train/learning_rate"


def test_tracking_uses_arch_slash_prefix() -> None:
    """Architecture params should use arch/ prefix."""
    src_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "minivess"
        / "observability"
        / "tracking.py"
    )
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Find _log_config method
    log_config_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_log_config":
            log_config_func = node
            break

    assert log_config_func is not None

    # Check for f"arch/{key}" pattern
    has_arch_slash = False
    for node in ast.walk(log_config_func):
        if isinstance(node, ast.JoinedStr):
            parts = []
            for val in node.values:
                if isinstance(val, ast.Constant):
                    parts.append(str(val.value))
            joined = "".join(parts)
            if "arch/" in joined:
                has_arch_slash = True
                break

    assert has_arch_slash, "_log_config should use arch/ prefix for architecture params"
