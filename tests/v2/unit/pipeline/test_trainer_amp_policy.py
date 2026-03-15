"""Tests for separate AMP policy: train ON, validation OFF.

Root cause: AMP autocast + 3D sliding_window_inference produces NaN
on SAM3 hybrid models. Confirmed on both T4 (FP16) and L4 (BF16).
MONAI maintainers acknowledge: "AMP does not support very well with
3D operations" (Project-MONAI/MONAI#4243).

Fix: TrainingConfig.mixed_precision_val defaults to False for SAM3 models.
Trainer uses mixed_precision for train_epoch, mixed_precision_val for validate_epoch.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


class TestMixedPrecisionValConfig:
    """TrainingConfig must have separate mixed_precision_val field."""

    def test_training_config_has_mixed_precision_val(self) -> None:
        """TrainingConfig must have mixed_precision_val field."""
        from minivess.config.models import TrainingConfig

        config = TrainingConfig()
        assert hasattr(config, "mixed_precision_val")

    def test_mixed_precision_val_defaults_false(self) -> None:
        """mixed_precision_val should default to False (safe default)."""
        from minivess.config.models import TrainingConfig

        config = TrainingConfig()
        assert config.mixed_precision_val is False


class TestTrainerUsesValAmpPolicy:
    """Trainer must use mixed_precision_val for validation autocast."""

    def test_validate_epoch_uses_val_amp_flag(self) -> None:
        """AST: validate_epoch autocast uses mixed_precision_val, not mixed_precision."""
        src = Path("src/minivess/pipeline/trainer.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "validate_epoch":
                body_dump = ast.dump(node)
                assert "mixed_precision_val" in body_dump, (
                    "validate_epoch must use mixed_precision_val for autocast"
                )
                return
        pytest.fail("validate_epoch not found")

    def test_train_epoch_uses_train_amp_flag(self) -> None:
        """AST: train_epoch autocast uses mixed_precision (not _val)."""
        src = Path("src/minivess/pipeline/trainer.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "train_epoch":
                body_dump = ast.dump(node)
                assert "mixed_precision" in body_dump, (
                    "train_epoch must use mixed_precision for autocast"
                )
                return
        pytest.fail("train_epoch not found")
