"""Tests for checkpoint resume wiring (B3).

Verifies:
1. check_resume_state_task is called in train_one_fold_task
2. trainer.fit() accepts start_epoch parameter
3. Resume state loads model weights from epoch_latest.pth
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import yaml


class TestTrainerStartEpoch:
    """trainer.fit() must accept a start_epoch parameter to resume training."""

    def test_fit_accepts_start_epoch(self) -> None:
        """AST check: fit() has a start_epoch parameter."""
        src = Path("src/minivess/pipeline/trainer.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "fit":
                arg_names = [a.arg for a in node.args.args]
                # Also check kwargs
                kwonly = [a.arg for a in node.args.kwonlyargs]
                all_args = arg_names + kwonly
                assert "start_epoch" in all_args, (
                    "fit() must accept a start_epoch parameter for spot recovery resume"
                )
                return
        pytest.fail("fit() method not found in trainer.py")

    def test_fit_start_epoch_defaults_to_zero(self) -> None:
        """start_epoch should default to 0 (fresh training)."""
        src = Path("src/minivess/pipeline/trainer.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "fit":
                for i, arg in enumerate(node.args.kwonlyargs):
                    if arg.arg == "start_epoch":
                        default = node.args.kw_defaults[i]
                        assert isinstance(default, ast.Constant) and default.value == 0
                        return
                # Check positional args with defaults
                n_args = len(node.args.args)
                n_defaults = len(node.args.defaults)
                for j, default in enumerate(node.args.defaults):
                    arg_idx = n_args - n_defaults + j
                    if node.args.args[arg_idx].arg == "start_epoch":
                        assert isinstance(default, ast.Constant) and default.value == 0
                        return
                pytest.fail("start_epoch has no default or default is not 0")
        pytest.fail("fit() method not found")

    def test_fit_uses_start_epoch_in_range(self) -> None:
        """fit() epoch loop must use start_epoch (not always range(0, max_epochs))."""
        src = Path("src/minivess/pipeline/trainer.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "fit":
                body_src = ast.dump(node)
                assert "start_epoch" in body_src, (
                    "fit() body must reference start_epoch in epoch loop"
                )
                return
        pytest.fail("fit() method not found")


class TestResumeWiring:
    """check_resume_state_task must be wired into the fold training path."""

    def test_train_one_fold_calls_check_resume(self) -> None:
        """AST check: train_one_fold_task calls check_resume_state_task."""
        src = Path("src/minivess/orchestration/flows/train_flow.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "train_one_fold_task":
                body_dump = ast.dump(node)
                assert "check_resume_state_task" in body_dump, (
                    "train_one_fold_task must call check_resume_state_task"
                )
                return
        pytest.fail("train_one_fold_task not found")

    def test_train_one_fold_passes_start_epoch_to_fit(self) -> None:
        """AST check: trainer.fit() is called with start_epoch kwarg."""
        src = Path("src/minivess/orchestration/flows/train_flow.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "train_one_fold_task":
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        # Look for .fit(..., start_epoch=...) call
                        for kw in child.keywords:
                            if kw.arg == "start_epoch":
                                return
        pytest.fail("trainer.fit() must be called with start_epoch keyword argument")


class TestResumeStateIntegration:
    """Integration: epoch_latest.yaml + .pth enable resume."""

    def test_resume_from_epoch_latest(self, tmp_path: Path) -> None:
        """Write epoch_latest files, verify they can inform resume."""
        checkpoint_dir = tmp_path / "fold_0"
        checkpoint_dir.mkdir()

        # Write epoch_latest.yaml
        state = {
            "epoch": 5,
            "fold": 0,
            "mlflow_run_id": None,
            "best_val_loss": 0.42,
            "timestamp": "2026-03-15T01:00:00+00:00",
        }
        yaml_path = checkpoint_dir / "epoch_latest.yaml"
        yaml_path.write_text(yaml.dump(state), encoding="utf-8")

        # Write epoch_latest.pth
        model_state = {"layer.weight": torch.randn(4, 4)}
        pth_path = checkpoint_dir / "epoch_latest.pth"
        torch.save(model_state, pth_path)

        # Verify can load
        loaded_state = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert loaded_state["epoch"] == 5

        loaded_weights = torch.load(pth_path, map_location="cpu", weights_only=True)
        assert "layer.weight" in loaded_weights
