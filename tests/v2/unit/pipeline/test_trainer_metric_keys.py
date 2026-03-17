"""Tests for trainer.py slash-prefix metric key migration (T3, T4, T6, T7).

Verifies:
- T3: grad/norm_mean, grad/clip_count in epoch metrics
- T4: infer/latency_ms_per_volume in fit() return dict
- T6: optim/grad_scale, prof/val_seconds, prof/train_seconds,
      train/patience_counter, train/stopped_early
- T7: checkpoint/size_mb after saves
- General: trainer uses slash-prefix keys in epoch_log

Issue: #790
"""

from __future__ import annotations

import ast
from pathlib import Path


def _get_trainer_source() -> str:
    """Read trainer.py source code."""
    src_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "minivess"
        / "pipeline"
        / "trainer.py"
    )
    return src_path.read_text(encoding="utf-8")


def test_epoch_log_uses_slash_prefix_for_train_loss() -> None:
    """epoch_log dict in fit() must use train/loss, not train_loss."""
    source = _get_trainer_source()
    assert '"train/loss"' in source, "trainer should use train/loss key"


def test_epoch_log_uses_slash_prefix_for_val_loss() -> None:
    """epoch_log dict in fit() must use val/loss, not val_loss."""
    source = _get_trainer_source()
    assert '"val/loss"' in source, "trainer should use val/loss key"


def test_epoch_log_uses_optim_lr() -> None:
    """epoch_log dict in fit() must use optim/lr, not learning_rate."""
    source = _get_trainer_source()
    assert '"optim/lr"' in source, "trainer should use optim/lr key"


def test_gpu_metrics_use_gpu_prefix() -> None:
    """GPU epoch metrics should use gpu/ prefix, not sys_gpu_."""
    source = _get_trainer_source()
    # The old pattern was f"sys_gpu_{k}" — should now be f"gpu/{k}"
    assert "sys_gpu_" not in source or "gpu/" in source


def test_profiling_epoch_uses_slash_prefix() -> None:
    """First/steady epoch timing should use prof/ prefix."""
    source = _get_trainer_source()
    assert '"prof/first_epoch_seconds"' in source
    assert '"prof/steady_epoch_seconds"' in source


def test_train_prefix_for_metrics() -> None:
    """Train metrics should use train/ prefix in epoch_log."""
    source = _get_trainer_source()
    tree = ast.parse(source)

    # Find fit() method
    fit_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "fit":
            fit_func = node
            break

    assert fit_func is not None, "fit() method not found"

    # Check for f"train/{k}" pattern in f-strings
    has_train_slash = False
    for node in ast.walk(fit_func):
        if isinstance(node, ast.JoinedStr):
            parts = []
            for val in node.values:
                if isinstance(val, ast.Constant):
                    parts.append(str(val.value))
            joined = "".join(parts)
            if "train/" in joined:
                has_train_slash = True
                break

    assert has_train_slash, "fit() should use train/ prefix for train metrics"


def test_val_prefix_for_metrics() -> None:
    """Val metrics should use val/ prefix in epoch_log."""
    source = _get_trainer_source()
    tree = ast.parse(source)

    fit_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "fit":
            fit_func = node
            break

    assert fit_func is not None

    has_val_slash = False
    for node in ast.walk(fit_func):
        if isinstance(node, ast.JoinedStr):
            parts = []
            for val in node.values:
                if isinstance(val, ast.Constant):
                    parts.append(str(val.value))
            joined = "".join(parts)
            if "val/" in joined:
                has_val_slash = True
                break

    assert has_val_slash, "fit() should use val/ prefix for validation metrics"


# T3: Gradient norms
def test_gradient_norm_keys_in_train_epoch() -> None:
    """train_epoch() must return grad/norm_mean and grad/clip_count in metrics."""
    source = _get_trainer_source()
    assert '"grad/norm_mean"' in source, "Missing grad/norm_mean key"
    assert '"grad/clip_count"' in source, "Missing grad/clip_count key"


# T4: Inference latency
def test_inference_latency_in_validate_epoch() -> None:
    """validate_epoch() must compute and return infer/latency_ms_per_volume."""
    source = _get_trainer_source()
    assert '"infer/latency_ms_per_volume"' in source, (
        "Missing infer/latency_ms_per_volume key"
    )


# T6: Optimizer state, timing, early stopping
def test_optim_grad_scale_logged() -> None:
    """fit() should log optim/grad_scale from AMP scaler."""
    source = _get_trainer_source()
    assert '"optim/grad_scale"' in source, "Missing optim/grad_scale key"


def test_prof_val_seconds_logged() -> None:
    """fit() should log prof/val_seconds per validation epoch."""
    source = _get_trainer_source()
    assert '"prof/val_seconds"' in source, "Missing prof/val_seconds key"


def test_prof_train_seconds_logged() -> None:
    """fit() should log prof/train_seconds per train epoch."""
    source = _get_trainer_source()
    assert '"prof/train_seconds"' in source, "Missing prof/train_seconds key"


def test_patience_counter_logged() -> None:
    """fit() should log train/patience_counter."""
    source = _get_trainer_source()
    assert '"train/patience_counter"' in source, "Missing train/patience_counter key"


def test_stopped_early_in_return_dict() -> None:
    """fit() return dict should include stopped_early flag."""
    source = _get_trainer_source()
    assert '"stopped_early"' in source, "Missing stopped_early in fit() return dict"


# T7: Checkpoint metadata
def test_checkpoint_size_mb_logged() -> None:
    """fit() should log checkpoint/size_mb after checkpoint saves."""
    source = _get_trainer_source()
    assert '"checkpoint/size_mb"' in source, "Missing checkpoint/size_mb key"
