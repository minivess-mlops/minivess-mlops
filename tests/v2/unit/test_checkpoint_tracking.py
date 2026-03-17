"""Tests for checkpoint size, format, and parameter count tracking.

Validates that checkpoint metadata is logged to MLflow as metrics.
"""

from __future__ import annotations

from pathlib import Path

import torch


def _make_temp_checkpoint(tmp_path: Path) -> Path:
    """Create a temporary checkpoint with known model state."""
    model = torch.nn.Linear(16, 4)
    state = {"model_state_dict": model.state_dict()}
    ckpt_path = tmp_path / "best.pt"
    torch.save(state, ckpt_path)
    return ckpt_path


class TestCheckpointSizeLogging:
    """T6: Checkpoint size in MB is computed correctly."""

    def test_checkpoint_size_logging(self, tmp_path: Path) -> None:
        from minivess.observability.checkpoint_tracking import (
            compute_checkpoint_metrics,
        )

        ckpt_path = _make_temp_checkpoint(tmp_path)
        metrics = compute_checkpoint_metrics(ckpt_path)

        assert "checkpoint/size_mb" in metrics
        assert metrics["checkpoint/size_mb"] > 0
        assert isinstance(metrics["checkpoint/size_mb"], float)


class TestCheckpointSizeFormatTag:
    """T6: Checkpoint format is detected."""

    def test_checkpoint_size_format_tag(self, tmp_path: Path) -> None:
        from minivess.observability.checkpoint_tracking import (
            compute_checkpoint_metrics,
        )

        ckpt_path = _make_temp_checkpoint(tmp_path)
        metrics = compute_checkpoint_metrics(ckpt_path)

        assert "checkpoint/format" in metrics
        assert metrics["checkpoint/format"] == "pt"


class TestCheckpointSizeNParams:
    """T6: Parameter count is logged."""

    def test_checkpoint_size_n_params(self, tmp_path: Path) -> None:
        from minivess.observability.checkpoint_tracking import (
            compute_checkpoint_metrics,
        )

        ckpt_path = _make_temp_checkpoint(tmp_path)
        metrics = compute_checkpoint_metrics(ckpt_path)

        assert "checkpoint/n_params" in metrics
        # Linear(16, 4) = 16*4 + 4 = 68 params
        assert metrics["checkpoint/n_params"] == 68
