"""Tests for _average_checkpoints safety checks.

Verifies empty list raises, key mismatch warns, and single checkpoint
returns identity.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch


class TestAverageCheckpoints:
    """_average_checkpoints must be safe against edge cases."""

    def test_empty_list_raises(self, tmp_path: Path) -> None:
        """Empty checkpoint list must raise ValueError (Rule 25)."""
        from minivess.orchestration.flows.post_training_flow import _average_checkpoints

        output = tmp_path / "averaged.pth"
        with pytest.raises(ValueError, match="Cannot average 0 checkpoints"):
            _average_checkpoints([], output)

    def test_mismatched_keys_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Checkpoints with different keys must log a warning."""
        from minivess.orchestration.flows.post_training_flow import _average_checkpoints

        # Checkpoint 1: has conv1 + conv2
        ckpt1 = {"model_state_dict": {"conv1.weight": torch.randn(4, 1, 3), "conv2.weight": torch.randn(2, 4, 3)}}
        ckpt2 = {"model_state_dict": {"conv1.weight": torch.randn(4, 1, 3), "conv3.weight": torch.randn(2, 4, 3)}}

        p1 = tmp_path / "ckpt1.pth"
        p2 = tmp_path / "ckpt2.pth"
        torch.save(ckpt1, p1)
        torch.save(ckpt2, p2)

        output = tmp_path / "averaged.pth"
        with caplog.at_level(logging.WARNING):
            _average_checkpoints([p1, p2], output)

        assert any(
            "mismatch" in r.message.lower() for r in caplog.records
        ), f"Expected key mismatch warning, got: {[r.message for r in caplog.records]}"

    def test_single_checkpoint_returns_identity(self, tmp_path: Path) -> None:
        """Single checkpoint should produce identical output."""
        from minivess.orchestration.flows.post_training_flow import _average_checkpoints

        w = torch.randn(4, 1, 3)
        ckpt = {"model_state_dict": {"conv.weight": w.clone()}}
        p = tmp_path / "ckpt.pth"
        torch.save(ckpt, p)

        output = tmp_path / "averaged.pth"
        _average_checkpoints([p], output)

        loaded = torch.load(output, weights_only=True)
        assert torch.allclose(loaded["model_state_dict"]["conv.weight"], w)

    def test_two_matching_checkpoints_average_correctly(self, tmp_path: Path) -> None:
        """Two checkpoints with same keys should average correctly."""
        from minivess.orchestration.flows.post_training_flow import _average_checkpoints

        w1 = torch.ones(4, 1, 3)
        w2 = torch.ones(4, 1, 3) * 3.0
        expected = torch.ones(4, 1, 3) * 2.0  # (1+3)/2

        torch.save({"model_state_dict": {"w": w1}}, tmp_path / "c1.pth")
        torch.save({"model_state_dict": {"w": w2}}, tmp_path / "c2.pth")

        output = tmp_path / "avg.pth"
        _average_checkpoints([tmp_path / "c1.pth", tmp_path / "c2.pth"], output)

        loaded = torch.load(output, weights_only=True)
        assert torch.allclose(loaded["model_state_dict"]["w"], expected)

    def test_corrupt_file_raises_clear_error(self, tmp_path: Path) -> None:
        """Corrupt checkpoint must raise RuntimeError with file path in message."""
        from minivess.orchestration.flows.post_training_flow import _average_checkpoints

        corrupt = tmp_path / "corrupt.pth"
        corrupt.write_bytes(b"not a valid pytorch checkpoint file")
        output = tmp_path / "averaged.pth"

        with pytest.raises(RuntimeError, match="corrupt"):
            _average_checkpoints([corrupt], output)

    def test_one_corrupt_one_valid_identifies_corrupt(self, tmp_path: Path) -> None:
        """When one checkpoint is corrupt, error must identify which file."""
        from minivess.orchestration.flows.post_training_flow import _average_checkpoints

        valid = tmp_path / "valid.pth"
        torch.save({"model_state_dict": {"w": torch.ones(4)}}, valid)

        corrupt = tmp_path / "corrupt.pth"
        corrupt.write_bytes(b"not valid")

        output = tmp_path / "averaged.pth"
        with pytest.raises(RuntimeError, match="corrupt"):
            _average_checkpoints([valid, corrupt], output)
