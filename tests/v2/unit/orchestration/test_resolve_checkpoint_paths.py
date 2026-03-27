"""Tests for resolve_checkpoint_paths_from_contract.

T6 from double-check plan: empty result with parent_run_id must raise ValueError.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestResolveCheckpointPaths:
    """resolve_checkpoint_paths_from_contract must fail loudly when checkpoints missing."""

    def test_none_parent_returns_empty(self) -> None:
        from minivess.orchestration.flows.post_training_flow import (
            resolve_checkpoint_paths_from_contract,
        )

        result = resolve_checkpoint_paths_from_contract(
            parent_run_id=None,
            tracking_uri="file:///tmp/mlruns",
        )
        assert result == []

    def test_missing_dirs_raises(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.post_training_flow import (
            resolve_checkpoint_paths_from_contract,
        )

        # Mock FlowContract to return infos with non-existent dirs
        fake_infos = [
            {"fold_id": 0, "checkpoint_dir": tmp_path / "nonexistent_fold_0"},
            {"fold_id": 1, "checkpoint_dir": tmp_path / "nonexistent_fold_1"},
        ]
        mock_fc = MagicMock()
        mock_fc.find_fold_checkpoints.return_value = fake_infos

        with patch(
            "minivess.orchestration.flow_contract.FlowContract",
            return_value=mock_fc,
        ), pytest.raises(ValueError, match="0 checkpoints"):
            resolve_checkpoint_paths_from_contract(
                parent_run_id="abc123",
                tracking_uri="file:///tmp/mlruns",
            )

    def test_valid_paths_returned(self, tmp_path: Path) -> None:
        from minivess.orchestration.flows.post_training_flow import (
            resolve_checkpoint_paths_from_contract,
        )

        # Create real checkpoint files
        fold_dir = tmp_path / "fold_0"
        fold_dir.mkdir()
        ckpt = fold_dir / "best_val_loss.pth"
        ckpt.write_bytes(b"fake checkpoint")

        fake_infos = [{"fold_id": 0, "checkpoint_dir": fold_dir}]
        mock_fc = MagicMock()
        mock_fc.find_fold_checkpoints.return_value = fake_infos

        with patch(
            "minivess.orchestration.flow_contract.FlowContract",
            return_value=mock_fc,
        ):
            result = resolve_checkpoint_paths_from_contract(
                parent_run_id="abc123",
                tracking_uri="file:///tmp/mlruns",
            )

        assert len(result) == 1
        assert result[0] == ckpt
