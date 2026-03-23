"""Tests for checkpoint resolution fallback chain (T0.1-T0.3).

Validates resolve_checkpoint_paths_from_contract() discovers checkpoints
in priority order: best_val_loss.pth → last.pth → epoch_*.ckpt.
(best_val_loss.pth removed — canonical filename is best_val_loss.pth per constants.py)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch


def _write_dummy_ckpt(path: Path) -> None:
    """Create a minimal valid checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": {"w": torch.randn(2, 2)}}, path)


def _mock_fold_infos(ckpt_dirs: list[Path]) -> list[dict]:
    """Build fold info dicts matching FlowContract.find_fold_checkpoints format."""
    return [{"fold_id": i, "checkpoint_dir": d} for i, d in enumerate(ckpt_dirs)]


class TestCheckpointFallbackChain:
    """Checkpoint resolution: best_val_loss.pth > last.pth > epoch_*."""

    def test_prefers_best_val_loss_pth(self, tmp_path: Path) -> None:
        """best_val_loss.pth (canonical) is preferred over all alternatives."""
        fold_dir = tmp_path / "fold_0"
        fold_dir.mkdir()
        _write_dummy_ckpt(fold_dir / "best_val_loss.pth")
        _write_dummy_ckpt(fold_dir / "last.pth")
        _write_dummy_ckpt(fold_dir / "epoch_001.ckpt")

        from minivess.orchestration.flows.post_training_flow import (
            resolve_checkpoint_paths_from_contract,
        )

        with patch("minivess.orchestration.flow_contract.FlowContract") as MockFC:
            MockFC.return_value.find_fold_checkpoints.return_value = _mock_fold_infos(
                [fold_dir]
            )
            paths = resolve_checkpoint_paths_from_contract(
                parent_run_id="test_run", tracking_uri="file:///tmp/mlruns"
            )

        assert len(paths) == 1
        assert paths[0].name == "best_val_loss.pth"

    def test_fallback_to_last(self, tmp_path: Path) -> None:
        """Falls back to last.pth when best_val_loss.pth absent."""
        fold_dir = tmp_path / "fold_0"
        fold_dir.mkdir()
        _write_dummy_ckpt(fold_dir / "last.pth")
        _write_dummy_ckpt(fold_dir / "epoch_001.ckpt")

        from minivess.orchestration.flows.post_training_flow import (
            resolve_checkpoint_paths_from_contract,
        )

        with patch("minivess.orchestration.flow_contract.FlowContract") as MockFC:
            MockFC.return_value.find_fold_checkpoints.return_value = _mock_fold_infos(
                [fold_dir]
            )
            paths = resolve_checkpoint_paths_from_contract(
                parent_run_id="test_run", tracking_uri="file:///tmp/mlruns"
            )

        assert len(paths) == 1
        assert paths[0].name == "last.pth"

    def test_fallback_to_latest_epoch(self, tmp_path: Path) -> None:
        """Falls back to lexicographically latest epoch_*.ckpt."""
        fold_dir = tmp_path / "fold_0"
        fold_dir.mkdir()
        _write_dummy_ckpt(fold_dir / "epoch_001.ckpt")
        _write_dummy_ckpt(fold_dir / "epoch_003.ckpt")
        _write_dummy_ckpt(fold_dir / "epoch_002.ckpt")

        from minivess.orchestration.flows.post_training_flow import (
            resolve_checkpoint_paths_from_contract,
        )

        with patch("minivess.orchestration.flow_contract.FlowContract") as MockFC:
            MockFC.return_value.find_fold_checkpoints.return_value = _mock_fold_infos(
                [fold_dir]
            )
            paths = resolve_checkpoint_paths_from_contract(
                parent_run_id="test_run", tracking_uri="file:///tmp/mlruns"
            )

        assert len(paths) == 1
        assert paths[0].name == "epoch_003.ckpt"

    def test_none_run_id_returns_empty(self) -> None:
        """None parent_run_id returns empty list."""
        from minivess.orchestration.flows.post_training_flow import (
            resolve_checkpoint_paths_from_contract,
        )

        paths = resolve_checkpoint_paths_from_contract(
            parent_run_id=None, tracking_uri="file:///tmp/mlruns"
        )
        assert paths == []


class TestMultiFoldDiscovery:
    """T0.3: Multi-fold checkpoint discovery across 3 folds."""

    def test_discovers_checkpoints_across_3_folds(self, tmp_path: Path) -> None:
        """Should find one checkpoint per fold directory."""
        fold_dirs = []
        for i in range(3):
            fold_dir = tmp_path / f"fold_{i}"
            fold_dir.mkdir()
            _write_dummy_ckpt(fold_dir / "best_val_loss.pth")
            fold_dirs.append(fold_dir)

        from minivess.orchestration.flows.post_training_flow import (
            resolve_checkpoint_paths_from_contract,
        )

        with patch("minivess.orchestration.flow_contract.FlowContract") as MockFC:
            MockFC.return_value.find_fold_checkpoints.return_value = _mock_fold_infos(
                fold_dirs
            )
            paths = resolve_checkpoint_paths_from_contract(
                parent_run_id="test_run", tracking_uri="file:///tmp/mlruns"
            )

        assert len(paths) == 3
        for p in paths:
            assert p.name == "best_val_loss.pth"

    def test_skips_missing_fold_dirs(self, tmp_path: Path) -> None:
        """Missing fold directories are silently skipped."""
        fold_0 = tmp_path / "fold_0"
        fold_0.mkdir()
        _write_dummy_ckpt(fold_0 / "best_val_loss.pth")
        fold_1_missing = tmp_path / "fold_1_does_not_exist"  # Does NOT exist

        from minivess.orchestration.flows.post_training_flow import (
            resolve_checkpoint_paths_from_contract,
        )

        with patch("minivess.orchestration.flow_contract.FlowContract") as MockFC:
            MockFC.return_value.find_fold_checkpoints.return_value = _mock_fold_infos(
                [fold_0, fold_1_missing]
            )
            paths = resolve_checkpoint_paths_from_contract(
                parent_run_id="test_run", tracking_uri="file:///tmp/mlruns"
            )

        assert len(paths) == 1  # Only fold_0 found
