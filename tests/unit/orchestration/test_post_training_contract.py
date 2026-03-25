"""Tests for post_training_flow checkpoint discovery via FlowContract — Issue #555.

Verifies:
- resolve_checkpoint_paths_from_contract() exists as a public helper
- Returns empty list when parent_run_id is None
- Calls FlowContract.find_fold_checkpoints() with given parent_run_id
- Prefers best_val_loss.pth over epoch_*.ckpt
- Returns empty list when checkpoint_dir does not exist on volume

Plan: docs/planning/prefect-flow-connectivity-execution-plan.xml Phase 0 (T0.4)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    import pytest


class TestResolveCheckpointPathsFromContract:
    def test_returns_empty_when_parent_run_id_is_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        from minivess.orchestration.flows.post_training_flow import (
            resolve_checkpoint_paths_from_contract,
        )

        paths = resolve_checkpoint_paths_from_contract(
            parent_run_id=None,
            tracking_uri=str(tmp_path / "mlruns"),
        )
        assert paths == []

    def test_returns_empty_when_find_fold_checkpoints_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        from minivess.orchestration.flows.post_training_flow import (
            resolve_checkpoint_paths_from_contract,
        )

        with patch(
            "minivess.orchestration.flow_contract.FlowContract.find_fold_checkpoints"
        ) as mock_find:
            mock_find.return_value = []
            paths = resolve_checkpoint_paths_from_contract(
                parent_run_id="fake_parent",
                tracking_uri=str(tmp_path / "mlruns"),
            )

        assert paths == []
        mock_find.assert_called_once_with(parent_run_id="fake_parent")

    def test_prefers_best_ckpt_over_epoch_ckpt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        from minivess.orchestration.flows.post_training_flow import (
            resolve_checkpoint_paths_from_contract,
        )

        ckpt_dir = tmp_path / "fold_0"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "epoch_010.ckpt").write_text("epoch", encoding="utf-8")
        (ckpt_dir / "best_val_loss.pth").write_text("best", encoding="utf-8")

        with patch(
            "minivess.orchestration.flow_contract.FlowContract.find_fold_checkpoints"
        ) as mock_find:
            mock_find.return_value = [
                {"fold_id": 0, "run_id": "r0", "checkpoint_dir": ckpt_dir}
            ]
            paths = resolve_checkpoint_paths_from_contract(
                parent_run_id="fake",
                tracking_uri=str(tmp_path / "mlruns"),
            )

        assert len(paths) == 1
        assert paths[0].name == "best_val_loss.pth"

    def test_falls_back_to_latest_epoch_ckpt_when_no_best(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        from minivess.orchestration.flows.post_training_flow import (
            resolve_checkpoint_paths_from_contract,
        )

        ckpt_dir = tmp_path / "fold_0"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "epoch_005.ckpt").write_text("e5", encoding="utf-8")
        (ckpt_dir / "epoch_010.ckpt").write_text("e10", encoding="utf-8")

        with patch(
            "minivess.orchestration.flow_contract.FlowContract.find_fold_checkpoints"
        ) as mock_find:
            mock_find.return_value = [
                {"fold_id": 0, "run_id": "r0", "checkpoint_dir": ckpt_dir}
            ]
            paths = resolve_checkpoint_paths_from_contract(
                parent_run_id="fake",
                tracking_uri=str(tmp_path / "mlruns"),
            )

        assert len(paths) == 1
        assert "epoch_010" in paths[0].name

    def test_skips_missing_checkpoint_dirs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        from minivess.orchestration.flows.post_training_flow import (
            resolve_checkpoint_paths_from_contract,
        )

        missing_dir = tmp_path / "nonexistent_fold"
        # Do NOT create the directory

        with patch(
            "minivess.orchestration.flow_contract.FlowContract.find_fold_checkpoints"
        ) as mock_find:
            mock_find.return_value = [
                {"fold_id": 0, "run_id": "r0", "checkpoint_dir": missing_dir}
            ]
            paths = resolve_checkpoint_paths_from_contract(
                parent_run_id="fake",
                tracking_uri=str(tmp_path / "mlruns"),
            )

        assert paths == []

    def test_resolves_multiple_folds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MINIVESS_ALLOW_HOST", "1")
        from minivess.orchestration.flows.post_training_flow import (
            resolve_checkpoint_paths_from_contract,
        )

        folds = []
        for i in range(3):
            ckpt_dir = tmp_path / f"fold_{i}"
            ckpt_dir.mkdir(parents=True)
            (ckpt_dir / "best_val_loss.pth").write_text(f"fold{i}", encoding="utf-8")
            folds.append({"fold_id": i, "run_id": f"r{i}", "checkpoint_dir": ckpt_dir})

        with patch(
            "minivess.orchestration.flow_contract.FlowContract.find_fold_checkpoints"
        ) as mock_find:
            mock_find.return_value = folds
            paths = resolve_checkpoint_paths_from_contract(
                parent_run_id="fake",
                tracking_uri=str(tmp_path / "mlruns"),
            )

        assert len(paths) == 3
        assert all(p.name == "best_val_loss.pth" for p in paths)
