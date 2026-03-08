"""Tests for post-training Prefect flow (Flow 2.5).

Phase 8 of post-training plugin architecture (#322).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path


def _make_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        "conv.weight": torch.randn(4, 1, 3, 3, 3, generator=gen),
        "conv.bias": torch.randn(4, generator=gen),
    }


def _write_checkpoint(path: Path, seed: int = 0) -> Path:
    torch.save({"state_dict": _make_state_dict(seed)}, path)
    return path


class TestPostTrainingFlow:
    """Post-training flow should orchestrate plugins based on config."""

    def test_flow_function_exists(self) -> None:
        from minivess.orchestration.flows.post_training_flow import post_training_flow

        assert callable(post_training_flow)

    def test_all_disabled_returns_empty(self, tmp_path: Path) -> None:
        from minivess.config.post_training_config import PostTrainingConfig
        from minivess.orchestration.flows.post_training_flow import post_training_flow

        config = PostTrainingConfig(
            swa={"enabled": False},  # type: ignore[arg-type]
            multi_swa={"enabled": False},  # type: ignore[arg-type]
            model_merging={"enabled": False},  # type: ignore[arg-type]
            calibration={"enabled": False},  # type: ignore[arg-type]
            crc_conformal={"enabled": False},  # type: ignore[arg-type]
            conseco_fp_control={"enabled": False},  # type: ignore[arg-type]
        )
        result = post_training_flow(
            config=config,
            checkpoint_paths=[],
            output_dir=tmp_path,
        )
        assert result.status == "completed"
        assert not result.swa_completed
        assert not result.calibration_completed

    def test_swa_plugin_executes(self, tmp_path: Path) -> None:
        from minivess.config.post_training_config import PostTrainingConfig
        from minivess.orchestration.flows.post_training_flow import post_training_flow

        ckpt1 = _write_checkpoint(tmp_path / "ckpt1.pt", seed=1)
        ckpt2 = _write_checkpoint(tmp_path / "ckpt2.pt", seed=2)

        config = PostTrainingConfig(
            multi_swa={"enabled": False},  # type: ignore[arg-type]
            model_merging={"enabled": False},  # type: ignore[arg-type]
            calibration={"enabled": False},  # type: ignore[arg-type]
            crc_conformal={"enabled": False},  # type: ignore[arg-type]
            conseco_fp_control={"enabled": False},  # type: ignore[arg-type]
        )
        result = post_training_flow(
            config=config,
            checkpoint_paths=[ckpt1, ckpt2],
            run_metadata=[
                {"loss_type": "dice_ce", "fold_id": 0},
                {"loss_type": "dice_ce", "fold_id": 1},
            ],
            output_dir=tmp_path,
        )
        assert result.status == "completed"
        assert result.swa_completed

    def test_plugin_failure_isolation(self, tmp_path: Path) -> None:
        """A failing plugin should not block other plugins."""
        from minivess.config.post_training_config import PostTrainingConfig
        from minivess.orchestration.flows.post_training_flow import post_training_flow

        # SWA will fail with empty checkpoints, but we still get a result
        config = PostTrainingConfig(
            multi_swa={"enabled": False},  # type: ignore[arg-type]
            model_merging={"enabled": False},  # type: ignore[arg-type]
            calibration={"enabled": False},  # type: ignore[arg-type]
            crc_conformal={"enabled": False},  # type: ignore[arg-type]
            conseco_fp_control={"enabled": False},  # type: ignore[arg-type]
        )
        result = post_training_flow(
            config=config,
            checkpoint_paths=[],  # Empty → SWA will handle gracefully
            output_dir=tmp_path,
        )
        # Flow should still return success overall (best-effort per plugin)
        assert result.status == "completed"

    def test_result_aggregation(self, tmp_path: Path) -> None:
        from minivess.config.post_training_config import PostTrainingConfig
        from minivess.orchestration.flows.post_training_flow import post_training_flow

        ckpts = [_write_checkpoint(tmp_path / f"ckpt{i}.pt", seed=i) for i in range(4)]

        config = PostTrainingConfig(
            swa={"enabled": True, "per_loss": True, "cross_loss": True},  # type: ignore[arg-type]
            multi_swa={"enabled": False},  # type: ignore[arg-type]
            model_merging={"enabled": False},  # type: ignore[arg-type]
            calibration={"enabled": False},  # type: ignore[arg-type]
            crc_conformal={"enabled": False},  # type: ignore[arg-type]
            conseco_fp_control={"enabled": False},  # type: ignore[arg-type]
        )
        result = post_training_flow(
            config=config,
            checkpoint_paths=ckpts,
            run_metadata=[
                {"loss_type": "dice_ce", "fold_id": 0},
                {"loss_type": "dice_ce", "fold_id": 1},
                {"loss_type": "cbdice", "fold_id": 0},
                {"loss_type": "cbdice", "fold_id": 1},
            ],
            output_dir=tmp_path,
        )
        assert result.swa_completed

    def test_trigger_source_propagated(self, tmp_path: Path) -> None:
        from minivess.config.post_training_config import PostTrainingConfig
        from minivess.orchestration.flows.post_training_flow import post_training_flow

        config = PostTrainingConfig(
            swa={"enabled": False},  # type: ignore[arg-type]
            multi_swa={"enabled": False},  # type: ignore[arg-type]
            model_merging={"enabled": False},  # type: ignore[arg-type]
            calibration={"enabled": False},  # type: ignore[arg-type]
            crc_conformal={"enabled": False},  # type: ignore[arg-type]
            conseco_fp_control={"enabled": False},  # type: ignore[arg-type]
        )
        result = post_training_flow(
            config=config,
            checkpoint_paths=[],
            output_dir=tmp_path,
            trigger_source="test",
        )
        assert result.status == "completed"

    def test_weight_and_data_plugins_separate(self, tmp_path: Path) -> None:
        """Weight-based and data-dependent plugins should both run when enabled."""
        import numpy as np

        from minivess.config.post_training_config import PostTrainingConfig
        from minivess.orchestration.flows.post_training_flow import post_training_flow

        ckpt1 = _write_checkpoint(tmp_path / "ckpt1.pt", seed=1)
        ckpt2 = _write_checkpoint(tmp_path / "ckpt2.pt", seed=2)

        # Build synthetic calibration data for CRC
        rng = np.random.default_rng(42)
        n, c, d, h, w = 20, 2, 4, 4, 4
        raw = rng.standard_normal((n, c, d, h, w)).astype(np.float32)
        exp = np.exp(raw - raw.max(axis=1, keepdims=True))
        scores = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
        labels = rng.integers(0, c, size=(n, d, h, w)).astype(np.int64)

        config = PostTrainingConfig(
            swa={"enabled": True, "per_loss": True, "cross_loss": False},  # type: ignore[arg-type]
            multi_swa={"enabled": False},  # type: ignore[arg-type]
            model_merging={"enabled": False},  # type: ignore[arg-type]
            calibration={"enabled": False},  # type: ignore[arg-type]
            crc_conformal={"enabled": True},  # type: ignore[arg-type]
            conseco_fp_control={"enabled": False},  # type: ignore[arg-type]
        )
        result = post_training_flow(
            config=config,
            checkpoint_paths=[ckpt1, ckpt2],
            run_metadata=[
                {"loss_type": "dice_ce", "fold_id": 0},
                {"loss_type": "dice_ce", "fold_id": 1},
            ],
            output_dir=tmp_path,
            calibration_data={"scores": scores, "labels": labels},
        )
        assert result.status == "completed"
        assert result.swa_completed
        assert result.conformal_completed
