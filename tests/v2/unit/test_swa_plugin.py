"""Tests for SWA post-training plugin.

Phase 2 of post-training plugin architecture (#316).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path

from minivess.pipeline.post_training_plugin import (
    PluginInput,
    PluginOutput,
    PostTrainingPlugin,
)


def _make_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    """Create a small state dict for testing."""
    gen = torch.Generator().manual_seed(seed)
    return {
        "conv.weight": torch.randn(4, 1, 3, 3, 3, generator=gen),
        "conv.bias": torch.randn(4, generator=gen),
    }


def _write_checkpoint(path: Path, seed: int = 0) -> Path:
    """Write a checkpoint file and return its path."""
    sd = _make_state_dict(seed)
    ckpt = {"state_dict": sd}
    torch.save(ckpt, path)
    return path


class TestSWAPlugin:
    """SWA plugin should wrap model_soup.uniform_swa()."""

    def test_implements_protocol(self) -> None:
        from minivess.pipeline.post_training_plugins.swa import SWAPlugin

        plugin = SWAPlugin()
        assert isinstance(plugin, PostTrainingPlugin)

    def test_name_property(self) -> None:
        from minivess.pipeline.post_training_plugins.swa import SWAPlugin

        assert SWAPlugin().name == "swa"

    def test_does_not_require_calibration_data(self) -> None:
        from minivess.pipeline.post_training_plugins.swa import SWAPlugin

        assert SWAPlugin().requires_calibration_data is False

    def test_validate_inputs_empty_checkpoints(self) -> None:
        from minivess.pipeline.post_training_plugins.swa import SWAPlugin

        plugin = SWAPlugin()
        pi = PluginInput(checkpoint_paths=[], config={"per_loss": True})
        errors = plugin.validate_inputs(pi)
        assert len(errors) > 0
        assert "checkpoint" in errors[0].lower()

    def test_per_loss_swa(self, tmp_path: Path) -> None:
        from minivess.pipeline.post_training_plugins.swa import SWAPlugin

        # Create checkpoints with metadata
        ckpt1 = _write_checkpoint(tmp_path / "ckpt1.pt", seed=1)
        ckpt2 = _write_checkpoint(tmp_path / "ckpt2.pt", seed=2)
        ckpt3 = _write_checkpoint(tmp_path / "ckpt3.pt", seed=3)

        pi = PluginInput(
            checkpoint_paths=[ckpt1, ckpt2, ckpt3],
            config={"per_loss": True, "cross_loss": False, "output_dir": str(tmp_path)},
            run_metadata=[
                {"loss_type": "dice_ce", "fold_id": 0},
                {"loss_type": "dice_ce", "fold_id": 1},
                {"loss_type": "cbdice", "fold_id": 0},
            ],
        )

        plugin = SWAPlugin()
        result = plugin.execute(pi)
        assert isinstance(result, PluginOutput)
        # Should produce one model per loss type
        assert len(result.model_paths) == 2  # dice_ce + cbdice

    def test_cross_loss_swa(self, tmp_path: Path) -> None:
        from minivess.pipeline.post_training_plugins.swa import SWAPlugin

        ckpt1 = _write_checkpoint(tmp_path / "ckpt1.pt", seed=1)
        ckpt2 = _write_checkpoint(tmp_path / "ckpt2.pt", seed=2)

        pi = PluginInput(
            checkpoint_paths=[ckpt1, ckpt2],
            config={"per_loss": False, "cross_loss": True, "output_dir": str(tmp_path)},
            run_metadata=[
                {"loss_type": "dice_ce", "fold_id": 0},
                {"loss_type": "cbdice", "fold_id": 0},
            ],
        )

        plugin = SWAPlugin()
        result = plugin.execute(pi)
        assert isinstance(result, PluginOutput)
        # Single cross-loss SWA model
        assert len(result.model_paths) == 1

    def test_single_checkpoint_passthrough(self, tmp_path: Path) -> None:
        from minivess.pipeline.post_training_plugins.swa import SWAPlugin

        ckpt = _write_checkpoint(tmp_path / "ckpt.pt", seed=1)

        pi = PluginInput(
            checkpoint_paths=[ckpt],
            config={"per_loss": True, "cross_loss": False, "output_dir": str(tmp_path)},
            run_metadata=[{"loss_type": "dice_ce", "fold_id": 0}],
        )

        plugin = SWAPlugin()
        result = plugin.execute(pi)
        assert len(result.model_paths) == 1
