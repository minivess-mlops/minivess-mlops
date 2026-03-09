"""Unit tests for ModelMergingPlugin — TDD for #535 (IndexError fix).

Tests verify:
1. validate_inputs catches <2 checkpoints (was returning early without checking)
2. execute() raises ValueError (not IndexError) when <2 checkpoints
3. Happy path: 2 checkpoints merge successfully with linear/slerp/layer_wise
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from minivess.pipeline.post_training_plugin import PluginInput, PluginOutput
from minivess.pipeline.post_training_plugins.model_merging import ModelMergingPlugin


def _make_state_dict(seed: int = 0) -> dict[str, Any]:
    """Return a minimal state dict with deterministic weights."""
    torch.manual_seed(seed)
    return {
        "layer.weight": torch.randn(4, 4),
        "layer.bias": torch.randn(4),
    }


def _make_ckpt_file(tmp_path: Path, seed: int, name: str = "ckpt.pt") -> Path:
    """Save a minimal checkpoint file and return its path."""
    path = tmp_path / name
    torch.save({"state_dict": _make_state_dict(seed)}, path)
    return path


def _make_plugin_input(
    checkpoint_paths: list[Path],
    config: dict | None = None,
    run_metadata: list[dict] | None = None,
) -> PluginInput:
    return PluginInput(
        checkpoint_paths=checkpoint_paths,
        config=config or {},
        run_metadata=run_metadata or [],
    )


# ─── validate_inputs ──────────────────────────────────────────────────────────


class TestValidateInputs:
    def test_zero_checkpoints_returns_error(self, tmp_path: Path) -> None:
        plugin = ModelMergingPlugin()
        inp = _make_plugin_input(checkpoint_paths=[])
        errors = plugin.validate_inputs(inp)
        assert any("2 checkpoints" in e or "checkpoint" in e.lower() for e in errors), (
            f"Expected checkpoint count error, got: {errors}"
        )

    def test_one_checkpoint_returns_error(self, tmp_path: Path) -> None:
        plugin = ModelMergingPlugin()
        ckpt = _make_ckpt_file(tmp_path, seed=0)
        inp = _make_plugin_input(checkpoint_paths=[ckpt])
        errors = plugin.validate_inputs(inp)
        assert any("2 checkpoints" in e or "checkpoint" in e.lower() for e in errors), (
            f"Expected checkpoint count error, got: {errors}"
        )

    def test_one_checkpoint_with_merge_pairs_still_errors(self, tmp_path: Path) -> None:
        """Bug: previously returned early when merge_pairs was truthy, skipping the count check."""
        plugin = ModelMergingPlugin()
        ckpt = _make_ckpt_file(tmp_path, seed=0)
        inp = _make_plugin_input(
            checkpoint_paths=[ckpt],
            config={"merge_pairs": [["cat_a", "cat_b"]]},
        )
        errors = plugin.validate_inputs(inp)
        assert any("checkpoint" in e.lower() for e in errors), (
            "Should still validate checkpoint count even when merge_pairs is set"
        )

    def test_two_checkpoints_no_error(self, tmp_path: Path) -> None:
        plugin = ModelMergingPlugin()
        ckpt1 = _make_ckpt_file(tmp_path, seed=0, name="a.pt")
        ckpt2 = _make_ckpt_file(tmp_path, seed=1, name="b.pt")
        inp = _make_plugin_input(checkpoint_paths=[ckpt1, ckpt2])
        errors = plugin.validate_inputs(inp)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_invalid_method_returns_error(self, tmp_path: Path) -> None:
        plugin = ModelMergingPlugin()
        ckpt1 = _make_ckpt_file(tmp_path, seed=0, name="a.pt")
        ckpt2 = _make_ckpt_file(tmp_path, seed=1, name="b.pt")
        inp = _make_plugin_input(
            checkpoint_paths=[ckpt1, ckpt2],
            config={"method": "invalid_method"},
        )
        errors = plugin.validate_inputs(inp)
        assert any("invalid_method" in e for e in errors)


# ─── execute ──────────────────────────────────────────────────────────────────


class TestExecute:
    def test_execute_raises_value_error_with_zero_checkpoints(
        self, tmp_path: Path
    ) -> None:
        """execute() must raise ValueError (not IndexError) when called with 0 checkpoints."""
        plugin = ModelMergingPlugin()
        inp = _make_plugin_input(
            checkpoint_paths=[], config={"output_dir": str(tmp_path)}
        )
        with pytest.raises(ValueError, match="checkpoint"):
            plugin.execute(inp)

    def test_execute_raises_value_error_with_one_checkpoint(
        self, tmp_path: Path
    ) -> None:
        plugin = ModelMergingPlugin()
        ckpt = _make_ckpt_file(tmp_path, seed=0)
        inp = _make_plugin_input(
            checkpoint_paths=[ckpt], config={"output_dir": str(tmp_path)}
        )
        with pytest.raises(ValueError, match="checkpoint"):
            plugin.execute(inp)

    def test_execute_linear_merge_two_checkpoints(self, tmp_path: Path) -> None:
        plugin = ModelMergingPlugin()
        ckpt1 = _make_ckpt_file(tmp_path, seed=0, name="a.pt")
        ckpt2 = _make_ckpt_file(tmp_path, seed=1, name="b.pt")
        inp = _make_plugin_input(
            checkpoint_paths=[ckpt1, ckpt2],
            config={"method": "linear", "t": 0.5, "output_dir": str(tmp_path)},
        )
        result = plugin.execute(inp)
        assert isinstance(result, PluginOutput)
        assert len(result.model_paths) == 1
        assert result.model_paths[0].exists()

    def test_execute_slerp_merge_two_checkpoints(self, tmp_path: Path) -> None:
        plugin = ModelMergingPlugin()
        ckpt1 = _make_ckpt_file(tmp_path, seed=0, name="a.pt")
        ckpt2 = _make_ckpt_file(tmp_path, seed=1, name="b.pt")
        inp = _make_plugin_input(
            checkpoint_paths=[ckpt1, ckpt2],
            config={"method": "slerp", "t": 0.5, "output_dir": str(tmp_path)},
        )
        result = plugin.execute(inp)
        assert isinstance(result, PluginOutput)
        assert len(result.model_paths) == 1

    def test_execute_layer_wise_merge_two_checkpoints(self, tmp_path: Path) -> None:
        plugin = ModelMergingPlugin()
        ckpt1 = _make_ckpt_file(tmp_path, seed=0, name="a.pt")
        ckpt2 = _make_ckpt_file(tmp_path, seed=1, name="b.pt")
        inp = _make_plugin_input(
            checkpoint_paths=[ckpt1, ckpt2],
            config={"method": "layer_wise", "t": 0.5, "output_dir": str(tmp_path)},
        )
        result = plugin.execute(inp)
        assert isinstance(result, PluginOutput)
        assert len(result.model_paths) == 1
