"""Tests for SWAG post-training plugin (Maddox et al. 2019).

TDD RED phase for SWAG plugin implementation.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from minivess.pipeline.post_training_plugin import (
    PluginInput,
    PluginOutput,
    PostTrainingPlugin,
)


def _make_simple_net(seed: int = 42) -> nn.Module:
    """Create a small deterministic network for testing."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv3d(1, 4, kernel_size=3, padding=1),
        nn.BatchNorm3d(4),
        nn.ReLU(),
        nn.Conv3d(4, 2, kernel_size=1),
    )


def _make_train_loader(n_samples: int = 4, seed: int = 42) -> DataLoader:
    """Create a tiny DataLoader for SWAG training."""
    torch.manual_seed(seed)
    images = torch.randn(n_samples, 1, 4, 4, 4)
    labels = torch.randint(0, 2, (n_samples, 2, 4, 4, 4)).float()
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=2)


def _write_checkpoint(path: Path, seed: int = 0) -> Path:
    """Write a checkpoint file with state_dict."""
    net = _make_simple_net(seed=seed)
    torch.save({"state_dict": net.state_dict()}, path)
    return path


class TestSWAGPlugin:
    """SWAG plugin should implement PostTrainingPlugin protocol."""

    def test_implements_protocol(self) -> None:
        from minivess.pipeline.post_training_plugins.swag import SWAGPlugin

        plugin = SWAGPlugin()
        assert isinstance(plugin, PostTrainingPlugin)

    def test_name_property(self) -> None:
        from minivess.pipeline.post_training_plugins.swag import SWAGPlugin

        assert SWAGPlugin().name == "swag"

    def test_requires_calibration_data(self) -> None:
        from minivess.pipeline.post_training_plugins.swag import SWAGPlugin

        assert SWAGPlugin().requires_calibration_data is True

    def test_validate_requires_checkpoint(self) -> None:
        from minivess.pipeline.post_training_plugins.swag import SWAGPlugin

        plugin = SWAGPlugin()
        pi = PluginInput(
            checkpoint_paths=[],
            config={},
            calibration_data={"train_loader": _make_train_loader()},
        )
        errors = plugin.validate_inputs(pi)
        assert len(errors) > 0
        assert "checkpoint" in errors[0].lower()

    def test_validate_requires_calibration_data(self) -> None:
        from minivess.pipeline.post_training_plugins.swag import SWAGPlugin

        plugin = SWAGPlugin()
        pi = PluginInput(
            checkpoint_paths=[Path("/tmp/ckpt.pt")],
            config={},
            calibration_data=None,
        )
        errors = plugin.validate_inputs(pi)
        assert len(errors) > 0
        assert "calibration_data" in errors[0].lower()

    def test_validate_requires_train_loader_key(self) -> None:
        from minivess.pipeline.post_training_plugins.swag import SWAGPlugin

        plugin = SWAGPlugin()
        pi = PluginInput(
            checkpoint_paths=[Path("/tmp/ckpt.pt")],
            config={},
            calibration_data={"logits": [], "labels": []},
        )
        errors = plugin.validate_inputs(pi)
        assert len(errors) > 0
        assert "train_loader" in errors[0].lower()

    def test_execute_produces_swag_model(self, tmp_path: Path) -> None:
        from minivess.pipeline.post_training_plugins.swag import SWAGPlugin

        ckpt_path = _write_checkpoint(tmp_path / "ckpt.pt", seed=1)

        pi = PluginInput(
            checkpoint_paths=[ckpt_path],
            config={
                "swa_lr": 0.001,
                "swa_epochs": 2,
                "max_rank": 3,
                "update_bn": True,
                "output_dir": str(tmp_path / "output"),
            },
            calibration_data={
                "train_loader": _make_train_loader(),
                "model": _make_simple_net(seed=1),
            },
        )

        plugin = SWAGPlugin()
        result = plugin.execute(pi)

        assert isinstance(result, PluginOutput)
        assert len(result.model_paths) == 1
        assert result.model_paths[0].exists()
        assert result.metrics["swag_epochs"] == 2.0
        assert result.metrics["swag_max_rank"] == 3.0

    def test_execute_metrics_logged(self, tmp_path: Path) -> None:
        from minivess.pipeline.post_training_plugins.swag import SWAGPlugin

        ckpt_path = _write_checkpoint(tmp_path / "ckpt.pt", seed=1)

        pi = PluginInput(
            checkpoint_paths=[ckpt_path],
            config={
                "swa_lr": 0.01,
                "swa_epochs": 3,
                "max_rank": 5,
                "update_bn": False,
                "output_dir": str(tmp_path / "output"),
            },
            calibration_data={
                "train_loader": _make_train_loader(),
                "model": _make_simple_net(seed=1),
            },
        )

        plugin = SWAGPlugin()
        result = plugin.execute(pi)

        assert "swag_epochs" in result.metrics
        assert "swag_lr" in result.metrics
        assert "swag_max_rank" in result.metrics
        assert "swag_n_models_collected" in result.metrics
        assert result.metrics["swag_n_models_collected"] == 3.0
