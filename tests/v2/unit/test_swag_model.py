"""Tests for SWAGModel posterior approximation (Maddox et al. 2019).

TDD RED phase for SWAG implementation.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def _make_simple_net(seed: int = 42) -> nn.Module:
    """Create a small deterministic network for testing."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv3d(1, 4, kernel_size=3, padding=1),
        nn.BatchNorm3d(4),
        nn.ReLU(),
        nn.Conv3d(4, 2, kernel_size=1),
    )


class TestSWAGModelCollect:
    """Test SWAG statistics collection."""

    def test_collect_updates_mean(self) -> None:
        """After collecting one model, mean should equal that model's weights."""
        from minivess.ensemble.swag import SWAGModel

        net = _make_simple_net(seed=1)
        swag = SWAGModel(net, max_rank=5)
        swag.collect_model(net)

        assert swag.n_models_collected == 1
        for name, param in net.named_parameters():
            torch.testing.assert_close(swag._mean[name], param.data)

    def test_collect_two_models_updates_sq_mean(self) -> None:
        """After collecting two models, sq_mean should be non-zero."""
        from minivess.ensemble.swag import SWAGModel

        net1 = _make_simple_net(seed=1)
        net2 = _make_simple_net(seed=2)
        swag = SWAGModel(net1, max_rank=5)
        swag.collect_model(net1)
        swag.collect_model(net2)

        assert swag.n_models_collected == 2
        assert swag.has_covariance

    def test_collect_stores_deviations(self) -> None:
        """After collecting 3+ models, deviations list should be populated."""
        from minivess.ensemble.swag import SWAGModel

        net = _make_simple_net(seed=1)
        swag = SWAGModel(net, max_rank=5)

        for i in range(5):
            model_i = _make_simple_net(seed=i)
            swag.collect_model(model_i)

        # First collection doesn't create a deviation (no previous mean)
        # So 5 collections -> 4 deviations
        assert len(swag._deviations) == 4

    def test_collect_respects_max_rank(self) -> None:
        """Deviations should not exceed max_rank."""
        from minivess.ensemble.swag import SWAGModel

        net = _make_simple_net(seed=1)
        swag = SWAGModel(net, max_rank=3)

        for i in range(10):
            model_i = _make_simple_net(seed=i)
            swag.collect_model(model_i)

        assert len(swag._deviations) <= 3


class TestSWAGModelSample:
    """Test posterior sampling."""

    def test_sample_produces_different_weights(self) -> None:
        """Two samples with different seeds should produce different weights."""
        from minivess.ensemble.swag import SWAGModel

        net = _make_simple_net(seed=1)
        swag = SWAGModel(net, max_rank=5)
        for i in range(5):
            swag.collect_model(_make_simple_net(seed=i))

        # Sample 1
        swag.sample(scale=1.0, seed=42)
        weights1 = {name: p.data.clone() for name, p in net.named_parameters()}

        # Sample 2
        swag.sample(scale=1.0, seed=99)
        weights2 = {name: p.data.clone() for name, p in net.named_parameters()}

        any_diff = any(
            not torch.equal(weights1[k], weights2[k])
            for k in weights1
            if weights1[k].is_floating_point()
        )
        assert any_diff, "Different seeds should produce different weight samples"

    def test_sample_scale_zero_returns_mean(self) -> None:
        """Sample with scale=0 should return the mean weights."""
        from minivess.ensemble.swag import SWAGModel

        net = _make_simple_net(seed=1)
        swag = SWAGModel(net, max_rank=5)
        for i in range(5):
            swag.collect_model(_make_simple_net(seed=i))

        mean_copy = {k: v.clone() for k, v in swag._mean.items()}

        swag.sample(scale=0.0, seed=42)
        for name, param in net.named_parameters():
            if name in mean_copy:
                torch.testing.assert_close(param.data, mean_copy[name])

    def test_sample_before_covariance_loads_mean(self) -> None:
        """Sampling before collecting 2+ models should load the mean."""
        from minivess.ensemble.swag import SWAGModel

        net = _make_simple_net(seed=1)
        swag = SWAGModel(net, max_rank=5)
        swag.collect_model(net)

        assert not swag.has_covariance
        swag.sample(scale=1.0)  # Should not crash, just loads mean


class TestSWAGModelPersistence:
    """Test save/load roundtrip."""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """SWAG model should be reconstructible from saved state."""
        from minivess.ensemble.swag import SWAGModel

        net = _make_simple_net(seed=1)
        swag = SWAGModel(net, max_rank=5)
        for i in range(5):
            swag.collect_model(_make_simple_net(seed=i))

        save_path = tmp_path / "swag.pt"
        swag.save(save_path)
        assert save_path.exists()

        # Load into a fresh model
        net2 = _make_simple_net(seed=1)
        swag2 = SWAGModel.load(save_path, net2)

        assert swag2.n_models_collected == swag.n_models_collected
        assert swag2.max_rank == swag.max_rank
        assert len(swag2._deviations) == len(swag._deviations)

        # Mean should match
        for name in swag._mean:
            torch.testing.assert_close(swag2._mean[name], swag._mean[name])

    def test_save_load_produces_same_samples(self, tmp_path: Path) -> None:
        """Samples from loaded SWAG should match samples from original."""
        from minivess.ensemble.swag import SWAGModel

        net = _make_simple_net(seed=1)
        swag = SWAGModel(net, max_rank=5)
        for i in range(5):
            swag.collect_model(_make_simple_net(seed=i))

        # Sample from original
        swag.sample(scale=1.0, seed=42)
        weights_original = {name: p.data.clone() for name, p in net.named_parameters()}

        # Save and load
        save_path = tmp_path / "swag.pt"
        swag.save(save_path)

        net2 = _make_simple_net(seed=1)
        swag2 = SWAGModel.load(save_path, net2)

        # Sample from loaded
        swag2.sample(scale=1.0, seed=42)
        weights_loaded = {name: p.data.clone() for name, p in net2.named_parameters()}

        for name in weights_original:
            torch.testing.assert_close(weights_loaded[name], weights_original[name])


class TestSWAGPluginSavesMeanModel:
    """SWAGPlugin must save a standard-format mean model alongside the SWAG posterior."""

    def test_swag_plugin_output_has_two_model_paths(self) -> None:
        """SWAGPlugin.execute() must return 2 model paths: mean + posterior."""
        from minivess.pipeline.post_training_plugin import PluginInput
        from minivess.pipeline.post_training_plugins.swag import SWAGPlugin

        net = _make_simple_net(seed=1)
        # Create a fake checkpoint
        ckpt_dir = Path("/tmp/test_swag_mean_model")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "best_val_loss.pth"
        torch.save({"model_state_dict": net.state_dict()}, ckpt_path)

        # Build a simple train loader (1 batch)
        dataset = torch.utils.data.TensorDataset(
            torch.randn(2, 1, 4, 4, 4), torch.randint(0, 2, (2, 1, 4, 4, 4))
        )
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        # Wrap to yield dicts (MONAI convention)
        class DictLoader:
            def __init__(self, loader: torch.utils.data.DataLoader) -> None:
                self.loader = loader

            def __iter__(self):  # noqa: ANN204
                for imgs, lbls in self.loader:
                    yield {"image": imgs, "label": lbls}

            def __len__(self) -> int:
                return len(self.loader)

        plugin = SWAGPlugin()
        plugin_input = PluginInput(
            checkpoint_paths=[ckpt_path],
            config={
                "swa_lr": 0.01,
                "swa_epochs": 1,
                "max_rank": 2,
                "update_bn": False,
                "output_dir": str(ckpt_dir / "swag_output"),
            },
            calibration_data={
                "train_loader": DictLoader(train_loader),
                "model": net,
            },
        )

        output = plugin.execute(plugin_input)
        assert len(output.model_paths) == 2, (
            f"Expected 2 model paths (mean + posterior), got {len(output.model_paths)}: "
            f"{output.model_paths}"
        )

    def test_swag_mean_model_is_standard_state_dict(self) -> None:
        """The first model path from SWAGPlugin must be a standard state_dict checkpoint."""
        from minivess.pipeline.post_training_plugin import PluginInput
        from minivess.pipeline.post_training_plugins.swag import SWAGPlugin

        net = _make_simple_net(seed=1)
        ckpt_dir = Path("/tmp/test_swag_mean_model_format")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "best_val_loss.pth"
        torch.save({"model_state_dict": net.state_dict()}, ckpt_path)

        dataset = torch.utils.data.TensorDataset(
            torch.randn(2, 1, 4, 4, 4), torch.randint(0, 2, (2, 1, 4, 4, 4))
        )

        class DictLoader:
            def __init__(self, loader: torch.utils.data.DataLoader) -> None:
                self.loader = loader

            def __iter__(self):  # noqa: ANN204
                for imgs, lbls in self.loader:
                    yield {"image": imgs, "label": lbls}

            def __len__(self) -> int:
                return len(self.loader)

        plugin = SWAGPlugin()
        plugin_input = PluginInput(
            checkpoint_paths=[ckpt_path],
            config={
                "swa_lr": 0.01,
                "swa_epochs": 1,
                "max_rank": 2,
                "update_bn": False,
                "output_dir": str(ckpt_dir / "swag_output2"),
            },
            calibration_data={
                "train_loader": DictLoader(
                    torch.utils.data.DataLoader(dataset, batch_size=2)
                ),
                "model": net,
            },
        )

        output = plugin.execute(plugin_input)
        mean_path = output.model_paths[0]
        assert mean_path.exists(), f"Mean model not found: {mean_path}"

        # Must be loadable as standard state_dict
        ckpt = torch.load(mean_path, weights_only=False)
        assert "model_state_dict" in ckpt, (
            f"Mean model must have 'model_state_dict' key, got: {list(ckpt.keys())}"
        )

        # Must be loadable into a fresh model
        fresh_net = _make_simple_net(seed=99)
        fresh_net.load_state_dict(ckpt["model_state_dict"])


class TestSWAGModelForward:
    """Test forward pass through SWAG model."""

    def test_forward_produces_finite_output(self) -> None:
        """Forward pass after sampling should produce finite output."""
        from minivess.ensemble.swag import SWAGModel

        net = _make_simple_net(seed=1)
        swag = SWAGModel(net, max_rank=5)
        for i in range(5):
            swag.collect_model(_make_simple_net(seed=i))

        swag.sample(scale=1.0, seed=42)

        x = torch.randn(1, 1, 4, 4, 4)
        with torch.no_grad():
            out = swag.forward(x)
        assert out.shape == (1, 2, 4, 4, 4)
        assert torch.isfinite(out).all()
