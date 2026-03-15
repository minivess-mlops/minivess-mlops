"""T02-T04 — RED phase: MlpChannel, BidirectionalMambaLayer, MambaBlock.

All tests are CPU-only and use MockMamba (nn.Linear stub) to avoid
requiring mamba-ssm installation. GPU tests with real Mamba are in
tests/gpu_instance/test_mambavesselnet_gpu.py.
"""

from __future__ import annotations

import torch

from tests.v2.unit.conftest import MockMamba


class TestMlpChannel:
    """T02: Conv3d channel-MLP with 4x inner expansion."""

    def test_output_shape_small(self) -> None:
        from minivess.adapters.mambavesselnet import MlpChannel

        mlp = MlpChannel(dim=48)
        x = torch.randn(2, 48, 4, 4, 4)
        out = mlp(x)
        assert out.shape == (2, 48, 4, 4, 4)

    def test_output_shape_bottleneck(self) -> None:
        from minivess.adapters.mambavesselnet import MlpChannel

        mlp = MlpChannel(dim=768)
        x = torch.randn(1, 768, 2, 2, 2)
        out = mlp(x)
        assert out.shape == (1, 768, 2, 2, 2)

    def test_inner_expansion_4x(self) -> None:
        from minivess.adapters.mambavesselnet import MlpChannel

        mlp = MlpChannel(dim=48)
        assert mlp.fc1.out_channels == 4 * 48

    def test_spatial_dims_unchanged(self) -> None:
        from minivess.adapters.mambavesselnet import MlpChannel

        mlp = MlpChannel(dim=96)
        x = torch.randn(1, 96, 5, 7, 3)
        out = mlp(x)
        assert out.shape[2:] == x.shape[2:]

    def test_nonlinear(self) -> None:
        from minivess.adapters.mambavesselnet import MlpChannel

        mlp = MlpChannel(dim=48)
        x = torch.randn(1, 48, 4, 4, 4)
        out = mlp(x)
        assert out.sum().item() != 0.0
        assert not torch.allclose(out, x)


class TestBidirMambaLayerShape:
    """T03: BidirectionalMambaLayer CPU shape tests with MockMamba."""

    def test_output_shape_standard(self) -> None:
        from minivess.adapters.mambavesselnet import BidirectionalMambaLayer

        layer = BidirectionalMambaLayer(dim=768, mamba_cls=MockMamba)
        x = torch.randn(2, 768, 4, 4, 4)
        out = layer(x)
        assert out.shape == x.shape

    def test_output_shape_asymmetric(self) -> None:
        """Non-cube spatial dims — verifies reshape logic is correct."""
        from minivess.adapters.mambavesselnet import BidirectionalMambaLayer

        layer = BidirectionalMambaLayer(dim=768, mamba_cls=MockMamba)
        x = torch.randn(1, 768, 3, 5, 7)
        out = layer(x)
        assert out.shape == x.shape

    def test_residual_applied(self) -> None:
        from minivess.adapters.mambavesselnet import BidirectionalMambaLayer

        layer = BidirectionalMambaLayer(dim=48, mamba_cls=MockMamba)
        x = torch.randn(1, 48, 4, 4, 4)
        out = layer(x)
        # Residual means output != input (unless weights are zero, which they aren't by default)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_has_fwd_bwd_submodules(self) -> None:
        from minivess.adapters.mambavesselnet import BidirectionalMambaLayer

        layer = BidirectionalMambaLayer(dim=48, mamba_cls=MockMamba)
        assert hasattr(layer, "fwd")
        assert hasattr(layer, "bwd")


class TestMambaBlockShape:
    """T04: MambaBlock CPU shape tests with MockMamba."""

    def test_depth1_shape(self) -> None:
        from minivess.adapters.mambavesselnet import MambaBlock

        block = MambaBlock(dim=48, depth=1, mamba_cls=MockMamba)
        x = torch.randn(2, 48, 4, 4, 4)
        out = block(x)
        assert out.shape == x.shape

    def test_depth2_shape(self) -> None:
        from minivess.adapters.mambavesselnet import MambaBlock

        block = MambaBlock(dim=48, depth=2, mamba_cls=MockMamba)
        x = torch.randn(2, 48, 4, 4, 4)
        out = block(x)
        assert out.shape == x.shape

    def test_bottleneck_dim(self) -> None:
        from minivess.adapters.mambavesselnet import MambaBlock

        block = MambaBlock(dim=768, depth=1, mamba_cls=MockMamba)
        x = torch.randn(1, 768, 2, 2, 2)
        out = block(x)
        assert out.shape == x.shape

    def test_transform_applied(self) -> None:
        from minivess.adapters.mambavesselnet import MambaBlock

        block = MambaBlock(dim=48, depth=1, mamba_cls=MockMamba)
        x = torch.randn(1, 48, 4, 4, 4)
        out = block(x)
        assert not torch.allclose(out, torch.zeros_like(out))
