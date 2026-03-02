"""Tests for SDF regression head module (T6 — #231)."""

from __future__ import annotations

import torch

from minivess.adapters.sdf_head import SDFHead


class TestSDFHead:
    """Tests for SDFHead nn.Module."""

    def test_sdf_head_output_shape(self) -> None:
        """[B, 1, D, H, W] output from [B, C, D, H, W] input."""
        head = SDFHead(in_channels=32)
        x = torch.randn(2, 32, 8, 16, 16)
        out = head(x)
        assert out.shape == (2, 1, 8, 16, 16)

    def test_sdf_head_unbounded_output(self) -> None:
        """Output can be negative and positive (no activation on final layer)."""
        head = SDFHead(in_channels=32)
        # Use large random input to produce both signs
        torch.manual_seed(42)
        x = torch.randn(4, 32, 8, 8, 8) * 10
        out = head(x)
        assert out.min() < 0, "SDF head output should include negative values"
        assert out.max() > 0, "SDF head output should include positive values"

    def test_sdf_head_gradient_flows(self) -> None:
        """Backward pass produces gradients."""
        head = SDFHead(in_channels=32)
        x = torch.randn(1, 32, 4, 8, 8, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Gradients should flow to input"
        for name, param in head.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_sdf_head_channel_reduction(self) -> None:
        """Intermediate channels = in_channels // 4."""
        head = SDFHead(in_channels=64)
        # Check the first conv layer's out_channels
        first_conv = head.net[0]
        assert first_conv.out_channels == 16, (
            f"Expected 64//4=16 intermediate channels, got {first_conv.out_channels}"
        )

    def test_sdf_head_param_count_small(self) -> None:
        """Fewer than 10K params for C=32."""
        head = SDFHead(in_channels=32)
        n_params = sum(p.numel() for p in head.parameters())
        assert n_params < 10_000, f"Expected <10K params, got {n_params}"

    def test_sdf_head_different_spatial_sizes(self) -> None:
        """Works with various (D, H, W) spatial dimensions."""
        head = SDFHead(in_channels=16)
        for spatial in [(4, 8, 8), (16, 32, 32), (8, 16, 16), (2, 4, 4)]:
            x = torch.randn(1, 16, *spatial)
            out = head(x)
            assert out.shape == (1, 1, *spatial), f"Failed for spatial={spatial}"

    def test_sdf_head_large_input_no_overflow(self) -> None:
        """Large feature values do not cause inf/nan."""
        head = SDFHead(in_channels=32)
        # Large values
        x = torch.randn(1, 32, 4, 8, 8) * 1000.0
        out = head(x)
        assert torch.isfinite(out).all(), "Output should be finite for large inputs"
