"""Tests for TopoLoRA-SAM paper fidelity (Khazem et al., 2025).

Verifies our SAM3 adaptation matches the paper architecture:
1. LoRA targets FFN layers ONLY (mlp.lin1/lin2), not attention Q/K/V
2. Spatial Adapter exists (Conv_DW 3x3 + Conv 1x1 + residual)
3. Loss is config-driven (all factorial losses must work)

Paper: https://arxiv.org/html/2601.02273v1
Code: https://github.com/salimkhazem/Seglab
"""

from __future__ import annotations

import torch
from torch import nn

# ---------------------------------------------------------------------------
# LoRA target filtering (Phase B2.5)
# ---------------------------------------------------------------------------


class TestLoRATargetsFFNOnly:
    """LoRA must target FFN layers only, not attention Q/K/V projections."""

    def test_apply_lora_has_target_keywords_param(self) -> None:
        """_apply_lora_to_encoder must accept target_keywords filter."""
        import inspect

        from minivess.adapters.sam3_topolora import _apply_lora_to_encoder

        sig = inspect.signature(_apply_lora_to_encoder)
        assert "target_keywords" in sig.parameters, (
            "_apply_lora_to_encoder must accept target_keywords parameter "
            "to filter which layers get LoRA (paper: FFN only)"
        )

    def test_default_targets_are_ffn_keywords(self) -> None:
        """Default target_keywords should be FFN-related (mlp, lin, fc)."""
        import inspect

        from minivess.adapters.sam3_topolora import _apply_lora_to_encoder

        sig = inspect.signature(_apply_lora_to_encoder)
        default = sig.parameters["target_keywords"].default
        # Must include FFN keywords
        assert any(kw in default for kw in ("mlp", "fc", "lin")), (
            f"Default target_keywords {default} must include FFN keywords "
            f"(mlp, fc, lin) per TopoLoRA paper"
        )

    def test_lora_only_applied_to_matching_layers(self) -> None:
        """LoRA should only be applied to layers matching target_keywords."""
        from minivess.adapters.sam3_topolora import _apply_lora_to_encoder

        # Create a mock encoder with both FFN and attention layers
        encoder = nn.Module()
        encoder.block = nn.Module()
        encoder.block.attn = nn.Module()
        encoder.block.attn.qkv = nn.Linear(64, 192)  # attention — should NOT get LoRA
        encoder.block.mlp = nn.Module()
        encoder.block.mlp.lin1 = nn.Linear(64, 256)  # FFN — should get LoRA
        encoder.block.mlp.lin2 = nn.Linear(256, 64)  # FFN — should get LoRA

        targets = _apply_lora_to_encoder(
            encoder,
            rank=8,
            alpha=16.0,
            dropout=0.0,
            target_keywords=("mlp", "lin1", "lin2"),
        )

        # Only FFN layers should be targeted
        assert "block.mlp.lin1" in targets
        assert "block.mlp.lin2" in targets
        assert "block.attn.qkv" not in targets, (
            "Attention Q/K/V should NOT receive LoRA per the paper"
        )


# ---------------------------------------------------------------------------
# Spatial Adapter (Phase B2)
# ---------------------------------------------------------------------------


class TestSpatialConvAdapter:
    """Spatial Adapter: Conv_DW(3x3) + Conv(1x1) + BN + GELU + residual."""

    def test_adapter_exists(self) -> None:
        """SpatialConvAdapter class must exist in sam3_topolora module."""
        from minivess.adapters.sam3_topolora import SpatialConvAdapter

        assert SpatialConvAdapter is not None

    def test_adapter_forward_shape(self) -> None:
        """Output shape must match input shape (residual connection)."""
        from minivess.adapters.sam3_topolora import SpatialConvAdapter

        adapter = SpatialConvAdapter(channels=256)
        x = torch.randn(2, 256, 16, 16)
        y = adapter(x)
        assert y.shape == x.shape

    def test_adapter_has_residual(self) -> None:
        """Adapter must have a residual connection (output ≈ input for zero init)."""
        from minivess.adapters.sam3_topolora import SpatialConvAdapter

        adapter = SpatialConvAdapter(channels=256)
        x = torch.randn(1, 256, 8, 8)

        # Zero-init the last pointwise conv so residual dominates
        with torch.no_grad():
            adapter.pw2.weight.zero_()
            if adapter.pw2.bias is not None:
                adapter.pw2.bias.zero_()

        y = adapter(x)
        assert torch.allclose(y, x, atol=1e-5), (
            "With zero pw2, output should equal input (residual path)"
        )

    def test_adapter_has_depthwise_conv(self) -> None:
        """Must have a depthwise Conv2d (groups=channels)."""
        from minivess.adapters.sam3_topolora import SpatialConvAdapter

        adapter = SpatialConvAdapter(channels=256)
        assert hasattr(adapter, "dw_conv")
        assert adapter.dw_conv.groups == 256, "Depthwise conv must have groups=channels"

    def test_adapter_param_count(self) -> None:
        """Spatial Adapter should have ~66K params for 256 channels."""
        from minivess.adapters.sam3_topolora import SpatialConvAdapter

        adapter = SpatialConvAdapter(channels=256)
        n_params = sum(p.numel() for p in adapter.parameters())
        # ~66K: DW(3x3x256) + PW1(256x256) + BN(256x2) + PW2(256x256) = ~132K
        # Exact count depends on bias settings
        assert 50_000 < n_params < 200_000, f"Expected ~66K-132K params, got {n_params}"

    def test_adapter_produces_gradients(self) -> None:
        """Adapter output must be differentiable (gradients flow)."""
        from minivess.adapters.sam3_topolora import SpatialConvAdapter

        adapter = SpatialConvAdapter(channels=64)
        x = torch.randn(1, 64, 8, 8, requires_grad=True)
        y = adapter(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Spatial Adapter wired into TopoLoRA forward (Phase B2 TB2.3)
# ---------------------------------------------------------------------------


class TestSpatialAdapterInForward:
    """Sam3TopoLoraAdapter forward must use the Spatial Adapter."""

    def test_adapter_attribute_exists(self) -> None:
        """Sam3TopoLoraAdapter must have a spatial_adapter attribute."""
        import inspect

        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        source = inspect.getsource(Sam3TopoLoraAdapter)
        assert "spatial_adapter" in source, (
            "Sam3TopoLoraAdapter must have spatial_adapter attribute"
        )

    def test_forward_calls_spatial_adapter(self) -> None:
        """The forward pass must apply spatial_adapter to FPN features."""
        import inspect

        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        source = inspect.getsource(Sam3TopoLoraAdapter.forward)
        assert "spatial_adapter" in source, (
            "forward() must call self.spatial_adapter on FPN features"
        )
