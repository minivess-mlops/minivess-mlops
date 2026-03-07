"""Tests for Sam3HybridAdapter.

V3: Frozen SAM3 features + DynUNet 3D + GatedFeatureFusion.

IMPORTANT: These tests require real SAM3 pretrained weights (GPU ≥16 GB).
TestSam3HybridAdapter is skipped in CI where SAM3 is not installed.
TestGatedFeatureFusion tests pure-PyTorch code and run in CI.
"""

from __future__ import annotations

import pytest
import torch

from minivess.adapters.model_builder import _sam3_package_available
from minivess.config.models import ModelConfig, ModelFamily

_sam3_skip = pytest.mark.skipif(
    not _sam3_package_available(), reason="SAM3 not installed"
)


@pytest.fixture()
def sam3_hybrid_config() -> ModelConfig:
    """SAM3 hybrid config."""
    return ModelConfig(
        family=ModelFamily.SAM3_HYBRID,
        name="sam3-hybrid-test",
        in_channels=1,
        out_channels=2,
        architecture_params={
            "filters": [16, 32, 64],
            "fusion_gate_init": 0.0,
        },
    )


class TestGatedFeatureFusion:
    """GatedFeatureFusion: f_3d + sigmoid(alpha) * proj(f_sam).

    These tests use only PyTorch — no SAM3 required.
    """

    def test_fusion_creates(self) -> None:
        from minivess.adapters.sam3_hybrid import GatedFeatureFusion

        fusion = GatedFeatureFusion(sam_channels=256, target_channels=128)
        assert fusion is not None

    def test_fusion_output_shape(self) -> None:
        from minivess.adapters.sam3_hybrid import GatedFeatureFusion

        fusion = GatedFeatureFusion(sam_channels=256, target_channels=128)
        f_3d = torch.randn(1, 128, 4, 8, 8)
        f_sam = torch.randn(1, 256, 4, 8, 8)
        fused = fusion(f_3d, f_sam)
        assert fused.shape == f_3d.shape

    def test_fusion_gate_init_zero(self) -> None:
        """Gate alpha initialized to 0 → pure DynUNet at start."""
        from minivess.adapters.sam3_hybrid import GatedFeatureFusion

        fusion = GatedFeatureFusion(
            sam_channels=256, target_channels=128, gate_init=0.0
        )
        assert fusion.gate_alpha.item() == pytest.approx(0.0)

    def test_fusion_with_zero_gate_passes_3d_unchanged(self) -> None:
        """When gate=-10 (sigmoid≈0) and f_sam=0, output ≈ f_3d."""
        from minivess.adapters.sam3_hybrid import GatedFeatureFusion

        fusion = GatedFeatureFusion(
            sam_channels=256, target_channels=128, gate_init=-10.0
        )
        f_3d = torch.randn(1, 128, 4, 8, 8)
        f_sam = torch.zeros(1, 256, 4, 8, 8)
        fused = fusion(f_3d, f_sam)
        assert torch.allclose(fused, f_3d, atol=1e-3)


@_sam3_skip
class TestSam3HybridAdapter:
    """Sam3HybridAdapter: SAM3 features + DynUNet fusion."""

    def test_adapter_creates(self, sam3_hybrid_config: ModelConfig) -> None:
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(config=sam3_hybrid_config)
        assert adapter is not None

    def test_forward_produces_segmentation_output(
        self, sam3_hybrid_config: ModelConfig
    ) -> None:
        from minivess.adapters.base import SegmentationOutput
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(config=sam3_hybrid_config)
        volume = torch.randn(1, 1, 4, 64, 64)
        output = adapter(volume)
        assert isinstance(output, SegmentationOutput)

    def test_forward_output_shape(self, sam3_hybrid_config: ModelConfig) -> None:
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(config=sam3_hybrid_config)
        volume = torch.randn(1, 1, 4, 64, 64)
        output = adapter(volume)
        assert output.logits.shape == (1, 2, 4, 64, 64)

    def test_sam_encoder_frozen(self, sam3_hybrid_config: ModelConfig) -> None:
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(config=sam3_hybrid_config)
        for p in adapter.sam_backbone.encoder.parameters():
            assert not p.requires_grad

    def test_dynunet_trainable(self, sam3_hybrid_config: ModelConfig) -> None:
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(config=sam3_hybrid_config)
        trainable = [p for p in adapter.dynunet.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_gradient_flows_through_dynunet(
        self, sam3_hybrid_config: ModelConfig
    ) -> None:
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(config=sam3_hybrid_config)
        volume = torch.randn(1, 1, 4, 64, 64)
        output = adapter(volume)
        loss = output.logits.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in adapter.dynunet.parameters()
        )
        assert has_grad

    def test_get_config(self, sam3_hybrid_config: ModelConfig) -> None:
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(config=sam3_hybrid_config)
        info = adapter.get_config()
        assert info.family == "sam3_hybrid"
        assert info.extras.get("variant") == "hybrid"
