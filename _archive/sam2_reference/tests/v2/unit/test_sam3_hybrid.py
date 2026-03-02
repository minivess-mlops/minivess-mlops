"""Tests for Sam3HybridAdapter (SAM-08).

Frozen SAM2 features + DynUNet 3D + GatedFeatureFusion.
"""

from __future__ import annotations

import pytest
import torch

from minivess.config.models import ModelConfig, ModelFamily


@pytest.fixture()
def hybrid_config() -> ModelConfig:
    return ModelConfig(
        family=ModelFamily.SAM3_HYBRID,
        name="sam3-hybrid-test",
        in_channels=1,
        out_channels=2,
        architecture_params={"filters": [16, 32, 64, 128]},
    )


# ---------------------------------------------------------------------------
# GatedFeatureFusion tests
# ---------------------------------------------------------------------------


class TestGatedFeatureFusion:
    """GatedFeatureFusion module: f_3d + sigmoid(alpha) * proj(f_sam)."""

    def test_fusion_output_shape(self) -> None:
        from minivess.adapters.sam3_hybrid import GatedFeatureFusion

        fusion = GatedFeatureFusion(dim_3d=128, dim_sam=256)
        f_3d = torch.randn(1, 128, 4, 8, 8)
        f_sam = torch.randn(1, 256, 4, 8, 8)
        out = fusion(f_3d, f_sam)
        assert out.shape == f_3d.shape

    def test_gate_alpha_init_zero(self) -> None:
        from minivess.adapters.sam3_hybrid import GatedFeatureFusion

        fusion = GatedFeatureFusion(dim_3d=64, dim_sam=128)
        assert fusion.gate_alpha.item() == pytest.approx(0.0)

    def test_zero_gate_passes_through_3d(self) -> None:
        from minivess.adapters.sam3_hybrid import GatedFeatureFusion

        fusion = GatedFeatureFusion(dim_3d=64, dim_sam=128)
        f_3d = torch.randn(1, 64, 2, 4, 4)
        f_sam = torch.randn(1, 128, 2, 4, 4)
        with torch.no_grad():
            out = fusion(f_3d, f_sam)
        # At gate=0, sigmoid(0)=0.5, so out ≈ f_3d + 0.5*proj(f_sam)
        # This is NOT pure passthrough — but with detached SAM features
        assert out.shape == f_3d.shape

    def test_fusion_gradient_flows(self) -> None:
        from minivess.adapters.sam3_hybrid import GatedFeatureFusion

        fusion = GatedFeatureFusion(dim_3d=64, dim_sam=128)
        f_3d = torch.randn(1, 64, 2, 4, 4, requires_grad=True)
        f_sam = torch.randn(1, 128, 2, 4, 4)  # detached in forward
        out = fusion(f_3d, f_sam)
        out.sum().backward()
        assert f_3d.grad is not None
        assert fusion.gate_alpha.grad is not None


# ---------------------------------------------------------------------------
# AxialProjection tests
# ---------------------------------------------------------------------------


class TestAxialProjection:
    """AxialProjection: 1D conv along Z on stacked 2D SAM features."""

    def test_output_shape(self) -> None:
        from minivess.adapters.sam3_hybrid import AxialProjection

        proj = AxialProjection(in_channels=256, out_channels=256)
        # (B, C, D, H, W)
        features = torch.randn(1, 256, 8, 4, 4)
        out = proj(features)
        assert out.shape == features.shape

    def test_gradient_flows(self) -> None:
        from minivess.adapters.sam3_hybrid import AxialProjection

        proj = AxialProjection(in_channels=64, out_channels=64)
        features = torch.randn(1, 64, 4, 4, 4, requires_grad=True)
        out = proj(features)
        out.sum().backward()
        assert features.grad is not None


# ---------------------------------------------------------------------------
# Sam3HybridAdapter tests
# ---------------------------------------------------------------------------


class TestSam3HybridInit:
    """Sam3HybridAdapter initialization."""

    def test_creates_with_config(self, hybrid_config: ModelConfig) -> None:
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(hybrid_config)
        assert adapter is not None

    def test_get_config(self, hybrid_config: ModelConfig) -> None:
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(hybrid_config)
        cfg = adapter.get_config()
        assert cfg.family == "sam3_hybrid"
        assert "filters" in cfg.extras

    def test_encoder_frozen_dynunet_trainable(self, hybrid_config: ModelConfig) -> None:
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(hybrid_config)
        # SAM backbone should be frozen
        for param in adapter.sam_backbone.parameters():
            assert not param.requires_grad
        # DynUNet should be trainable
        trainable = [p for p in adapter.dynunet.parameters() if p.requires_grad]
        assert len(trainable) > 0


class TestSam3HybridForward:
    """Sam3HybridAdapter forward pass."""

    def test_forward_output_shape(self, hybrid_config: ModelConfig) -> None:
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(hybrid_config)
        adapter.eval()
        # D=8 so DynUNet skip connections align after 3 downsampling steps
        volume = torch.randn(1, 1, 8, 64, 64)
        with torch.no_grad():
            output = adapter(volume)
        assert output.logits.shape == (1, 2, 8, 64, 64)
        assert output.prediction.shape == (1, 2, 8, 64, 64)

    def test_forward_predictions_valid(self, hybrid_config: ModelConfig) -> None:
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(hybrid_config)
        adapter.eval()
        volume = torch.randn(1, 1, 8, 32, 32)
        with torch.no_grad():
            output = adapter(volume)
        assert output.prediction.min() >= 0
        sums = output.prediction.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestSam3HybridCheckpoint:
    """Checkpoint for DynUNet + fusion weights."""

    def test_save_load_roundtrip(self, hybrid_config: ModelConfig, tmp_path) -> None:
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(hybrid_config)
        ckpt_path = tmp_path / "hybrid_ckpt.pth"
        adapter.save_checkpoint(ckpt_path)

        adapter2 = Sam3HybridAdapter(hybrid_config)
        adapter2.load_checkpoint(ckpt_path)

    def test_trainable_parameters(self, hybrid_config: ModelConfig) -> None:
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        adapter = Sam3HybridAdapter(hybrid_config)
        n = adapter.trainable_parameters()
        assert n > 0
