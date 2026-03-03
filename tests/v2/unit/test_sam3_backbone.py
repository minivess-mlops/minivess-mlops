"""Tests for SAM3 backbone wrapper (T1).

Validates Sam3Backbone: ViT-32L perception encoder wrapping,
stub encoder for testing, feature extraction, and caching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from minivess.config.models import ModelConfig, ModelFamily

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def sam3_config() -> ModelConfig:
    """Minimal SAM3 config for testing."""
    return ModelConfig(
        family=ModelFamily.SAM3_VANILLA,
        name="sam3-test",
        in_channels=1,
        out_channels=2,
    )


class TestSam3Constants:
    """SAM3 model constants and metadata."""

    def test_sam3_input_size_is_1008(self) -> None:
        from minivess.adapters.sam3_backbone import SAM3_INPUT_SIZE

        assert SAM3_INPUT_SIZE == 1008

    def test_sam3_checkpoint_info_defined(self) -> None:
        from minivess.adapters.sam3_backbone import SAM3_CKPT_NAME, SAM3_HF_MODEL_ID

        assert SAM3_HF_MODEL_ID == "facebook/sam3"
        assert SAM3_CKPT_NAME == "sam3.pt"

    def test_sam3_normalization_constants(self) -> None:
        from minivess.adapters.sam3_backbone import SAM3_IMAGE_MEAN, SAM3_IMAGE_STD

        assert SAM3_IMAGE_MEAN == (0.5, 0.5, 0.5)
        assert SAM3_IMAGE_STD == (0.5, 0.5, 0.5)


class TestStubSam3Encoder:
    """Stub encoder for testing without SAM3 package installed."""

    def test_stub_encoder_creates(self) -> None:
        from minivess.adapters.sam3_backbone import _StubSam3Encoder

        encoder = _StubSam3Encoder(embed_dim=1024)
        assert encoder is not None

    def test_stub_encoder_output_shape_1024(self) -> None:
        from minivess.adapters.sam3_backbone import _StubSam3Encoder

        encoder = _StubSam3Encoder(embed_dim=1024)
        # Input: (B, 3, 1008, 1008)
        x = torch.randn(1, 3, 1008, 1008)
        features = encoder(x)
        # ViT-32L output: (B, 1024, H_feat, W_feat) where H=W=1008/14=72
        assert features.shape[0] == 1
        assert features.shape[1] == 1024
        # Feature map spatial dims depend on patch_size
        assert features.shape[2] > 0
        assert features.shape[3] > 0

    def test_stub_encoder_grayscale_to_3ch(self) -> None:
        from minivess.adapters.sam3_backbone import _StubSam3Encoder

        encoder = _StubSam3Encoder(embed_dim=1024)
        # Grayscale input: (B, 1, 1008, 1008) — should be expanded to 3ch internally
        x = torch.randn(1, 1, 1008, 1008)
        features = encoder(x)
        assert features.shape[1] == 1024

    def test_stub_encoder_native_resolution(self) -> None:
        from minivess.adapters.sam3_backbone import _StubSam3Encoder

        encoder = _StubSam3Encoder(embed_dim=1024)
        # Stub operates at native resolution (no 1008x1008 upscale)
        x = torch.randn(1, 3, 56, 56)
        features = encoder(x)
        assert features.shape[1] == 1024
        # 56 / 14 = 4 (native patch embedding stride)
        assert features.shape[2] == 4
        assert features.shape[3] == 4


class TestSam3Backbone:
    """Sam3Backbone wrapper for the full perception encoder."""

    def test_backbone_creates_with_stub(self, sam3_config: ModelConfig) -> None:
        from minivess.adapters.sam3_backbone import Sam3Backbone

        backbone = Sam3Backbone(config=sam3_config, use_stub=True)
        assert backbone is not None

    def test_backbone_frozen_by_default(self, sam3_config: ModelConfig) -> None:
        from minivess.adapters.sam3_backbone import Sam3Backbone

        backbone = Sam3Backbone(config=sam3_config, use_stub=True)
        for param in backbone.encoder.parameters():
            assert not param.requires_grad, "Encoder should be frozen by default"

    def test_backbone_feature_shape_1024(self, sam3_config: ModelConfig) -> None:
        from minivess.adapters.sam3_backbone import Sam3Backbone

        backbone = Sam3Backbone(config=sam3_config, use_stub=True)
        x = torch.randn(1, 1, 1008, 1008)
        features = backbone.extract_features(x)
        assert features.shape[1] == 1024, "ViT-32L produces 1024-dim features"

    def test_backbone_fpn_feature_shape_256(self, sam3_config: ModelConfig) -> None:
        from minivess.adapters.sam3_backbone import Sam3Backbone

        backbone = Sam3Backbone(config=sam3_config, use_stub=True)
        x = torch.randn(1, 1, 1008, 1008)
        fpn_features = backbone.extract_fpn_features(x)
        # FPN neck outputs 256-dim features
        assert fpn_features.shape[1] == 256

    def test_backbone_out_channels_property(self, sam3_config: ModelConfig) -> None:
        from minivess.adapters.sam3_backbone import Sam3Backbone

        backbone = Sam3Backbone(config=sam3_config, use_stub=True)
        assert backbone.out_channels == 1024
        assert backbone.fpn_channels == 256

    def test_backbone_get_volume_embeddings(self, sam3_config: ModelConfig) -> None:
        """Extract features for all Z-slices of a 3D volume."""
        from minivess.adapters.sam3_backbone import Sam3Backbone

        backbone = Sam3Backbone(config=sam3_config, use_stub=True)
        # 3D volume: (B, C, D, H, W) = (1, 1, 4, 64, 64)
        volume = torch.randn(1, 1, 4, 64, 64)
        embeddings = backbone.get_volume_embeddings(volume)
        # Should return per-slice features stacked along D
        assert embeddings.shape[0] == 1  # batch
        assert embeddings.shape[1] == 1024  # channels
        assert embeddings.shape[2] == 4  # depth (Z slices)


class TestSam3FeatureCache:
    """Offline feature caching for 8GB VRAM workflow."""

    def test_cache_save_load_roundtrip(
        self, sam3_config: ModelConfig, tmp_path: Path
    ) -> None:
        from minivess.adapters.sam3_backbone import Sam3Backbone

        backbone = Sam3Backbone(config=sam3_config, use_stub=True)
        x = torch.randn(1, 1, 1008, 1008)
        features = backbone.extract_features(x)

        # Save
        cache_path = tmp_path / "test_features.pt"
        backbone.save_cached_features(features, cache_path)
        assert cache_path.exists()

        # Load
        loaded = backbone.load_cached_features(cache_path)
        assert torch.allclose(features, loaded, atol=1e-6)
