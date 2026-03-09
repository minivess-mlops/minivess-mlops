"""Tests for SAM3 backbone wrapper.

Validates Sam3Backbone: ViT-32L perception encoder wrapping,
feature extraction, and caching.

IMPORTANT: These tests require real SAM3 pretrained weights (GPU ≥16 GB).
They are skipped in CI where SAM3 is not installed. The _StubSam3Encoder
has been permanently removed — use pytest.mark.skipif for any test that
instantiates Sam3Backbone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from minivess.adapters.model_builder import _sam3_package_available
from minivess.config.models import ModelConfig, ModelFamily

if TYPE_CHECKING:
    from pathlib import Path

_sam3_skip = pytest.mark.skipif(
    not _sam3_package_available(), reason="SAM3 not installed"
)


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


_gpu_skip = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SAM3 backbone forward pass requires GPU (CPU takes >300s per encoder call)",
)


@_sam3_skip
@_gpu_skip
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.gpu_heavy
class TestSam3Backbone:
    """Sam3Backbone wrapper for the full perception encoder."""

    def test_backbone_creates(self, sam3_config: ModelConfig) -> None:
        from minivess.adapters.sam3_backbone import Sam3Backbone

        backbone = Sam3Backbone(config=sam3_config)
        assert backbone is not None

    def test_backbone_frozen_by_default(self, sam3_config: ModelConfig) -> None:
        from minivess.adapters.sam3_backbone import Sam3Backbone

        backbone = Sam3Backbone(config=sam3_config)
        for param in backbone.encoder.parameters():
            assert not param.requires_grad, "Encoder should be frozen by default"

    def test_backbone_feature_shape_matches_out_channels(
        self, sam3_config: ModelConfig
    ) -> None:
        from minivess.adapters.sam3_backbone import Sam3Backbone

        backbone = Sam3Backbone(config=sam3_config)
        x = torch.randn(1, 1, 1008, 1008)
        features = backbone.extract_features(x)
        assert features.shape[1] == backbone.out_channels, (
            f"extract_features dim {features.shape[1]} != out_channels {backbone.out_channels}"
        )

    def test_backbone_fpn_feature_shape_256(self, sam3_config: ModelConfig) -> None:
        from minivess.adapters.sam3_backbone import Sam3Backbone

        backbone = Sam3Backbone(config=sam3_config)
        x = torch.randn(1, 1, 1008, 1008)
        fpn_features = backbone.extract_fpn_features(x)
        assert fpn_features.shape[1] == backbone.fpn_channels

    def test_backbone_out_channels_property(self, sam3_config: ModelConfig) -> None:
        from minivess.adapters.sam3_backbone import Sam3Backbone

        backbone = Sam3Backbone(config=sam3_config)
        # HF path: out_channels == fpn_channels == 256 (FPN integrated)
        # Native path: out_channels == 1024, fpn_channels == 256
        assert backbone.out_channels in (256, 1024)
        assert backbone.fpn_channels == 256

    @pytest.mark.timeout(600)
    def test_backbone_get_volume_embeddings(self, sam3_config: ModelConfig) -> None:
        """Extract features for all Z-slices of a 3D volume.

        Uses only 2 slices to keep test feasible on CPU (~200s).
        """
        from minivess.adapters.sam3_backbone import Sam3Backbone

        backbone = Sam3Backbone(config=sam3_config)
        volume = torch.randn(1, 1, 2, 64, 64)
        embeddings = backbone.get_volume_embeddings(volume)
        assert embeddings.shape[0] == 1  # batch
        assert embeddings.shape[1] == backbone.out_channels  # channels
        assert embeddings.shape[2] == 2  # depth (Z slices)


@_sam3_skip
@_gpu_skip
@pytest.mark.slow
@pytest.mark.gpu_heavy
class TestSam3FeatureCache:
    """Offline feature caching for 8GB VRAM workflow."""

    def test_cache_save_load_roundtrip(
        self, sam3_config: ModelConfig, tmp_path: Path
    ) -> None:
        from minivess.adapters.sam3_backbone import Sam3Backbone

        backbone = Sam3Backbone(config=sam3_config)
        x = torch.randn(1, 1, 1008, 1008)
        features = backbone.extract_features(x)

        cache_path = tmp_path / "test_features.pt"
        backbone.save_cached_features(features, cache_path)
        assert cache_path.exists()

        loaded = backbone.load_cached_features(cache_path)
        assert torch.allclose(features, loaded, atol=1e-6)
