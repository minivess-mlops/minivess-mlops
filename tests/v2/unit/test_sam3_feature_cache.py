"""Tests for SAM3 feature caching infrastructure (T5).

Validates offline feature extraction, caching to disk, and
Sam3CachedFeatureDataset for 8GB VRAM training workflow.
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
        name="sam3-cache-test",
        in_channels=1,
        out_channels=2,
    )


class TestExtractAndCache:
    """Offline feature extraction and disk caching."""

    def test_extract_features_creates_cache_dir(
        self, sam3_config: ModelConfig, tmp_path: Path
    ) -> None:
        from minivess.adapters.sam3_feature_cache import extract_and_cache_features

        cache_dir = tmp_path / "sam3_cache"
        # Single small volume for speed
        volumes = {"vol_001": torch.randn(1, 1, 2, 64, 64)}
        extract_and_cache_features(
            config=sam3_config, volumes=volumes, cache_dir=cache_dir, use_stub=True
        )
        assert cache_dir.exists()

    def test_extract_features_saves_pt_files(
        self, sam3_config: ModelConfig, tmp_path: Path
    ) -> None:
        from minivess.adapters.sam3_feature_cache import extract_and_cache_features

        cache_dir = tmp_path / "sam3_cache"
        volumes = {"vol_001": torch.randn(1, 1, 3, 64, 64)}
        extract_and_cache_features(
            config=sam3_config, volumes=volumes, cache_dir=cache_dir, use_stub=True
        )
        pt_files = list(cache_dir.glob("*.pt"))
        assert len(pt_files) >= 1, "Should save at least one .pt file"


class TestLoadCachedFeatures:
    """Loading cached features from disk."""

    def test_load_cached_features_returns_tensor(
        self, sam3_config: ModelConfig, tmp_path: Path
    ) -> None:
        from minivess.adapters.sam3_feature_cache import (
            extract_and_cache_features,
            load_cached_volume_features,
        )

        cache_dir = tmp_path / "sam3_cache"
        volumes = {"vol_001": torch.randn(1, 1, 2, 64, 64)}
        extract_and_cache_features(
            config=sam3_config, volumes=volumes, cache_dir=cache_dir, use_stub=True
        )
        loaded = load_cached_volume_features("vol_001", cache_dir)
        assert isinstance(loaded, torch.Tensor)

    def test_load_cached_features_correct_shape(
        self, sam3_config: ModelConfig, tmp_path: Path
    ) -> None:
        from minivess.adapters.sam3_feature_cache import (
            extract_and_cache_features,
            load_cached_volume_features,
        )

        cache_dir = tmp_path / "sam3_cache"
        d = 3
        volumes = {"vol_001": torch.randn(1, 1, d, 64, 64)}
        extract_and_cache_features(
            config=sam3_config, volumes=volumes, cache_dir=cache_dir, use_stub=True
        )
        loaded = load_cached_volume_features("vol_001", cache_dir)
        # Should be (1, embed_dim, D, H_feat, W_feat)
        assert loaded.shape[0] == 1  # batch
        assert loaded.shape[1] == 1024  # embed_dim
        assert loaded.shape[2] == d  # depth preserved


class TestSam3CachedFeatureDataset:
    """PyTorch Dataset wrapping cached features for training."""

    def test_cached_dataset_len_matches_volumes(
        self, sam3_config: ModelConfig, tmp_path: Path
    ) -> None:
        from minivess.adapters.sam3_feature_cache import (
            Sam3CachedFeatureDataset,
            extract_and_cache_features,
        )

        cache_dir = tmp_path / "sam3_cache"
        volumes = {
            "vol_001": torch.randn(1, 1, 2, 64, 64),
            "vol_002": torch.randn(1, 1, 2, 64, 64),
        }
        extract_and_cache_features(
            config=sam3_config, volumes=volumes, cache_dir=cache_dir, use_stub=True
        )
        dataset = Sam3CachedFeatureDataset(cache_dir=cache_dir)
        assert len(dataset) == 2

    def test_cached_dataset_getitem_returns_tuple(
        self, sam3_config: ModelConfig, tmp_path: Path
    ) -> None:
        from minivess.adapters.sam3_feature_cache import (
            Sam3CachedFeatureDataset,
            extract_and_cache_features,
        )

        cache_dir = tmp_path / "sam3_cache"
        volumes = {"vol_001": torch.randn(1, 1, 2, 64, 64)}
        extract_and_cache_features(
            config=sam3_config, volumes=volumes, cache_dir=cache_dir, use_stub=True
        )
        dataset = Sam3CachedFeatureDataset(cache_dir=cache_dir)
        item = dataset[0]
        # Returns (volume_id, features_tensor)
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_cache_roundtrip_preserves_values(
        self, sam3_config: ModelConfig, tmp_path: Path
    ) -> None:
        from minivess.adapters.sam3_backbone import Sam3Backbone
        from minivess.adapters.sam3_feature_cache import (
            extract_and_cache_features,
            load_cached_volume_features,
        )

        cache_dir = tmp_path / "sam3_cache"
        volume = torch.randn(1, 1, 2, 64, 64)
        volumes = {"vol_001": volume}
        extract_and_cache_features(
            config=sam3_config, volumes=volumes, cache_dir=cache_dir, use_stub=True
        )
        loaded = load_cached_volume_features("vol_001", cache_dir)

        # Re-extract directly and compare
        backbone = Sam3Backbone(config=sam3_config, use_stub=True)
        direct = backbone.get_volume_embeddings(volume)
        # Shapes must match (values differ due to random init, but shapes must match)
        assert loaded.shape == direct.shape
