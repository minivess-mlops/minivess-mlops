"""Tests for SAM3 feature caching infrastructure.

Validates offline feature extraction, caching to disk, and
Sam3CachedFeatureDataset for training workflow.

IMPORTANT: Tests that instantiate Sam3Backbone (extract_and_cache_features)
require real SAM3 pretrained weights and are skipped in CI.
TestSam3CachedFeatureDataset tests that only need pre-saved .pt files
and run in CI.
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
        name="sam3-cache-test",
        in_channels=1,
        out_channels=2,
    )


@_sam3_skip
class TestExtractAndCache:
    """Offline feature extraction and disk caching."""

    def test_extract_features_creates_cache_dir(
        self, sam3_config: ModelConfig, tmp_path: Path
    ) -> None:
        from minivess.adapters.sam3_feature_cache import extract_and_cache_features

        cache_dir = tmp_path / "sam3_cache"
        volumes = {"vol_001": torch.randn(1, 1, 2, 64, 64)}
        extract_and_cache_features(
            config=sam3_config, volumes=volumes, cache_dir=cache_dir
        )
        assert cache_dir.exists()

    def test_extract_features_saves_pt_files(
        self, sam3_config: ModelConfig, tmp_path: Path
    ) -> None:
        from minivess.adapters.sam3_feature_cache import extract_and_cache_features

        cache_dir = tmp_path / "sam3_cache"
        volumes = {"vol_001": torch.randn(1, 1, 3, 64, 64)}
        extract_and_cache_features(
            config=sam3_config, volumes=volumes, cache_dir=cache_dir
        )
        pt_files = list(cache_dir.glob("*.pt"))
        assert len(pt_files) >= 1, "Should save at least one .pt file"


@_sam3_skip
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
            config=sam3_config, volumes=volumes, cache_dir=cache_dir
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
            config=sam3_config, volumes=volumes, cache_dir=cache_dir
        )
        loaded = load_cached_volume_features("vol_001", cache_dir)
        assert loaded.shape[0] == 1  # batch
        assert loaded.shape[1] == 1024  # embed_dim
        assert loaded.shape[2] == d  # depth preserved


class TestSam3CachedFeatureDataset:
    """PyTorch Dataset wrapping pre-cached features for training.

    These tests pre-create .pt files manually so they work in CI
    without SAM3 installed.
    """

    def test_cached_dataset_len_matches_volumes(self, tmp_path: Path) -> None:
        from minivess.adapters.sam3_feature_cache import Sam3CachedFeatureDataset

        cache_dir = tmp_path / "sam3_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Create dummy cached features
        torch.save(torch.randn(1, 1024, 2, 4, 4), cache_dir / "vol_001.pt")
        torch.save(torch.randn(1, 1024, 2, 4, 4), cache_dir / "vol_002.pt")

        dataset = Sam3CachedFeatureDataset(cache_dir=cache_dir)
        assert len(dataset) == 2

    def test_cached_dataset_getitem_returns_tuple(self, tmp_path: Path) -> None:
        from minivess.adapters.sam3_feature_cache import Sam3CachedFeatureDataset

        cache_dir = tmp_path / "sam3_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.randn(1, 1024, 2, 4, 4), cache_dir / "vol_001.pt")

        dataset = Sam3CachedFeatureDataset(cache_dir=cache_dir)
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_load_nonexistent_volume_raises(self, tmp_path: Path) -> None:
        from minivess.adapters.sam3_feature_cache import load_cached_volume_features

        cache_dir = tmp_path / "empty"
        cache_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError):
            load_cached_volume_features("no_such_vol", cache_dir)
