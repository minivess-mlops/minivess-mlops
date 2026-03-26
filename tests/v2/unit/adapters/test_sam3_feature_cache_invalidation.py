"""Tests for sam3_feature_cache invalidation — T23 regression test.

Bug: Sam3FeatureCache saves .pt files without model version metadata.
If the model changes, cached features become stale with no detection.
"""

from __future__ import annotations

import json
from pathlib import Path


class TestCacheInvalidation:
    """T23: Cache must track metadata for invalidation."""

    def test_cache_creates_metadata_file(self, tmp_path: Path):
        """_save_cache_metadata must create a .meta.json sidecar."""
        from minivess.adapters.sam3_feature_cache import _save_cache_metadata

        cache_path = tmp_path / "vol_001.pt"
        cache_path.touch()
        _save_cache_metadata(cache_path, "abc123", (1, 256, 3, 72, 72))

        meta_path = cache_path.with_suffix(".meta.json")
        assert meta_path.exists()
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        assert metadata["config_hash"] == "abc123"
        assert metadata["feature_shape"] == [1, 256, 3, 72, 72]
        assert "created_at" in metadata

    def test_cache_invalidated_when_config_changes(self, tmp_path: Path):
        """Validation should fail when config hash doesn't match."""
        from minivess.adapters.sam3_feature_cache import (
            _save_cache_metadata,
            _validate_cache_metadata,
        )

        cache_path = tmp_path / "vol_001.pt"
        cache_path.touch()
        _save_cache_metadata(cache_path, "old_hash", (1, 256, 3, 72, 72))

        assert not _validate_cache_metadata(cache_path, "new_hash")

    def test_cache_valid_when_config_matches(self, tmp_path: Path):
        """Validation should pass when config hash matches."""
        from minivess.adapters.sam3_feature_cache import (
            _save_cache_metadata,
            _validate_cache_metadata,
        )

        cache_path = tmp_path / "vol_001.pt"
        cache_path.touch()
        _save_cache_metadata(cache_path, "same_hash", (1, 256, 3, 72, 72))

        assert _validate_cache_metadata(cache_path, "same_hash")

    def test_cache_invalid_when_no_metadata(self, tmp_path: Path):
        """Old cache files without metadata should be treated as invalid."""
        from minivess.adapters.sam3_feature_cache import _validate_cache_metadata

        cache_path = tmp_path / "vol_001.pt"
        cache_path.touch()  # No metadata file

        assert not _validate_cache_metadata(cache_path, "any_hash")
