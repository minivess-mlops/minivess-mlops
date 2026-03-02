"""Tests for SAM2 backbone wrapper (SAM-01).

Validates Sam2Backbone initialization, feature extraction,
weight management, and optional dependency gating.
"""

from __future__ import annotations

import pytest
import torch

from minivess.config.models import ModelConfig, ModelFamily

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sam_config() -> ModelConfig:
    return ModelConfig(
        family=ModelFamily.SAM3_VANILLA,
        name="sam2-test",
        in_channels=1,
        out_channels=2,
    )


# ---------------------------------------------------------------------------
# Test: SAM2 constants and imports
# ---------------------------------------------------------------------------


class TestSam2Constants:
    """SAM2 variant registry and checkpoint metadata."""

    def test_sam2_variants_contains_hiera_tiny(self) -> None:
        from minivess.adapters.sam2_backbone import SAM2_VARIANTS

        assert "hiera_tiny" in SAM2_VARIANTS

    def test_sam2_checkpoint_urls_defined(self) -> None:
        from minivess.adapters.sam2_backbone import SAM2_CHECKPOINT_URLS

        assert "hiera_tiny" in SAM2_CHECKPOINT_URLS
        assert SAM2_CHECKPOINT_URLS["hiera_tiny"].startswith("https://")

    def test_sam2_checksums_defined(self) -> None:
        from minivess.adapters.sam2_backbone import SAM2_CHECKSUMS

        assert "hiera_tiny" in SAM2_CHECKSUMS
        assert len(SAM2_CHECKSUMS["hiera_tiny"]) == 64  # SHA256 hex

    def test_sam2_variants_feature_dims(self) -> None:
        from minivess.adapters.sam2_backbone import SAM2_VARIANTS

        assert SAM2_VARIANTS["hiera_tiny"]["embed_dim"] > 0
        assert SAM2_VARIANTS["hiera_tiny"]["num_params_m"] > 0


# ---------------------------------------------------------------------------
# Test: Sam2Backbone initialization (mocked — no real weights)
# ---------------------------------------------------------------------------


class TestSam2BackboneInit:
    """Sam2Backbone initializes with mock encoder (no real SAM2 download)."""

    def test_backbone_creates_with_variant(self, sam_config: ModelConfig) -> None:
        from minivess.adapters.sam2_backbone import Sam2Backbone

        backbone = Sam2Backbone(variant="hiera_tiny", pretrained=False)
        assert backbone.variant == "hiera_tiny"

    def test_backbone_embed_dim(self) -> None:
        from minivess.adapters.sam2_backbone import Sam2Backbone

        backbone = Sam2Backbone(variant="hiera_tiny", pretrained=False)
        assert backbone.embed_dim > 0

    def test_invalid_variant_raises(self) -> None:
        from minivess.adapters.sam2_backbone import Sam2Backbone

        with pytest.raises(ValueError, match="Unknown SAM2 variant"):
            Sam2Backbone(variant="nonexistent_model", pretrained=False)

    def test_backbone_is_frozen_by_default(self) -> None:
        from minivess.adapters.sam2_backbone import Sam2Backbone

        backbone = Sam2Backbone(variant="hiera_tiny", pretrained=False)
        for param in backbone.parameters():
            assert not param.requires_grad


# ---------------------------------------------------------------------------
# Test: Feature extraction
# ---------------------------------------------------------------------------


class TestSam2FeatureExtraction:
    """Sam2Backbone feature extraction from 2D slices."""

    def test_extract_features_shape(self) -> None:
        from minivess.adapters.sam2_backbone import Sam2Backbone

        backbone = Sam2Backbone(variant="hiera_tiny", pretrained=False)
        # SAM expects 1024x1024 input; test with smaller mock
        image_2d = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            features = backbone.extract_features(image_2d)
        assert features.ndim == 4  # (B, C, H', W')
        assert features.shape[0] == 1

    def test_get_image_embeddings_from_volume(self) -> None:
        """Slice-by-slice extraction from a 3D volume."""
        from minivess.adapters.sam2_backbone import Sam2Backbone

        backbone = Sam2Backbone(variant="hiera_tiny", pretrained=False)
        # Small volume: (B=1, C=1, D=4, H=64, W=64)
        volume = torch.randn(1, 1, 4, 64, 64)
        with torch.no_grad():
            embeddings = backbone.get_image_embeddings(volume)
        # Returns list of D feature maps
        assert len(embeddings) == 4
        for emb in embeddings:
            assert emb.ndim == 4  # (1, C, H', W')


# ---------------------------------------------------------------------------
# Test: Checksum verification
# ---------------------------------------------------------------------------


class TestChecksumVerification:
    """SHA256 checksum utility for weight integrity."""

    def test_verify_checksum_correct(self) -> None:
        from minivess.adapters.sam2_backbone import verify_checkpoint_integrity

        data = b"test data for checksum"
        import hashlib

        expected = hashlib.sha256(data).hexdigest()
        assert verify_checkpoint_integrity(data, expected)

    def test_verify_checksum_incorrect(self) -> None:
        from minivess.adapters.sam2_backbone import verify_checkpoint_integrity

        data = b"test data"
        assert not verify_checkpoint_integrity(data, "0" * 64)
