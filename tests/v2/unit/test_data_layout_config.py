"""Tests for configurable data layout keys (Code Review R4.3).

Validates that DataConfig supports custom image/label keys and that
transforms and loaders respect the configured keys.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# T1: DataConfig has configurable keys
# ---------------------------------------------------------------------------


class TestDataLayoutConfig:
    """Test that DataConfig supports configurable image/label keys."""

    def test_default_keys(self) -> None:
        """DataConfig should default to 'image' and 'label' keys."""
        from minivess.config.models import DataConfig

        config = DataConfig(dataset_name="test")
        assert config.image_key == "image"
        assert config.label_key == "label"

    def test_custom_keys(self) -> None:
        """DataConfig should accept custom image/label keys."""
        from minivess.config.models import DataConfig

        config = DataConfig(
            dataset_name="test",
            image_key="vol",
            label_key="seg",
        )
        assert config.image_key == "vol"
        assert config.label_key == "seg"


# ---------------------------------------------------------------------------
# T2: Transforms use configured keys
# ---------------------------------------------------------------------------


class TestTransformsUseConfiguredKeys:
    """Test that transform builders pass through the configured keys."""

    def test_train_transforms_use_custom_keys(self) -> None:
        """Train transforms should use keys from DataConfig."""
        from minivess.config.models import DataConfig
        from minivess.data.transforms import build_train_transforms

        config = DataConfig(dataset_name="test", image_key="vol", label_key="seg")
        transforms = build_train_transforms(config)

        # Verify the first transform (LoadImaged) uses custom keys
        load_transform = transforms.transforms[0]
        assert "vol" in load_transform.keys
        assert "seg" in load_transform.keys

    def test_val_transforms_use_custom_keys(self) -> None:
        """Val transforms should use keys from DataConfig."""
        from minivess.config.models import DataConfig
        from minivess.data.transforms import build_val_transforms

        config = DataConfig(dataset_name="test", image_key="vol", label_key="seg")
        transforms = build_val_transforms(config)

        load_transform = transforms.transforms[0]
        assert "vol" in load_transform.keys
        assert "seg" in load_transform.keys
