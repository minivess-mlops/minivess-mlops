"""Tests for D2C integration with MONAI data pipeline (T4 — #229)."""

from __future__ import annotations

from minivess.config.models import DataConfig
from minivess.data.disconnect_augmentation import DisconnectToConnectd
from minivess.data.transforms import build_train_transforms, build_val_transforms


class TestD2CIntegration:
    """Tests for D2C integration with transform pipeline."""

    def test_d2c_in_training_pipeline(self) -> None:
        """D2C transform present in training transforms when enabled."""
        config = DataConfig(
            dataset_name="test",
            d2c_enabled=True,
            d2c_probability=0.3,
        )
        transforms = build_train_transforms(config)
        transform_types = [type(t) for t in transforms.transforms]
        assert DisconnectToConnectd in transform_types

    def test_d2c_not_in_validation_pipeline(self) -> None:
        """D2C transform absent from validation transforms."""
        config = DataConfig(
            dataset_name="test",
            d2c_enabled=True,
        )
        transforms = build_val_transforms(config)
        transform_types = [type(t) for t in transforms.transforms]
        assert DisconnectToConnectd not in transform_types

    def test_d2c_before_random_crop(self) -> None:
        """D2C positioned before RandCropByPosNegLabeld."""
        from monai.transforms import RandCropByPosNegLabeld

        config = DataConfig(
            dataset_name="test",
            d2c_enabled=True,
        )
        transforms = build_train_transforms(config)
        transform_types = [type(t) for t in transforms.transforms]
        d2c_idx = transform_types.index(DisconnectToConnectd)
        crop_idx = transform_types.index(RandCropByPosNegLabeld)
        assert d2c_idx < crop_idx, "D2C must come before random cropping"

    def test_d2c_config_default_disabled(self) -> None:
        """Default config has D2C disabled."""
        config = DataConfig(dataset_name="test")
        assert config.d2c_enabled is False

    def test_d2c_config_enabled(self) -> None:
        """Enabled config adds D2C transform."""
        config = DataConfig(
            dataset_name="test",
            d2c_enabled=True,
            d2c_probability=0.5,
            d2c_mode="noise",
        )
        assert config.d2c_enabled is True
        assert config.d2c_probability == 0.5
        assert config.d2c_mode == "noise"

    def test_d2c_config_disabled(self) -> None:
        """Disabled config does not add D2C transform."""
        config = DataConfig(
            dataset_name="test",
            d2c_enabled=False,
        )
        transforms = build_train_transforms(config)
        transform_types = [type(t) for t in transforms.transforms]
        assert DisconnectToConnectd not in transform_types
