from __future__ import annotations

from typing import TYPE_CHECKING

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    Spacingd,
)

if TYPE_CHECKING:
    from minivess.config.models import DataConfig


def build_train_transforms(config: DataConfig) -> Compose:
    """Build MONAI training transform chain with spatial augmentation."""
    ik, lk = config.image_key, config.label_key
    keys = [ik, lk]
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Spacingd(
                keys=keys,
                pixdim=config.voxel_spacing,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=ik, nonzero=True),
            RandRotate90d(keys=keys, prob=0.3, spatial_axes=(0, 1)),
            RandFlipd(keys=keys, prob=0.3, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.3, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.3, spatial_axis=2),
            RandCropByPosNegLabeld(
                keys=keys,
                label_key=lk,
                spatial_size=config.patch_size,
                pos=1,
                neg=1,
                num_samples=4,
            ),
        ]
    )


def build_val_transforms(config: DataConfig) -> Compose:
    """Build MONAI validation transform chain (no augmentation)."""
    ik, lk = config.image_key, config.label_key
    keys = [ik, lk]
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Spacingd(
                keys=keys,
                pixdim=config.voxel_spacing,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=ik, nonzero=True),
        ]
    )
