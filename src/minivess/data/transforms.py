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
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"],
                pixdim=config.voxel_spacing,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True),
            RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(0, 1)),
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=2),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=config.patch_size,
                pos=1,
                neg=1,
                num_samples=4,
            ),
        ]
    )


def build_val_transforms(config: DataConfig) -> Compose:
    """Build MONAI validation transform chain (no augmentation)."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"],
                pixdim=config.voxel_spacing,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True),
        ]
    )
