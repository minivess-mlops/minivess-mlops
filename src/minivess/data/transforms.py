from __future__ import annotations

from typing import TYPE_CHECKING

from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    Spacingd,
    SpatialPadd,
)

if TYPE_CHECKING:
    from minivess.config.models import DataConfig


def _build_spacing_transform(
    keys: list[str],
    voxel_spacing: tuple[float, float, float],
) -> Spacingd | None:
    """Build Spacingd only if voxel_spacing is not the no-op sentinel (0,0,0).

    MiniVess has heterogeneous voxel spacings (0.31–4.97 μm/voxel).
    Resampling outlier volumes (e.g. mv02 at 4.97 μm) to 1.0 μm isotropic
    creates 2545×2545×305 arrays (~8 GB each), causing OOM.

    Set ``voxel_spacing=(0, 0, 0)`` in DataConfig to skip resampling entirely
    and train at native resolution (recommended for this dataset).
    """
    if voxel_spacing == (0, 0, 0) or voxel_spacing == (0.0, 0.0, 0.0):
        return None
    return Spacingd(
        keys=keys,
        pixdim=voxel_spacing,
        mode=("bilinear", "nearest"),
    )


def build_train_transforms(config: DataConfig) -> Compose:
    """Build MONAI training transform chain with spatial augmentation."""
    ik, lk = config.image_key, config.label_key
    keys = [ik, lk]

    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
    ]

    spacing = _build_spacing_transform(keys, config.voxel_spacing)
    if spacing is not None:
        transforms.append(spacing)

    transforms.extend([
        NormalizeIntensityd(keys=ik, nonzero=True),
        SpatialPadd(keys=keys, spatial_size=config.patch_size),
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
    ])

    return Compose(transforms)


def build_val_transforms(config: DataConfig) -> Compose:
    """Build MONAI validation transform chain (no augmentation)."""
    ik, lk = config.image_key, config.label_key
    keys = [ik, lk]

    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
    ]

    spacing = _build_spacing_transform(keys, config.voxel_spacing)
    if spacing is not None:
        transforms.append(spacing)

    transforms.extend([
        NormalizeIntensityd(keys=ik, nonzero=True),
        # DynUNet with 4 levels has 3 stages of 2x downsampling → divisor = 2^3 = 8.
        # Full-volume validation requires all spatial dims divisible by 8.
        DivisiblePadd(keys=keys, k=8),
    ])

    return Compose(transforms)
