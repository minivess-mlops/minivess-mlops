from __future__ import annotations

from typing import TYPE_CHECKING, Any

from monai.transforms import (  # type: ignore[attr-defined]
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
    from pathlib import Path

    from minivess.config.models import DataConfig
    from minivess.data.multitask_targets import AuxTargetConfig


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


def build_train_transforms(
    config: DataConfig,
    aux_configs: list[AuxTargetConfig] | None = None,
    precomputed_dir: Path | None = None,
) -> Compose:
    """Build MONAI training transform chain with spatial augmentation.

    Parameters
    ----------
    config:
        Data configuration with transform parameters.
    aux_configs:
        Optional list of auxiliary target configs for multi-task training.
        When provided, ``LoadAuxiliaryTargetsd`` is injected after loading
        and the auxiliary keys are included in all spatial transforms.
    precomputed_dir:
        Directory containing precomputed auxiliary NIfTI files.
    """
    ik, lk = config.image_key, config.label_key
    keys = [ik, lk]

    transforms: list[Any] = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
    ]

    spacing = _build_spacing_transform(keys, config.voxel_spacing)
    if spacing is not None:
        transforms.append(spacing)

    transforms.append(NormalizeIntensityd(keys=ik, nonzero=True))

    # Inject auxiliary target loading after image/label are loaded + normalized
    aux_keys: list[str] = []
    if aux_configs:
        from minivess.data.multitask_targets import LoadAuxiliaryTargetsd

        transforms.append(
            LoadAuxiliaryTargetsd(
                label_key=lk,
                aux_configs=aux_configs,
                precomputed_dir=precomputed_dir,
            )
        )
        aux_keys = [c.name for c in aux_configs]
        # Aux targets are raw (D,H,W) arrays — add channel dim for MONAI compat
        transforms.append(EnsureChannelFirstd(keys=aux_keys, channel_dim="no_channel"))

    # All spatial keys: image + label + any auxiliary targets
    spatial_keys = keys + aux_keys

    transforms.extend(
        [
            SpatialPadd(keys=spatial_keys, spatial_size=config.patch_size),
            RandRotate90d(keys=spatial_keys, prob=0.3, spatial_axes=(0, 1)),
            RandFlipd(keys=spatial_keys, prob=0.3, spatial_axis=0),
            RandFlipd(keys=spatial_keys, prob=0.3, spatial_axis=1),
            RandFlipd(keys=spatial_keys, prob=0.3, spatial_axis=2),
        ]
    )

    # D2C augmentation: operates on full volumes before random cropping
    if config.d2c_enabled:
        from minivess.data.disconnect_augmentation import DisconnectToConnectd

        transforms.append(
            DisconnectToConnectd(
                keys=[ik],
                label_key=lk,
                prob=config.d2c_probability,
                mode=config.d2c_mode,
                max_segment_length=config.d2c_max_segment_length,
                max_junctions=config.d2c_max_junctions,
                dilation_radius=config.d2c_dilation_radius,
            )
        )

    transforms.append(
        RandCropByPosNegLabeld(
            keys=spatial_keys,
            label_key=lk,
            spatial_size=config.patch_size,
            pos=1,
            neg=1,
            num_samples=4,
        ),
    )

    return Compose(transforms)


def build_val_transforms(
    config: DataConfig,
    aux_configs: list[AuxTargetConfig] | None = None,
    precomputed_dir: Path | None = None,
) -> Compose:
    """Build MONAI validation transform chain (no augmentation).

    Parameters
    ----------
    config:
        Data configuration with transform parameters.
    aux_configs:
        Optional list of auxiliary target configs for multi-task validation.
    precomputed_dir:
        Directory containing precomputed auxiliary NIfTI files.
    """
    ik, lk = config.image_key, config.label_key
    keys = [ik, lk]

    transforms: list[Any] = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
    ]

    spacing = _build_spacing_transform(keys, config.voxel_spacing)
    if spacing is not None:
        transforms.append(spacing)

    transforms.append(NormalizeIntensityd(keys=ik, nonzero=True))

    # Inject auxiliary target loading after image/label are loaded
    aux_keys: list[str] = []
    if aux_configs:
        from minivess.data.multitask_targets import LoadAuxiliaryTargetsd

        transforms.append(
            LoadAuxiliaryTargetsd(
                label_key=lk,
                aux_configs=aux_configs,
                precomputed_dir=precomputed_dir,
            )
        )
        aux_keys = [c.name for c in aux_configs]
        # Aux targets are raw (D,H,W) arrays — add channel dim for MONAI compat
        transforms.append(EnsureChannelFirstd(keys=aux_keys, channel_dim="no_channel"))

    spatial_keys = keys + aux_keys

    transforms.append(
        # DynUNet with 4 levels has 3 stages of 2x downsampling → divisor = 2^3 = 8.
        # Full-volume validation requires all spatial dims divisible by 8.
        DivisiblePadd(keys=spatial_keys, k=8),
    )

    return Compose(transforms)
