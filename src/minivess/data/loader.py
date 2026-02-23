from __future__ import annotations

from typing import TYPE_CHECKING

from monai.data import CacheDataset, DataLoader, list_data_collate

from minivess.data.transforms import build_train_transforms, build_val_transforms

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.config.models import DataConfig

_DEFAULT_BATCH_SIZE: int = 2


def discover_nifti_pairs(data_dir: Path) -> list[dict[str, str]]:
    """Discover image/label NIfTI pairs from a directory.

    Expects structure:
        data_dir/
            images/ or imagesTr/   -> *.nii.gz
            labels/ or labelsTr/   -> *.nii.gz (matching filenames)
    """
    # Try standard naming conventions
    for img_dir_name in ("imagesTr", "images", "imagesTs"):
        img_dir = data_dir / img_dir_name
        if img_dir.exists():
            break
    else:
        msg = f"No image directory found in {data_dir}"
        raise FileNotFoundError(msg)

    lbl_dir_name = img_dir_name.replace("images", "labels")
    lbl_dir = data_dir / lbl_dir_name
    if not lbl_dir.exists():
        msg = f"Label directory {lbl_dir} not found"
        raise FileNotFoundError(msg)

    pairs = []
    for img_path in sorted(img_dir.glob("*.nii.gz")):
        lbl_path = lbl_dir / img_path.name
        if lbl_path.exists():
            pairs.append({"image": str(img_path), "label": str(lbl_path)})

    if not pairs:
        msg = f"No matching NIfTI pairs found in {data_dir}"
        raise FileNotFoundError(msg)

    return pairs


def create_train_loader(
    data_dicts: list[dict[str, str]],
    config: DataConfig,
    *,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    cache_rate: float = 0.5,
) -> DataLoader:
    """Create training DataLoader with MONAI CacheDataset.

    Parameters
    ----------
    data_dicts:
        List of {"image": path, "label": path} dictionaries.
    config:
        Data configuration with transform parameters.
    batch_size:
        Training batch size (default 2). Typically sourced from
        ``TrainingConfig.batch_size`` by the caller.
    cache_rate:
        Fraction of data to cache in memory (0.0-1.0).
    """
    transforms = build_train_transforms(config)
    dataset = CacheDataset(
        data=data_dicts,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=config.num_workers,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=list_data_collate,
    )


def create_val_loader(
    data_dicts: list[dict[str, str]],
    config: DataConfig,
    *,
    cache_rate: float = 1.0,
) -> DataLoader:
    """Create validation DataLoader with full caching.

    Parameters
    ----------
    data_dicts:
        List of {"image": path, "label": path} dictionaries.
    config:
        Data configuration with transform parameters.
    cache_rate:
        Fraction of data to cache in memory (0.0-1.0).
    """
    transforms = build_val_transforms(config)
    dataset = CacheDataset(
        data=data_dicts,
        transform=transforms,
        cache_rate=cache_rate,
        num_workers=config.num_workers,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
