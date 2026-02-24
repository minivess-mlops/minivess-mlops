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

    Supports two layouts:

    1. **Medical Decathlon** (imagesTr/labelsTr):
        data_dir/imagesTr/*.nii.gz + data_dir/labelsTr/*.nii.gz
        Labels have matching filenames.

    2. **EBRAINS MiniVess** (raw/seg):
        data_dir/raw/mvXX.nii.gz + data_dir/seg/mvXX_y.nii.gz
        Labels have ``_y`` suffix before the extension.
    """
    # Layout 1: Medical Decathlon convention
    for img_dir_name in ("imagesTr", "images", "imagesTs"):
        img_dir = data_dir / img_dir_name
        if img_dir.exists():
            lbl_dir_name = img_dir_name.replace("images", "labels")
            lbl_dir = data_dir / lbl_dir_name
            if lbl_dir.exists():
                return _discover_matching_names(img_dir, lbl_dir)

    # Layout 2: EBRAINS raw/seg convention (labels have _y suffix)
    raw_dir = data_dir / "raw"
    seg_dir = data_dir / "seg"
    if raw_dir.exists() and seg_dir.exists():
        return _discover_suffix_labels(raw_dir, seg_dir, label_suffix="_y")

    msg = f"No image directory found in {data_dir}. Expected imagesTr/ or raw/"
    raise FileNotFoundError(msg)


def _discover_matching_names(
    img_dir: Path, lbl_dir: Path
) -> list[dict[str, str]]:
    """Discover pairs where image and label share the same filename."""
    pairs = []
    for img_path in sorted(img_dir.glob("*.nii.gz")):
        lbl_path = lbl_dir / img_path.name
        if lbl_path.exists():
            pairs.append({"image": str(img_path), "label": str(lbl_path)})

    if not pairs:
        msg = f"No matching NIfTI pairs found in {img_dir} + {lbl_dir}"
        raise FileNotFoundError(msg)

    return pairs


def _discover_suffix_labels(
    img_dir: Path,
    lbl_dir: Path,
    label_suffix: str = "_y",
) -> list[dict[str, str]]:
    """Discover pairs where labels have a suffix (e.g. mv01 → mv01_y)."""
    pairs = []
    for img_path in sorted(img_dir.glob("*.nii.gz")):
        stem = img_path.name.replace(".nii.gz", "")
        lbl_name = f"{stem}{label_suffix}.nii.gz"
        lbl_path = lbl_dir / lbl_name
        if lbl_path.exists():
            pairs.append({"image": str(img_path), "label": str(lbl_path)})

    if not pairs:
        msg = f"No matching NIfTI pairs found in {img_dir} + {lbl_dir}"
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
