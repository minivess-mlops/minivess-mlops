"""Generate synthetic NIfTI files for testing data loading pipelines.

Creates small 3D volumes with realistic-ish properties:
- Images: random intensity with vessel-like bright structures
- Labels: binary masks with ~5-15% foreground (sparse vessels)
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def create_synthetic_nifti_dataset(
    root_dir: Path,
    n_volumes: int = 3,
    spatial_size: tuple[int, int, int] = (32, 32, 8),
    *,
    seed: int = 42,
) -> Path:
    """Create a synthetic NIfTI dataset in imagesTr/labelsTr layout.

    Parameters
    ----------
    root_dir:
        Parent directory (typically pytest tmp_path).
    n_volumes:
        Number of image/label pairs to generate.
    spatial_size:
        (X, Y, Z) dimensions for each volume.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Path to the dataset directory containing imagesTr/ and labelsTr/.
    """
    rng = np.random.default_rng(seed)

    data_dir = root_dir / "dataset"
    img_dir = data_dir / "imagesTr"
    lbl_dir = data_dir / "labelsTr"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    affine = np.eye(4)  # 1mm isotropic

    for i in range(1, n_volumes + 1):
        name = f"vol_{i:03d}.nii.gz"

        # Image: random uint16-like intensity
        img_data = rng.integers(0, 65535, size=spatial_size, dtype=np.uint16).astype(
            np.float32
        )
        img_nii = nib.Nifti1Image(img_data, affine)
        nib.save(img_nii, img_dir / name)

        # Label: sparse binary mask (~5-15% foreground)
        lbl_data = (rng.random(spatial_size) < rng.uniform(0.05, 0.15)).astype(
            np.uint8
        )
        lbl_nii = nib.Nifti1Image(lbl_data, affine)
        nib.save(lbl_nii, lbl_dir / name)

    return data_dir


def create_synthetic_nifti_dataset_ebrains(
    root_dir: Path,
    n_volumes: int = 3,
    spatial_size: tuple[int, int, int] = (32, 32, 8),
    *,
    seed: int = 42,
) -> Path:
    """Create a synthetic NIfTI dataset in EBRAINS raw/seg layout.

    EBRAINS MiniVess structure:
        raw/mv01.nii.gz, raw/mv02.nii.gz, ...
        seg/mv01_y.nii.gz, seg/mv02_y.nii.gz, ...

    Parameters
    ----------
    root_dir:
        Parent directory (typically pytest tmp_path).
    n_volumes:
        Number of image/label pairs to generate.
    spatial_size:
        (X, Y, Z) dimensions for each volume.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Path to the dataset directory containing raw/ and seg/.
    """
    rng = np.random.default_rng(seed)

    data_dir = root_dir / "dataset_ebrains"
    raw_dir = data_dir / "raw"
    seg_dir = data_dir / "seg"
    raw_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    affine = np.diag([0.001, 0.001, 0.001, 1.0])  # µm-scale spacing

    for i in range(1, n_volumes + 1):
        img_name = f"mv{i:02d}.nii.gz"
        lbl_name = f"mv{i:02d}_y.nii.gz"

        img_data = rng.integers(0, 65535, size=spatial_size, dtype=np.uint16).astype(
            np.float32
        )
        img_nii = nib.Nifti1Image(img_data, affine)
        nib.save(img_nii, raw_dir / img_name)

        lbl_data = (rng.random(spatial_size) < rng.uniform(0.05, 0.15)).astype(
            np.uint8
        )
        lbl_nii = nib.Nifti1Image(lbl_data, affine)
        nib.save(lbl_nii, seg_dir / lbl_name)

    return data_dir
