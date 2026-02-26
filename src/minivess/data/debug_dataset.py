"""Debug dataset generator for development and testing.

Creates synthetic NIfTI volumes in EBRAINS MiniVess layout (raw/ + seg/).
Used for debugging evaluation pipelines while waiting for real external test sets.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def create_debug_dataset(
    output_dir: Path,
    *,
    n_volumes: int = 4,
    volume_shape: tuple[int, int, int] = (64, 64, 16),
    seed: int = 42,
) -> Path:
    """Create a synthetic debug dataset with random volumes and binary labels.

    Creates NIfTI files in EBRAINS MiniVess layout (raw/ + seg/) with
    ``_y`` suffix labels, compatible with ``discover_nifti_pairs()``.

    Parameters
    ----------
    output_dir:
        Root directory to create the dataset in.
    n_volumes:
        Number of synthetic volumes to create.
    volume_shape:
        Shape of each volume (X, Y, Z).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Path to the created dataset directory.
    """
    rng = np.random.default_rng(seed)

    raw_dir = output_dir / "raw"
    seg_dir = output_dir / "seg"
    raw_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    affine = np.eye(4)

    for i in range(n_volumes):
        stem = f"debug{i:02d}"

        # Create random image volume (float32, similar to microscopy data)
        image = rng.random(volume_shape, dtype=np.float32)
        img_nii = nib.Nifti1Image(image, affine)
        nib.save(img_nii, str(raw_dir / f"{stem}.nii.gz"))

        # Create binary label with vessel-like tube structures
        label = _create_vessel_label(rng, volume_shape)
        label_nii = nib.Nifti1Image(label, affine)
        nib.save(label_nii, str(seg_dir / f"{stem}_y.nii.gz"))

    logger.info("Created debug dataset with %d volumes at %s", n_volumes, output_dir)
    return output_dir


def _create_vessel_label(
    rng: np.random.Generator,
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    """Create a binary label volume with random vessel-like tube structures.

    Generates 3 random tubes that wander through the volume by random-walking
    a center point and painting a small radius around it at each Z-slice.

    Parameters
    ----------
    rng:
        NumPy random generator instance.
    volume_shape:
        (X, Y, Z) dimensions of the volume.

    Returns
    -------
    Binary uint8 array with vessel-like foreground (0/1 only).
    """
    label = np.zeros(volume_shape, dtype=np.uint8)
    shape_arr = np.array(volume_shape)

    for _ in range(3):
        # Random starting point in the first half of the volume
        start = rng.integers(0, shape_arr // 2, size=3)
        pos = start.copy()

        for z in range(volume_shape[2]):
            radius = int(rng.integers(2, 5))
            y_lo = max(0, pos[1] - radius)
            y_hi = min(volume_shape[1], pos[1] + radius)
            x_lo = max(0, pos[0] - radius)
            x_hi = min(volume_shape[0], pos[0] + radius)
            label[x_lo:x_hi, y_lo:y_hi, z] = 1

            # Random walk the position
            pos = pos + rng.integers(-2, 3, size=3)
            pos = np.clip(pos, 0, shape_arr - 1)

    return label
