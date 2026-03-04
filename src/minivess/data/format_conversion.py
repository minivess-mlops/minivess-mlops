"""Format conversion utilities — TIFF → NIfTI.

Uses ``tifffile`` for reading TIFF/OME-TIFF stacks and ``nibabel`` for
writing NIfTI-1 files. Voxel spacing is taken from the acquisition registry
(not from TIFF metadata, which is often unreliable for microscopy data).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import nibabel as nib

if TYPE_CHECKING:
    from pathlib import Path
import numpy as np
import tifffile

logger = logging.getLogger(__name__)

_TIFF_EXTENSIONS = frozenset({".tif", ".tiff"})


def convert_tiff_to_nifti(
    input_path: Path,
    output_path: Path,
    voxel_spacing: tuple[float, float, float],
    *,
    skip_existing: bool = False,
) -> Path:
    """Convert a single TIFF stack to NIfTI format.

    Parameters
    ----------
    input_path:
        Path to the input TIFF file.
    output_path:
        Path for the output NIfTI file (typically ``.nii.gz``).
    voxel_spacing:
        Voxel spacing in physical units ``(x, y, z)``.
    skip_existing:
        If True and output_path exists, skip conversion.

    Returns
    -------
    Path to the output NIfTI file.
    """
    if skip_existing and output_path.exists():
        logger.info("Skipping existing: %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = tifffile.imread(str(input_path))
    affine = np.diag([*voxel_spacing, 1.0])
    nii_img = nib.Nifti1Image(data, affine=affine)  # type: ignore[attr-defined, no-untyped-call]
    nib.save(nii_img, str(output_path))  # type: ignore[attr-defined]

    logger.info(
        "Converted %s → %s (shape=%s, spacing=%s)",
        input_path.name,
        output_path.name,
        data.shape,
        voxel_spacing,
    )
    return output_path


def convert_dataset_formats(
    dataset_name: str,
    input_dir: Path,
    output_dir: Path,
    voxel_spacing: tuple[float, float, float],
    *,
    skip_existing: bool = False,
) -> list[str]:
    """Batch-convert all TIFF files in a dataset to NIfTI.

    Walks ``images/`` and ``labels/`` subdirectories of ``input_dir``,
    converts each ``.tif``/``.tiff`` to ``.nii.gz`` in ``output_dir``.

    Parameters
    ----------
    dataset_name:
        Dataset identifier (for logging).
    input_dir:
        Root directory with ``images/`` and ``labels/`` subdirectories.
    output_dir:
        Output directory (mirrors ``images/``/``labels/`` structure).
    voxel_spacing:
        Voxel spacing for all volumes in this dataset.
    skip_existing:
        If True, skip files that already exist.

    Returns
    -------
    List of conversion log messages (empty if nothing to convert).
    """
    if not input_dir.is_dir():
        logger.warning("Input directory does not exist: %s", input_dir)
        return []

    log: list[str] = []

    for subdir_name in ("images", "labels"):
        subdir = input_dir / subdir_name
        if not subdir.is_dir():
            continue

        for tiff_path in sorted(subdir.iterdir()):
            if tiff_path.suffix.lower() not in _TIFF_EXTENSIONS:
                continue

            nifti_name = tiff_path.stem + ".nii.gz"
            out_path = output_dir / subdir_name / nifti_name

            convert_tiff_to_nifti(
                input_path=tiff_path,
                output_path=out_path,
                voxel_spacing=voxel_spacing,
                skip_existing=skip_existing,
            )

            msg = f"{dataset_name}: {tiff_path.name} → {nifti_name}"
            log.append(msg)
            logger.info(msg)

    return log
