"""3D-to-2D slice adapter for DeepChecks Vision integration.

DeepChecks Vision expects 2D images (H, W, C). MinIVess data is 3D NIfTI
(D, H, W). This adapter extracts representative 2D slices for property-based
validation checks (brightness, contrast, label distribution).

Strategy options:
- 'middle': Middle axial slice (default -- fast, deterministic)
- 'max_foreground': Slice with most foreground voxels in label (slower, more informative)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def extract_representative_slices(
    volume: np.ndarray,
    label: np.ndarray | None = None,
    *,
    strategy: str = "middle",
) -> list[np.ndarray]:
    """Extract representative 2D slices from a 3D volume.

    Parameters
    ----------
    volume:
        3D array (X, Y, Z).
    label:
        Optional 3D label array (same shape as volume).
    strategy:
        'middle' for middle axial slice, 'max_foreground' for slice with
        most foreground voxels in label.

    Returns
    -------
    List of 2D arrays (X, Y). Empty list if volume is empty.
    """
    if volume.size == 0:
        return []

    if volume.ndim != 3:
        logger.warning("Expected 3D volume, got %dD — returning empty", volume.ndim)
        return []

    z_dim = volume.shape[2]

    if strategy == "max_foreground" and label is not None:
        # Find axial slice with most foreground voxels
        foreground_counts = [
            int(np.count_nonzero(label[:, :, z])) for z in range(z_dim)
        ]
        best_z = int(np.argmax(foreground_counts))
        return [volume[:, :, best_z]]

    # Default: middle axial slice
    mid_z = z_dim // 2
    return [volume[:, :, mid_z]]


def build_deepchecks_dataset(
    pairs: list[dict[str, str]],
    *,
    strategy: str = "middle",
) -> list[dict[str, Any]]:
    """Build a list of 2D image/label dicts from 3D NIfTI pairs.

    Parameters
    ----------
    pairs:
        Image/label pairs with path strings.
    strategy:
        Slice extraction strategy ('middle' or 'max_foreground').

    Returns
    -------
    List of dicts with 'image' (2D ndarray) and 'label' (2D ndarray) keys.
    """
    import nibabel as nib

    dataset: list[dict[str, Any]] = []

    for pair in pairs:
        try:
            img = nib.load(pair["image"])
            img_data = np.asarray(img.dataobj)  # type: ignore[attr-defined]

            lbl_data: np.ndarray | None = None
            if pair.get("label"):
                lbl = nib.load(pair["label"])
                lbl_data = np.asarray(lbl.dataobj)  # type: ignore[attr-defined]

            img_slices = extract_representative_slices(
                img_data, label=lbl_data, strategy=strategy
            )

            for i, img_slice in enumerate(img_slices):
                entry: dict[str, Any] = {"image": img_slice}
                if lbl_data is not None:
                    lbl_slices = extract_representative_slices(
                        lbl_data.astype(np.float32), strategy=strategy
                    )
                    if i < len(lbl_slices):
                        entry["label"] = lbl_slices[i]
                    else:
                        entry["label"] = np.zeros_like(img_slice, dtype=np.int16)
                else:
                    entry["label"] = np.zeros_like(img_slice, dtype=np.int16)

                dataset.append(entry)
        except Exception:
            logger.warning(
                "Failed to process pair %s",
                pair.get("image", "?"),
                exc_info=True,
            )

    return dataset
