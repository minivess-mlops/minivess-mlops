"""Dataset profiler: scan NIfTI volumes and compute safe patch sizes.

Provides VolumeStats (per-volume) and DatasetProfile (aggregate) along with
scan_volume, scan_dataset and compute_safe_patch_sizes utilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VolumeStats:
    """Statistics for a single NIfTI volume."""

    path: Path
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    size_bytes: int
    intensity_min: float
    intensity_max: float
    intensity_mean: float
    is_anisotropic: bool  # True if max_spacing / min_spacing > 2.0


@dataclass
class DatasetProfile:
    """Aggregated statistics for an entire dataset."""

    num_volumes: int
    min_shape: tuple[int, ...]
    max_shape: tuple[int, ...]
    median_shape: tuple[int, ...]
    min_spacing: tuple[float, ...]
    max_spacing: tuple[float, ...]
    median_spacing: tuple[float, ...]
    total_size_bytes: int
    volume_stats: list[VolumeStats]
    outlier_volumes: list[str] = field(default_factory=list)

    def to_mlflow_params(self) -> dict[str, str]:
        """Return flat dict of dataset params for ``mlflow.log_params()``.

        All keys are prefixed with ``data_``.
        """
        return {
            "data_n_volumes": str(self.num_volumes),
            "data_total_size_gb": f"{self.total_size_bytes / (1024**3):.2f}",
            "data_min_shape": str(self.min_shape),
            "data_max_shape": str(self.max_shape),
            "data_median_shape": str(self.median_shape),
            "data_min_spacing": str(self.min_spacing),
            "data_max_spacing": str(self.max_spacing),
            "data_median_spacing": str(self.median_spacing),
            "data_n_outlier_volumes": str(len(self.outlier_volumes)),
        }

    def to_json_dict(self) -> dict[str, object]:
        """Return JSON-serializable dict for artifact logging."""
        return {
            "num_volumes": self.num_volumes,
            "min_shape": list(self.min_shape),
            "max_shape": list(self.max_shape),
            "median_shape": list(self.median_shape),
            "min_spacing": list(self.min_spacing),
            "max_spacing": list(self.max_spacing),
            "median_spacing": list(self.median_spacing),
            "total_size_bytes": self.total_size_bytes,
            "total_size_gb": round(self.total_size_bytes / (1024**3), 2),
            "outlier_volumes": self.outlier_volumes,
        }


def scan_volume(path: Path) -> VolumeStats:
    """Read a NIfTI file and compute per-volume statistics.

    Parameters
    ----------
    path:
        Absolute path to a ``.nii`` or ``.nii.gz`` file.

    Returns
    -------
    VolumeStats
        Dataclass with shape, voxel spacing, intensity statistics and an
        ``is_anisotropic`` flag (True when max / min spacing ratio > 2.0).
    """
    img = nib.load(str(path))
    data: np.ndarray = np.asarray(img.dataobj, dtype=np.float32)

    # Shape as plain integers (avoids numpy int64 in frozen dataclass)
    shape: tuple[int, ...] = tuple(int(s) for s in data.shape)

    # Voxel spacings from the affine diagonal (absolute values, first 3 dims)
    zooms = img.header.get_zooms()
    spacing: tuple[float, ...] = tuple(float(abs(z)) for z in zooms[:3])

    # Intensity statistics over the raw float32 array
    intensity_min = float(data.min())
    intensity_max = float(data.max())
    intensity_mean = float(data.mean())

    # Raw voxel count * 4 bytes (float32) — uncompressed array size
    size_bytes = int(data.size * 4)

    # Anisotropy: ratio of max-to-min spacing across axes
    is_anisotropic = bool(max(spacing) / max(min(spacing), 1e-9) > 2.0)

    return VolumeStats(
        path=path,
        shape=shape,
        spacing=spacing,
        size_bytes=size_bytes,
        intensity_min=intensity_min,
        intensity_max=intensity_max,
        intensity_mean=intensity_mean,
        is_anisotropic=is_anisotropic,
    )


def scan_dataset(data_dir: Path) -> DatasetProfile:
    """Scan all NIfTI volumes under *data_dir* and return aggregated stats.

    Uses :func:`minivess.data.loader.discover_nifti_pairs` to locate
    image/label pairs, then calls :func:`scan_volume` on each image file.

    Parameters
    ----------
    data_dir:
        Root directory of the dataset (e.g. containing ``imagesTr/``).

    Returns
    -------
    DatasetProfile
        Aggregate statistics including min/max/median shapes and spacings, plus
        a list of outlier volume names (spacing deviates >3× from the median).
    """
    from minivess.data.loader import discover_nifti_pairs

    pairs = discover_nifti_pairs(data_dir)
    volume_stats: list[VolumeStats] = []

    for pair in pairs:
        img_path = Path(pair["image"])
        logger.debug("Scanning volume: %s", img_path)
        stats = scan_volume(img_path)
        volume_stats.append(stats)

    n = len(volume_stats)

    # --- Shape aggregation (element-wise) ---
    shapes_array = np.array([list(vs.shape) for vs in volume_stats], dtype=np.int64)
    min_shape: tuple[int, ...] = tuple(int(v) for v in shapes_array.min(axis=0))
    max_shape: tuple[int, ...] = tuple(int(v) for v in shapes_array.max(axis=0))
    median_shape: tuple[int, ...] = tuple(
        int(v) for v in np.median(shapes_array, axis=0).astype(np.int64)
    )

    # --- Spacing aggregation (element-wise) ---
    spacings_array = np.array(
        [list(vs.spacing) for vs in volume_stats], dtype=np.float64
    )
    min_spacing: tuple[float, ...] = tuple(float(v) for v in spacings_array.min(axis=0))
    max_spacing: tuple[float, ...] = tuple(float(v) for v in spacings_array.max(axis=0))
    median_spacing: tuple[float, ...] = tuple(
        float(v) for v in np.median(spacings_array, axis=0)
    )

    # --- Total size ---
    total_size_bytes = int(sum(vs.size_bytes for vs in volume_stats))

    # --- Outlier detection: any spacing axis > 3× median ---
    median_arr = np.median(spacings_array, axis=0)  # shape: (ndim_spacing,)
    outlier_volumes: list[str] = []
    for vs in volume_stats:
        sp = np.array(list(vs.spacing), dtype=np.float64)
        # Avoid division by zero for zero-spacing edge cases
        ratio = sp / np.maximum(median_arr, 1e-9)
        if np.any(ratio > 3.0) or np.any(ratio < 1.0 / 3.0):
            outlier_volumes.append(str(vs.path))

    return DatasetProfile(
        num_volumes=n,
        min_shape=min_shape,
        max_shape=max_shape,
        median_shape=median_shape,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        median_spacing=median_spacing,
        total_size_bytes=total_size_bytes,
        volume_stats=volume_stats,
        outlier_volumes=outlier_volumes,
    )


def compute_safe_patch_sizes(
    profile: DatasetProfile,
    model_divisor: int = 8,
    max_patch_xy: int = 128,
) -> tuple[int, int, int]:
    """Compute patch sizes that are safe for the entire dataset.

    Rules applied per dimension *d*:
    1. candidate = min_shape[d]  (must fit inside every volume)
    2. floor candidate to the nearest multiple of *model_divisor*
    3. If the floored value is 0 (i.e. min_shape < model_divisor), fall back
       to min_shape[d] unchanged — the caller is responsible for using a
       compatible model configuration.
    4. For the XY dimensions (0 and 1) also apply ``min(result, max_patch_xy)``
       to avoid excessive memory usage on large-FOV volumes.

    Parameters
    ----------
    profile:
        DatasetProfile returned by :func:`scan_dataset`.
    model_divisor:
        All patch dimensions must be divisible by this value (e.g. 8 for
        DynUNet with 4 pooling levels).
    max_patch_xy:
        Upper cap for the XY patch dimensions (default 128).

    Returns
    -------
    tuple[int, int, int]
        (patch_x, patch_y, patch_z) guaranteed > 0 and <= min_shape per dim.
    """

    def _floor_to_divisor(value: int, divisor: int) -> int:
        """Return the largest multiple of *divisor* that is <= *value*.

        If *value* < *divisor*, returns *value* itself (non-zero fallback).
        """
        floored = (value // divisor) * divisor
        return floored if floored > 0 else value

    min_x, min_y, min_z = (
        profile.min_shape[0],
        profile.min_shape[1],
        profile.min_shape[2],
    )

    patch_x = min(_floor_to_divisor(min_x, model_divisor), max_patch_xy)
    patch_y = min(_floor_to_divisor(min_y, model_divisor), max_patch_xy)
    patch_z = _floor_to_divisor(min_z, model_divisor)

    return (patch_x, patch_y, patch_z)
