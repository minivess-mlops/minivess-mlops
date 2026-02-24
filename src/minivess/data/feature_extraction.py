"""Image feature extraction for drift detection.

Extracts statistical features from 3D volumes that can be monitored
for distribution shifts using Evidently or other drift detectors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy.stats import entropy

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def extract_volume_features(volume: NDArray[np.float32]) -> dict[str, float]:
    """Extract statistical features from a single 3D volume.

    Parameters
    ----------
    volume:
        3D array of shape (D, H, W) as float32.

    Returns
    -------
    Dictionary of feature name → float value.
    """
    flat = volume.ravel()
    p5, p95 = float(np.percentile(flat, 5)), float(np.percentile(flat, 95))
    std = float(np.std(flat))

    # SNR: mean / std (avoid division by zero)
    snr = float(np.mean(flat)) / max(std, 1e-10)

    # Histogram entropy (binned)
    hist, _ = np.histogram(flat, bins=64, density=True)
    hist = hist + 1e-10  # avoid log(0)
    hist_entropy = float(entropy(hist))

    return {
        "mean": float(np.mean(flat)),
        "std": std,
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "p5": p5,
        "p95": p95,
        "snr": snr,
        "contrast": p95 - p5,
        "entropy": hist_entropy,
    }


def extract_batch_features(
    volumes: list[NDArray[np.float32]],
) -> pd.DataFrame:
    """Extract features from a batch of volumes.

    Parameters
    ----------
    volumes:
        List of 3D arrays.

    Returns
    -------
    DataFrame with one row per volume, columns are feature names.
    """
    records = [extract_volume_features(v) for v in volumes]
    return pd.DataFrame(records)


def compute_cl_dice_proxy(mask: NDArray[np.float32]) -> float:
    """Simplified clDice proxy measuring vessel connectivity.

    Uses connected component analysis as a proxy for centreline-based
    Dice. A well-connected vessel tree should have fewer connected
    components relative to its total foreground volume.

    Parameters
    ----------
    mask:
        Binary 3D mask (0/1 or float with threshold at 0.5).

    Returns
    -------
    Connectivity score in [0, 1]. Higher = more connected.
    """
    binary = (mask > 0.5).astype(np.uint8)
    foreground_count = int(binary.sum())

    if foreground_count == 0:
        return 0.0

    # Count connected components (3D, 26-connectivity)
    struct = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connectivity
    labeled, n_components = label(binary, structure=struct)

    if n_components == 0:
        return 0.0

    # Score: 1 component = perfect connectivity = 1.0
    # More components = less connectivity → lower score
    # Normalize: 1/n_components gives [0, 1] range
    return 1.0 / float(n_components)
