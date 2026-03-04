"""Signed Distance Field (SDF) ground truth generation from binary masks.

Uses scipy.ndimage.distance_transform_edt (library-first per CLAUDE.md Rule #3).
Sign convention: negative inside vessel, positive outside (Wu et al., SDF-TopoNet).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_sdf_from_mask(mask: np.ndarray) -> np.ndarray:
    """Compute signed distance field from a binary 3D mask.

    Args:
        mask: Binary 3D array (0 = background, >0 = foreground/vessel).

    Returns:
        SDF array (float32). Negative inside vessel, positive outside.
        Magnitude equals Euclidean distance to nearest boundary voxel.
    """
    binary = (mask > 0).astype(np.uint8)

    # Distance from each background voxel to nearest foreground voxel
    dist_outside = distance_transform_edt(1 - binary).astype(np.float32)

    # Distance from each foreground voxel to nearest background voxel
    dist_inside = distance_transform_edt(binary).astype(np.float32)

    # Sign convention: negative inside, positive outside
    sdf = dist_outside - dist_inside

    return np.asarray(sdf)


def normalize_sdf(sdf: np.ndarray, max_dist: float = 10.0) -> np.ndarray:
    """Clip SDF to [-max_dist, max_dist] and normalize to [-1, 1].

    Args:
        sdf: Signed distance field array.
        max_dist: Maximum distance for clipping before normalization.

    Returns:
        Normalized SDF in [-1, 1] range (float32).
    """
    clipped = np.clip(sdf, -max_dist, max_dist)
    normalized = (clipped / max_dist).astype(np.float32)
    return np.asarray(normalized)
