"""Signed distance transform utilities for conformal prediction.

Reusable distance functions for both distance-transform and morphological
conformal prediction metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import distance_transform_edt

if TYPE_CHECKING:
    from numpy.typing import NDArray


def signed_distance_transform(mask: NDArray) -> NDArray[np.float64]:
    """Compute signed distance transform: positive inside, negative outside.

    Parameters
    ----------
    mask:
        Binary mask (any shape).

    Returns
    -------
    SDT array with positive values inside, negative outside, ~0 at boundary.
    """
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return -distance_transform_edt(~mask_bool)
    if mask_bool.all():
        return distance_transform_edt(mask_bool)

    # Distance from boundary: positive inside, negative outside
    dist_inside = distance_transform_edt(mask_bool)
    dist_outside = distance_transform_edt(~mask_bool)
    return dist_inside - dist_outside


def boundary_distance(
    mask_a: NDArray,
    mask_b: NDArray,
) -> float:
    """Symmetric boundary distance between two binary masks.

    Computes the maximum of the two directed Hausdorff distances
    (symmetric Hausdorff distance) at the boundary.

    Parameters
    ----------
    mask_a:
        First binary mask.
    mask_b:
        Second binary mask.

    Returns
    -------
    Symmetric boundary distance in voxels.
    """
    a_bool = mask_a.astype(bool)
    b_bool = mask_b.astype(bool)

    if not a_bool.any() and not b_bool.any():
        return 0.0
    if not a_bool.any() or not b_bool.any():
        # One empty: distance is the max extent of the other
        nonempty = a_bool if a_bool.any() else b_bool
        return float(distance_transform_edt(~nonempty).max())

    # Distance from each mask to the boundary of the other
    d_a = _directed_hausdorff_mean(a_bool, b_bool)
    d_b = _directed_hausdorff_mean(b_bool, a_bool)
    return max(d_a, d_b)


def _directed_hausdorff_mean(
    source: NDArray[np.bool_],
    target: NDArray[np.bool_],
) -> float:
    """Directed Hausdorff: max distance from source boundary to nearest target."""
    # Distance from every voxel to nearest target voxel
    dist_to_target = distance_transform_edt(~target)
    # Only look at source boundary voxels
    source_vals = dist_to_target[source]
    return float(source_vals.max()) if source_vals.size > 0 else 0.0


def asymmetric_hausdorff_percentile(
    ground_truth: NDArray,
    prediction: NDArray,
    percentile: float = 95,
) -> float:
    """Asymmetric Hausdorff distance at given percentile.

    Measures the percentile of distances from GT boundary voxels to the
    nearest prediction boundary voxel. Robust to outliers when percentile < 100.

    Parameters
    ----------
    ground_truth:
        Binary ground truth mask.
    prediction:
        Binary prediction mask.
    percentile:
        Percentile (0-100) for robust distance.

    Returns
    -------
    Hausdorff distance at given percentile.
    """
    gt_bool = ground_truth.astype(bool)
    pred_bool = prediction.astype(bool)

    if not gt_bool.any() or not pred_bool.any():
        return 0.0

    # Distance from every voxel to nearest prediction voxel
    dist_to_pred = distance_transform_edt(~pred_bool)
    # Distances at GT voxels
    gt_distances = dist_to_pred[gt_bool]

    return float(np.percentile(gt_distances, percentile))
