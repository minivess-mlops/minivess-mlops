"""VessQC-style uncertainty-guided annotation curation.

Bridges UQ outputs (MC Dropout, Deep Ensemble) with annotation curation
workflow. Flags high-uncertainty regions for expert review and computes
error detection metrics following Terms et al. (2025).

Reference: Terms et al. (2025). "VessQC: Bridging Deep Learning Predictions
and Expert Curation." arxiv:2511.22236
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class CurationFlag:
    """A flagged region in a single sample requiring expert review.

    Parameters
    ----------
    sample_id:
        Identifier for the sample.
    voxel_count:
        Number of flagged voxels.
    mean_uncertainty:
        Mean uncertainty across flagged voxels.
    max_uncertainty:
        Maximum uncertainty in flagged region.
    volume_fraction:
        Fraction of total volume that is flagged (0-1).
    """

    sample_id: str
    voxel_count: int
    mean_uncertainty: float
    max_uncertainty: float
    volume_fraction: float


@dataclass
class CurationReport:
    """Summary of curation flagging across a batch.

    Parameters
    ----------
    flags:
        Per-sample CurationFlags.
    total_flagged_voxels:
        Total flagged voxels across all samples.
    total_voxels:
        Total voxels across all samples.
    flagged_fraction:
        Overall fraction of voxels flagged.
    uncertainty_threshold:
        Threshold used for flagging.
    """

    flags: list[CurationFlag]
    total_flagged_voxels: int
    total_voxels: int
    flagged_fraction: float
    uncertainty_threshold: float


def flag_uncertain_regions(
    uncertainty_maps: NDArray[np.float32],
    *,
    threshold: float | None = None,
    percentile: float = 95.0,
    sample_ids: list[str] | None = None,
) -> CurationReport:
    """Flag high-uncertainty regions for expert curation.

    Parameters
    ----------
    uncertainty_maps:
        Uncertainty maps (B, 1, D, H, W) from MC Dropout or Deep Ensemble.
    threshold:
        Explicit uncertainty threshold. If None, auto-computed from percentile.
    percentile:
        Percentile for auto-threshold (default: 95th = top 5% flagged).
    sample_ids:
        Sample identifiers. Auto-generated if None.

    Returns
    -------
    CurationReport with per-sample flags and summary statistics.
    """
    n_samples = uncertainty_maps.shape[0]

    if sample_ids is None:
        sample_ids = [f"sample_{i}" for i in range(n_samples)]

    # Compute threshold from percentile if not explicit
    if threshold is None:
        threshold = float(np.percentile(uncertainty_maps, percentile))

    # Squeeze channel dim: (B, 1, D, H, W) → (B, D, H, W)
    maps = uncertainty_maps[:, 0]
    spatial_size = int(np.prod(maps.shape[1:]))

    flags: list[CurationFlag] = []
    total_flagged = 0

    for i in range(n_samples):
        sample_map = maps[i]
        mask = sample_map > threshold
        voxel_count = int(mask.sum())
        total_flagged += voxel_count

        flagged_values = sample_map[mask]
        mean_unc = float(flagged_values.mean()) if voxel_count > 0 else 0.0
        max_unc = float(flagged_values.max()) if voxel_count > 0 else 0.0

        flags.append(
            CurationFlag(
                sample_id=sample_ids[i],
                voxel_count=voxel_count,
                mean_uncertainty=mean_unc,
                max_uncertainty=max_unc,
                volume_fraction=voxel_count / max(spatial_size, 1),
            )
        )

    total_voxels = n_samples * spatial_size

    return CurationReport(
        flags=flags,
        total_flagged_voxels=total_flagged,
        total_voxels=total_voxels,
        flagged_fraction=total_flagged / max(total_voxels, 1),
        uncertainty_threshold=threshold,
    )


def compute_error_detection_metrics(
    flagged_mask: NDArray[np.bool_],
    error_mask: NDArray[np.bool_],
) -> dict[str, float]:
    """Compute error detection recall and precision.

    Parameters
    ----------
    flagged_mask:
        Boolean mask of flagged voxels (B, D, H, W).
    error_mask:
        Boolean mask of ground truth annotation errors (B, D, H, W).

    Returns
    -------
    Dict with recall, precision, and f1 scores.
    """
    flagged_flat = flagged_mask.ravel()
    error_flat = error_mask.ravel()

    true_positives = int((flagged_flat & error_flat).sum())
    total_errors = int(error_flat.sum())
    total_flagged = int(flagged_flat.sum())

    # Recall: fraction of real errors that were flagged
    recall = true_positives / max(total_errors, 1) if total_errors > 0 else 1.0

    # Precision: fraction of flagged voxels that are real errors
    precision = true_positives / max(total_flagged, 1) if total_flagged > 0 else 1.0

    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "true_positives": true_positives,
        "total_errors": total_errors,
        "total_flagged": total_flagged,
    }


def rank_samples_by_uncertainty(
    uncertainty_maps: NDArray[np.float32],
    sample_ids: list[str] | None = None,
    *,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Rank samples by mean uncertainty for review prioritization.

    Parameters
    ----------
    uncertainty_maps:
        Uncertainty maps (B, 1, D, H, W).
    sample_ids:
        Sample identifiers. Auto-generated if None.
    top_k:
        Return only top-k most uncertain samples.

    Returns
    -------
    List of (sample_id, mean_uncertainty) sorted descending.
    """
    n_samples = uncertainty_maps.shape[0]

    if sample_ids is None:
        sample_ids = [f"sample_{i}" for i in range(n_samples)]

    # Mean uncertainty per sample (over all spatial dims)
    mean_per_sample = [
        float(uncertainty_maps[i].mean()) for i in range(n_samples)
    ]

    # Sort descending by uncertainty
    ranked = sorted(
        zip(sample_ids, mean_per_sample, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )

    if top_k is not None:
        ranked = ranked[:top_k]

    return ranked
