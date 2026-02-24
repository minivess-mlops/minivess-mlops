"""AtlasSegFM one-shot foundation model customization (Zhang et al., 2025).

Provides atlas registration infrastructure for one-shot adaptation of
segmentation foundation models to new anatomical targets. The atlas
serves as a spatial prior that guides segmentation without full fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AtlasRegistrationMethod(StrEnum):
    """Registration methods for atlas-to-target alignment."""

    AFFINE = "affine"
    DEFORMABLE = "deformable"
    LANDMARK = "landmark"


@dataclass
class AtlasConfig:
    """Configuration for atlas-guided segmentation.

    Parameters
    ----------
    atlas_name:
        Identifier for the atlas volume.
    registration_method:
        Method for atlas-to-target registration.
    spatial_dims:
        Number of spatial dimensions (2 or 3).
    num_atlas_channels:
        Number of atlas label channels injected as priors.
    """

    atlas_name: str
    registration_method: str = "affine"
    spatial_dims: int = 3
    num_atlas_channels: int = 1


@dataclass
class AtlasRegistrationResult:
    """Result of atlas-to-target registration.

    Parameters
    ----------
    warped_atlas:
        Atlas volume warped to target space.
    similarity_score:
        Normalized cross-correlation between warped atlas and target.
    method:
        Registration method used.
    deformation_field:
        Dense displacement field (only for deformable registration).
    """

    warped_atlas: NDArray
    similarity_score: float
    method: str
    deformation_field: NDArray | None = None


def _normalized_cross_correlation(a: NDArray, b: NDArray) -> float:
    """Compute normalized cross-correlation between two volumes."""
    a_norm = a - np.mean(a)
    b_norm = b - np.mean(b)
    denom = np.std(a) * np.std(b) * a.size
    if denom < 1e-10:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.sum(a_norm * b_norm) / denom)


def _affine_register(atlas: NDArray, target: NDArray) -> NDArray:
    """Simple affine registration via intensity-based alignment.

    Rescales atlas intensity range to match target statistics.
    For production use, replace with MONAI's AffineRegistration or ANTsPy.
    """
    atlas_min, atlas_max = atlas.min(), atlas.max()
    target_min, target_max = target.min(), target.max()

    if atlas_max - atlas_min < 1e-10:
        return np.full_like(target, np.mean(target))

    # Normalize atlas to [0, 1] then rescale to target range
    normalized = (atlas - atlas_min) / (atlas_max - atlas_min)
    warped = normalized * (target_max - target_min) + target_min
    return warped.astype(target.dtype)


def _deformable_register(
    atlas: NDArray, target: NDArray, *, seed: int | None = None,
) -> tuple[NDArray, NDArray]:
    """Simple deformable registration prototype.

    Applies affine registration followed by a small random deformation
    field. For production use, replace with MONAI's DenseFieldRegistration
    or VoxelMorph.
    """
    warped = _affine_register(atlas, target)

    rng = np.random.default_rng(seed)
    ndims = len(target.shape)
    deformation = rng.normal(0, 0.5, size=(*target.shape, ndims)).astype(
        np.float32,
    )
    return warped, deformation


def register_atlas(
    atlas: NDArray,
    target: NDArray,
    *,
    method: str = "affine",
    seed: int | None = None,
) -> AtlasRegistrationResult:
    """Register an atlas volume to a target volume.

    Parameters
    ----------
    atlas:
        Reference atlas volume.
    target:
        Target volume to register to.
    method:
        Registration method: "affine" or "deformable".
    seed:
        Random seed for deformable registration.
    """
    if method == "deformable":
        warped, deformation = _deformable_register(atlas, target, seed=seed)
        similarity = _normalized_cross_correlation(warped, target)
        return AtlasRegistrationResult(
            warped_atlas=warped,
            similarity_score=max(0.0, min(1.0, similarity)),
            method=method,
            deformation_field=deformation,
        )

    # Default: affine
    warped = _affine_register(atlas, target)
    similarity = _normalized_cross_correlation(warped, target)
    return AtlasRegistrationResult(
        warped_atlas=warped,
        similarity_score=max(0.0, min(1.0, similarity)),
        method="affine",
    )
