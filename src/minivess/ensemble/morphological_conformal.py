"""Morphological conformal prediction for 3D segmentation.

Produces inner/outer contour prediction sets via calibrated morphological
dilation and erosion, providing spatially meaningful prediction bands with
distribution-free coverage guarantees.

Based on: Mossina & Friedrich (2025), "ConSeMa: Conformal Semantic Image
Segmentation with Mask-level Guarantees," MICCAI 2025.

Extended to 3D with ball structuring elements for vascular segmentation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class MorphologicalConformalResult:
    """Result of morphological conformal prediction.

    Parameters
    ----------
    inner_contour:
        High-confidence vessel core (eroded prediction).
    outer_contour:
        Maximum vessel extent (dilated prediction).
    band_width:
        Mean prediction band width in voxels.
    dilation_radius:
        Calibrated dilation radius (iterations).
    erosion_radius:
        Calibrated erosion radius (iterations).
    alpha:
        Significance level used.
    """

    inner_contour: NDArray[np.bool_]
    outer_contour: NDArray[np.bool_]
    band_width: float
    dilation_radius: int
    erosion_radius: int
    alpha: float


def _covers(dilated: NDArray, ground_truth: NDArray) -> bool:
    """Check if dilated mask fully covers ground truth."""
    gt_bool = ground_truth.astype(bool)
    if not gt_bool.any():
        return True
    return bool(np.all(gt_bool <= dilated))


def _contained_in(eroded: NDArray, ground_truth: NDArray) -> bool:
    """Check if eroded mask is fully contained in ground truth."""
    eroded_bool = eroded.astype(bool)
    if not eroded_bool.any():
        return True
    return bool(np.all(eroded_bool <= ground_truth.astype(bool)))


def _find_min_dilation(
    prediction: NDArray,
    ground_truth: NDArray,
    structuring_element: NDArray,
    max_radius: int,
) -> int:
    """Find minimal dilation iterations for prediction to cover ground truth."""
    pred_bool = prediction.astype(bool)
    if _covers(pred_bool, ground_truth):
        return 0

    dilated = pred_bool.copy()
    for lam in range(1, max_radius + 1):
        dilated = binary_dilation(dilated, structure=structuring_element)
        if _covers(dilated, ground_truth):
            return lam

    return max_radius


def _find_min_erosion(
    prediction: NDArray,
    ground_truth: NDArray,
    structuring_element: NDArray,
    max_radius: int,
) -> int:
    """Find minimal erosion iterations for eroded prediction to be inside GT."""
    pred_bool = prediction.astype(bool)
    if _contained_in(pred_bool, ground_truth):
        return 0

    eroded = pred_bool.copy()
    for mu in range(1, max_radius + 1):
        eroded = binary_erosion(eroded, structure=structuring_element)
        if _contained_in(eroded, ground_truth):
            return mu

    return max_radius


class MorphologicalConformalPredictor:
    """Morphological conformal prediction for binary 3D segmentation.

    Calibrates dilation and erosion radii on holdout volumes to produce
    inner/outer contour prediction sets with coverage guarantees.

    Parameters
    ----------
    alpha:
        Significance level (e.g., 0.1 for 90% coverage).
    max_radius:
        Maximum dilation/erosion iterations to try.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        max_radius: int = 20,
    ) -> None:
        self.alpha = alpha
        self.max_radius = max_radius
        self._dilation_radius: int | None = None
        self._erosion_radius: int | None = None
        self._struct_elem: NDArray | None = None

    @property
    def is_calibrated(self) -> bool:
        """Whether calibrate() has been called."""
        return self._dilation_radius is not None

    @property
    def dilation_radius(self) -> int:
        """Calibrated dilation radius."""
        if self._dilation_radius is None:
            msg = "Not calibrated yet"
            raise RuntimeError(msg)
        return self._dilation_radius

    @property
    def erosion_radius(self) -> int:
        """Calibrated erosion radius."""
        if self._erosion_radius is None:
            msg = "Not calibrated yet"
            raise RuntimeError(msg)
        return self._erosion_radius

    def calibrate(
        self,
        predictions: list[NDArray],
        labels: list[NDArray],
    ) -> None:
        """Calibrate dilation and erosion radii on holdout volumes.

        Parameters
        ----------
        predictions:
            List of binary prediction masks (D, H, W).
        labels:
            List of binary ground truth masks (D, H, W).
        """
        if not predictions:
            msg = "Need at least one calibration volume"
            raise ValueError(msg)

        ndim = predictions[0].ndim
        self._struct_elem = generate_binary_structure(rank=ndim, connectivity=1)

        # Find minimal dilation for each calibration volume
        dilation_lambdas: list[int] = []
        erosion_mus: list[int] = []

        for pred, gt in zip(predictions, labels, strict=True):
            lam = _find_min_dilation(pred, gt, self._struct_elem, self.max_radius)
            dilation_lambdas.append(lam)

            mu = _find_min_erosion(pred, gt, self._struct_elem, self.max_radius)
            erosion_mus.append(mu)

        # Quantile with finite-sample correction
        n = len(dilation_lambdas)
        level = min(math.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)

        # For dilation: take the quantile of the lambdas
        self._dilation_radius = int(np.quantile(dilation_lambdas, min(level, 1.0)))
        # For erosion: take the quantile of the mus
        self._erosion_radius = int(np.quantile(erosion_mus, min(level, 1.0)))

        # Cap at max_radius
        self._dilation_radius = min(self._dilation_radius, self.max_radius)
        self._erosion_radius = min(self._erosion_radius, self.max_radius)

        logger.info(
            "Morphological conformal calibrated: dilation=%d, erosion=%d, "
            "alpha=%.2f, n_volumes=%d",
            self._dilation_radius,
            self._erosion_radius,
            self.alpha,
            n,
        )

    def predict(
        self,
        prediction: NDArray,
    ) -> MorphologicalConformalResult:
        """Produce inner/outer contour prediction sets.

        Parameters
        ----------
        prediction:
            Binary prediction mask (D, H, W).

        Returns
        -------
        MorphologicalConformalResult with inner/outer contours.
        """
        if not self.is_calibrated:
            msg = "Must calibrate() before predict()"
            raise RuntimeError(msg)

        pred_bool = prediction.astype(bool)

        # Outer contour: dilate by calibrated radius
        if self._dilation_radius > 0:
            outer = binary_dilation(
                pred_bool,
                structure=self._struct_elem,
                iterations=self._dilation_radius,
            )
        else:
            outer = pred_bool.copy()

        # Inner contour: erode by calibrated radius
        if self._erosion_radius > 0:
            inner = binary_erosion(
                pred_bool,
                structure=self._struct_elem,
                iterations=self._erosion_radius,
            )
        else:
            inner = pred_bool.copy()

        # Compute band width: mean distance from inner to outer boundary
        band = outer & ~inner
        band_width = float(band.sum()) / max(float(pred_bool.sum()), 1.0)

        return MorphologicalConformalResult(
            inner_contour=np.asarray(inner, dtype=bool),
            outer_contour=np.asarray(outer, dtype=bool),
            band_width=band_width,
            dilation_radius=self._dilation_radius,
            erosion_radius=self._erosion_radius,
            alpha=self.alpha,
        )


def compute_morphological_metrics(
    result: MorphologicalConformalResult,
    ground_truth: NDArray,
) -> dict[str, float]:
    """Compute morphological conformal prediction metrics.

    Parameters
    ----------
    result:
        MorphologicalConformalResult from predict().
    ground_truth:
        Binary ground truth mask (D, H, W).

    Returns
    -------
    Dict with: outer_coverage, inner_precision, mean_band_width,
    band_volume_ratio.
    """
    gt_bool = ground_truth.astype(bool)

    # Outer coverage: fraction of GT voxels inside outer contour
    gt_sum = float(gt_bool.sum())
    if gt_sum > 0:
        outer_coverage = float((gt_bool & result.outer_contour).sum()) / gt_sum
    else:
        outer_coverage = 1.0

    # Inner precision: fraction of inner contour voxels that are in GT
    inner_sum = float(result.inner_contour.sum())
    if inner_sum > 0:
        inner_precision = float((result.inner_contour & gt_bool).sum()) / inner_sum
    else:
        inner_precision = 1.0

    # Mean band width (already computed in result, but re-derive for clarity)
    mean_band_width = result.band_width

    # Band volume ratio: band volume / prediction volume
    band = result.outer_contour & ~result.inner_contour
    outer_sum = float(result.outer_contour.sum())
    band_volume_ratio = float(band.sum()) / outer_sum if outer_sum > 0 else 0.0

    return {
        "outer_coverage": outer_coverage,
        "inner_precision": inner_precision,
        "mean_band_width": mean_band_width,
        "band_volume_ratio": band_volume_ratio,
    }
