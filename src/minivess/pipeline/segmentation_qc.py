"""Segmentation quality control framework.

Automated quality scoring for segmentation outputs without ground truth
labels, inspired by nnQC (Marciano et al., 2025). Provides heuristic
checks for fragmentation, volume ratio, confidence, and border contact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from numpy.typing import NDArray


class QCFlag(StrEnum):
    """Quality control outcome flag."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class QCResult:
    """Result of a segmentation quality check.

    Parameters
    ----------
    flag:
        Overall quality flag.
    confidence_score:
        Mean softmax probability in the foreground region.
    num_components:
        Number of connected components in the mask.
    volume_ratio:
        Fraction of the volume occupied by foreground.
    reasons:
        List of reasons for WARNING or FAIL.
    """

    flag: QCFlag
    confidence_score: float
    num_components: int
    volume_ratio: float
    reasons: list[str] = field(default_factory=list)


class SegmentationQC:
    """Quality control checks for segmentation masks.

    Parameters
    ----------
    max_components:
        Maximum allowed connected components before WARNING.
    max_volume_ratio:
        Maximum allowed foreground fraction before WARNING.
    min_confidence:
        Minimum mean confidence before WARNING.
    """

    def __init__(
        self,
        *,
        max_components: int = 10,
        max_volume_ratio: float = 0.30,
        min_confidence: float = 0.5,
    ) -> None:
        self.max_components = max_components
        self.max_volume_ratio = max_volume_ratio
        self.min_confidence = min_confidence

    def count_connected_components(self, mask: NDArray) -> int:
        """Count the number of connected components in a binary mask."""
        labeled, n_components = ndimage.label(mask)
        return int(n_components)

    def compute_volume_ratio(self, mask: NDArray) -> float:
        """Compute the fraction of voxels that are foreground."""
        total = mask.size
        if total == 0:
            return 0.0
        return float(np.sum(mask > 0)) / total

    def compute_confidence(self, prob_map: NDArray, mask: NDArray) -> float:
        """Compute mean probability in the foreground region.

        Parameters
        ----------
        prob_map:
            Probability map (e.g., softmax output for foreground class).
        mask:
            Binary segmentation mask.
        """
        fg_voxels = prob_map[mask > 0]
        if fg_voxels.size == 0:
            return 0.0
        return float(np.mean(fg_voxels))

    def check_border_touching(self, mask: NDArray) -> bool:
        """Check if the foreground mask touches the volume boundaries."""
        if mask.ndim < 3:  # noqa: PLR2004
            return False
        # Check all six faces of the 3D volume
        faces = [
            mask[0, :, :],
            mask[-1, :, :],
            mask[:, 0, :],
            mask[:, -1, :],
            mask[:, :, 0],
            mask[:, :, -1],
        ]
        return any(np.any(face > 0) for face in faces)


def evaluate_segmentation_quality(
    mask: NDArray,
    prob_map: NDArray,
    *,
    max_components: int = 10,
    max_volume_ratio: float = 0.30,
    min_confidence: float = 0.5,
) -> QCResult:
    """Evaluate overall segmentation quality.

    Parameters
    ----------
    mask:
        Binary segmentation mask (3D).
    prob_map:
        Foreground probability map (3D).
    """
    qc = SegmentationQC(
        max_components=max_components,
        max_volume_ratio=max_volume_ratio,
        min_confidence=min_confidence,
    )
    reasons: list[str] = []

    # Empty mask check
    volume_ratio = qc.compute_volume_ratio(mask)
    if volume_ratio == 0.0:
        return QCResult(
            flag=QCFlag.FAIL,
            confidence_score=0.0,
            num_components=0,
            volume_ratio=0.0,
            reasons=["Empty segmentation mask"],
        )

    num_components = qc.count_connected_components(mask)
    confidence = qc.compute_confidence(prob_map, mask)
    border_touching = qc.check_border_touching(mask)

    if num_components > max_components:
        reasons.append(
            f"Fragmented: {num_components} components (max {max_components})"
        )
    if volume_ratio > max_volume_ratio:
        reasons.append(f"Volume ratio {volume_ratio:.3f} exceeds {max_volume_ratio}")
    if confidence < min_confidence:
        reasons.append(f"Low confidence {confidence:.3f} (min {min_confidence})")
    if border_touching:
        reasons.append("Mask touches volume boundary")

    if len(reasons) >= 2:  # noqa: PLR2004
        flag = QCFlag.FAIL
    elif len(reasons) == 1:
        flag = QCFlag.WARNING
    else:
        flag = QCFlag.PASS

    return QCResult(
        flag=flag,
        confidence_score=confidence,
        num_components=num_components,
        volume_ratio=volume_ratio,
        reasons=reasons,
    )
