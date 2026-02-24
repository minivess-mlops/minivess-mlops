"""MAPIE-based conformal prediction for 3D segmentation.

Wraps MAPIE's SplitConformalClassifier for voxel-level conformal
prediction sets with distribution-free coverage guarantees.
Flattens 3D volumes for MAPIE calibration, then reshapes back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np  # noqa: TC002
from mapie.classification import SplitConformalClassifier
from sklearn.linear_model import LogisticRegression

from minivess.ensemble.conformal import ConformalResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ConformalMetrics:
    """Conformal prediction coverage metrics.

    Parameters
    ----------
    coverage:
        Fraction of voxels where the true class is in the prediction set.
    mean_set_size:
        Average number of classes in each prediction set.
    """

    coverage: float
    mean_set_size: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dict for MLflow logging."""
        return {
            "conformal_coverage": self.coverage,
            "conformal_mean_set_size": self.mean_set_size,
        }


class MapieConformalSegmentation:
    """MAPIE-based conformal prediction for 3D segmentation.

    Calibrates MAPIE's SplitConformalClassifier on flattened voxel
    probabilities, then produces 3D prediction sets with coverage
    guarantees.

    Parameters
    ----------
    alpha:
        Significance level (e.g., 0.1 for 90% coverage).
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._mapie_clf: SplitConformalClassifier | None = None
        self._n_classes: int | None = None

    @property
    def is_calibrated(self) -> bool:
        """Whether calibrate() has been called."""
        return self._mapie_clf is not None

    def calibrate(
        self,
        cal_probs: NDArray[np.float32],
        cal_labels: NDArray[np.int64],
    ) -> None:
        """Calibrate using MAPIE on holdout volumes.

        Parameters
        ----------
        cal_probs:
            Softmax probabilities (N, C, D, H, W).
        cal_labels:
            Ground truth class indices (N, D, H, W).
        """
        n_volumes, n_classes = cal_probs.shape[:2]
        self._n_classes = n_classes

        # Flatten: (N, C, D, H, W) → (N*D*H*W, C)
        probs_flat = cal_probs.transpose(0, 2, 3, 4, 1).reshape(-1, n_classes)
        labels_flat = cal_labels.reshape(-1)

        # Train a simple LogisticRegression on the probs as features
        # This gives MAPIE a fitted estimator to wrap
        base_clf = LogisticRegression(max_iter=200, random_state=42)
        base_clf.fit(probs_flat, labels_flat)

        # Create MAPIE wrapper and conformalize
        self._mapie_clf = SplitConformalClassifier(
            estimator=base_clf,
            confidence_level=1.0 - self.alpha,
            prefit=True,
        )
        self._mapie_clf.conformalize(probs_flat, labels_flat)

        logger.info(
            "MAPIE conformal calibrated: alpha=%.2f, n_voxels=%d, n_classes=%d",
            self.alpha,
            len(labels_flat),
            n_classes,
        )

    def predict(
        self,
        test_probs: NDArray[np.float32],
    ) -> ConformalResult:
        """Produce voxel-level prediction sets.

        Parameters
        ----------
        test_probs:
            Softmax probabilities (B, C, D, H, W).

        Returns
        -------
        ConformalResult with boolean prediction sets (B, C, D, H, W).
        """
        if not self.is_calibrated:
            msg = "Must calibrate() before predict()"
            raise RuntimeError(msg)

        n_volumes, n_classes = test_probs.shape[:2]
        spatial = test_probs.shape[2:]

        # Flatten: (B, C, D, H, W) → (B*D*H*W, C)
        probs_flat = test_probs.transpose(0, 2, 3, 4, 1).reshape(-1, n_classes)

        # MAPIE predict_set returns (point_preds, pred_sets)
        # pred_sets shape: (N, C, 1) boolean
        _, pred_sets_raw = self._mapie_clf.predict_set(probs_flat)
        pred_sets_2d = pred_sets_raw[:, :, 0]  # (N, C) boolean

        # Reshape back: (B*D*H*W, C) → (B, D, H, W, C) → (B, C, D, H, W)
        pred_sets_5d = pred_sets_2d.reshape(n_volumes, *spatial, n_classes)
        pred_sets_5d = pred_sets_5d.transpose(0, 4, 1, 2, 3)

        return ConformalResult(
            prediction_sets=pred_sets_5d,
            quantile=1.0 - self.alpha,
            alpha=self.alpha,
        )


def compute_coverage_metrics(
    prediction_sets: NDArray[np.bool_],
    labels: NDArray[np.int64],
) -> ConformalMetrics:
    """Compute empirical coverage and mean set size.

    Parameters
    ----------
    prediction_sets:
        Boolean prediction sets (B, C, D, H, W).
    labels:
        Ground truth class indices (B, D, H, W).

    Returns
    -------
    ConformalMetrics with coverage and mean_set_size.
    """
    n_volumes = labels.shape[0]
    spatial = labels.shape[1:]

    # Check if true class is in prediction set for each voxel
    covered = 0
    total = 0
    for i in range(n_volumes):
        for d in range(spatial[0]):
            for h in range(spatial[1]):
                for w in range(spatial[2]):
                    true_class = labels[i, d, h, w]
                    if prediction_sets[i, true_class, d, h, w]:
                        covered += 1
                    total += 1

    coverage = covered / max(total, 1)
    mean_set_size = float(prediction_sets.sum(axis=1).mean())

    return ConformalMetrics(coverage=coverage, mean_set_size=mean_set_size)
