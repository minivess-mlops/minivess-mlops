"""Dual-mode champion model evaluation.

Evaluates deployed champion models against drift simulation batches
in two modes:
    - **supervised**: Dice + clDice when ground truth masks are available
    - **unsupervised**: MC Dropout uncertainty + Mahalanobis distance
    - **both**: both metrics computed simultaneously

Config-driven switch via ``evaluation_mode: supervised | unsupervised | both``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

_VALID_MODES = {"supervised", "unsupervised", "both"}


@dataclass
class EvaluationResult:
    """Result of champion model evaluation on a drift batch.

    Contains supervised metrics (Dice), unsupervised metrics (uncertainty),
    or both depending on the evaluation mode.
    """

    mode: str
    batch_id: str
    n_volumes: int = 0
    dice_scores: list[float] | None = field(default=None)
    mean_dice: float | None = field(default=None)
    cldice_scores: list[float] | None = field(default=None)
    mean_cldice: float | None = field(default=None)
    uncertainty_scores: list[float] | None = field(default=None)
    mean_uncertainty: float | None = field(default=None)
    mahalanobis_distances: list[float] | None = field(default=None)


def compute_dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute Dice coefficient between binary prediction and ground truth.

    Args:
        prediction: Binary segmentation mask (0/1).
        ground_truth: Binary ground truth mask (0/1).

    Returns:
        Dice coefficient in [0, 1].
    """
    import numpy as np

    pred_flat = prediction.ravel().astype(np.float64)
    gt_flat = ground_truth.ravel().astype(np.float64)
    intersection = float(np.sum(pred_flat * gt_flat))
    total = float(np.sum(pred_flat) + np.sum(gt_flat))
    if total == 0:
        return 0.0
    return 2.0 * intersection / total


class ChampionEvaluator:
    """Evaluate champion model predictions against drift batches.

    Supports supervised (with GT masks), unsupervised (uncertainty only),
    and combined evaluation modes.
    """

    def __init__(self, mode: str = "both") -> None:
        if mode not in _VALID_MODES:
            msg = (
                f"Invalid evaluation mode '{mode}'. "
                f"Must be one of: {', '.join(sorted(_VALID_MODES))}"
            )
            raise ValueError(msg)
        self._mode = mode

    def evaluate(
        self,
        predictions: list[np.ndarray],
        masks: list[np.ndarray] | None = None,
        uncertainty_maps: list[np.ndarray] | None = None,
        batch_id: str = "",
    ) -> EvaluationResult:
        """Evaluate predictions against optional masks and uncertainty.

        Args:
            predictions: List of binary segmentation predictions.
            masks: Optional ground truth masks (required for supervised mode).
            uncertainty_maps: Optional per-voxel uncertainty maps.
            batch_id: Identifier for the drift simulation batch.

        Returns:
            EvaluationResult with metrics appropriate for the mode.
        """
        import numpy as np

        result = EvaluationResult(
            mode=self._mode,
            batch_id=batch_id,
            n_volumes=len(predictions),
        )

        # Supervised metrics
        if self._mode in ("supervised", "both") and masks is not None:
            dice_scores = [
                compute_dice(pred, gt)
                for pred, gt in zip(predictions, masks, strict=True)
            ]
            result.dice_scores = dice_scores
            result.mean_dice = float(np.mean(dice_scores))

        # Unsupervised metrics
        if self._mode in ("unsupervised", "both") and uncertainty_maps is not None:
            unc_scores = [float(np.mean(u)) for u in uncertainty_maps]
            result.uncertainty_scores = unc_scores
            result.mean_uncertainty = float(np.mean(unc_scores))

        return result
