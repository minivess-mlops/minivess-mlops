"""Evaluation-time calibration metrics for binary segmentation.

Computes Brier score, O:E ratio, IPA, and calibration slope at the voxel level.
These are logged to MLflow during the analysis flow, not used as training losses.

References
----------
- Van Calster et al. (2016). "A calibration hierarchy for risk models." Stat Med.
- Steyerberg et al. (2010). "Assessing the performance of prediction models." Epidemiology.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from minivess.pipeline.biostatistics_types import CalibrationMetricsResult


def compute_calibration_metrics(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    *,
    max_voxels: int = 100_000,
    seed: int = 42,
) -> CalibrationMetricsResult:
    """Compute calibration metrics for binary segmentation predictions.

    Parameters
    ----------
    y_true:
        Ground truth binary labels (0 or 1), flat array.
    p_pred:
        Predicted probabilities for the positive class, flat array.
    max_voxels:
        Maximum voxels to use (subsamples if exceeded for efficiency).
    seed:
        Random seed for subsampling.

    Returns
    -------
    CalibrationMetricsResult with brier_score, oe_ratio, ipa, calibration_slope.
    """
    rng = np.random.default_rng(seed)

    # Subsample if needed
    if len(y_true) > max_voxels:
        idx = rng.choice(len(y_true), size=max_voxels, replace=False)
        y_true = y_true[idx]
        p_pred = p_pred[idx]

    # Brier score: mean squared error of probabilities
    brier = float(brier_score_loss(y_true, p_pred))

    # O:E ratio: observed / expected event rate
    observed = float(y_true.sum())
    expected = float(p_pred.sum())
    oe_ratio = observed / expected if expected > 0 else 0.0

    # IPA (Index of Prediction Accuracy): 1 - Brier / Brier_null
    prevalence = float(y_true.mean())
    brier_null = prevalence * (1 - prevalence)
    ipa = 1.0 - (brier / brier_null) if brier_null > 0 else 0.0

    # Calibration slope via logistic regression of outcome on log-odds
    log_odds = np.log(
        np.clip(p_pred, 1e-7, 1 - 1e-7) / (1 - np.clip(p_pred, 1e-7, 1 - 1e-7))
    )
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(log_odds.reshape(-1, 1), y_true)
    calibration_slope = float(lr.coef_[0, 0])

    return CalibrationMetricsResult(
        brier_score=brier,
        oe_ratio=oe_ratio,
        ipa=ipa,
        calibration_slope=calibration_slope,
    )
