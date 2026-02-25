"""Prediction-Powered Risk Monitoring (PPRM).

Semi-supervised drift detection with formal false alarm guarantees.
Uses a small labeled calibration set to rectify risk estimates from
unlabeled deployment data.

Reference: Zhang et al. (2026). "Prediction-Powered Risk Monitoring."
arxiv:2602.02229
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats as sp_stats

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@dataclass
class RiskEstimate:
    """Result of PPRM monitoring step.

    Parameters
    ----------
    risk:
        Estimated deployment risk (prediction-powered).
    ci_lower:
        Lower bound of confidence interval.
    ci_upper:
        Upper bound of confidence interval.
    alarm:
        Whether risk exceeds threshold with statistical significance.
    threshold:
        Risk threshold used for alarm decision.
    n_calibration:
        Number of labeled calibration samples.
    n_deployment:
        Number of unlabeled deployment samples.
    """

    risk: float
    ci_lower: float
    ci_upper: float
    alarm: bool
    threshold: float
    n_calibration: int
    n_deployment: int

    def to_dict(self) -> dict[str, float]:
        """Convert to flat dict for Prometheus/MLflow logging."""
        return {
            "pprm_risk": self.risk,
            "pprm_ci_lower": self.ci_lower,
            "pprm_ci_upper": self.ci_upper,
            "pprm_alarm": 1.0 if self.alarm else 0.0,
            "pprm_threshold": self.threshold,
            "pprm_n_calibration": float(self.n_calibration),
            "pprm_n_deployment": float(self.n_deployment),
        }


class PPRMDetector:
    """Prediction-Powered Risk Monitor.

    Implements Algorithm 1 from Zhang et al. (2026):
    1. Calibrate: compute rectifier from labeled calibration set
    2. Monitor: estimate risk on unlabeled deployment data with CI

    Parameters
    ----------
    threshold:
        Risk threshold — alarm triggers if CI lower bound exceeds this.
    alpha:
        Significance level for confidence interval (default 0.05 = 95% CI).
    """

    def __init__(
        self,
        threshold: float = 0.20,
        alpha: float = 0.05,
    ) -> None:
        self.threshold = threshold
        self.alpha = alpha
        self._rectifier: float | None = None
        self._rectifier_var: float | None = None
        self._n_cal: int = 0
        self._cal_pred_mean: float = 0.0

    @property
    def is_calibrated(self) -> bool:
        """Whether the detector has been calibrated."""
        return self._rectifier is not None

    def calibrate(
        self,
        cal_predictions: NDArray[np.floating],
        cal_labels: NDArray[np.floating],
        *,
        risk_fn: Callable[
            [NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]
        ],
    ) -> None:
        """Calibrate the rectifier from a labeled calibration set.

        Parameters
        ----------
        cal_predictions:
            Model predictions on calibration set (n_cal,).
        cal_labels:
            Ground truth labels for calibration set (n_cal,).
        risk_fn:
            Per-sample risk function: risk_fn(predictions, labels) → risks.
        """
        cal_predictions = np.asarray(cal_predictions)
        cal_labels = np.asarray(cal_labels)

        self._n_cal = len(cal_predictions)
        self._cal_pred_mean = float(np.mean(cal_predictions))

        # True per-sample risk on calibration set
        true_risk = risk_fn(cal_predictions, cal_labels)

        # Proxy: |prediction - calibration_mean| as deviation-based risk
        proxy = np.abs(cal_predictions - self._cal_pred_mean)

        # Rectifier: corrects bias between true risk and proxy
        rectifier_samples = true_risk - proxy
        self._rectifier = float(np.mean(rectifier_samples))
        self._rectifier_var = float(np.var(rectifier_samples, ddof=1))

    def monitor(
        self,
        deployment_predictions: NDArray[np.floating],
    ) -> RiskEstimate:
        """Estimate deployment risk from unlabeled predictions.

        Parameters
        ----------
        deployment_predictions:
            Model predictions on deployment data (n_deploy,).

        Returns
        -------
        RiskEstimate with risk, CI, and alarm status.
        """
        if not self.is_calibrated:
            msg = "Detector must be calibrated before monitoring. Call calibrate() first."
            raise RuntimeError(msg)

        deployment_predictions = np.asarray(deployment_predictions)
        n_deploy = len(deployment_predictions)

        # Proxy risk on deployment data (deviation from calibration mean)
        proxy_risk = np.abs(
            deployment_predictions - self._cal_pred_mean
        )
        proxy_mean = float(np.mean(proxy_risk))
        proxy_var = float(np.var(proxy_risk, ddof=1)) if n_deploy > 1 else 0.0

        # Prediction-powered risk estimate
        risk = proxy_mean + self._rectifier

        # Combined standard error (calibration + deployment)
        se = np.sqrt(
            self._rectifier_var / self._n_cal + proxy_var / n_deploy
        )

        # Confidence interval via CLT
        z = sp_stats.norm.ppf(1.0 - self.alpha / 2)
        ci_lower = risk - z * se
        ci_upper = risk + z * se

        # Alarm: CI lower bound exceeds threshold
        alarm = ci_lower > self.threshold

        return RiskEstimate(
            risk=float(risk),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            alarm=bool(alarm),
            threshold=self.threshold,
            n_calibration=self._n_cal,
            n_deployment=n_deploy,
        )


def compute_prediction_risk(
    predictions: NDArray[np.floating],
    labels: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute per-sample prediction risk (1 - accuracy proxy).

    For continuous predictions, risk = |prediction - label|.
    For segmentation, this would be (1 - Dice) per sample.

    Parameters
    ----------
    predictions:
        Model predictions (n_samples,).
    labels:
        Ground truth labels (n_samples,).

    Returns
    -------
    Per-sample risk values (n_samples,).
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return np.abs(predictions - labels)
