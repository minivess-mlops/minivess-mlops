"""Confidence interval reporting for segmentation metrics.

Implements percentile bootstrap and BCa bootstrap following
André et al. (2026) recommendations for medical image segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@dataclass
class ConfidenceInterval:
    """Confidence interval for a single metric.

    Parameters
    ----------
    point_estimate:
        The observed value of the statistic.
    lower:
        Lower bound of the confidence interval.
    upper:
        Upper bound of the confidence interval.
    confidence_level:
        Confidence level (e.g., 0.95 for 95% CI).
    method:
        Method used to compute the CI.
    """

    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            msg = f"lower ({self.lower}) must not exceed upper ({self.upper})"
            raise ValueError(msg)

    @property
    def width(self) -> float:
        """Width of the confidence interval."""
        return self.upper - self.lower

    def to_dict(self, metric_name: str) -> dict[str, float]:
        """Convert to flat dict for MLflow logging.

        Parameters
        ----------
        metric_name:
            Name prefix for the keys.
        """
        return {
            metric_name: self.point_estimate,
            f"{metric_name}_ci_lower": self.lower,
            f"{metric_name}_ci_upper": self.upper,
            f"{metric_name}_ci_level": self.confidence_level,
        }


def bootstrap_ci(
    samples: NDArray[np.floating],
    statistic: Callable[[NDArray[np.floating]], float] = np.mean,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> ConfidenceInterval:
    """Percentile bootstrap confidence interval.

    Parameters
    ----------
    samples:
        1-D array of per-sample metric values.
    statistic:
        Function to compute the statistic of interest.
    confidence_level:
        Confidence level (e.g., 0.95).
    n_resamples:
        Number of bootstrap resamples.
    seed:
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples)
    n = len(samples)

    point_estimate = float(statistic(samples))

    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        resample = samples[rng.integers(0, n, size=n)]
        boot_stats[i] = statistic(resample)

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return ConfidenceInterval(
        point_estimate=point_estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        method="percentile_bootstrap",
    )


def bca_bootstrap_ci(
    samples: NDArray[np.floating],
    statistic: Callable[[NDArray[np.floating]], float] = np.mean,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> ConfidenceInterval:
    """BCa (bias-corrected and accelerated) bootstrap confidence interval.

    Better for small sample sizes than percentile bootstrap.

    Parameters
    ----------
    samples:
        1-D array of per-sample metric values.
    statistic:
        Function to compute the statistic of interest.
    confidence_level:
        Confidence level (e.g., 0.95).
    n_resamples:
        Number of bootstrap resamples.
    seed:
        Random seed for reproducibility.
    """
    from scipy import stats as sp_stats

    rng = np.random.default_rng(seed)
    samples = np.asarray(samples)
    n = len(samples)

    point_estimate = float(statistic(samples))

    # Bootstrap distribution
    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        resample = samples[rng.integers(0, n, size=n)]
        boot_stats[i] = statistic(resample)

    # Bias correction factor (z0)
    proportion_below = np.mean(boot_stats < point_estimate)
    z0 = sp_stats.norm.ppf(max(proportion_below, 1e-10))

    # Acceleration factor (a) via jackknife
    jackknife_stats = np.empty(n)
    for i in range(n):
        jack_sample = np.delete(samples, i)
        jackknife_stats[i] = statistic(jack_sample)

    jack_mean = np.mean(jackknife_stats)
    jack_diff = jack_mean - jackknife_stats
    numerator = np.sum(jack_diff**3)
    denominator = 6.0 * (np.sum(jack_diff**2)) ** 1.5
    a = numerator / denominator if abs(denominator) > 1e-15 else 0.0

    # Adjusted percentiles
    alpha = 1.0 - confidence_level
    z_alpha_lower = sp_stats.norm.ppf(alpha / 2)
    z_alpha_upper = sp_stats.norm.ppf(1 - alpha / 2)

    def _adjusted_percentile(z_alpha: float) -> float:
        numerator_adj = z0 + z_alpha
        adjusted_z = z0 + numerator_adj / (1 - a * numerator_adj)
        return float(sp_stats.norm.cdf(adjusted_z)) * 100

    lower_pct = _adjusted_percentile(z_alpha_lower)
    upper_pct = _adjusted_percentile(z_alpha_upper)

    # Clamp percentiles to valid range
    lower_pct = max(0.0, min(100.0, lower_pct))
    upper_pct = max(0.0, min(100.0, upper_pct))

    lower = float(np.percentile(boot_stats, lower_pct))
    upper = float(np.percentile(boot_stats, upper_pct))

    return ConfidenceInterval(
        point_estimate=point_estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        method="bca_bootstrap",
    )


def compute_metrics_with_ci(
    per_sample_metrics: dict[str, NDArray[np.floating]],
    *,
    confidence_level: float = 0.95,
    method: str = "percentile",
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> dict[str, ConfidenceInterval]:
    """Compute confidence intervals for multiple metrics.

    Parameters
    ----------
    per_sample_metrics:
        Dict mapping metric names to 1-D arrays of per-sample values.
    confidence_level:
        Confidence level for all CIs.
    method:
        ``"percentile"`` or ``"bca"``.
    n_resamples:
        Number of bootstrap resamples.
    seed:
        Random seed for reproducibility.
    """
    ci_fn = bca_bootstrap_ci if method == "bca" else bootstrap_ci
    result: dict[str, ConfidenceInterval] = {}

    for name, values in per_sample_metrics.items():
        result[name] = ci_fn(
            values,
            statistic=np.mean,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=seed,
        )

    return result
