from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from minivess.pipeline.ci import ConfidenceInterval, compute_metrics_with_ci

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Aggregated evaluation results for one fold.

    Parameters
    ----------
    per_volume_metrics:
        Dict mapping metric names to per-volume value arrays.
    aggregated:
        Dict mapping metric names to ConfidenceInterval objects.
    """

    per_volume_metrics: dict[str, list[float]] = field(default_factory=dict)
    aggregated: dict[str, ConfidenceInterval] = field(default_factory=dict)


class EvaluationRunner:
    """Post-training evaluation using MetricsReloaded.

    Wraps MetricsReloaded's BinaryPairwiseMeasures to compute
    clinically-relevant segmentation metrics on CPU.

    Parameters
    ----------
    metrics:
        List of metric names to compute. Defaults to the primary
        set: centreline_dsc, dsc, measured_masd.
    include_expensive:
        If True, also compute HD95 and NSD (slower).
    """

    # Primary metrics (always computed)
    PRIMARY_METRICS = ("centreline_dsc", "dsc", "measured_masd")
    # Expensive metrics (optional, for test-time only)
    EXPENSIVE_METRICS = (
        "measured_hausdorff_distance_perc",
        "normalised_surface_distance",
    )

    def __init__(
        self,
        *,
        include_expensive: bool = False,
    ) -> None:
        self.metric_names = list(self.PRIMARY_METRICS)
        if include_expensive:
            self.metric_names.extend(self.EXPENSIVE_METRICS)

    def evaluate_volume(
        self,
        pred_np: np.ndarray,
        label_np: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate a single volume pair.

        Parameters
        ----------
        pred_np:
            Binary prediction array (D, H, W) or (H, W).
        label_np:
            Binary ground truth array (D, H, W) or (H, W).

        Returns
        -------
        dict[str, float]
            Metric name to value mapping.
        """
        from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures

        # Pass unflattened arrays to preserve spatial structure
        # (required for skeleton-based metrics like centreline_dsc)
        pred_int = pred_np.astype(int)
        label_int = label_np.astype(int)

        bpm = BinaryPairwiseMeasures(pred_int, label_int)

        results: dict[str, float] = {}
        for name in self.metric_names:
            method = getattr(bpm, name, None)
            if method is None:
                logger.warning("Metric %s not found in BinaryPairwiseMeasures", name)
                continue
            try:
                value = method()
                results[name] = float(value) if value is not None else float("nan")
            except Exception:
                logger.exception("Failed to compute metric %s", name)
                results[name] = float("nan")

        return results

    def evaluate_fold(
        self,
        predictions: list[np.ndarray],
        labels: list[np.ndarray],
        *,
        confidence_level: float = 0.95,
        n_resamples: int = 10_000,
        seed: int = 42,
    ) -> FoldResult:
        """Evaluate all volumes in a fold and aggregate with bootstrap CI.

        Parameters
        ----------
        predictions:
            List of binary prediction arrays.
        labels:
            List of binary ground truth arrays.
        confidence_level:
            CI confidence level.
        n_resamples:
            Number of bootstrap resamples.
        seed:
            Random seed for bootstrap.

        Returns
        -------
        FoldResult
            Per-volume metrics and aggregated CIs.
        """
        if len(predictions) != len(labels):
            msg = f"predictions ({len(predictions)}) and labels ({len(labels)}) length mismatch"
            raise ValueError(msg)

        per_volume: dict[str, list[float]] = {name: [] for name in self.metric_names}

        for i, (pred, label) in enumerate(zip(predictions, labels, strict=True)):
            vol_metrics = self.evaluate_volume(pred, label)
            for name in self.metric_names:
                per_volume[name].append(vol_metrics.get(name, float("nan")))
            logger.debug("Volume %d: %s", i, vol_metrics)

        # Aggregate with bootstrap CI
        per_sample_arrays = {
            name: np.array(values) for name, values in per_volume.items()
        }
        aggregated = compute_metrics_with_ci(
            per_sample_arrays,
            confidence_level=confidence_level,
            method="percentile",
            n_resamples=n_resamples,
            seed=seed,
        )

        return FoldResult(per_volume_metrics=per_volume, aggregated=aggregated)
