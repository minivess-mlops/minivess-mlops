from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 — used at runtime in save/load methods
from typing import Any

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
    volume_ids:
        Ordered list of volume identifiers matching per-volume arrays.
    """

    per_volume_metrics: dict[str, list[float]] = field(default_factory=dict)
    aggregated: dict[str, ConfidenceInterval] = field(default_factory=dict)
    volume_ids: list[str] = field(default_factory=list)

    def to_per_volume_json(self) -> list[dict[str, Any]]:
        """Convert per-volume metrics to a list of dicts for JSON serialization.

        Returns
        -------
        List of ``{"volume_id": str, metric_1: float|None, ...}`` dicts.
        NaN values are converted to None for JSON compatibility.
        """
        records: list[dict[str, Any]] = []
        n_volumes = len(self.volume_ids)
        for i in range(n_volumes):
            entry: dict[str, Any] = {"volume_id": self.volume_ids[i]}
            for metric_name, values in self.per_volume_metrics.items():
                if i < len(values):
                    val = values[i]
                    entry[metric_name] = None if math.isnan(val) else val
                else:
                    entry[metric_name] = None
            records.append(entry)
        return records

    def save_per_volume_json(self, path: Path) -> Path:
        """Save per-volume metrics as JSON file.

        Parameters
        ----------
        path:
            Output file path.

        Returns
        -------
        The written path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        records = self.to_per_volume_json()
        path.write_text(
            json.dumps(records, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved per-volume metrics to %s (%d volumes)", path, len(records))
        return path

    @staticmethod
    def load_per_volume_json(path: Path) -> list[dict[str, Any]]:
        """Load per-volume metrics from JSON file.

        Parameters
        ----------
        path:
            Path to JSON file.

        Returns
        -------
        List of per-volume metric dicts.
        """
        result: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return result

    def best_volume(
        self, metric: str, *, higher_is_better: bool = True
    ) -> tuple[str, float]:
        """Find the best-performing volume for a metric.

        Parameters
        ----------
        metric:
            Metric name.
        higher_is_better:
            If True, highest value is best.

        Returns
        -------
        Tuple of (volume_id, metric_value).

        Raises
        ------
        KeyError
            If the metric is not found.
        """
        if metric not in self.per_volume_metrics:
            msg = f"Metric '{metric}' not found in per_volume_metrics"
            raise KeyError(msg)
        values = self.per_volume_metrics[metric]
        best_idx: int | None = None
        best_val = float("-inf") if higher_is_better else float("inf")
        for i, v in enumerate(values):
            if math.isnan(v):
                continue
            if (higher_is_better and v > best_val) or (
                not higher_is_better and v < best_val
            ):
                best_idx = i
                best_val = v
        if best_idx is None:
            msg = f"All values are NaN for metric '{metric}'"
            raise ValueError(msg)
        return self.volume_ids[best_idx], best_val

    def worst_volume(
        self, metric: str, *, higher_is_better: bool = True
    ) -> tuple[str, float]:
        """Find the worst-performing volume for a metric.

        Parameters
        ----------
        metric:
            Metric name.
        higher_is_better:
            If True, lowest value is worst.

        Returns
        -------
        Tuple of (volume_id, metric_value).
        """
        return self.best_volume(metric, higher_is_better=not higher_is_better)


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
        seed: int | None = None,
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
