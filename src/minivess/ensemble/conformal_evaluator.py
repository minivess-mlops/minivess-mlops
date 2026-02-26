"""Unified conformal evaluator for comparing CP methods.

Runs voxel-level, morphological, distance-transform, and risk-controlling
conformal prediction on evaluation data and produces a comparison report.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from minivess.ensemble.conformal import ConformalPredictor
from minivess.ensemble.distance_conformal import (
    DistanceTransformConformalPredictor,
    compute_distance_metrics,
)
from minivess.ensemble.morphological_conformal import (
    MorphologicalConformalPredictor,
    compute_morphological_metrics,
)
from minivess.ensemble.risk_control import RiskControllingPredictor, fnr_risk

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Default methods to run
_DEFAULT_METHODS = ("voxel", "morphological", "distance")


class ConformalEvaluator:
    """Unified conformal prediction evaluator.

    Runs multiple CP methods on the same data and produces a comparison
    report with metrics for each method.

    Parameters
    ----------
    alpha:
        Significance level for all CP methods.
    methods:
        List of CP methods to run. Options: "voxel", "morphological",
        "distance", "risk_control".
    max_dilation_radius:
        Maximum dilation radius for morphological CP.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        methods: list[str] | None = None,
        max_dilation_radius: int = 20,
    ) -> None:
        self.alpha = alpha
        self.methods = list(methods) if methods else list(_DEFAULT_METHODS)
        self.max_dilation_radius = max_dilation_radius

    def evaluate(
        self,
        predictions: list[NDArray],
        softmax_probs: list[NDArray],
        labels: list[NDArray],
        *,
        calibration_fraction: float = 0.3,
    ) -> dict[str, dict[str, float]]:
        """Run all enabled CP methods and compute metrics.

        Splits data into calibration and test sets, calibrates each
        CP method on calibration data, and evaluates on test data.

        Parameters
        ----------
        predictions:
            List of binary prediction masks (D, H, W).
        softmax_probs:
            List of probability maps (D, H, W).
        labels:
            List of binary ground truth masks (D, H, W).
        calibration_fraction:
            Fraction of data to use for calibration.

        Returns
        -------
        Dict mapping method name to dict of metrics.
        """
        n = len(predictions)
        n_cal = max(1, int(n * calibration_fraction))

        cal_preds = predictions[:n_cal]
        cal_probs = softmax_probs[:n_cal]
        cal_labels = labels[:n_cal]
        test_preds = predictions[n_cal:]
        test_probs = softmax_probs[n_cal:]
        test_labels = labels[n_cal:]

        # Fall back to calibration data for testing if not enough data
        if not test_preds:
            test_preds = cal_preds
            test_probs = cal_probs
            test_labels = cal_labels

        results: dict[str, dict[str, float]] = {}

        if "voxel" in self.methods:
            results["voxel"] = self._run_voxel(
                cal_probs, cal_labels, test_probs, test_labels
            )

        if "morphological" in self.methods:
            results["morphological"] = self._run_morphological(
                cal_preds, cal_labels, test_preds, test_labels
            )

        if "distance" in self.methods:
            results["distance"] = self._run_distance(
                cal_preds, cal_labels, test_preds, test_labels
            )

        if "risk_control" in self.methods:
            results["risk_control"] = self._run_risk_control(
                cal_probs, cal_labels, test_probs, test_labels
            )

        return results

    def flatten_results(
        self,
        results: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Flatten nested results to a flat dict for MLflow logging.

        Parameters
        ----------
        results:
            Nested dict from evaluate().

        Returns
        -------
        Flat dict with prefixed keys (e.g., "conformal_voxel_coverage").
        """
        flat: dict[str, float] = {}
        for method, metrics in results.items():
            for metric_name, value in metrics.items():
                flat[f"conformal_{method}_{metric_name}"] = value
        return flat

    def to_markdown(
        self,
        results: dict[str, dict[str, float]],
    ) -> str:
        """Generate a markdown comparison table.

        Parameters
        ----------
        results:
            Nested dict from evaluate().

        Returns
        -------
        Markdown string.
        """
        lines = [
            f"## Conformal Prediction Comparison (alpha={self.alpha})",
            "",
        ]

        if not results:
            lines.append("*No results.*")
            return "\n".join(lines)

        # Collect all metric names
        all_metrics: set[str] = set()
        for metrics in results.values():
            all_metrics.update(metrics.keys())
        sorted_metrics = sorted(all_metrics)

        # Header
        header = "| Method | " + " | ".join(sorted_metrics) + " |"
        sep = "| --- | " + " | ".join(["---"] * len(sorted_metrics)) + " |"
        lines.extend([header, sep])

        # Rows
        for method in sorted(results.keys()):
            cells = [method]
            for m in sorted_metrics:
                val = results[method].get(m)
                cells.append(f"{val:.4f}" if val is not None else "N/A")
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal method runners
    # ------------------------------------------------------------------

    def _run_voxel(
        self,
        cal_probs: list[NDArray],
        cal_labels: list[NDArray],
        test_probs: list[NDArray],
        test_labels: list[NDArray],
    ) -> dict[str, float]:
        """Run voxel-level split conformal prediction."""
        try:
            predictor = ConformalPredictor(alpha=self.alpha)
            # Stack into (N, C=1, D, H, W) for binary segmentation
            # ConformalPredictor expects (N, C, D, H, W) probs
            cal_stacked = np.stack(
                [np.stack([1 - p, p], axis=0) for p in cal_probs], axis=0
            )
            cal_lab = np.stack([lab.astype(np.int64) for lab in cal_labels], axis=0)
            predictor.calibrate(cal_stacked, cal_lab)

            # Evaluate on test
            test_stacked = np.stack(
                [np.stack([1 - p, p], axis=0) for p in test_probs], axis=0
            )
            test_lab = np.stack([lab.astype(np.int64) for lab in test_labels], axis=0)
            result = predictor.predict(test_stacked)

            # Compute coverage
            labels_idx = test_lab[:, np.newaxis, ...]
            covered = np.take_along_axis(result.prediction_sets, labels_idx, axis=1)
            coverage = float(covered.mean())

            return {
                "coverage": coverage,
                "quantile": result.quantile,
                "mean_set_size": float(result.prediction_sets.sum(axis=1).mean()),
            }
        except Exception:
            logger.exception("Voxel CP failed")
            return {"error": 1.0}

    def _run_morphological(
        self,
        cal_preds: list[NDArray],
        cal_labels: list[NDArray],
        test_preds: list[NDArray],
        test_labels: list[NDArray],
    ) -> dict[str, float]:
        """Run morphological conformal prediction."""
        try:
            predictor = MorphologicalConformalPredictor(
                alpha=self.alpha,
                max_radius=self.max_dilation_radius,
            )
            predictor.calibrate(cal_preds, cal_labels)

            # Evaluate on each test volume, aggregate metrics
            all_metrics: list[dict[str, float]] = []
            for pred, gt in zip(test_preds, test_labels, strict=True):
                result = predictor.predict(pred)
                metrics = compute_morphological_metrics(result, gt)
                all_metrics.append(metrics)

            # Aggregate: mean over test volumes
            aggregated: dict[str, float] = {}
            for key in all_metrics[0]:
                aggregated[key] = float(np.mean([m[key] for m in all_metrics]))
            aggregated["dilation_radius"] = float(predictor.dilation_radius)
            aggregated["erosion_radius"] = float(predictor.erosion_radius)

            return aggregated
        except Exception:
            logger.exception("Morphological CP failed")
            return {"error": 1.0}

    def _run_distance(
        self,
        cal_preds: list[NDArray],
        cal_labels: list[NDArray],
        test_preds: list[NDArray],
        test_labels: list[NDArray],
    ) -> dict[str, float]:
        """Run distance-transform conformal prediction."""
        try:
            predictor = DistanceTransformConformalPredictor(alpha=self.alpha)
            predictor.calibrate(cal_preds, cal_labels)

            all_metrics: list[dict[str, float]] = []
            for pred, gt in zip(test_preds, test_labels, strict=True):
                pred_set = predictor.predict(pred)
                metrics = compute_distance_metrics(
                    pred_set, gt, threshold=predictor.calibrated_threshold
                )
                all_metrics.append(metrics)

            aggregated: dict[str, float] = {}
            for key in all_metrics[0]:
                aggregated[key] = float(np.mean([m[key] for m in all_metrics]))
            aggregated["calibrated_threshold"] = predictor.calibrated_threshold

            return aggregated
        except Exception:
            logger.exception("Distance CP failed")
            return {"error": 1.0}

    def _run_risk_control(
        self,
        cal_probs: list[NDArray],
        cal_labels: list[NDArray],
        test_probs: list[NDArray],
        test_labels: list[NDArray],
    ) -> dict[str, float]:
        """Run risk-controlling prediction sets."""
        try:
            predictor = RiskControllingPredictor(alpha=self.alpha, risk_fn=fnr_risk)
            predictor.calibrate(cal_probs, cal_labels)

            fnrs: list[float] = []
            for probs, gt in zip(test_probs, test_labels, strict=True):
                pred_set = predictor.predict(probs)
                fnr = fnr_risk(pred_set, gt)
                fnrs.append(fnr)

            return {
                "mean_fnr": float(np.mean(fnrs)),
                "optimal_threshold": predictor.optimal_threshold,
            }
        except Exception:
            logger.exception("Risk control CP failed")
            return {"error": 1.0}
