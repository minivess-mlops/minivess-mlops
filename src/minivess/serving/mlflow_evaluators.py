"""MLflow custom evaluators for 3D segmentation metrics.

Provides custom metrics for ``mlflow.evaluate()`` using **path-based
indirection**: predictions and labels are stored as ``.npz`` files on disk,
and evaluator functions load them by path from a ``pd.DataFrame``.

This avoids shoehorning 3D medical imaging volumes into MLflow's tabular
evaluation framework while still leveraging the MLflow UI for comparison
dashboards.

References
----------
* MLflow custom metrics: https://mlflow.org/docs/latest/python_api/mlflow.metrics.html
* MLflow dataset evaluation: https://mlflow.org/docs/latest/ml/evaluation/dataset-eval/
"""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003 — used at runtime in build_evaluation_dataframe
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.metrics import MetricValue, make_metric

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# .npz loading helpers
# ---------------------------------------------------------------------------


def load_npz_prediction(path: str) -> np.ndarray:
    """Load a hard prediction array from a .npz file.

    Parameters
    ----------
    path:
        Path to ``.npz`` file containing a ``hard_pred`` key
        (matching :func:`prediction_store.save_volume_prediction` format).

    Returns
    -------
    Integer prediction array ``(D, H, W)``.
    """
    data = np.load(path)
    return data["hard_pred"]


def load_npz_label(path: str) -> np.ndarray:
    """Load a label array from a .npz file.

    Parameters
    ----------
    path:
        Path to ``.npz`` file containing an ``arr_0`` key.

    Returns
    -------
    Integer label array ``(D, H, W)``.
    """
    data = np.load(path)
    return data["arr_0"]


# ---------------------------------------------------------------------------
# Build evaluation DataFrame
# ---------------------------------------------------------------------------


def build_evaluation_dataframe(
    predictions_dir: Path,
    labels_dir: Path,
) -> pd.DataFrame:
    """Build a DataFrame mapping prediction/label .npz file paths.

    Matches files by stem name across both directories.

    Parameters
    ----------
    predictions_dir:
        Directory containing prediction ``.npz`` files.
    labels_dir:
        Directory containing label ``.npz`` files.

    Returns
    -------
    DataFrame with columns: ``prediction_path``, ``label_path``,
    ``volume_name``.
    """
    pred_files = {p.stem: p for p in sorted(predictions_dir.glob("*.npz"))}
    label_files = {p.stem: p for p in sorted(labels_dir.glob("*.npz"))}

    # Intersect by stem name
    common_names = sorted(set(pred_files.keys()) & set(label_files.keys()))

    if not common_names:
        return pd.DataFrame(columns=["prediction_path", "label_path", "volume_name"])

    rows = []
    for name in common_names:
        rows.append(
            {
                "prediction_path": str(pred_files[name]),
                "label_path": str(label_files[name]),
                "volume_name": name,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metric evaluation functions
# ---------------------------------------------------------------------------


def _compute_dice(pred: np.ndarray, label: np.ndarray) -> float:
    """Compute Dice coefficient between binary arrays."""
    pred_bool = pred.astype(bool)
    label_bool = label.astype(bool)

    intersection = np.logical_and(pred_bool, label_bool).sum()
    total = pred_bool.sum() + label_bool.sum()

    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2.0 * intersection / total)


def dice_eval_fn(
    predictions: pd.Series,
    targets: pd.Series,
    metrics: dict[str, Any],
    **kwargs: Any,
) -> MetricValue:
    """Evaluate Dice coefficient from .npz file paths.

    Parameters
    ----------
    predictions:
        Series of prediction file paths.
    targets:
        Series of label file paths.
    metrics:
        Dict of already-computed built-in metrics (unused).

    Returns
    -------
    MetricValue with per-volume scores and aggregate mean.
    """
    scores: list[float] = []
    for pred_path, label_path in zip(predictions, targets, strict=True):
        pred = load_npz_prediction(str(pred_path))
        label = load_npz_label(str(label_path))
        scores.append(_compute_dice(pred, label))

    mean_dice = float(np.mean(scores)) if scores else 0.0
    return MetricValue(
        scores=scores,
        aggregate_results={"mean_dice": mean_dice},
    )


def compound_eval_fn(
    predictions: pd.Series,
    targets: pd.Series,
    metrics: dict[str, Any],
    **kwargs: Any,
) -> MetricValue:
    """Evaluate compound MASD+clDice metric from .npz file paths.

    Uses simplified Dice as the base (no skeleton computation) for
    the evaluation framework.  Full MetricsReloaded metrics including
    skeleton-based clDice and MASD are computed by the
    :class:`UnifiedEvaluationRunner` separately.

    The compound score is: ``0.5 * dice + 0.5 * dice`` as a placeholder
    for ``0.5 * normalize_masd(masd) + 0.5 * cldice`` when the full
    MetricsReloaded pipeline is not available.

    Parameters
    ----------
    predictions:
        Series of prediction file paths.
    targets:
        Series of label file paths.
    metrics:
        Dict of already-computed built-in metrics.

    Returns
    -------
    MetricValue with per-volume compound scores and aggregate mean.
    """
    scores: list[float] = []
    for pred_path, label_path in zip(predictions, targets, strict=True):
        pred = load_npz_prediction(str(pred_path))
        label = load_npz_label(str(label_path))
        dice = _compute_dice(pred, label)
        # Compound = 0.5 * normalized_distance + 0.5 * overlap
        # Using Dice as proxy for both components in the MLflow evaluator;
        # real compound metric uses MetricsReloaded MASD + clDice
        compound = float(dice)
        scores.append(compound)

    mean_compound = float(np.mean(scores)) if scores else 0.0
    return MetricValue(
        scores=scores,
        aggregate_results={"mean_compound": mean_compound},
    )


# ---------------------------------------------------------------------------
# Metric objects (created via make_metric)
# ---------------------------------------------------------------------------


dice_metric = make_metric(
    eval_fn=dice_eval_fn,
    greater_is_better=True,
    name="dice_coefficient",
)

compound_metric = make_metric(
    eval_fn=compound_eval_fn,
    greater_is_better=True,
    name="compound_masd_cldice",
)


# ---------------------------------------------------------------------------
# Top-level evaluation wrapper
# ---------------------------------------------------------------------------


def run_mlflow_evaluation(
    predictions_dir: Path,
    labels_dir: Path,
    *,
    include_expensive: bool = False,
) -> Any:
    """Run mlflow.evaluate() with custom segmentation metrics.

    Parameters
    ----------
    predictions_dir:
        Directory containing prediction ``.npz`` files.
    labels_dir:
        Directory containing label ``.npz`` files.
    include_expensive:
        If ``True``, include skeleton-based metrics (clDice, MASD).
        Currently only Dice and compound are included regardless.

    Returns
    -------
    ``mlflow.models.EvaluationResult``.
    """
    df = build_evaluation_dataframe(predictions_dir, labels_dir)

    extra_metrics = [dice_metric, compound_metric]

    result = mlflow.evaluate(
        data=df,
        predictions="prediction_path",
        targets="label_path",
        model_type=None,
        extra_metrics=extra_metrics,
    )

    logger.info(
        "MLflow evaluation complete: %d volumes, metrics: %s",
        len(df),
        list(result.metrics.keys()) if hasattr(result, "metrics") else "N/A",
    )
    return result
