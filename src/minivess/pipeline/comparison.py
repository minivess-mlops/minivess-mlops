"""Cross-loss comparison with paired bootstrap significance testing.

Builds comparison tables across loss functions from per-fold evaluation results,
performs paired bootstrap hypothesis testing, and identifies the best loss function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from minivess.pipeline.evaluation import FoldResult

logger = logging.getLogger(__name__)


@dataclass
class MetricSummary:
    """Summary statistics for one metric across folds.

    Parameters
    ----------
    mean:
        Mean of per-fold point estimates.
    std:
        Standard deviation of per-fold point estimates.
    ci_lower:
        Average lower CI bound across folds.
    ci_upper:
        Average upper CI bound across folds.
    per_fold:
        Per-fold point estimates (one per fold).
    """

    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    per_fold: list[float]


@dataclass
class LossResult:
    """Results for one loss function across all folds.

    Parameters
    ----------
    loss_name:
        Name of the loss function (e.g., ``"dice_ce"``).
    num_folds:
        Number of cross-validation folds.
    metrics:
        Mapping from metric name to :class:`MetricSummary`.
    """

    loss_name: str
    num_folds: int
    metrics: dict[str, MetricSummary]


@dataclass
class ComparisonTable:
    """Cross-loss comparison table.

    Parameters
    ----------
    losses:
        One :class:`LossResult` per loss function.
    metric_names:
        Sorted list of all metric names present across all losses.
    """

    losses: list[LossResult]
    metric_names: list[str] = field(default_factory=list)


def _summarise_metric(
    fold_results: list[FoldResult],
    metric_name: str,
) -> MetricSummary:
    """Compute :class:`MetricSummary` for *metric_name* across all folds.

    Each fold contributes its ``aggregated[metric_name].point_estimate``.

    Parameters
    ----------
    fold_results:
        Per-fold evaluation results.
    metric_name:
        The metric to summarise.

    Returns
    -------
    MetricSummary
    """
    per_fold: list[float] = []
    ci_lowers: list[float] = []
    ci_uppers: list[float] = []

    for fold in fold_results:
        ci = fold.aggregated.get(metric_name)
        if ci is None:
            logger.warning(
                "Metric %s not found in fold aggregated results", metric_name
            )
            continue
        per_fold.append(float(ci.point_estimate))
        ci_lowers.append(float(ci.lower))
        ci_uppers.append(float(ci.upper))

    if not per_fold:
        return MetricSummary(
            mean=float("nan"),
            std=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            per_fold=[],
        )

    arr = np.array(per_fold)
    return MetricSummary(
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0),
        ci_lower=float(np.mean(ci_lowers)),
        ci_upper=float(np.mean(ci_uppers)),
        per_fold=per_fold,
    )


def build_comparison_table(
    eval_results: dict[str, list[FoldResult]],
) -> ComparisonTable:
    """Build a :class:`ComparisonTable` from per-fold evaluation results.

    Parameters
    ----------
    eval_results:
        Mapping from loss function name to list of per-fold
        :class:`~minivess.pipeline.evaluation.FoldResult` objects.

    Returns
    -------
    ComparisonTable
        Table with one :class:`LossResult` per loss function.
    """
    if not eval_results:
        return ComparisonTable(losses=[], metric_names=[])

    # Collect all metric names from the first non-empty fold result
    all_metric_names: set[str] = set()
    for fold_list in eval_results.values():
        for fold in fold_list:
            all_metric_names.update(fold.aggregated.keys())

    sorted_metrics = sorted(all_metric_names)

    loss_results: list[LossResult] = []
    for loss_name, fold_list in eval_results.items():
        metrics: dict[str, MetricSummary] = {}
        for metric_name in sorted_metrics:
            metrics[metric_name] = _summarise_metric(fold_list, metric_name)

        loss_results.append(
            LossResult(
                loss_name=loss_name,
                num_folds=len(fold_list),
                metrics=metrics,
            )
        )

    return ComparisonTable(losses=loss_results, metric_names=sorted_metrics)


def find_best_loss(
    table: ComparisonTable,
    metric: str = "dsc",
    *,
    higher_is_better: bool = True,
) -> str:
    """Find the best loss function by a given metric.

    Parameters
    ----------
    table:
        Cross-loss comparison table.
    metric:
        Metric name to rank by.
    higher_is_better:
        If ``True``, higher metric value is better (e.g., DSC).
        If ``False``, lower is better (e.g., MASD).

    Returns
    -------
    str
        Name of the best loss function.

    Raises
    ------
    ValueError
        If ``table.losses`` is empty or ``metric`` not found.
    """
    if not table.losses:
        msg = "ComparisonTable has no losses to compare"
        raise ValueError(msg)

    best: LossResult | None = None
    best_value: float | None = None

    for lr in table.losses:
        summary = lr.metrics.get(metric)
        if summary is None:
            logger.warning("Metric %s not found for loss %s", metric, lr.loss_name)
            continue

        value = summary.mean
        is_better = higher_is_better and value > (best_value or float("-inf"))
        is_better_lower = not higher_is_better and value < (
            best_value if best_value is not None else float("inf")
        )
        if best_value is None or is_better or is_better_lower:
            best = lr
            best_value = value

    if best is None:
        msg = (
            f"Could not determine best loss; metric '{metric}' missing from all results"
        )
        raise ValueError(msg)

    return best.loss_name


def paired_bootstrap_test(
    scores_a: NDArray[np.floating],
    scores_b: NDArray[np.floating],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided paired bootstrap significance test.

    Tests the null hypothesis H0: mean(a) == mean(b).

    Algorithm
    ---------
    1. Compute observed difference: ``d_obs = mean(a) - mean(b)``.
    2. For each of *n_resamples* bootstrap iterations:
       a. Draw indices with replacement.
       b. Compute ``d_boot = mean(a[idx]) - mean(b[idx])``.
    3. Shift distribution to be centred at 0 under H0:
       ``d_shifted = d_boot - mean(d_boot)``.
    4. p-value = fraction of ``|d_shifted| >= |d_obs|``.

    Parameters
    ----------
    scores_a:
        Per-sample scores for method A (1-D array).
    scores_b:
        Per-sample scores for method B (1-D array, same length).
    n_resamples:
        Number of bootstrap resamples.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    float
        Two-sided p-value in [0, 1].

    Raises
    ------
    ValueError
        If ``scores_a`` and ``scores_b`` have different lengths.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    if len(scores_a) != len(scores_b):
        msg = (
            f"scores_a and scores_b must have the same length, "
            f"got {len(scores_a)} and {len(scores_b)}"
        )
        raise ValueError(msg)

    n = len(scores_a)
    rng = np.random.default_rng(seed)

    d_obs = float(np.mean(scores_a) - np.mean(scores_b))

    d_boot = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        d_boot[i] = np.mean(scores_a[idx]) - np.mean(scores_b[idx])

    # Shift to centre under H0 (mean difference = 0)
    d_shifted = d_boot - np.mean(d_boot)

    p_value = float(np.mean(np.abs(d_shifted) >= np.abs(d_obs)))
    return p_value


def format_comparison_markdown(table: ComparisonTable) -> str:
    """Format a :class:`ComparisonTable` as a markdown table.

    Columns: Loss | <metric_1> | <metric_2> | ...
    Each cell shows ``mean ± std`` for that metric.

    Parameters
    ----------
    table:
        Cross-loss comparison table.

    Returns
    -------
    str
        Markdown-formatted comparison table.
    """
    if not table.losses:
        return "*No results to display.*"

    metrics = table.metric_names
    header = "| Loss | " + " | ".join(metrics) + " |"
    separator = "| --- | " + " | ".join(["---"] * len(metrics)) + " |"

    rows: list[str] = [header, separator]

    for lr in table.losses:
        cells: list[str] = [lr.loss_name]
        for metric_name in metrics:
            summary = lr.metrics.get(metric_name)
            if summary is None or np.isnan(summary.mean):
                cells.append("N/A")
            else:
                cells.append(f"{summary.mean:.4f} ± {summary.std:.4f}")
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)
