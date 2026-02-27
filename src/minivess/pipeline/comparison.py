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


def holm_bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[tuple[float, bool]]:
    """Apply Holm-Bonferroni step-down multiple comparison correction.

    The procedure sorts p-values ascending, then for each position *i*
    (0-indexed) checks whether ``p_i <= alpha / (m - i)`` where *m* is
    the total number of tests.  Rejection stops at the first non-rejected
    hypothesis; all subsequent hypotheses are also not rejected.

    The adjusted p-value for position *i* is::

        adj_p_i = max(p_j * (m - j) for j in (0.0).i)

    clamped to 1.0.

    Parameters
    ----------
    p_values:
        Raw p-values, one per test.  Order is arbitrary.
    alpha:
        Family-wise error rate threshold (default 0.05).

    Returns
    -------
    list[tuple[float, bool]]
        Parallel list to *p_values*.  Each element is
        ``(adjusted_p_value, is_significant)`` in the **original** input
        order.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort by p-value ascending, keeping track of original indices
    order = sorted(range(m), key=lambda i: p_values[i])
    sorted_ps = [p_values[i] for i in order]

    # Step-down: find the first position where we fail to reject
    rejected = [False] * m
    stopped = False
    for rank, _orig_idx in enumerate(order):
        if stopped:
            rejected[rank] = False
        elif sorted_ps[rank] <= alpha / (m - rank):
            rejected[rank] = True
        else:
            stopped = True
            rejected[rank] = False

    # Compute adjusted p-values in sorted order using running max
    adjusted_sorted: list[float] = [0.0] * m
    running_max = 0.0
    for rank in range(m):
        raw_adj = sorted_ps[rank] * (m - rank)
        running_max = max(running_max, raw_adj)
        adjusted_sorted[rank] = min(running_max, 1.0)

    # Map back to original order
    result: list[tuple[float, bool]] = [(0.0, False)] * m
    for rank, orig_idx in enumerate(order):
        result[orig_idx] = (adjusted_sorted[rank], rejected[rank])

    return result


def cohens_d(
    scores_a: NDArray[np.floating],
    scores_b: NDArray[np.floating],
) -> float:
    """Compute paired Cohen's d effect size using pooled standard deviation.

    .. math::

        d = \\frac{\\bar{a} - \\bar{b}}{s_{\\text{pooled}}}

    where :math:`s_{\\text{pooled}} = \\sqrt{(\\text{var}(a) + \\text{var}(b)) / 2}`.

    Parameters
    ----------
    scores_a:
        Per-sample scores for method A (1-D array).
    scores_b:
        Per-sample scores for method B (1-D array).

    Returns
    -------
    float
        Cohen's d.  Returns 0.0 when pooled standard deviation is zero.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    mean_diff = float(np.mean(a) - np.mean(b))
    s_pooled = float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0))

    if s_pooled == 0.0:
        return 0.0

    return mean_diff / s_pooled


@dataclass
class PairwiseComparison:
    """Result of a single pairwise comparison between two loss functions.

    Parameters
    ----------
    loss_a:
        Name of the first loss function.
    loss_b:
        Name of the second loss function.
    metric:
        Metric used for comparison.
    p_value:
        Raw two-sided p-value from paired bootstrap test.
    adjusted_p_value:
        Holm-Bonferroni-adjusted p-value.
    is_significant:
        Whether the comparison is statistically significant after correction.
    effect_size:
        Cohen's d effect size (positive means A > B).
    direction:
        Human-readable direction: ``"A > B"``, ``"A < B"``, or ``"A \u2248 B"``.
    """

    loss_a: str
    loss_b: str
    metric: str
    p_value: float
    adjusted_p_value: float
    is_significant: bool
    effect_size: float
    direction: str


def compute_all_pairwise_comparisons(
    table: ComparisonTable,
    metric: str,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
    effect_threshold: float = 0.2,
) -> list[PairwiseComparison]:
    """Compute all C(n, 2) pairwise comparisons for *metric* across losses.

    Steps
    -----
    1. Extract per-fold scores for each loss function.
    2. Run :func:`paired_bootstrap_test` for every pair.
    3. Apply :func:`holm_bonferroni_correction` to all raw p-values.
    4. Compute :func:`cohens_d` for every pair.
    5. Assign direction string.

    Parameters
    ----------
    table:
        Cross-loss comparison table.
    metric:
        Metric name to compare on.
    n_resamples:
        Bootstrap resamples per test.
    seed:
        Random seed (incremented per pair for independence).
    alpha:
        Family-wise error rate for Holm-Bonferroni correction.
    effect_threshold:
        |d| below this value → ``"A \u2248 B"`` direction.

    Returns
    -------
    list[PairwiseComparison]
        One entry per unique pair, sorted by p-value ascending.
    """
    losses = table.losses
    n = len(losses)

    # Extract per-fold score arrays
    score_arrays: dict[str, NDArray[np.floating]] = {}
    for lr in losses:
        summary = lr.metrics.get(metric)
        if summary is not None:
            score_arrays[lr.loss_name] = np.array(summary.per_fold, dtype=float)

    # Build ordered list of pairs
    pairs: list[tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            name_a = losses[i].loss_name
            name_b = losses[j].loss_name
            if name_a in score_arrays and name_b in score_arrays:
                pairs.append((name_a, name_b))

    if not pairs:
        return []

    # Run bootstrap tests
    raw_p_values: list[float] = []
    for pair_idx, (name_a, name_b) in enumerate(pairs):
        p = paired_bootstrap_test(
            score_arrays[name_a],
            score_arrays[name_b],
            n_resamples=n_resamples,
            seed=seed + pair_idx,
        )
        raw_p_values.append(p)

    # Apply Holm-Bonferroni correction
    corrected = holm_bonferroni_correction(raw_p_values, alpha=alpha)

    # Assemble results
    results: list[PairwiseComparison] = []
    for pair_idx, (name_a, name_b) in enumerate(pairs):
        adj_p, is_sig = corrected[pair_idx]
        d = cohens_d(score_arrays[name_a], score_arrays[name_b])

        if abs(d) < effect_threshold:
            direction = "A \u2248 B"
        elif d > 0:
            direction = "A > B"
        else:
            direction = "A < B"

        results.append(
            PairwiseComparison(
                loss_a=name_a,
                loss_b=name_b,
                metric=metric,
                p_value=raw_p_values[pair_idx],
                adjusted_p_value=adj_p,
                is_significant=is_sig,
                effect_size=d,
                direction=direction,
            )
        )

    results.sort(key=lambda c: c.p_value)
    return results


def format_significance_matrix_markdown(
    comparisons: list[PairwiseComparison],
) -> str:
    """Format pairwise comparisons as a significance matrix in Markdown.

    Each cell shows ``p_adj`` and a ``*`` marker when significant.

    Parameters
    ----------
    comparisons:
        List returned by :func:`compute_all_pairwise_comparisons`.

    Returns
    -------
    str
        Markdown table with one row/column per loss function.
        Significant pairs are marked with ``*``.
    """
    if not comparisons:
        return "*No pairwise comparisons to display.*"

    # Collect ordered unique loss names (preserve insertion order from pairs)
    seen: dict[str, None] = {}
    for cmp in comparisons:
        seen[cmp.loss_a] = None
        seen[cmp.loss_b] = None
    loss_names = list(seen.keys())

    # Build lookup: (a, b) -> PairwiseComparison (symmetric)
    lookup: dict[tuple[str, str], PairwiseComparison] = {}
    for cmp in comparisons:
        lookup[(cmp.loss_a, cmp.loss_b)] = cmp
        lookup[(cmp.loss_b, cmp.loss_a)] = cmp

    header = "| | " + " | ".join(loss_names) + " |"
    separator = "| --- | " + " | ".join(["---"] * len(loss_names)) + " |"

    rows: list[str] = [header, separator]
    for row_name in loss_names:
        cells: list[str] = [row_name]
        for col_name in loss_names:
            if row_name == col_name:
                cells.append("—")
            else:
                maybe_cmp: PairwiseComparison | None = lookup.get((row_name, col_name))
                if maybe_cmp is None:
                    cells.append("N/A")
                else:
                    star = " *" if maybe_cmp.is_significant else ""
                    cells.append(f"p={maybe_cmp.adjusted_p_value:.3f}{star}")
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)
