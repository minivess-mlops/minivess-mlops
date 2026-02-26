"""Extended comparison reporting for evaluation results.

Builds on the existing comparison.py module to support:
- Ensemble vs single model comparison
- Per-dataset breakdown tables
- Statistical significance between strategies
- Compound metric as default sort column
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from minivess.pipeline.comparison import paired_bootstrap_test

if TYPE_CHECKING:
    from minivess.pipeline.evaluation_runner import EvaluationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModelComparisonEntry:
    """One row in the comparison table -- can be a single model or ensemble.

    Parameters
    ----------
    model_name:
        Descriptive name of the model or ensemble.
    model_type:
        Either ``"single"`` or ``"ensemble"``.
    strategy:
        Ensemble strategy name (e.g. ``"mean"``, ``"voting"``), or ``None``
        for single models.
    per_dataset_metrics:
        ``{dataset_name: {metric_name: value}}``.
    overall_metrics:
        Metrics averaged across all datasets.
    per_dataset_per_volume:
        ``{dataset_name: {metric_name: [per-volume values]}}``.
        Stored for significance testing.
    """

    model_name: str
    model_type: str
    strategy: str | None
    per_dataset_metrics: dict[str, dict[str, float]]
    overall_metrics: dict[str, float]
    per_dataset_per_volume: dict[str, dict[str, list[float]]] = field(
        default_factory=dict,
    )


@dataclass
class ExtendedComparisonTable:
    """Cross-model comparison table supporting both single models and ensembles.

    Parameters
    ----------
    entries:
        One :class:`ModelComparisonEntry` per model/ensemble.
    metric_names:
        Sorted list of all metric names present.
    dataset_names:
        Sorted list of all dataset names present.
    primary_metric:
        Metric used for sorting and selection.
    primary_metric_direction:
        ``"maximize"`` or ``"minimize"``.
    """

    entries: list[ModelComparisonEntry]
    metric_names: list[str]
    dataset_names: list[str]
    primary_metric: str
    primary_metric_direction: str


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _infer_model_type(model_name: str) -> tuple[str, str | None]:
    """Infer model_type and strategy from the model name.

    Simple heuristic: names starting with ``"ensemble"`` are ensemble models;
    the part after the first underscore is the strategy name.

    Returns
    -------
    tuple[str, str | None]
        ``(model_type, strategy)``.
    """
    lower = model_name.lower()
    if lower.startswith("ensemble"):
        parts = model_name.split("_", maxsplit=1)
        strategy = parts[1] if len(parts) > 1 else None
        return "ensemble", strategy
    return "single", None


def _aggregate_subset_metrics(
    subset_results: dict[str, EvaluationResult],
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Aggregate metrics across subsets for one dataset.

    For simplicity, we average point estimates across subsets and concatenate
    per-volume arrays.

    Returns
    -------
    tuple[dict[str, float], dict[str, list[float]]]
        ``(aggregated_metrics, per_volume_metrics)``.
    """
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    per_volume: dict[str, list[float]] = {}

    for eval_result in subset_results.values():
        for metric_name, ci in eval_result.fold_result.aggregated.items():
            val = ci.point_estimate
            if math.isnan(val):
                continue
            metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + val
            metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1

        for metric_name, values in eval_result.fold_result.per_volume_metrics.items():
            per_volume.setdefault(metric_name, []).extend(values)

    aggregated: dict[str, float] = {}
    for metric_name in metric_sums:
        count = metric_counts[metric_name]
        aggregated[metric_name] = (
            metric_sums[metric_name] / count if count > 0 else float("nan")
        )

    return aggregated, per_volume


def build_extended_comparison(
    all_results: dict[str, dict[str, dict[str, EvaluationResult]]],
    *,
    primary_metric: str = "compound_masd_cldice",
    primary_metric_direction: str = "maximize",
) -> ExtendedComparisonTable:
    """Build extended comparison from evaluation results.

    Parameters
    ----------
    all_results:
        ``{model_name: {dataset_name: {subset_name: EvaluationResult}}}``.
    primary_metric:
        Metric to sort entries by.
    primary_metric_direction:
        ``"maximize"`` (default) or ``"minimize"``.

    Returns
    -------
    ExtendedComparisonTable
    """
    if not all_results:
        return ExtendedComparisonTable(
            entries=[],
            metric_names=[],
            dataset_names=[],
            primary_metric=primary_metric,
            primary_metric_direction=primary_metric_direction,
        )

    all_metric_names: set[str] = set()
    all_dataset_names: set[str] = set()
    entries: list[ModelComparisonEntry] = []

    for model_name, ds_results in all_results.items():
        model_type, strategy = _infer_model_type(model_name)

        per_dataset_metrics: dict[str, dict[str, float]] = {}
        per_dataset_per_volume: dict[str, dict[str, list[float]]] = {}

        for ds_name, subset_results in ds_results.items():
            all_dataset_names.add(ds_name)
            ds_agg, ds_per_vol = _aggregate_subset_metrics(subset_results)
            per_dataset_metrics[ds_name] = ds_agg
            per_dataset_per_volume[ds_name] = ds_per_vol
            all_metric_names.update(ds_agg.keys())

        # Compute overall metrics as the mean across datasets
        overall_metrics: dict[str, float] = {}
        for metric_name in all_metric_names:
            values = [
                ds_metrics[metric_name]
                for ds_metrics in per_dataset_metrics.values()
                if metric_name in ds_metrics and not math.isnan(ds_metrics[metric_name])
            ]
            if values:
                overall_metrics[metric_name] = sum(values) / len(values)
            else:
                overall_metrics[metric_name] = float("nan")

        entries.append(
            ModelComparisonEntry(
                model_name=model_name,
                model_type=model_type,
                strategy=strategy,
                per_dataset_metrics=per_dataset_metrics,
                overall_metrics=overall_metrics,
                per_dataset_per_volume=per_dataset_per_volume,
            )
        )

    # Sort entries by primary metric
    reverse = primary_metric_direction == "maximize"

    def _sort_key(entry: ModelComparisonEntry) -> float:
        val = entry.overall_metrics.get(primary_metric, float("nan"))
        # Push NaN to the end
        if math.isnan(val):
            return float("-inf") if reverse else float("inf")
        return val

    entries.sort(key=_sort_key, reverse=reverse)

    return ExtendedComparisonTable(
        entries=entries,
        metric_names=sorted(all_metric_names),
        dataset_names=sorted(all_dataset_names),
        primary_metric=primary_metric,
        primary_metric_direction=primary_metric_direction,
    )


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------


def format_extended_markdown(
    table: ExtendedComparisonTable,
    *,
    include_per_dataset: bool = True,
) -> str:
    """Format extended comparison as rich markdown.

    Produces:
    1. Overall comparison table (sorted by primary metric)
    2. Per-dataset breakdown tables (if *include_per_dataset*)

    Parameters
    ----------
    table:
        Extended comparison table from :func:`build_extended_comparison`.
    include_per_dataset:
        Include per-dataset breakdown sections.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    if not table.entries:
        return "*No results to display.*"

    lines: list[str] = []

    # --- Overall comparison ---
    lines.append("## Overall Comparison")
    lines.append("")
    lines.append(
        _build_markdown_table(
            table.entries,
            table.metric_names,
            source="overall",
        )
    )

    # --- Per-dataset breakdown ---
    if include_per_dataset and table.dataset_names:
        lines.append("")
        for ds_name in table.dataset_names:
            lines.append(f"### Per-dataset: {ds_name}")
            lines.append("")
            lines.append(
                _build_markdown_table(
                    table.entries,
                    table.metric_names,
                    source="per_dataset",
                    dataset_name=ds_name,
                )
            )
            lines.append("")

    return "\n".join(lines)


def _build_markdown_table(
    entries: list[ModelComparisonEntry],
    metric_names: list[str],
    *,
    source: str = "overall",
    dataset_name: str | None = None,
) -> str:
    """Build a markdown table from entries.

    Parameters
    ----------
    entries:
        Model comparison entries.
    metric_names:
        List of metric column names.
    source:
        ``"overall"`` or ``"per_dataset"``.
    dataset_name:
        Required when *source* is ``"per_dataset"``.

    Returns
    -------
    str
        Markdown table string.
    """
    header = "| Model | Type | " + " | ".join(metric_names) + " |"
    separator = "| --- | --- | " + " | ".join(["---"] * len(metric_names)) + " |"

    rows: list[str] = [header, separator]

    for entry in entries:
        if source == "per_dataset" and dataset_name is not None:
            metrics = entry.per_dataset_metrics.get(dataset_name, {})
        else:
            metrics = entry.overall_metrics

        cells: list[str] = [entry.model_name, entry.model_type]
        for metric_name in metric_names:
            val = metrics.get(metric_name, float("nan"))
            if math.isnan(val):
                cells.append("N/A")
            else:
                cells.append(f"{val:.4f}")
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------


def compute_significance_matrix(
    table: ExtendedComparisonTable,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[tuple[str, str], float]:
    """Compute pairwise p-values between all model pairs.

    Uses the primary metric's per-volume scores and
    :func:`~minivess.pipeline.comparison.paired_bootstrap_test`.

    Parameters
    ----------
    table:
        Extended comparison table.
    n_resamples:
        Number of bootstrap resamples.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict[tuple[str, str], float]
        ``{(model_a, model_b): p_value}`` for all unique pairs.
    """
    if len(table.entries) < 2:
        return {}

    primary = table.primary_metric
    pvalues: dict[tuple[str, str], float] = {}

    for entry_a, entry_b in itertools.combinations(table.entries, 2):
        scores_a = _collect_per_volume_scores(entry_a, primary)
        scores_b = _collect_per_volume_scores(entry_b, primary)

        if scores_a is None or scores_b is None:
            logger.warning(
                "Cannot compute significance for %s vs %s: missing per-volume scores for %s",
                entry_a.model_name,
                entry_b.model_name,
                primary,
            )
            pvalues[(entry_a.model_name, entry_b.model_name)] = float("nan")
            continue

        # Align lengths to the shorter array
        min_len = min(len(scores_a), len(scores_b))
        if min_len == 0:
            pvalues[(entry_a.model_name, entry_b.model_name)] = float("nan")
            continue

        p = paired_bootstrap_test(
            scores_a[:min_len],
            scores_b[:min_len],
            n_resamples=n_resamples,
            seed=seed,
        )
        pvalues[(entry_a.model_name, entry_b.model_name)] = p

    return pvalues


def _collect_per_volume_scores(
    entry: ModelComparisonEntry,
    metric_name: str,
) -> np.ndarray | None:
    """Collect per-volume scores across all datasets for a given metric.

    Returns
    -------
    np.ndarray | None
        Concatenated per-volume scores, or ``None`` if no data available.
    """
    all_scores: list[float] = []
    for ds_per_vol in entry.per_dataset_per_volume.values():
        scores = ds_per_vol.get(metric_name, [])
        all_scores.extend(scores)

    if not all_scores:
        return None

    return np.array(all_scores, dtype=float)


def format_significance_markdown(
    pvalues: dict[tuple[str, str], float],
    model_names: list[str],
    *,
    alpha: float = 0.05,
) -> str:
    """Format significance matrix as markdown table.

    Uses significance stars:
    - ``**`` for p < 0.01
    - ``*`` for p < *alpha*
    - ``ns`` for p >= *alpha*

    Parameters
    ----------
    pvalues:
        ``{(model_a, model_b): p_value}`` dict.
    model_names:
        Ordered list of model names for rows/columns.
    alpha:
        Significance threshold. Default 0.05.

    Returns
    -------
    str
        Markdown-formatted significance matrix.
    """
    if not pvalues:
        return "*No significance results to display.*"

    # Build lookup dict (both directions)
    lookup: dict[tuple[str, str], float] = {}
    for key, val in pvalues.items():
        lookup[key] = val
        lookup[(key[1], key[0])] = val

    # Header
    header = "| | " + " | ".join(model_names) + " |"
    separator = "| --- | " + " | ".join(["---"] * len(model_names)) + " |"
    rows: list[str] = [header, separator]

    for row_name in model_names:
        cells: list[str] = [row_name]
        for col_name in model_names:
            if row_name == col_name:
                cells.append("-")
            else:
                p = lookup.get((row_name, col_name), float("nan"))
                if math.isnan(p):
                    cells.append("N/A")
                elif p < 0.01:
                    cells.append(f"**({p:.3f})")
                elif p < alpha:
                    cells.append(f"*({p:.3f})")
                else:
                    cells.append(f"ns({p:.3f})")
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Best model selection
# ---------------------------------------------------------------------------


def find_best_model_overall(
    table: ExtendedComparisonTable,
) -> str:
    """Find the best model by the primary metric.

    Parameters
    ----------
    table:
        Extended comparison table (entries should already be sorted).

    Returns
    -------
    str
        Name of the best model.

    Raises
    ------
    ValueError
        If the table has no entries.
    """
    if not table.entries:
        msg = "No entries in comparison table; cannot find best model"
        raise ValueError(msg)

    # Table entries are pre-sorted by build_extended_comparison; first entry is best
    return table.entries[0].model_name
