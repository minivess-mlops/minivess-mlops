"""Loss comparison visualizations (F1.1 box/violin, F1.4 forest plot).

F1.1: Box/violin plots — one subplot per metric, grouped by loss function.
F1.4: Forest plot — CI bars with effect sizes from pairwise comparisons.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from minivess.pipeline.viz.figure_dimensions import get_figsize
from minivess.pipeline.viz.plot_config import COLORS, LOSS_LABELS, setup_style

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from minivess.pipeline.comparison import ComparisonTable, PairwiseComparison

logger = logging.getLogger(__name__)


def plot_loss_comparison(
    table: ComparisonTable,
    pairwise_comparisons: list[PairwiseComparison] | None = None,
) -> Figure:
    """Create box/violin plots of per-fold metrics across loss functions (F1.1).

    Parameters
    ----------
    table:
        Cross-loss comparison table with per-fold values.
    pairwise_comparisons:
        Optional pairwise significance results for annotation.

    Returns
    -------
    matplotlib Figure with one subplot per metric.
    """
    setup_style()

    n_metrics = len(table.metric_names)
    n_cols = min(n_metrics, 3)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    if n_metrics == 1:
        figsize = get_figsize("single")
    elif n_metrics <= 3:
        figsize = get_figsize("double")
    else:
        figsize = get_figsize("triple")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for idx, metric_name in enumerate(table.metric_names):
        ax = axes_flat[idx]
        _draw_metric_boxplot(ax, table, metric_name)

        if pairwise_comparisons:
            _annotate_significance(ax, table, pairwise_comparisons, metric_name)

    # Hide unused axes
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Loss Function Comparison (Per-Fold)", fontsize=14, fontweight="bold")
    return fig


def _draw_metric_boxplot(
    ax: plt.Axes,
    table: ComparisonTable,
    metric_name: str,
) -> None:
    """Draw a single box plot for one metric across all losses."""
    data: list[list[float]] = []
    labels: list[str] = []
    colors: list[str] = []

    for lr in table.losses:
        summary = lr.metrics.get(metric_name)
        if summary is None or not summary.per_fold:
            continue
        data.append(summary.per_fold)
        labels.append(LOSS_LABELS.get(lr.loss_name, lr.loss_name))
        colors.append(COLORS.get(lr.loss_name, "#999999"))

    if not data:
        ax.set_title(metric_name)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    positions = list(range(len(data)))
    bp = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markersize": 6},
    )

    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual fold points
    for i, fold_vals in enumerate(data):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(fold_vals))
        ax.scatter(
            [positions[i] + j for j in jitter],
            fold_vals,
            color=colors[i],
            edgecolor="black",
            s=40,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(metric_name.replace("_", " ").title())
    ax.set_ylabel(metric_name)


def _annotate_significance(
    ax: plt.Axes,
    table: ComparisonTable,
    pairwise: list[PairwiseComparison],
    metric_name: str,
) -> None:
    """Add significance stars above compared pairs."""
    metric_pairs = [p for p in pairwise if p.metric == metric_name]
    if not metric_pairs:
        return

    loss_names = [lr.loss_name for lr in table.losses]
    y_max = ax.get_ylim()[1]
    step = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08

    for i, pair in enumerate(metric_pairs):
        if pair.loss_a not in loss_names or pair.loss_b not in loss_names:
            continue
        x1 = loss_names.index(pair.loss_a)
        x2 = loss_names.index(pair.loss_b)
        y = y_max + step * (i + 1)

        star = (
            "***"
            if pair.adjusted_p_value < 0.001
            else (
                "**"
                if pair.adjusted_p_value < 0.01
                else ("*" if pair.is_significant else "ns")
            )
        )

        ax.plot(
            [x1, x1, x2, x2],
            [y - step * 0.2, y, y, y - step * 0.2],
            color="black",
            lw=1,
        )
        ax.text((x1 + x2) / 2, y, star, ha="center", va="bottom", fontsize=10)


def plot_forest_comparison(
    table: ComparisonTable,
    pairwise_comparisons: list[PairwiseComparison],
    metric: str | None = None,
) -> Figure:
    """Create a forest plot with CI bars and effect sizes (F1.4).

    Parameters
    ----------
    table:
        Cross-loss comparison table.
    pairwise_comparisons:
        Pairwise significance results with effect sizes.
    metric:
        Metric to show. Defaults to the first metric in the table.

    Returns
    -------
    matplotlib Figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=get_figsize("forest"))

    target_metric = metric or (table.metric_names[0] if table.metric_names else "dsc")

    # Plot per-loss CI bars
    y_positions: list[float] = []
    y_labels: list[str] = []
    y_idx = 0.0

    for lr in table.losses:
        summary = lr.metrics.get(target_metric)
        if summary is None:
            continue

        color = COLORS.get(lr.loss_name, "#999999")
        label = LOSS_LABELS.get(lr.loss_name, lr.loss_name)

        ax.errorbar(
            x=summary.mean,
            y=y_idx,
            xerr=[[summary.mean - summary.ci_lower], [summary.ci_upper - summary.mean]],
            fmt="o",
            color=color,
            markersize=8,
            capsize=4,
            linewidth=2,
        )
        y_positions.append(y_idx)
        y_labels.append(label)
        y_idx += 1.0

    # Add vertical reference line at overall mean
    if y_positions:
        all_means = [
            lr.metrics[target_metric].mean
            for lr in table.losses
            if target_metric in lr.metrics
        ]
        if all_means:
            ax.axvline(
                np.mean(all_means),
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="Overall mean",
            )

    # Annotate pairwise effect sizes on the right
    if pairwise_comparisons:
        metric_pairs = [p for p in pairwise_comparisons if p.metric == target_metric]
        loss_names = [lr.loss_name for lr in table.losses]
        for pair in metric_pairs:
            if pair.loss_a in loss_names and pair.loss_b in loss_names:
                mid_y = (
                    loss_names.index(pair.loss_a) + loss_names.index(pair.loss_b)
                ) / 2
                sig_marker = "*" if pair.is_significant else ""
                ax.annotate(
                    f"d={pair.effect_size:.2f}{sig_marker}",
                    xy=(ax.get_xlim()[1], mid_y),
                    fontsize=9,
                    ha="left",
                    va="center",
                )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(target_metric.replace("_", " ").title())
    ax.set_title(
        f"Forest Plot — {target_metric.replace('_', ' ').title()}", fontweight="bold"
    )
    ax.invert_yaxis()

    return fig
