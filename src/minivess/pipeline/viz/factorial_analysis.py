"""Factorial design visualizations (F2.1–F2.5).

F2.1: Parallel coordinates — config dimensions as axes.
F2.2: Specification curve — sorted configs with CI + binary matrix.
F2.3: Sensitivity heatmap — loss × metric performance matrix.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from minivess.pipeline.viz.figure_dimensions import get_figsize
from minivess.pipeline.viz.plot_config import COLORS, LOSS_LABELS, setup_style

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

    from minivess.pipeline.comparison import ComparisonTable

logger = logging.getLogger(__name__)


def plot_specification_curve(
    config_df: pd.DataFrame,
    metric_col: str = "primary_metric",
) -> Figure:
    """Create a specification curve plot (F2.2).

    Parameters
    ----------
    config_df:
        DataFrame with columns: ``loss_function``, ``fold_id``,
        ``architecture``, ``primary_metric``, ``ci_lower``, ``ci_upper``.
    metric_col:
        Column name for the primary metric.

    Returns
    -------
    matplotlib Figure with two panels: scatter (top) + binary matrix (bottom).
    """
    setup_style()

    df = config_df.sort_values(metric_col, ascending=False).reset_index(drop=True)
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=get_figsize("specification_curve"),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Top panel: sorted metric values with CI
    x = np.arange(len(df))
    colors_list = [COLORS.get(loss, "#999999") for loss in df["loss_function"]]

    ax_top.scatter(x, df[metric_col], c=colors_list, s=40, zorder=3)
    if "ci_lower" in df.columns and "ci_upper" in df.columns:
        for i, row in df.iterrows():
            ax_top.plot(
                [i, i],
                [row["ci_lower"], row["ci_upper"]],
                color=colors_list[int(i)],
                alpha=0.5,
                linewidth=1.5,
            )

    ax_top.set_ylabel(metric_col.replace("_", " ").title())
    ax_top.set_title("Specification Curve", fontsize=14, fontweight="bold")

    # Bottom panel: binary matrix (which loss, which fold)
    loss_names = sorted(df["loss_function"].unique())
    matrix = np.zeros((len(loss_names), len(df)))
    for col_idx, (_, row) in enumerate(df.iterrows()):
        row_idx = loss_names.index(row["loss_function"])
        matrix[row_idx, col_idx] = 1

    ax_bot.imshow(matrix, aspect="auto", cmap="Greys", interpolation="none")
    ax_bot.set_yticks(range(len(loss_names)))
    ax_bot.set_yticklabels([LOSS_LABELS.get(n, n) for n in loss_names], fontsize=8)
    ax_bot.set_xlabel("Configuration (sorted by metric)")

    return fig


def plot_parallel_coordinates(
    config_df: pd.DataFrame,
    metric_col: str = "primary_metric",
) -> Figure:
    """Create a parallel coordinates plot (F2.1).

    Parameters
    ----------
    config_df:
        DataFrame with columns: ``loss_function``, ``fold_id``,
        ``architecture``, ``primary_metric``.
    metric_col:
        Column for color mapping.

    Returns
    -------
    matplotlib Figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=get_figsize("double"))

    # Encode categorical columns as numeric
    df = config_df.copy()
    loss_names = sorted(df["loss_function"].unique())
    arch_names = (
        sorted(df["architecture"].unique()) if "architecture" in df.columns else []
    )
    df["loss_num"] = df["loss_function"].map({n: i for i, n in enumerate(loss_names)})
    if arch_names:
        df["arch_num"] = df["architecture"].map(
            {n: i for i, n in enumerate(arch_names)}
        )
        dims = ["loss_num", "arch_num", "fold_id", metric_col]
    else:
        dims = ["loss_num", "fold_id", metric_col]

    # Normalize each dimension to [0, 1]
    normalized = df[dims].copy()
    for dim in dims:
        col_min = normalized[dim].min()
        col_max = normalized[dim].max()
        if col_max > col_min:
            normalized[dim] = (normalized[dim] - col_min) / (col_max - col_min)

    # Draw lines
    x_positions = np.arange(len(dims))
    for _, row in normalized.iterrows():
        color = COLORS.get(
            config_df.loc[row.name, "loss_function"]
            if hasattr(row, "name") and row.name in config_df.index
            else "",
            "#999999",
        )
        ax.plot(
            x_positions, [row[d] for d in dims], color=color, alpha=0.5, linewidth=1.5
        )

    ax.set_xticks(x_positions)
    dim_labels = [d.replace("_", " ").title() for d in dims]
    ax.set_xticklabels(dim_labels)
    ax.set_title("Parallel Coordinates", fontsize=14, fontweight="bold")

    return fig


def plot_sensitivity_heatmap(
    table: ComparisonTable,
) -> Figure:
    """Create a sensitivity heatmap of loss × metric (F2.3).

    Parameters
    ----------
    table:
        Cross-loss comparison table.

    Returns
    -------
    matplotlib Figure.
    """
    setup_style()

    # Build matrix: rows=losses, cols=metrics
    loss_names = [lr.loss_name for lr in table.losses]
    data = np.zeros((len(loss_names), len(table.metric_names)))
    for i, lr in enumerate(table.losses):
        for j, metric in enumerate(table.metric_names):
            summary = lr.metrics.get(metric)
            data[i, j] = summary.mean if summary else float("nan")

    fig, ax = plt.subplots(figsize=get_figsize("matrix"))
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        xticklabels=[m.replace("_", " ").title() for m in table.metric_names],
        yticklabels=[LOSS_LABELS.get(n, n) for n in loss_names],
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Sensitivity Heatmap (Loss × Metric)", fontsize=14, fontweight="bold")
    return fig
