"""External generalization visualizations (F5.1–F5.3).

F5.1: Domain gap degradation — grouped bars with reference line.
F5.2: Per-volume scatter — DSC vs clDice colored by loss.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from minivess.pipeline.viz.figure_dimensions import get_figsize
from minivess.pipeline.viz.plot_config import COLORS, LOSS_LABELS, setup_style

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def plot_domain_gap(
    external_df: pd.DataFrame,
    metric_col: str = "value",
) -> Figure:
    """Create grouped bar chart of external dataset performance (F5.1).

    Parameters
    ----------
    external_df:
        DataFrame with columns: ``dataset``, ``loss_function``,
        ``metric_name``, ``value``, ``reference_cv_mean``.
    metric_col:
        Column for metric values.

    Returns
    -------
    matplotlib Figure.
    """
    setup_style()

    metrics = sorted(external_df["metric_name"].unique())
    datasets = sorted(external_df["dataset"].unique())
    losses = sorted(external_df["loss_function"].unique())

    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=get_figsize("triple" if n_metrics >= 3 else "double"),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        metric_df = external_df[external_df["metric_name"] == metric]

        n_groups = len(datasets)
        n_bars = len(losses)
        bar_width = 0.8 / max(n_bars, 1)
        x = np.arange(n_groups)

        for bar_idx, loss in enumerate(losses):
            values = []
            for dataset in datasets:
                mask = (metric_df["dataset"] == dataset) & (
                    metric_df["loss_function"] == loss
                )
                val = metric_df.loc[mask, metric_col].values
                values.append(float(val[0]) if len(val) > 0 else 0.0)

            color = COLORS.get(loss, "#999999")
            label = LOSS_LABELS.get(loss, loss)
            offset = (bar_idx - n_bars / 2 + 0.5) * bar_width
            ax.bar(x + offset, values, bar_width, color=color, label=label)

        # Reference line from MiniVess CV mean
        ref_vals = metric_df["reference_cv_mean"].unique()
        if len(ref_vals) > 0:
            ax.axhline(
                float(np.mean(ref_vals)),
                color="#CC6677",
                linestyle="--",
                linewidth=1.5,
                label="MiniVess CV mean",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha="right")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric)
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide extra axes
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("External Generalization", fontsize=14, fontweight="bold")
    return fig


def plot_per_volume_scatter(
    volume_df: pd.DataFrame,
    x_col: str = "dsc",
    y_col: str = "centreline_dsc",
) -> Figure:
    """Create per-volume scatter of DSC vs clDice (F5.2).

    Parameters
    ----------
    volume_df:
        DataFrame with columns: ``volume_id``, ``loss_function``,
        ``dsc``, ``centreline_dsc``.
    x_col:
        Column for x-axis metric.
    y_col:
        Column for y-axis metric.

    Returns
    -------
    matplotlib Figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=get_figsize("single"))

    for loss_name in sorted(volume_df["loss_function"].unique()):
        mask = volume_df["loss_function"] == loss_name
        color = COLORS.get(loss_name, "#999999")
        label = LOSS_LABELS.get(loss_name, loss_name)
        ax.scatter(
            volume_df.loc[mask, x_col],
            volume_df.loc[mask, y_col],
            color=color,
            label=label,
            s=50,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title("Per-Volume Performance", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8)

    return fig
