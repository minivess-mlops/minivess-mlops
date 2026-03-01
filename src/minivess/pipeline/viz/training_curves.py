"""Training curves visualization (F1.3).

Line plots of training metrics over epochs, one subplot per metric,
one line per loss function with ±1 std bands across folds.
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


def plot_training_curves(
    history: pd.DataFrame,
) -> Figure:
    """Plot training curves with mean ± std bands per loss function (F1.3).

    Parameters
    ----------
    history:
        DataFrame with columns: ``epoch``, ``loss_function``, ``metric_name``,
        ``value``, ``fold_id``.

    Returns
    -------
    matplotlib Figure with one subplot per unique ``metric_name``.
    """
    setup_style()

    metric_names = sorted(history["metric_name"].unique())
    n_metrics = len(metric_names)
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

    loss_functions = sorted(history["loss_function"].unique())

    for idx, metric in enumerate(metric_names):
        ax = axes_flat[idx]
        metric_df = history[history["metric_name"] == metric]

        for loss_name in loss_functions:
            loss_df = metric_df[metric_df["loss_function"] == loss_name]
            if loss_df.empty:
                continue

            # Aggregate across folds: mean ± std per epoch
            grouped = loss_df.groupby("epoch")["value"]
            means = grouped.mean()
            stds = grouped.std().fillna(0)
            epochs = means.index.values

            color = COLORS.get(loss_name, "#999999")
            label = LOSS_LABELS.get(loss_name, loss_name)

            ax.plot(epochs, means.values, color=color, label=label, linewidth=2)
            ax.fill_between(
                epochs,
                (means - stds).values,
                (means + stds).values,
                color=color,
                alpha=0.2,
            )

            # Mark best epoch
            is_loss_metric = "loss" in metric.lower()
            best_epoch = int(
                np.argmin(means.values) if is_loss_metric else np.argmax(means.values)
            )
            best_val = means.values[best_epoch]
            ax.axvline(
                epochs[best_epoch],
                color=color,
                linestyle=":",
                alpha=0.5,
            )
            ax.scatter(
                [epochs[best_epoch]],
                [best_val],
                color=color,
                marker="*",
                s=100,
                zorder=5,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.legend(fontsize=8, loc="best")

    # Hide unused axes
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Training Curves (Mean ± Std across Folds)", fontsize=14, fontweight="bold"
    )
    return fig
