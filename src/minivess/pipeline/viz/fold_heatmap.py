"""Fold × metric heatmap visualization (F1.2).

Heatmap with rows = loss × fold, columns = metrics.
Cell values show per-fold point estimates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from minivess.pipeline.viz.figure_dimensions import get_figsize
from minivess.pipeline.viz.plot_config import LOSS_LABELS, setup_style

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from minivess.pipeline.comparison import ComparisonTable

logger = logging.getLogger(__name__)


def plot_fold_heatmap(
    table: ComparisonTable,
) -> Figure:
    """Create a heatmap of per-fold metric values across losses (F1.2).

    Parameters
    ----------
    table:
        Cross-loss comparison table with per-fold values in each MetricSummary.

    Returns
    -------
    matplotlib Figure with a single heatmap.
    """
    setup_style()

    # Build the data matrix: rows = loss×fold, columns = metrics
    row_labels: list[str] = []
    data_rows: list[list[float]] = []

    for lr in table.losses:
        label = LOSS_LABELS.get(lr.loss_name, lr.loss_name)
        n_folds = lr.num_folds
        for fold_idx in range(n_folds):
            row_labels.append(f"{label} F{fold_idx}")
            row_values: list[float] = []
            for metric in table.metric_names:
                summary = lr.metrics.get(metric)
                if summary is not None and fold_idx < len(summary.per_fold):
                    row_values.append(summary.per_fold[fold_idx])
                else:
                    row_values.append(float("nan"))
            data_rows.append(row_values)

    matrix = np.array(data_rows)
    fig, ax = plt.subplots(figsize=get_figsize("matrix"))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        xticklabels=[m.replace("_", " ").title() for m in table.metric_names],
        yticklabels=row_labels,
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Per-Fold Metric Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Loss Function × Fold")

    return fig
