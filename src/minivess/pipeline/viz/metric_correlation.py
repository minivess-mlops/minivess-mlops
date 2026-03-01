"""Metric correlation heatmap visualization (F1.5).

Pairwise Pearson correlation between all metrics, computed
from per-fold point estimates across all loss functions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from minivess.pipeline.viz.figure_dimensions import get_figsize
from minivess.pipeline.viz.plot_config import setup_style

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from minivess.pipeline.comparison import ComparisonTable

logger = logging.getLogger(__name__)


def plot_metric_correlation(
    table: ComparisonTable,
) -> Figure:
    """Create a metric-vs-metric correlation heatmap (F1.5).

    Collects per-fold values for each metric from all losses,
    then computes pairwise Pearson correlation.

    Parameters
    ----------
    table:
        Cross-loss comparison table with per-fold values.

    Returns
    -------
    matplotlib Figure with a symmetric correlation heatmap.
    """
    setup_style()

    # Collect per-fold values for each metric across all losses
    metric_vectors: dict[str, list[float]] = {m: [] for m in table.metric_names}
    for lr in table.losses:
        for metric in table.metric_names:
            summary = lr.metrics.get(metric)
            if summary is not None:
                metric_vectors[metric].extend(summary.per_fold)

    # Build correlation matrix
    n = len(table.metric_names)
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            vi = np.array(metric_vectors[table.metric_names[i]])
            vj = np.array(metric_vectors[table.metric_names[j]])
            min_len = min(len(vi), len(vj))
            if min_len >= 2:
                r = float(np.corrcoef(vi[:min_len], vj[:min_len])[0, 1])
            else:
                r = float("nan")
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    labels = [m.replace("_", " ").title() for m in table.metric_names]
    fig, ax = plt.subplots(figsize=get_figsize("matrix"))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        square=True,
    )

    ax.set_title("Metric Correlation Matrix", fontsize=14, fontweight="bold")
    return fig
