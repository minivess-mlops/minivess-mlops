"""MLOps/observability visualizations (F4.1–F4.5).

F4.1: Metric drift over time — scatter with color bands.
F4.3: GPU/memory utilization — multi-panel line charts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from minivess.pipeline.viz.figure_dimensions import get_figsize
from minivess.pipeline.viz.plot_config import COLORS, setup_style

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def plot_metric_drift(
    drift_df: pd.DataFrame,
    metric_col: str = "primary_metric",
) -> Figure:
    """Create metric drift scatter with quality bands (F4.1).

    Parameters
    ----------
    drift_df:
        DataFrame with columns: ``run_index``, ``primary_metric``,
        ``loss_function``.
    metric_col:
        Column for metric values.

    Returns
    -------
    matplotlib Figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=get_figsize("double"))

    # Color bands (green/yellow/red)
    ax.axhspan(0.9, 1.0, alpha=0.1, color="green", label="Good (>0.9)")
    ax.axhspan(0.8, 0.9, alpha=0.1, color="gold", label="Warning (0.8–0.9)")
    ax.axhspan(0.0, 0.8, alpha=0.1, color="red", label="Critical (<0.8)")

    # Scatter points colored by loss function
    for loss_name in sorted(drift_df["loss_function"].unique()):
        mask = drift_df["loss_function"] == loss_name
        color = COLORS.get(loss_name, "#999999")
        ax.scatter(
            drift_df.loc[mask, "run_index"],
            drift_df.loc[mask, metric_col],
            color=color,
            label=loss_name.replace("_", " "),
            s=50,
            zorder=3,
        )

    # Rolling average
    sorted_df = drift_df.sort_values("run_index")
    rolling = sorted_df[metric_col].rolling(window=3, min_periods=1).mean()
    ax.plot(
        sorted_df["run_index"],
        rolling,
        color="black",
        linewidth=2,
        alpha=0.7,
        label="Rolling avg (3)",
    )

    ax.set_xlabel("Run Index (Chronological)")
    ax.set_ylabel(metric_col.replace("_", " ").title())
    ax.set_title("Metric Drift Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")

    return fig


def plot_resource_utilization(
    resource_df: pd.DataFrame,
) -> Figure:
    """Create multi-panel resource utilization chart (F4.3).

    Parameters
    ----------
    resource_df:
        DataFrame with columns: ``epoch``, ``loss_function``,
        ``gpu_util_pct``, ``gpu_mem_mb``, ``cpu_util_pct``, ``ram_mb``.

    Returns
    -------
    matplotlib Figure.
    """
    setup_style()

    metrics = [
        ("gpu_util_pct", "GPU Utilization (%)"),
        ("gpu_mem_mb", "GPU Memory (MB)"),
        ("cpu_util_pct", "CPU Utilization (%)"),
        ("ram_mb", "System RAM (MB)"),
    ]
    available = [(col, label) for col, label in metrics if col in resource_df.columns]
    n = len(available)

    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True, squeeze=False)
    axes_flat = axes.flatten()

    for idx, (col, label) in enumerate(available):
        ax = axes_flat[idx]
        for loss_name in sorted(resource_df["loss_function"].unique()):
            mask = resource_df["loss_function"] == loss_name
            loss_df = resource_df.loc[mask].sort_values("epoch")
            color = COLORS.get(loss_name, "#999999")
            ax.plot(
                loss_df["epoch"],
                loss_df[col],
                color=color,
                label=loss_name.replace("_", " "),
                linewidth=1.5,
            )
        ax.set_ylabel(label)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    axes_flat[-1].set_xlabel("Epoch")
    fig.suptitle("Resource Utilization", fontsize=14, fontweight="bold")

    return fig
