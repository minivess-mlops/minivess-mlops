"""Master figure generation orchestrator.

Registry-based system for generating all paper-quality figures.
Each registered figure has a name, generator function, and category.

Usage::

    uv run python -m minivess.pipeline.viz.generate_all_figures
    uv run python -m minivess.pipeline.viz.generate_all_figures --list
    uv run python -m minivess.pipeline.viz.generate_all_figures --figure loss_comparison
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from minivess.pipeline.comparison import (
    ComparisonTable,
    LossResult,
    MetricSummary,
)
from minivess.pipeline.viz.figure_export import save_figure

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic demo data generators (used when no real data is available)
# ---------------------------------------------------------------------------


def _demo_metric_summary(mean: float, std: float = 0.02) -> MetricSummary:
    """Create a demo MetricSummary."""
    rng = np.random.default_rng(42)
    per_fold = [mean + rng.normal(0, std) for _ in range(3)]
    return MetricSummary(
        mean=mean,
        std=std,
        ci_lower=mean - 0.04,
        ci_upper=mean + 0.04,
        per_fold=per_fold,
    )


def _demo_comparison_table() -> ComparisonTable:
    """Create a demo ComparisonTable."""
    losses = []
    metric_names = ["dsc", "centreline_dsc", "measured_masd"]
    base_values = {
        "dice_ce": {"dsc": 0.82, "centreline_dsc": 0.83, "measured_masd": 0.15},
        "cbdice": {"dsc": 0.78, "centreline_dsc": 0.80, "measured_masd": 0.18},
        "dice_ce_cldice": {"dsc": 0.80, "centreline_dsc": 0.88, "measured_masd": 0.12},
        "cbdice_cldice": {"dsc": 0.79, "centreline_dsc": 0.91, "measured_masd": 0.13},
    }
    for loss_name, values in base_values.items():
        metrics = {m: _demo_metric_summary(values[m]) for m in metric_names}
        losses.append(LossResult(loss_name=loss_name, num_folds=3, metrics=metrics))
    return ComparisonTable(losses=losses, metric_names=metric_names)


# ---------------------------------------------------------------------------
# Figure generators (each returns a Figure)
# ---------------------------------------------------------------------------


def _gen_loss_comparison() -> plt.Figure:
    from minivess.pipeline.viz.loss_comparison import plot_loss_comparison

    return plot_loss_comparison(_demo_comparison_table())


def _gen_fold_heatmap() -> plt.Figure:
    from minivess.pipeline.viz.fold_heatmap import plot_fold_heatmap

    return plot_fold_heatmap(_demo_comparison_table())


def _gen_metric_correlation() -> plt.Figure:
    from minivess.pipeline.viz.metric_correlation import plot_metric_correlation

    return plot_metric_correlation(_demo_comparison_table())


def _gen_sensitivity_heatmap() -> plt.Figure:
    from minivess.pipeline.viz.factorial_analysis import plot_sensitivity_heatmap

    return plot_sensitivity_heatmap(_demo_comparison_table())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FIGURE_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "loss_comparison",
        "generator": _gen_loss_comparison,
        "category": "model_performance",
    },
    {
        "name": "fold_heatmap",
        "generator": _gen_fold_heatmap,
        "category": "model_performance",
    },
    {
        "name": "metric_correlation",
        "generator": _gen_metric_correlation,
        "category": "model_performance",
    },
    {
        "name": "sensitivity_heatmap",
        "generator": _gen_sensitivity_heatmap,
        "category": "factorial_design",
    },
]


def list_figures() -> list[str]:
    """Return list of all registered figure names."""
    return [entry["name"] for entry in FIGURE_REGISTRY]


def generate_figure(
    name: str,
    output_dir: Path | None = None,
) -> Path | None:
    """Generate a single figure by name.

    Parameters
    ----------
    name:
        Registered figure name.
    output_dir:
        Output directory. Defaults to cwd.

    Returns
    -------
    Path to the saved figure, or None if not found.
    """
    for entry in FIGURE_REGISTRY:
        if entry["name"] == name:
            try:
                fig = entry["generator"]()
                result_path: Path | None = save_figure(fig, name, output_dir=output_dir)
                plt.close(fig)
                return result_path
            except Exception:
                logger.exception("Failed to generate figure: %s", name)
                plt.close("all")
                return None
    logger.warning("Unknown figure name: %s", name)
    return None


def generate_all_figures(
    output_dir: Path | None = None,
) -> dict[str, list[str]]:
    """Generate all registered figures.

    Parameters
    ----------
    output_dir:
        Output directory for all figures.

    Returns
    -------
    Summary dict with 'succeeded' and 'failed' lists of figure names.
    """
    succeeded: list[str] = []
    failed: list[str] = []

    for entry in FIGURE_REGISTRY:
        name = entry["name"]
        try:
            fig = entry["generator"]()
            save_figure(fig, name, output_dir=output_dir)
            plt.close(fig)
            succeeded.append(name)
            logger.info("Generated: %s", name)
        except Exception:
            logger.exception("Failed: %s", name)
            plt.close("all")
            failed.append(name)

    logger.info(
        "Figure generation complete: %d succeeded, %d failed",
        len(succeeded),
        len(failed),
    )
    return {"succeeded": succeeded, "failed": failed}


if __name__ == "__main__":
    import argparse
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(description="Generate paper-quality figures")
    parser.add_argument("--figure", help="Generate a specific figure by name")
    parser.add_argument(
        "--list", action="store_true", help="List all registered figures"
    )
    parser.add_argument("--output-dir", default="docs/figures", help="Output directory")
    args = parser.parse_args()

    if args.list:
        for name in list_figures():
            entry = next(e for e in FIGURE_REGISTRY if e["name"] == name)
            print(f"  {name:30s}  [{entry['category']}]")
    elif args.figure:
        result = generate_figure(args.figure, output_dir=_Path(args.output_dir))
        if result:
            print(f"Saved: {result}")
        else:
            print(f"Failed or unknown: {args.figure}")
    else:
        summary = generate_all_figures(output_dir=_Path(args.output_dir))
        print(
            f"Succeeded: {len(summary['succeeded'])}, Failed: {len(summary['failed'])}"
        )
        if summary["failed"]:
            print(f"Failed figures: {summary['failed']}")
