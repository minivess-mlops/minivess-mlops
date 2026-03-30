"""Figure generation engine for the biostatistics flow.

Publication figures rendered by R/ggplot2 (src/minivess/pipeline/r_scripts/).
matplotlib retained ONLY for generate_cost_breakdown_figure() (exploratory).

Rule 26: Greenfield — no dual rendering. R is the ONLY publication renderer.
6 matplotlib generators deleted: effect_size_heatmap, forest_plot,
distribution_plot, interaction_plot, variance_lollipop, instability_plot.

Pure functions — no Prefect, no Docker dependency.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt

from minivess.pipeline.biostatistics_types import (
    FigureArtifact,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from minivess.pipeline.biostatistics_types import (
        FactorialAnovaResult,
        PairwiseResult,
        RankingResult,
        VarianceDecompositionResult,
    )

logger = logging.getLogger(__name__)

# Okabe-Ito colorblind-safe palette
OKABE_ITO = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

# Publication rcParams (used by cost_breakdown only)
_RC_PARAMS = {
    "font.size": 9,
    "pdf.fonttype": 42,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def generate_figures(
    per_volume_data: dict[str, dict[str, dict[int, np.ndarray]]],
    pairwise: list[PairwiseResult],
    variance: list[VarianceDecompositionResult],
    rankings: list[RankingResult],
    output_dir: Path,
    *,
    anova_results: list[FactorialAnovaResult] | None = None,
    instability_results: list[dict[str, Any]] | None = None,
) -> list[FigureArtifact]:
    """Generate biostatistics figures.

    Publication figures are rendered by R/ggplot2 (src/minivess/pipeline/r_scripts/).
    Returns empty list — R renderer produces all publication output.
    Kept for API compatibility with the biostatistics flow.
    """
    logger.info(
        "Publication figures rendered by R/ggplot2 — matplotlib generators removed (Rule 26). "
        "Run docker compose run r_figures to generate publication output."
    )
    return []


def write_sidecar(data: dict[str, Any], path: Path) -> Path:
    """Write a JSON sidecar file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def _save_figure(fig: Any, output_dir: Path, fig_id: str) -> list[Path]:
    """Save figure as PNG and SVG."""
    from pathlib import Path as P

    paths = []
    for ext in ("png", "svg"):
        path = P(str(output_dir / f"{fig_id}.{ext}"))
        fig.savefig(str(path))
        paths.append(path)
    return paths


def generate_cost_breakdown_figure(
    cost_summary: dict[str, Any],
    output_dir: Path,
) -> FigureArtifact:
    """Generate cost breakdown stacked bar chart (exploratory — not publication).

    This is the only matplotlib figure retained — no R equivalent exists.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cost_by_model = cost_summary.get("cost_by_model", {})
    total_spot = cost_summary.get("total_spot_cost_usd", 0.0)
    savings_pct = cost_summary.get("savings_pct", 0.0)

    models = sorted(cost_by_model.keys())
    costs = [cost_by_model[m] for m in models]

    with matplotlib.rc_context(_RC_PARAMS):
        fig, ax = plt.subplots(figsize=(8, 5))

        colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(models))]
        bars = ax.bar(models, costs, color=colors, edgecolor="black", linewidth=0.5)

        ax.set_xlabel("Model Family")
        ax.set_ylabel("Cost (USD)")
        ax.set_title(
            f"Training Cost by Model (total=${total_spot:.2f}, "
            f"spot savings={savings_pct:.1f}%)"
        )

        for bar, cost in zip(bars, costs, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.05,
                f"${cost:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        fig.tight_layout()

    fig_id = "cost_breakdown"
    paths = _save_figure(fig, output_dir, fig_id)
    plt.close(fig)

    sidecar_path = output_dir / f"{fig_id}.json"
    sidecar_data = {
        "figure_id": fig_id,
        "title": "Training Cost Breakdown",
        "generated_at": datetime.now(UTC).isoformat(),
        "models": models,
        "costs": costs,
        "total_spot_cost_usd": total_spot,
        "savings_pct": savings_pct,
    }
    write_sidecar(sidecar_data, sidecar_path)

    return FigureArtifact(
        figure_id=fig_id,
        title="Training Cost Breakdown",
        paths=paths,
        sidecar_path=sidecar_path,
    )
