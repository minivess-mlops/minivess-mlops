"""Figure generation engine for the biostatistics flow.

Publication-quality figures with JSON sidecars. Uses matplotlib + seaborn.
All output to a config-driven output_dir (Docker volume-mounted).

Pure functions — no Prefect, no Docker dependency.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from minivess.pipeline.biostatistics_types import (
    FigureArtifact,
    PairwiseResult,
    RankingResult,
    VarianceDecompositionResult,
)

if TYPE_CHECKING:
    from pathlib import Path

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

# Publication rcParams
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
) -> list[FigureArtifact]:
    """Generate all biostatistics figures.

    Parameters
    ----------
    per_volume_data:
        {metric: {condition: {fold: scores}}}.
    pairwise:
        Pairwise comparison results.
    variance:
        Variance decomposition results.
    rankings:
        Ranking results.
    output_dir:
        Directory for figure outputs.

    Returns
    -------
    List of FigureArtifact references.
    """
    matplotlib.use("Agg")
    plt.rcParams.update(_RC_PARAMS)
    sns.set_palette(OKABE_ITO)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[FigureArtifact] = []

    # F2: Effect size heatmap
    if pairwise:
        fig = _generate_effect_size_heatmap(pairwise, output_dir)
        if fig is not None:
            figures.append(fig)

    # F4: Forest plot
    if pairwise:
        fig = _generate_forest_plot(pairwise, output_dir)
        if fig is not None:
            figures.append(fig)

    # F1: Box/violin plot per condition (simplified raincloud)
    for metric, metric_data in per_volume_data.items():
        fig = _generate_distribution_plot(metric, metric_data, output_dir)
        if fig is not None:
            figures.append(fig)

    logger.info("Generated %d figures in %s", len(figures), output_dir)
    return figures


def write_sidecar(data: dict[str, Any], path: Path) -> Path:
    """Write a JSON sidecar file.

    Parameters
    ----------
    data:
        Sidecar data dict.
    path:
        Output path.

    Returns
    -------
    Path to the written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Internal figure generators
# ---------------------------------------------------------------------------


def _generate_effect_size_heatmap(
    pairwise: list[PairwiseResult],
    output_dir: Path,
) -> FigureArtifact | None:
    """Generate effect size (Cliff's delta) heatmap."""
    conditions = sorted(
        {r.condition_a for r in pairwise} | {r.condition_b for r in pairwise}
    )
    n = len(conditions)
    if n < 2:
        return None

    cond_idx = {c: i for i, c in enumerate(conditions)}
    matrix = np.zeros((n, n))
    for r in pairwise:
        i, j = cond_idx[r.condition_a], cond_idx[r.condition_b]
        matrix[i, j] = r.cliffs_delta
        matrix[j, i] = -r.cliffs_delta

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        xticklabels=conditions,
        yticklabels=conditions,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
    )
    ax.set_title("Effect Size (Cliff's delta)")

    fig_id = "effect_size_heatmap"
    paths = _save_figure(fig, output_dir, fig_id)
    plt.close(fig)

    sidecar_path = output_dir / f"{fig_id}.json"
    sidecar_data = {
        "figure_id": fig_id,
        "title": "Effect Size (Cliff's delta) Heatmap",
        "generated_at": datetime.now(UTC).isoformat(),
        "data": {
            "conditions": conditions,
            "matrix": matrix.tolist(),
        },
    }
    write_sidecar(sidecar_data, sidecar_path)

    return FigureArtifact(
        figure_id=fig_id,
        title="Effect Size (Cliff's delta) Heatmap",
        paths=paths,
        sidecar_path=sidecar_path,
    )


def _generate_forest_plot(
    pairwise: list[PairwiseResult],
    output_dir: Path,
) -> FigureArtifact | None:
    """Generate forest plot of effect sizes with significance markers."""
    if not pairwise:
        return None

    labels = [f"{r.condition_a} vs {r.condition_b}" for r in pairwise]
    effects = [r.cohens_d for r in pairwise]
    sig = ["*" if r.significant else "" for r in pairwise]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.5)))
    y_pos = range(len(labels))
    ax.barh(y_pos, effects, color=OKABE_ITO[0], alpha=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f"{label} {s}" for label, s in zip(labels, sig, strict=True)])
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Cohen's d")
    ax.set_title("Pairwise Effect Sizes (Forest Plot)")

    fig_id = "forest_plot"
    paths = _save_figure(fig, output_dir, fig_id)
    plt.close(fig)

    sidecar_path = output_dir / f"{fig_id}.json"
    sidecar_data = {
        "figure_id": fig_id,
        "title": "Pairwise Effect Sizes (Forest Plot)",
        "generated_at": datetime.now(UTC).isoformat(),
        "data": {
            "labels": labels,
            "effects": effects,
            "significant": [r.significant for r in pairwise],
        },
    }
    write_sidecar(sidecar_data, sidecar_path)

    return FigureArtifact(
        figure_id=fig_id,
        title="Pairwise Effect Sizes (Forest Plot)",
        paths=paths,
        sidecar_path=sidecar_path,
    )


def _generate_distribution_plot(
    metric: str,
    metric_data: dict[str, dict[int, np.ndarray]],
    output_dir: Path,
) -> FigureArtifact | None:
    """Generate box + strip distribution plot per condition."""
    conditions = sorted(metric_data.keys())
    if not conditions:
        return None

    all_values = []
    all_labels = []
    for cond in conditions:
        fold_data = metric_data[cond]
        scores = np.concatenate([fold_data[k] for k in sorted(fold_data.keys())])
        all_values.extend(scores.tolist())
        all_labels.extend([cond] * len(scores))

    fig, ax = plt.subplots(figsize=(8, 5))
    import pandas as pd

    df = pd.DataFrame({"Condition": all_labels, metric: all_values})
    sns.boxplot(
        data=df, x="Condition", y=metric, ax=ax, palette=OKABE_ITO[: len(conditions)]
    )
    sns.stripplot(
        data=df, x="Condition", y=metric, ax=ax, color="black", alpha=0.3, size=3
    )
    ax.set_title(f"Distribution: {metric}")

    fig_id = f"distribution_{metric}"
    paths = _save_figure(fig, output_dir, fig_id)
    plt.close(fig)

    sidecar_path = output_dir / f"{fig_id}.json"
    sidecar_data = {
        "figure_id": fig_id,
        "title": f"Distribution: {metric}",
        "generated_at": datetime.now(UTC).isoformat(),
        "data": {
            "conditions": conditions,
            "n_samples": len(all_values),
        },
    }
    write_sidecar(sidecar_data, sidecar_path)

    return FigureArtifact(
        figure_id=fig_id,
        title=f"Distribution: {metric}",
        paths=paths,
        sidecar_path=sidecar_path,
    )


def _save_figure(fig: Any, output_dir: Path, fig_id: str) -> list[Path]:
    """Save figure as PNG and SVG."""
    from pathlib import Path as P

    paths = []
    for ext in ("png", "svg"):
        path = P(str(output_dir / f"{fig_id}.{ext}"))
        fig.savefig(str(path))
        paths.append(path)
    return paths
