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
    FactorialAnovaResult,
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
    *,
    anova_results: list[FactorialAnovaResult] | None = None,
    instability_results: list[dict[str, Any]] | None = None,
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
    anova_results:
        Factorial ANOVA results for interaction and variance figures.
    instability_results:
        Riley bootstrap instability results for instability figures.

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

    # F5: Factorial ANOVA interaction plots and variance lollipops (Task 2.13)
    if anova_results:
        for anova_result in anova_results:
            fig = _generate_interaction_plot(
                anova_result, per_volume_data, output_dir
            )
            if fig is not None:
                figures.append(fig)
            fig = _generate_variance_lollipop(anova_result, output_dir)
            if fig is not None:
                figures.append(fig)

    # F6: Riley bootstrap instability plots (Task 2.13)
    if instability_results:
        for inst_result in instability_results:
            fig = _generate_instability_plot(inst_result, output_dir)
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


# ---------------------------------------------------------------------------
# Factorial ANOVA figure generators (PR-A)
# ---------------------------------------------------------------------------


def _generate_interaction_plot(
    anova_result: FactorialAnovaResult,
    per_volume_data: dict[str, dict[str, dict[int, np.ndarray]]],
    output_dir: Path,
    metrics: list[str] | None = None,
) -> FigureArtifact | None:
    """Generate Model x Loss interaction plot.

    Layout: multi-panel figure (one panel per metric).
    X-axis: Loss function. Y-axis: Mean metric value with 95% bootstrap CI.
    Lines: One per model, color-coded with Okabe-Ito palette.

    Parameters
    ----------
    anova_result:
        FactorialAnovaResult from compute_factorial_anova().
    per_volume_data:
        ``{metric: {condition_key: {fold_id: np.ndarray}}}``.
    output_dir:
        Output directory for figure files.
    metrics:
        Which metrics to plot. Defaults to ``["cldice", "masd"]``.

    Returns
    -------
    FigureArtifact or None if no data available.
    """
    if metrics is None:
        metrics = ["cldice", "masd"]

    # Filter to metrics that actually exist in the data
    available_metrics = [m for m in metrics if m in per_volume_data]
    if not available_metrics:
        return None

    n_panels = len(available_metrics)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    output_dir.mkdir(parents=True, exist_ok=True)

    for panel_idx, metric in enumerate(available_metrics):
        ax = axes[panel_idx]
        metric_data = per_volume_data[metric]

        # Parse condition keys into (model, loss) pairs
        models_set: set[str] = set()
        losses_set: set[str] = set()
        for condition_key in metric_data:
            parts = condition_key.split("__")
            models_set.add(parts[0])
            if len(parts) > 1:
                losses_set.add(parts[1])

        models = sorted(models_set)
        losses = sorted(losses_set)

        for model_idx, model in enumerate(models):
            means = []
            ci_lowers = []
            ci_uppers = []

            for loss in losses:
                condition_key = f"{model}__{loss}"
                if condition_key not in metric_data:
                    means.append(float("nan"))
                    ci_lowers.append(float("nan"))
                    ci_uppers.append(float("nan"))
                    continue

                # Pool scores across folds
                fold_data = metric_data[condition_key]
                scores = np.concatenate(
                    [fold_data[k] for k in sorted(fold_data.keys())]
                )
                mean = float(np.mean(scores))
                # 95% CI via bootstrap (quick percentile method)
                rng = np.random.default_rng(42 + model_idx)
                boot_means = [
                    float(np.mean(rng.choice(scores, size=len(scores), replace=True)))
                    for _ in range(1000)
                ]
                ci_lower = float(np.percentile(boot_means, 2.5))
                ci_upper = float(np.percentile(boot_means, 97.5))
                means.append(mean)
                ci_lowers.append(ci_lower)
                ci_uppers.append(ci_upper)

            color = OKABE_ITO[model_idx % len(OKABE_ITO)]
            x_pos = np.arange(len(losses))
            ax.plot(x_pos, means, marker="o", color=color, label=model, linewidth=1.5)
            ax.fill_between(x_pos, ci_lowers, ci_uppers, alpha=0.15, color=color)

        ax.set_xticks(np.arange(len(losses)))
        ax.set_xticklabels(losses, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f"Interaction: {metric}")
        ax.legend(fontsize=7, loc="best")

    fig.tight_layout()

    fig_id = "interaction_plot"
    paths = _save_figure(fig, output_dir, fig_id)
    plt.close(fig)

    sidecar_path = output_dir / f"{fig_id}.json"
    sidecar_data = {
        "figure_id": fig_id,
        "title": "Model x Loss Interaction Plot",
        "generated_at": datetime.now(UTC).isoformat(),
        "data": {
            "n_panels": n_panels,
            "metrics": available_metrics,
            "models": models,
            "losses": losses,
            "palette": "okabe_ito",
        },
    }
    write_sidecar(sidecar_data, sidecar_path)

    return FigureArtifact(
        figure_id=fig_id,
        title="Model x Loss Interaction Plot",
        paths=paths,
        sidecar_path=sidecar_path,
    )


def _generate_variance_lollipop(
    anova_result: FactorialAnovaResult,
    output_dir: Path,
) -> FigureArtifact | None:
    """Generate variance lollipop chart for partial eta-squared.

    Horizontal lollipop chart with factors on Y-axis and partial eta-squared
    on X-axis. Shows all 5 factors: Model, Loss, Model:Loss, Fold, Residual.

    Parameters
    ----------
    anova_result:
        FactorialAnovaResult from compute_factorial_anova().
    output_dir:
        Output directory for figure files.

    Returns
    -------
    FigureArtifact or None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect factors and their eta-squared values
    factor_order = ["Model", "Loss", "Model:Loss", "Fold", "Residual"]
    factors = []
    values = []
    for factor in factor_order:
        eta = anova_result.eta_squared_partial.get(factor)
        if eta is not None:
            factors.append(factor)
            values.append(eta)

    if not factors:
        return None

    fig, ax = plt.subplots(figsize=(8, max(3, len(factors) * 0.7)))

    y_pos = np.arange(len(factors))
    colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(factors))]

    # Lollipop: stem + dot
    ax.hlines(y_pos, 0, values, colors=colors, linewidth=2)
    ax.scatter(values, y_pos, c=colors, s=80, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(factors)
    ax.set_xlabel(r"Partial $\eta^2$")
    ax.set_title(f"Variance Decomposition: {anova_result.metric}")
    ax.set_xlim(left=0)

    fig.tight_layout()

    fig_id = "variance_lollipop"
    paths = _save_figure(fig, output_dir, fig_id)
    plt.close(fig)

    sidecar_path = output_dir / f"{fig_id}.json"
    sidecar_data = {
        "figure_id": fig_id,
        "title": f"Variance Decomposition: {anova_result.metric}",
        "generated_at": datetime.now(UTC).isoformat(),
        "data": {
            "factors": factors,
            "eta_squared_partial": values,
            "metric": anova_result.metric,
        },
    }
    write_sidecar(sidecar_data, sidecar_path)

    return FigureArtifact(
        figure_id=fig_id,
        title=f"Variance Decomposition: {anova_result.metric}",
        paths=paths,
        sidecar_path=sidecar_path,
    )


def _generate_instability_plot(
    instability_result: dict[str, Any],
    output_dir: Path,
) -> FigureArtifact | None:
    """Generate Riley bootstrap instability plot.

    Shows model rank trajectories across bootstrap iterations.
    X = bootstrap iteration, Y = rank, one line per model.

    Parameters
    ----------
    instability_result:
        Dict from compute_riley_instability() with keys:
        - rank_matrix: np.ndarray of shape (n_bootstrap, n_models)
        - model_names: list[str]
        - stability_fractions: dict[str, float]
        - metric_name: str
    output_dir:
        Output directory for figure files.

    Returns
    -------
    FigureArtifact or None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_matrix = instability_result["rank_matrix"]
    model_names = instability_result["model_names"]
    stability = instability_result["stability_fractions"]
    metric_name = instability_result.get("metric_name", "metric")

    n_bootstrap, n_models = rank_matrix.shape

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot rank trajectory for each model (subsample for readability)
    max_display = min(200, n_bootstrap)
    display_indices = np.linspace(0, n_bootstrap - 1, max_display, dtype=int)

    for model_idx, model_name in enumerate(model_names):
        color = OKABE_ITO[model_idx % len(OKABE_ITO)]
        frac = stability.get(model_name, 0.0)
        label = f"{model_name} (stability={frac:.2f})"
        ax.plot(
            display_indices,
            rank_matrix[display_indices, model_idx],
            color=color,
            alpha=0.6,
            linewidth=0.8,
            label=label,
        )

    ax.set_xlabel("Bootstrap iteration")
    ax.set_ylabel("Rank")
    ax.set_title(f"Ranking Instability ({metric_name})")
    ax.set_yticks(range(1, n_models + 1))
    ax.invert_yaxis()  # Rank 1 at top
    ax.legend(fontsize=7, loc="best")

    fig.tight_layout()

    fig_id = "instability_plot"
    paths = _save_figure(fig, output_dir, fig_id)
    plt.close(fig)

    sidecar_path = output_dir / f"{fig_id}.json"
    sidecar_data = {
        "figure_id": fig_id,
        "title": f"Ranking Instability ({metric_name})",
        "generated_at": datetime.now(UTC).isoformat(),
        "data": {
            "n_bootstrap": n_bootstrap,
            "n_models": n_models,
            "model_names": model_names,
            "stability_fractions": stability,
            "metric_name": metric_name,
        },
    }
    write_sidecar(sidecar_data, sidecar_path)

    return FigureArtifact(
        figure_id=fig_id,
        title=f"Ranking Instability ({metric_name})",
        paths=paths,
        sidecar_path=sidecar_path,
    )


def generate_cost_breakdown_figure(
    cost_summary: dict[str, Any],
    output_dir: Path,
) -> FigureArtifact:
    """Generate cost breakdown stacked bar chart.

    X-axis: model families, Y-axis: cost in USD.
    Stacked segments for each phase (training, post_training, debug, etc.).

    Parameters
    ----------
    cost_summary:
        Dict with ``cost_by_model`` and ``cost_by_phase`` keys.
    output_dir:
        Output directory for figure files.

    Returns
    -------
    FigureArtifact with paths to PNG and JSON sidecar.
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

        # Add value labels on bars
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
