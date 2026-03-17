"""Champion model selection from factorial experiment results.

Selects the best model from a factorial evaluation by ranking with a
compound metric: ``0.5 * clDice + 0.5 * normalize(MASD)``.

CV-average fold strategy is preferred over single-fold results.

PR-D T1 (Issue #825).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_compound_metric(
    cldice: float,
    masd: float,
    masd_max: float,
) -> float:
    """Compute the compound champion selection metric.

    Parameters
    ----------
    cldice:
        clDice score in [0, 1].
    masd:
        Mean Average Surface Distance (lower is better).
    masd_max:
        Maximum MASD across all runs (for normalization).

    Returns
    -------
    Compound metric in [0, 1]. Higher is better.
    """
    masd_normalized = 1.0 if masd_max <= 0 else 1.0 - masd / masd_max
    return 0.5 * cldice + 0.5 * masd_normalized


def select_factorial_champion(
    runs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select the best model from factorial evaluation results.

    Selection logic:
    1. Prefer cv_average fold strategy over single_fold.
    2. Among preferred strategy, rank by compound metric.
    3. Return the best run dict with compound_metric added.

    Parameters
    ----------
    runs:
        List of evaluation run dicts. Each must have keys:
        ``cldice``, ``masd``, ``fold_strategy``.

    Returns
    -------
    Best run dict with ``compound_metric`` key added, or ``None`` if
    ``runs`` is empty.
    """
    if not runs:
        return None

    # Compute max MASD for normalization
    masd_max = max(r["masd"] for r in runs)

    # Add compound metric to each run
    for run in runs:
        run["compound_metric"] = compute_compound_metric(
            cldice=run["cldice"],
            masd=run["masd"],
            masd_max=masd_max,
        )

    # Separate cv_average vs other strategies
    cv_runs = [r for r in runs if r.get("fold_strategy") == "cv_average"]
    other_runs = [r for r in runs if r.get("fold_strategy") != "cv_average"]

    # Prefer cv_average if any exist
    candidates = cv_runs if cv_runs else other_runs

    # Sort by compound metric descending
    candidates.sort(key=lambda r: r["compound_metric"], reverse=True)

    champion = candidates[0]
    logger.info(
        "Selected factorial champion: run=%s model=%s loss=%s compound=%.4f",
        champion.get("run_id"),
        champion.get("model"),
        champion.get("loss"),
        champion["compound_metric"],
    )
    return champion


def build_champion_tags(champion: dict[str, Any]) -> dict[str, str]:
    """Build MLflow tags dict for the champion model.

    Parameters
    ----------
    champion:
        Champion run dict from :func:`select_factorial_champion`.

    Returns
    -------
    Dict of ``champion/*`` tags for MLflow.
    """
    return {
        "champion/model": str(champion.get("model", "unknown")),
        "champion/loss": str(champion.get("loss", "unknown")),
        "champion/aux_calib": str(champion.get("aux_calib", False)),
        "champion/fold_strategy": str(champion.get("fold_strategy", "unknown")),
        "champion/ensemble": str(champion.get("ensemble", "none")),
        "champion/cldice": str(champion.get("cldice", 0.0)),
        "champion/masd": str(champion.get("masd", 0.0)),
        "champion/dsc": str(champion.get("dsc", 0.0)),
        "champion/compound_metric": str(champion.get("compound_metric", 0.0)),
    }


def prepare_registry_promotion(champion: dict[str, Any]) -> dict[str, Any]:
    """Prepare MLflow Model Registry promotion payload.

    Parameters
    ----------
    champion:
        Champion run dict from :func:`select_factorial_champion`.

    Returns
    -------
    Dict with ``model_name``, ``run_id``, and ``tags``.
    """
    tags = build_champion_tags(champion)
    return {
        "model_name": "minivess-champion",
        "run_id": champion["run_id"],
        "tags": tags,
    }
