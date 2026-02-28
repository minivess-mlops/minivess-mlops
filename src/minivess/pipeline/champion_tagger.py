"""Champion model tagging system for MLflow training runs.

Tags the best models across three categories directly on the MLflow
filesystem so they can be filtered with pandas/polars:

- ``champion_best_single_fold``  — best individual fold (any loss)
- ``champion_best_cv_mean``      — loss function with best 3-fold CV mean
- ``champion_best_ensemble``     — all member runs of the best ensemble

Tags are idempotent: every invocation clears ALL ``champion_*`` tags
before writing new ones.

Pattern reference: ``mlruns_enhancement.py`` (filesystem-first tags).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHAMPION_TAG_KEYS: frozenset[str] = frozenset(
    {
        "champion_best_single_fold",
        "champion_best_cv_mean",
        "champion_best_ensemble",
        "champion_metric_name",
        "champion_metric_value",
        "champion_fold_id",
        "champion_ensemble_strategy",
        "champion_tagged_at",
    }
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SingleFoldChampion:
    """Best individual fold across all losses."""

    model_name: str
    loss_function: str
    fold_id: int
    metric_value: float


@dataclass(frozen=True)
class CvMeanChampion:
    """Loss function with best cross-validation mean."""

    loss_function: str
    metric_value: float


@dataclass(frozen=True)
class EnsembleChampion:
    """Best ensemble strategy."""

    ensemble_strategy: str
    metric_value: float
    member_run_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChampionSelection:
    """Complete champion selection across all three categories."""

    primary_metric: str
    best_single_fold: SingleFoldChampion | None
    best_cv_mean: CvMeanChampion | None
    best_ensemble: EnsembleChampion | None


# ---------------------------------------------------------------------------
# Pure selection logic (no I/O)
# ---------------------------------------------------------------------------


def select_champions(
    analysis_entries: list[dict[str, Any]],
    *,
    primary_metric: str,
    maximize: bool,
) -> ChampionSelection:
    """Select champion models from analysis entries (pure logic, no I/O).

    Parameters
    ----------
    analysis_entries:
        Flat list of entry dicts from ``create_analysis_experiment()``.
        Each has ``entry_type``, ``model_name``, ``loss_function``,
        ``fold_id``, ``primary_metric_value``.
    primary_metric:
        Name of the primary metric used for ranking.
    maximize:
        Whether higher is better (``True``) or lower (``False``).

    Returns
    -------
    :class:`ChampionSelection` with winners for each category, or
    ``None`` for categories with no valid candidates.
    """
    best_single_fold: SingleFoldChampion | None = None
    best_cv_mean: CvMeanChampion | None = None
    best_ensemble: EnsembleChampion | None = None

    single_fold_score = float("-inf") if maximize else float("inf")
    cv_mean_score = float("-inf") if maximize else float("inf")
    ensemble_score = float("-inf") if maximize else float("inf")

    for entry in analysis_entries:
        val = entry["primary_metric_value"]
        if math.isnan(val):
            continue

        entry_type = entry["entry_type"]

        if entry_type == "per_fold":
            if _is_better(val, single_fold_score, maximize):
                single_fold_score = val
                best_single_fold = SingleFoldChampion(
                    model_name=entry["model_name"],
                    loss_function=entry["loss_function"],
                    fold_id=entry["fold_id"],
                    metric_value=val,
                )

        elif entry_type == "cv_mean":
            if _is_better(val, cv_mean_score, maximize):
                cv_mean_score = val
                best_cv_mean = CvMeanChampion(
                    loss_function=entry["loss_function"],
                    metric_value=val,
                )

        elif entry_type == "ensemble" and _is_better(val, ensemble_score, maximize):
            ensemble_score = val
            best_ensemble = EnsembleChampion(
                ensemble_strategy=entry["model_name"],
                metric_value=val,
                member_run_ids=[],  # Populated later by caller
            )

    return ChampionSelection(
        primary_metric=primary_metric,
        best_single_fold=best_single_fold,
        best_cv_mean=best_cv_mean,
        best_ensemble=best_ensemble,
    )


def _is_better(new: float, current: float, maximize: bool) -> bool:
    """Return True if *new* is strictly better than *current*."""
    if maximize:
        return new > current
    return new < current


def rank_then_aggregate(
    entries: list[dict[str, Any]],
    *,
    maximize_metrics: list[str],
    minimize_metrics: list[str],
    topology_metric: str = "cldice",
    overlap_metric: str = "dsc",
) -> dict[str, str]:
    """Select champions via rank-then-aggregate across multiple metrics.

    For each metric, models are ranked (1 = best). NaN values get worst rank.
    The balanced champion has the lowest mean rank across all metrics.
    Topology champion is the best on `topology_metric`.
    Overlap champion is the best on `overlap_metric`.

    Parameters
    ----------
    entries:
        List of dicts with 'model_id' and metric values as keys.
    maximize_metrics:
        Metrics where higher is better.
    minimize_metrics:
        Metrics where lower is better.
    topology_metric:
        Metric for topology champion (default: 'cldice').
    overlap_metric:
        Metric for overlap champion (default: 'dsc').

    Returns
    -------
    Dict with keys 'balanced', 'topology', 'overlap', each mapping to
    the winning model_id.
    """
    if not entries:
        return {"balanced": "", "topology": "", "overlap": ""}

    n = len(entries)
    all_metrics = [*maximize_metrics, *minimize_metrics]
    model_ids = [e["model_id"] for e in entries]

    # Compute ranks per metric
    ranks: dict[str, list[float]] = {}
    for metric in all_metrics:
        values = []
        for e in entries:
            val = e.get(metric, float("nan"))
            values.append(val)

        # Determine sort direction
        reverse = metric in maximize_metrics  # maximize → descending sort

        # Sort indices, NaN always last
        indexed = list(enumerate(values))
        indexed.sort(
            key=lambda iv: (
                math.isnan(iv[1]),
                -iv[1] if reverse else iv[1],
            )
        )

        # Assign ranks (1-based)
        metric_ranks = [0.0] * n
        for rank_pos, (orig_idx, _val) in enumerate(indexed):
            metric_ranks[orig_idx] = float(rank_pos + 1)
        ranks[metric] = metric_ranks

    # Compute mean rank per model
    mean_ranks = []
    for i in range(n):
        total = sum(ranks[m][i] for m in all_metrics)
        mean_ranks.append(total / len(all_metrics))

    # Balanced champion: lowest mean rank (first in input order for ties)
    best_balanced_idx = min(range(n), key=lambda i: mean_ranks[i])

    # Topology champion: best on topology_metric
    best_topo_idx = _best_by_metric(
        entries, topology_metric, topology_metric in maximize_metrics
    )

    # Overlap champion: best on overlap_metric
    best_overlap_idx = _best_by_metric(
        entries, overlap_metric, overlap_metric in maximize_metrics
    )

    return {
        "balanced": model_ids[best_balanced_idx],
        "topology": model_ids[best_topo_idx],
        "overlap": model_ids[best_overlap_idx],
    }


def _best_by_metric(entries: list[dict[str, Any]], metric: str, maximize: bool) -> int:
    """Find the index of the entry with the best value for a single metric."""
    best_idx = 0
    best_val = float("-inf") if maximize else float("inf")

    for i, e in enumerate(entries):
        val = e.get(metric, float("nan"))
        if math.isnan(val):
            continue
        if maximize and val > best_val or not maximize and val < best_val:
            best_val = val
            best_idx = i

    return best_idx


# ---------------------------------------------------------------------------
# Filesystem operations
# ---------------------------------------------------------------------------


def clear_champion_tags_filesystem(
    mlruns_dir: Path,
    experiment_id: str,
) -> int:
    """Delete all ``champion_*`` tag files from every run in the experiment.

    Parameters
    ----------
    mlruns_dir:
        Root mlruns directory.
    experiment_id:
        MLflow experiment ID string.

    Returns
    -------
    Number of tag files deleted.
    """
    experiment_dir = mlruns_dir / experiment_id
    if not experiment_dir.is_dir():
        return 0

    deleted = 0
    for run_dir in experiment_dir.iterdir():
        if not run_dir.is_dir():
            continue
        tags_dir = run_dir / "tags"
        if not tags_dir.is_dir():
            continue
        for tag_file in tags_dir.iterdir():
            if tag_file.is_file() and tag_file.name.startswith("champion_"):
                tag_file.unlink()
                deleted += 1

    logger.info("Cleared %d champion tag(s) from experiment %s", deleted, experiment_id)
    return deleted


def write_champion_tags_filesystem(
    mlruns_dir: Path,
    experiment_id: str,
    selection: ChampionSelection,
    *,
    runs: list[dict[str, Any]] | None = None,
    ensembles: dict[str, Any] | None = None,
) -> int:
    """Write champion tags to the filesystem.

    Parameters
    ----------
    mlruns_dir:
        Root mlruns directory.
    experiment_id:
        MLflow experiment ID string.
    selection:
        Champion selection result from :func:`select_champions`.
    runs:
        Training run info dicts for loss_function → run_id mapping.
    ensembles:
        Ensemble specs for member run_id resolution.

    Returns
    -------
    Number of tag files written.
    """
    runs = runs or []
    now = datetime.now(UTC).isoformat()
    written = 0

    # Build loss_function → list[run_id] mapping
    loss_to_run_ids: dict[str, list[str]] = {}
    for run in runs:
        loss = run["loss_type"]
        loss_to_run_ids.setdefault(loss, []).append(run["run_id"])

    # --- best_single_fold ---
    if selection.best_single_fold is not None:
        sf = selection.best_single_fold
        run_ids = loss_to_run_ids.get(sf.loss_function, [])
        for run_id in run_ids:
            tags = {
                "champion_best_single_fold": "true",
                "champion_metric_name": selection.primary_metric,
                "champion_metric_value": str(sf.metric_value),
                "champion_fold_id": str(sf.fold_id),
                "champion_tagged_at": now,
            }
            written += _write_tags_to_run(mlruns_dir, experiment_id, run_id, tags)

    # --- best_cv_mean ---
    if selection.best_cv_mean is not None:
        cv = selection.best_cv_mean
        run_ids = loss_to_run_ids.get(cv.loss_function, [])
        for run_id in run_ids:
            tags = {
                "champion_best_cv_mean": "true",
                "champion_metric_name": selection.primary_metric,
                "champion_metric_value": str(cv.metric_value),
                "champion_tagged_at": now,
            }
            written += _write_tags_to_run(mlruns_dir, experiment_id, run_id, tags)

    # --- best_ensemble ---
    if selection.best_ensemble is not None:
        ens = selection.best_ensemble
        member_ids = ens.member_run_ids

        # If member_run_ids is empty, try to resolve from ensembles dict
        if not member_ids and ensembles:
            spec = ensembles.get(ens.ensemble_strategy)
            if spec is not None and hasattr(spec, "members"):
                member_ids = [m.run_id for m in spec.members]

        for run_id in member_ids:
            tags = {
                "champion_best_ensemble": "true",
                "champion_metric_name": selection.primary_metric,
                "champion_metric_value": str(ens.metric_value),
                "champion_ensemble_strategy": ens.ensemble_strategy,
                "champion_tagged_at": now,
            }
            written += _write_tags_to_run(mlruns_dir, experiment_id, run_id, tags)

    logger.info(
        "Wrote %d champion tag file(s) to experiment %s", written, experiment_id
    )
    return written


def _write_tags_to_run(
    mlruns_dir: Path,
    experiment_id: str,
    run_id: str,
    tags: dict[str, str],
) -> int:
    """Write tag files to a single run's tags directory.

    Overwrites existing champion tags (called after clear).

    Returns number of files written.
    """
    tags_dir = mlruns_dir / experiment_id / run_id / "tags"
    if not tags_dir.is_dir():
        return 0

    written = 0
    for key, value in tags.items():
        tag_file = tags_dir / key
        # Don't overwrite non-champion tags that happen to share a name
        if tag_file.exists() and not key.startswith("champion_"):
            continue
        tag_file.write_text(value, encoding="utf-8")
        written += 1

    return written


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------


def tag_champions(
    mlruns_dir: Path,
    experiment_id: str,
    analysis_entries: list[dict[str, Any]],
    *,
    runs: list[dict[str, Any]] | None = None,
    ensembles: dict[str, Any] | None = None,
    primary_metric: str = "val_compound_masd_cldice",
    maximize: bool = True,
) -> ChampionSelection:
    """End-to-end champion tagging: select, clear, write.

    Parameters
    ----------
    mlruns_dir:
        Root mlruns directory.
    experiment_id:
        MLflow experiment ID string.
    analysis_entries:
        Flat list of analysis entry dicts.
    runs:
        Training run info dicts.
    ensembles:
        Ensemble specs.
    primary_metric:
        Name of the primary ranking metric.
    maximize:
        Whether higher is better.

    Returns
    -------
    :class:`ChampionSelection` with the selected champions.
    """
    # 1. Select champions (pure logic)
    selection = select_champions(
        analysis_entries,
        primary_metric=primary_metric,
        maximize=maximize,
    )

    # 2. Clear all existing champion tags (idempotent)
    clear_champion_tags_filesystem(mlruns_dir, experiment_id)

    # 3. Write new champion tags
    write_champion_tags_filesystem(
        mlruns_dir,
        experiment_id,
        selection,
        runs=runs,
        ensembles=ensembles,
    )

    return selection
