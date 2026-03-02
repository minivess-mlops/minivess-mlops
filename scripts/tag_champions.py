"""Tag champion models on real dynunet_loss_variation_v2 experiment.

Reads eval metrics from MLflow filesystem, selects champions using both
single-metric (select_champions) and multi-metric (rank_then_aggregate)
strategies, and writes champion tags to run directories.

Run:
    uv run python scripts/tag_champions.py
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from minivess.pipeline.champion_tagger import (
    rank_then_aggregate,
    tag_champions,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT_ID = "843896622863223169"
N_FOLDS = 3

# Metrics available per fold in the experiment
EVAL_METRICS = ["dsc", "centreline_dsc", "measured_masd"]


def read_metric(run_id: str, metric_name: str) -> float:
    """Read the last metric value from MLflow filesystem."""
    metric_file = MLRUNS_DIR / EXPERIMENT_ID / run_id / "metrics" / metric_name
    if not metric_file.exists():
        return float("nan")
    lines = metric_file.read_text(encoding="utf-8").strip().split("\n")
    if lines:
        parts = lines[-1].split()
        if len(parts) >= 2:
            return float(parts[1])
    return float("nan")


def read_tag(run_id: str, tag_key: str) -> str:
    """Read a tag value from MLflow filesystem."""
    tag_file = MLRUNS_DIR / EXPERIMENT_ID / run_id / "tags" / tag_key
    if not tag_file.exists():
        return ""
    return tag_file.read_text(encoding="utf-8").strip()


def compute_cv_mean(run_id: str, metric_name: str) -> float:
    """Compute cross-validation mean of a metric across all folds."""
    values = [
        read_metric(run_id, f"eval_fold{fold}_{metric_name}") for fold in range(N_FOLDS)
    ]
    valid = [v for v in values if not math.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def find_complete_runs() -> list[str]:
    """Find all run directories with complete 3-fold evaluation."""
    exp_dir = MLRUNS_DIR / EXPERIMENT_ID
    complete: list[str] = []
    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "meta.yaml").exists():
            continue
        # Check for fold 2 metric (implies all 3 folds)
        metrics_dir = run_dir / "metrics"
        if metrics_dir.exists():
            has_fold2 = any(
                f.name.startswith("eval_fold2") for f in metrics_dir.iterdir()
            )
            if has_fold2:
                complete.append(run_dir.name)
    return complete


def write_rank_aggregate_tags(
    champions: dict[str, str],
    loss_to_run_ids: dict[str, list[str]],
) -> int:
    """Write rank-then-aggregate champion tags (balanced/topology/overlap)."""
    now = datetime.now(UTC).isoformat()
    written = 0

    category_map = {
        "balanced": "champion_rank_balanced",
        "topology": "champion_rank_topology",
        "overlap": "champion_rank_overlap",
    }

    for category, tag_key in category_map.items():
        loss_name = champions.get(category, "")
        if not loss_name:
            continue
        run_ids = loss_to_run_ids.get(loss_name, [])
        for run_id in run_ids:
            tags_dir = MLRUNS_DIR / EXPERIMENT_ID / run_id / "tags"
            tags_dir.mkdir(parents=True, exist_ok=True)

            tag_file = tags_dir / tag_key
            tag_file.write_text("true", encoding="utf-8")
            written += 1

            ts_file = tags_dir / f"{tag_key}_at"
            ts_file.write_text(now, encoding="utf-8")
            written += 1

            logger.info("  Tagged %s: %s (%s)", tag_key, loss_name, run_id[:8])

    return written


def main() -> int:
    """Run champion tagging on real experiment data."""
    logger.info("=" * 70)
    logger.info("CHAMPION TAGGING: dynunet_loss_variation_v2")
    logger.info("=" * 70)

    # 1. Find complete runs
    complete_runs = find_complete_runs()
    logger.info("\n[1/4] Complete runs: %d", len(complete_runs))

    if not complete_runs:
        logger.error("No complete runs found!")
        return 1

    # 2. Build analysis entries + rank-aggregate entries
    analysis_entries: list[dict[str, object]] = []
    rank_entries: list[dict[str, object]] = []
    loss_to_run_ids: dict[str, list[str]] = {}

    logger.info("\n[2/4] Reading metrics per run/fold")
    for run_id in complete_runs:
        loss = read_tag(run_id, "loss_function")
        loss_to_run_ids.setdefault(loss, []).append(run_id)

        logger.info("\n  Run %s (%s):", run_id[:8], loss)

        # Per-fold entries (for select_champions)
        for fold in range(N_FOLDS):
            fold_metrics: dict[str, float] = {}
            for metric in EVAL_METRICS:
                val = read_metric(run_id, f"eval_fold{fold}_{metric}")
                fold_metrics[metric] = val

            dsc = fold_metrics.get("dsc", float("nan"))
            logger.info(
                "    Fold %d: DSC=%.4f, clDSC=%.4f, MASD=%.4f",
                fold,
                fold_metrics.get("dsc", float("nan")),
                fold_metrics.get("centreline_dsc", float("nan")),
                fold_metrics.get("measured_masd", float("nan")),
            )

            analysis_entries.append(
                {
                    "entry_type": "per_fold",
                    "model_name": "dynunet",
                    "loss_function": loss,
                    "fold_id": fold,
                    "primary_metric_value": dsc,
                }
            )

        # CV mean entry (for select_champions)
        cv_means: dict[str, float] = {}
        for metric in EVAL_METRICS:
            cv_means[metric] = compute_cv_mean(run_id, metric)

        dsc_cv = cv_means.get("dsc", float("nan"))
        logger.info(
            "    CV Mean: DSC=%.4f, clDSC=%.4f, MASD=%.4f",
            cv_means.get("dsc", float("nan")),
            cv_means.get("centreline_dsc", float("nan")),
            cv_means.get("measured_masd", float("nan")),
        )

        analysis_entries.append(
            {
                "entry_type": "cv_mean",
                "model_name": "dynunet",
                "loss_function": loss,
                "fold_id": -1,
                "primary_metric_value": dsc_cv,
            }
        )

        # Rank-aggregate entry (multi-metric)
        rank_entries.append(
            {
                "model_id": loss,
                "dsc": cv_means["dsc"],
                "centreline_dsc": cv_means["centreline_dsc"],
                "measured_masd": cv_means["measured_masd"],
            }
        )

    # Build runs list in the format tag_champions expects
    runs_for_tagger: list[dict[str, object]] = [
        {"run_id": run_id, "loss_type": read_tag(run_id, "loss_function")}
        for run_id in complete_runs
    ]

    # 3. Select champions
    logger.info("\n[3/4] Selecting champions")

    # Strategy A: Single-metric (DSC)
    selection = tag_champions(
        MLRUNS_DIR,
        EXPERIMENT_ID,
        analysis_entries,
        runs=runs_for_tagger,
        primary_metric="dsc",
        maximize=True,
    )
    logger.info("\n  [Single-metric selection (DSC)]:")
    logger.info("    Best single fold: %s", selection.best_single_fold)
    logger.info("    Best CV mean:     %s", selection.best_cv_mean)
    logger.info("    Best ensemble:    %s", selection.best_ensemble)

    # Strategy B: Rank-then-aggregate (multi-metric)
    rank_champions = rank_then_aggregate(
        rank_entries,
        maximize_metrics=["dsc", "centreline_dsc"],
        minimize_metrics=["measured_masd"],
        topology_metric="centreline_dsc",
        overlap_metric="dsc",
    )
    logger.info("\n  [Rank-then-aggregate (multi-metric)]:")
    logger.info("    Balanced:  %s", rank_champions["balanced"])
    logger.info("    Topology:  %s", rank_champions["topology"])
    logger.info("    Overlap:   %s", rank_champions["overlap"])

    # 4. Write rank-aggregate tags
    logger.info("\n[4/4] Writing rank-aggregate tags")
    n_written = write_rank_aggregate_tags(rank_champions, loss_to_run_ids)
    logger.info("  Written %d rank-aggregate tag files", n_written)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("  Runs tagged:         %d", len(complete_runs))
    logger.info(
        "  Single-metric champ: %s (DSC: %.4f)",
        selection.best_cv_mean.loss_function if selection.best_cv_mean else "none",
        selection.best_cv_mean.metric_value if selection.best_cv_mean else 0.0,
    )
    logger.info("  Balanced (rank):     %s", rank_champions["balanced"])
    logger.info("  Topology (rank):     %s", rank_champions["topology"])
    logger.info("  Overlap (rank):      %s", rank_champions["overlap"])

    # Verify tags were written
    tag_count = 0
    exp_dir = MLRUNS_DIR / EXPERIMENT_ID
    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir():
            continue
        tags_dir = run_dir / "tags"
        if tags_dir.is_dir():
            for tag_file in tags_dir.iterdir():
                if tag_file.name.startswith("champion_"):
                    tag_count += 1
    logger.info("  Total champion tags: %d", tag_count)

    logger.info("\nCHAMPION TAGGING COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
