"""Diagnostic script: verify MLflow runs, checkpoints, metrics, and data splits.

Run:
    uv run python scripts/diagnose_mlruns.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "minivess"
SPLITS_PATH = PROJECT_ROOT / "configs" / "splits" / "3fold_seed42.json"


def diagnose_experiments() -> dict[str, dict]:
    """List all MLflow experiments with run counts."""
    experiments = {}
    for exp_dir in sorted(MLRUNS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        meta_path = exp_dir / "meta.yaml"
        if not meta_path.exists():
            continue
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        name = meta.get("name", "unknown")
        exp_id = meta.get("experiment_id", exp_dir.name)

        # Count runs (directories with meta.yaml inside)
        runs = []
        for item in exp_dir.iterdir():
            if item.is_dir() and (item / "meta.yaml").exists():
                runs.append(item.name)

        experiments[exp_id] = {
            "name": name,
            "n_runs": len(runs),
            "run_ids": runs,
        }
        logger.info("  Experiment %-35s (ID: %s): %d runs", name, exp_id, len(runs))

    return experiments


def diagnose_checkpoints(experiment_id: str) -> list[dict]:
    """Verify checkpoints loadable for an experiment."""
    exp_dir = MLRUNS_DIR / experiment_id
    results = []

    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "meta.yaml").exists():
            continue

        run_id = run_dir.name
        ckpt_dir = run_dir / "artifacts" / "checkpoints"
        loss_tag_path = run_dir / "tags" / "loss_function"
        loss_name = (
            loss_tag_path.read_text(encoding="utf-8").strip()
            if loss_tag_path.exists()
            else "unknown"
        )

        if not ckpt_dir.exists():
            logger.warning(
                "  Run %s (%s): NO checkpoints directory", run_id[:8], loss_name
            )
            results.append(
                {
                    "run_id": run_id,
                    "loss": loss_name,
                    "checkpoints": 0,
                    "loadable": False,
                }
            )
            continue

        ckpt_files = list(ckpt_dir.glob("*.pth"))
        all_loadable = True

        for ckpt in ckpt_files:
            try:
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                n_params = sum(v.numel() for v in state.values() if hasattr(v, "numel"))
                logger.info(
                    "    %s: %-40s %6.1f MB, %d params",
                    run_id[:8],
                    ckpt.name,
                    ckpt.stat().st_size / 1e6,
                    n_params,
                )
            except Exception as e:
                logger.error("    %s: %-40s FAILED: %s", run_id[:8], ckpt.name, e)
                all_loadable = False

        results.append(
            {
                "run_id": run_id,
                "loss": loss_name,
                "checkpoints": len(ckpt_files),
                "loadable": all_loadable,
            }
        )

    return results


def diagnose_metrics(experiment_id: str) -> dict[str, list[str]]:
    """Check which eval metrics exist per run."""
    exp_dir = MLRUNS_DIR / experiment_id
    results: dict[str, list[str]] = {}

    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "meta.yaml").exists():
            continue

        run_id = run_dir.name
        metrics_dir = run_dir / "metrics"
        if not metrics_dir.exists():
            results[run_id] = []
            continue

        eval_metrics = sorted(
            f.name for f in metrics_dir.iterdir() if f.name.startswith("eval_")
        )
        results[run_id] = eval_metrics

    return results


def diagnose_champion_tags(experiment_id: str) -> dict[str, str]:
    """Check for champion tags."""
    exp_dir = MLRUNS_DIR / experiment_id
    tags_found = {}

    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "tags").is_dir():
            continue

        run_id = run_dir.name
        for tag_file in (run_dir / "tags").iterdir():
            if tag_file.name.startswith("champion_"):
                value = tag_file.read_text(encoding="utf-8").strip()
                tags_found[f"{run_id[:8]}/{tag_file.name}"] = value

    return tags_found


def diagnose_data_splits() -> dict:
    """Verify data splits match real volumes on disk."""
    if not SPLITS_PATH.exists():
        return {"error": f"Splits file not found: {SPLITS_PATH}"}

    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))

    # Collect all volume IDs from splits
    # Format: list of folds, each fold = {"train": [{"image": ..., "label": ...}], "val": [...]}
    split_volume_ids: set[str] = set()
    fold_list = splits if isinstance(splits, list) else list(splits.values())
    for fold_data in fold_list:
        if isinstance(fold_data, dict):
            for subset_pairs in fold_data.values():
                if isinstance(subset_pairs, list):
                    for pair in subset_pairs:
                        if isinstance(pair, dict) and "image" in pair:
                            vol_id = Path(pair["image"]).stem.replace(".nii", "")
                            split_volume_ids.add(vol_id)
                        elif isinstance(pair, str):
                            split_volume_ids.add(pair)

    # Collect all volume IDs from disk
    disk_volume_ids: set[str] = set()
    images_dir = DATA_DIR / "imagesTr"
    if images_dir.exists():
        for f in images_dir.iterdir():
            if f.suffix in (".gz", ".nii"):
                vol_id = f.name.replace(".nii.gz", "").replace(".nii", "")
                disk_volume_ids.add(vol_id)

    labels_dir = DATA_DIR / "labelsTr"
    label_ids: set[str] = set()
    if labels_dir.exists():
        for f in labels_dir.iterdir():
            if f.suffix in (".gz", ".nii"):
                vol_id = f.name.replace(".nii.gz", "").replace(".nii", "")
                label_ids.add(vol_id)

    missing_from_disk = split_volume_ids - disk_volume_ids
    orphaned_on_disk = disk_volume_ids - split_volume_ids
    missing_labels = disk_volume_ids - label_ids

    result = {
        "splits_file": str(SPLITS_PATH),
        "n_folds": len(splits),
        "n_volumes_in_splits": len(split_volume_ids),
        "n_volumes_on_disk": len(disk_volume_ids),
        "n_labels_on_disk": len(label_ids),
        "missing_from_disk": sorted(missing_from_disk),
        "orphaned_on_disk": sorted(orphaned_on_disk),
        "missing_labels": sorted(missing_labels),
    }

    logger.info("\n  Splits file: %s", SPLITS_PATH)
    logger.info("  Folds: %d", result["n_folds"])
    logger.info("  Volumes in splits: %d", result["n_volumes_in_splits"])
    logger.info(
        "  Volumes on disk:   %d (images), %d (labels)",
        len(disk_volume_ids),
        len(label_ids),
    )
    logger.info("  Missing from disk: %s", result["missing_from_disk"] or "NONE")
    logger.info("  Orphaned on disk:  %s", result["orphaned_on_disk"] or "NONE")
    logger.info("  Missing labels:    %s", result["missing_labels"] or "NONE")

    return result


def main() -> int:
    """Run full diagnostics."""
    logger.info("=" * 70)
    logger.info("MLRUNS DIAGNOSTICS")
    logger.info("=" * 70)

    # 1. Experiments
    logger.info("\n[1/5] Experiments")
    experiments = diagnose_experiments()

    # 2. Checkpoints for primary experiment
    primary_exp = "843896622863223169"
    logger.info("\n[2/5] Checkpoints (dynunet_loss_variation_v2)")
    ckpt_results = diagnose_checkpoints(primary_exp)
    total_ckpts = sum(r["checkpoints"] for r in ckpt_results)
    all_loadable = all(r["loadable"] for r in ckpt_results)
    logger.info("  Total checkpoints: %d, All loadable: %s", total_ckpts, all_loadable)

    # 3. Metrics
    logger.info("\n[3/5] Eval Metrics (dynunet_loss_variation_v2)")
    metrics = diagnose_metrics(primary_exp)
    for run_id, metric_list in metrics.items():
        eval_count = len(metric_list)
        logger.info("  Run %s: %d eval metrics", run_id[:8], eval_count)

    # 4. Champion tags
    logger.info("\n[4/5] Champion Tags (dynunet_loss_variation_v2)")
    tags = diagnose_champion_tags(primary_exp)
    if tags:
        for tag_key, tag_val in tags.items():
            logger.info("  %s = %s", tag_key, tag_val)
    else:
        logger.info("  NO champion tags found (expected — Phase 1 will create them)")

    # 5. Data splits
    logger.info("\n[5/5] Data Splits vs Disk")
    split_result = diagnose_data_splits()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("  Experiments:    %d active", len(experiments))
    logger.info("  Primary runs:   %d", len(ckpt_results))
    logger.info(
        "  Checkpoints:    %d total, all loadable: %s", total_ckpts, all_loadable
    )
    logger.info("  Champion tags:  %d (expect 0 before Phase 1)", len(tags))
    logger.info(
        "  Data volumes:   %d on disk", split_result.get("n_volumes_on_disk", 0)
    )
    logger.info(
        "  Split match:    %s",
        "YES" if not split_result.get("missing_from_disk") else "NO",
    )

    if not all_loadable:
        logger.error("\nFAILED: Some checkpoints not loadable!")
        return 1
    if split_result.get("missing_from_disk"):
        logger.error("\nFAILED: Volumes missing from disk!")
        return 1

    logger.info("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
