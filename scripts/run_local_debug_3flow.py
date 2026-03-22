"""Local 3-flow debug: Post-Training → Evaluation → Biostatistics.

Usage:
    MINIVESS_ALLOW_HOST=1 uv run python scripts/run_local_debug_3flow.py

Validates the full 3-flow pipeline locally using existing DynUNet checkpoints
from dynunet_loss_variation_v2 experiment BEFORE spending on GCP.

This is a flow orchestrator script — it chains Prefect flows together, not a
bypass of Prefect. See CLAUDE.md Rule #17 distinction: scripts that CALL flows
are orchestrators; scripts that BYPASS flows are banned.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Environment setup (must happen before any minivess imports)
# ---------------------------------------------------------------------------
os.environ.setdefault("MINIVESS_ALLOW_HOST", "1")
os.environ.setdefault("MLFLOW_TRACKING_URI", "mlruns")
os.environ.setdefault("POST_TRAINING_OUTPUT_DIR", "outputs/post_training")
os.environ.setdefault("ANALYSIS_OUTPUT_DIR", "outputs/analysis")

# Suppress MONAI/torch warnings for cleaner output
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="monai")
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("local_debug_3flow")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = REPO_ROOT / "mlruns"
SPLITS_FILE = REPO_ROOT / "configs" / "splits" / "3fold_seed42.json"
DATA_DIR = REPO_ROOT / "data" / "raw" / "minivess"
OUTPUT_DIR = REPO_ROOT / "outputs"
BIOSTATS_CONFIG = REPO_ROOT / "configs" / "biostatistics" / "smoke_local.yaml"

# Training experiment with existing DynUNet checkpoints
TRAINING_EXPERIMENT = "dynunet_loss_variation_v2"
TRAINING_EXPERIMENT_ID = "843896622863223169"

# All losses available in the training experiment
LOSS_FUNCTIONS = ["cbdice_cldice", "dice_ce", "dice_ce_cldice", "cbdice"]

# Post-training methods (registry decision: only "none" and "swag")
POST_TRAINING_METHODS = ["none", "swag"]

# New MLflow experiments for this debug run
POST_TRAINING_EXPERIMENT = "smoke_local_post_training"
EVALUATION_EXPERIMENT = "smoke_local_evaluation"

# Timing tracker
TIMINGS: dict[str, float] = {}


def _time(label: str) -> float:
    """Record a timing point and return elapsed since start."""
    now = time.perf_counter()
    TIMINGS[label] = now
    return now


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0: Pre-flight verification
# ═══════════════════════════════════════════════════════════════════════════


def phase_0_preflight() -> dict[str, Any]:
    """Verify all prerequisites. No guessing — query everything."""
    logger.info("=" * 70)
    logger.info("PHASE 0: Pre-flight verification")
    logger.info("=" * 70)

    results: dict[str, Any] = {}

    # T0.1: Verify checkpoint loadability
    import torch

    run_dirs = list((MLRUNS_DIR / TRAINING_EXPERIMENT_ID).iterdir())
    if not run_dirs:
        raise RuntimeError(f"No runs found in {MLRUNS_DIR / TRAINING_EXPERIMENT_ID}")

    # Find runs matching our loss functions
    import mlflow

    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    client = mlflow.MlflowClient()
    training_runs: dict[str, dict[str, Any]] = {}  # loss_function → run info

    runs = client.search_runs(
        experiment_ids=[TRAINING_EXPERIMENT_ID],
        filter_string="attributes.status = 'FINISHED'",
    )
    for run in runs:
        tags = run.data.tags
        loss = tags.get("loss_function", "")
        if loss in LOSS_FUNCTIONS:
            ckpt_dir = (
                MLRUNS_DIR
                / TRAINING_EXPERIMENT_ID
                / run.info.run_id
                / "artifacts"
                / "checkpoints"
            )
            ckpt_path = ckpt_dir / "best_val_loss.pth"
            if ckpt_path.exists():
                training_runs[loss] = {
                    "run_id": run.info.run_id,
                    "loss_function": loss,
                    "model_family": tags.get("model_family", "dynunet"),
                    "fold_id": tags.get("fold_id", "0"),
                    "with_aux_calib": tags.get("with_aux_calib", "false"),
                    "checkpoint_path": ckpt_path,
                    "checkpoint_dir": ckpt_dir,
                }

    if len(training_runs) < len(LOSS_FUNCTIONS):
        found = list(training_runs.keys())
        missing = [loss for loss in LOSS_FUNCTIONS if loss not in found]
        raise RuntimeError(
            f"Missing training runs for losses: {missing}. Found: {found}"
        )

    logger.info("Found %d training runs:", len(training_runs))
    for loss, info in training_runs.items():
        logger.info(
            "  %s: run_id=%s, checkpoint=%s",
            loss,
            info["run_id"][:10],
            info["checkpoint_path"],
        )

    # Test checkpoint loadability
    sample_ckpt = next(iter(training_runs.values()))["checkpoint_path"]
    t0 = time.perf_counter()
    ckpt = torch.load(sample_ckpt, weights_only=False, map_location="cpu")
    t_load = time.perf_counter() - t0
    assert "model_state_dict" in ckpt, (
        f"Unexpected checkpoint format: {list(ckpt.keys())}"
    )
    n_tensors = len(ckpt["model_state_dict"])
    logger.info("Checkpoint loadable: %d tensors in %.2fs", n_tensors, t_load)
    results["checkpoint_load_time_s"] = round(t_load, 3)
    results["n_tensors"] = n_tensors

    # T0.2: Verify MiniVess data
    images_dir = DATA_DIR / "imagesTr"
    labels_dir = DATA_DIR / "labelsTr"
    n_images = len(list(images_dir.glob("*.nii.gz")))
    n_labels = len(list(labels_dir.glob("*.nii.gz")))
    logger.info("MiniVess data: %d images, %d labels", n_images, n_labels)
    assert n_images >= 60, f"Expected ≥60 images, got {n_images}"
    assert n_labels >= 60, f"Expected ≥60 labels, got {n_labels}"
    results["n_images"] = n_images

    # T0.3: Verify splits file
    assert SPLITS_FILE.exists(), f"Splits file not found: {SPLITS_FILE}"
    splits_data = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))
    n_folds = len(splits_data)
    n_val_fold0 = len(splits_data[0]["val"])
    logger.info(
        "Splits: %d folds, fold-0 has %d validation volumes", n_folds, n_val_fold0
    )
    results["n_folds"] = n_folds
    results["n_val_fold0"] = n_val_fold0

    # T0.4: Verify GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("GPU: %s (%.1f GB)", gpu_name, gpu_mem)
        results["gpu"] = gpu_name
    else:
        logger.warning("No GPU available — inference will be slow on CPU")
        results["gpu"] = "CPU"

    # T0.5: Clean previous smoke test MLflow runs
    for exp_name in [POST_TRAINING_EXPERIMENT, EVALUATION_EXPERIMENT]:
        try:
            exp = client.get_experiment_by_name(exp_name)
            if exp is not None:
                for old_run in client.search_runs(experiment_ids=[exp.experiment_id]):
                    client.delete_run(old_run.info.run_id)
                logger.info("Cleaned previous runs from experiment: %s", exp_name)
        except Exception:
            pass  # Experiment doesn't exist yet

    results["training_runs"] = training_runs
    logger.info("Phase 0 PASSED — all prerequisites verified")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Post-Training Flow
# ═══════════════════════════════════════════════════════════════════════════


def phase_1_post_training(preflight: dict[str, Any]) -> list[dict[str, Any]]:
    """Run post-training: 4 losses × {none, swag} = 8 MLflow runs."""
    logger.info("=" * 70)
    logger.info("PHASE 1: Post-Training Flow")
    logger.info("=" * 70)

    from minivess.adapters.model_builder import build_adapter
    from minivess.config.models import DataConfig, ModelConfig, ModelFamily
    from minivess.data.loader import build_train_loader
    from minivess.data.splits import load_splits
    from minivess.orchestration.flows.post_training_flow import (
        run_factorial_post_training,
    )

    training_runs = preflight["training_runs"]
    all_results: list[dict[str, Any]] = []
    phase_start = time.perf_counter()

    # Build train DataLoader once for SWAG (shared across all losses)
    splits = load_splits(SPLITS_FILE)
    fold0_train = splits[0].train
    data_config = DataConfig(
        dataset_name="minivess",
        data_dir=DATA_DIR,
        patch_size=(64, 64, 16),
        num_workers=0,
    )
    train_loader = build_train_loader(
        fold0_train,
        data_config,
        batch_size=2,
        cache_rate=0.0,
    )
    logger.info("Built train DataLoader for SWAG: %d volumes", len(fold0_train))

    # Build model architecture for SWAG (DynUNet — all runs use the same arch)
    model_config = ModelConfig(
        family=ModelFamily.MONAI_DYNUNET,
        name="dynunet",
        in_channels=1,
        out_channels=2,
    )
    adapter = build_adapter(model_config)
    calibration_data: dict[str, Any] = {
        "train_loader": train_loader,
        "model": adapter,  # ModelAdapter IS an nn.Module (has .net inside)
    }
    logger.info("Built DynUNet model for SWAG calibration_data")

    for loss_fn in LOSS_FUNCTIONS:
        run_info = training_runs[loss_fn]
        ckpt_dir = run_info["checkpoint_dir"]

        # Collect ALL checkpoints from this run
        checkpoint_paths = sorted(ckpt_dir.glob("*.pth"))
        logger.info(
            "Loss %s: %d checkpoints found",
            loss_fn,
            len(checkpoint_paths),
        )

        output_dir = OUTPUT_DIR / "post_training" / loss_fn
        t0 = time.perf_counter()

        results = run_factorial_post_training(
            checkpoint_paths=checkpoint_paths,
            methods=POST_TRAINING_METHODS,
            output_dir=output_dir,
            tracking_uri=str(MLRUNS_DIR),
            experiment_name=POST_TRAINING_EXPERIMENT,
            upstream_run_id=run_info["run_id"],
            upstream_tags={
                "model_family": run_info["model_family"],
                "loss_function": loss_fn,
                "fold_id": run_info["fold_id"],
                "with_aux_calib": run_info["with_aux_calib"],
            },
            calibration_data=calibration_data,
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Loss %s: %d methods completed in %.1fs",
            loss_fn,
            len(results),
            elapsed,
        )
        for r in results:
            r["loss_function"] = loss_fn
            r["upstream_run_id"] = run_info["run_id"]
            r["upstream_tags"] = {
                "model_family": run_info["model_family"],
                "loss_function": loss_fn,
                "fold_id": run_info["fold_id"],
                "with_aux_calib": run_info["with_aux_calib"],
            }
            all_results.append(r)

    phase_elapsed = time.perf_counter() - phase_start
    TIMINGS["phase1_total_s"] = phase_elapsed
    logger.info(
        "Phase 1 COMPLETE: %d post-training variants in %.1fs",
        len(all_results),
        phase_elapsed,
    )

    # Verify MLflow runs created
    import mlflow

    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(POST_TRAINING_EXPERIMENT)
    if exp:
        pt_runs = client.search_runs(experiment_ids=[exp.experiment_id])
        logger.info(
            "MLflow: %d post-training runs in %s",
            len(pt_runs),
            POST_TRAINING_EXPERIMENT,
        )
        for r in pt_runs:
            tags = r.data.tags
            logger.info(
                "  %s: %s__%s__%s",
                r.info.run_id[:10],
                tags.get("loss_function", "?"),
                tags.get("post_training_method", "?"),
                tags.get("model_family", "?"),
            )

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Analysis / Evaluation
# ═══════════════════════════════════════════════════════════════════════════


def phase_2_evaluation(
    post_training_results: list[dict[str, Any]],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate each post-training checkpoint on MiniVess validation.

    This is the REAL test — DynUNet inference with sliding-window on actual
    MiniVess volumes using the RTX 2070 Super GPU.
    """
    logger.info("=" * 70)
    logger.info("PHASE 2: Analysis / Evaluation (Real Inference)")
    logger.info("=" * 70)

    import numpy as np
    import torch

    from minivess.adapters.model_builder import build_adapter
    from minivess.config.models import DataConfig, ModelConfig, ModelFamily
    from minivess.data.loader import build_val_loader
    from minivess.data.splits import load_splits
    from minivess.pipeline.inference import SlidingWindowInferenceRunner

    phase_start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build MiniVess validation DataLoader (fold 0)
    logger.info("Loading MiniVess validation data (fold 0)...")
    splits = load_splits(SPLITS_FILE)
    fold0_val = splits[0].val
    logger.info("Fold 0 validation: %d volumes", len(fold0_val))

    data_config = DataConfig(
        dataset_name="minivess",
        data_dir=DATA_DIR,
        patch_size=(128, 128, 16),
        voxel_spacing=(0.0, 0.0, 0.0),  # Native spacing — mv02 safe
        num_workers=0,
    )

    t0 = time.perf_counter()
    val_loader = build_val_loader(fold0_val, data_config, cache_rate=0.0)
    data_load_time = time.perf_counter() - t0
    logger.info("Validation loader built in %.1fs (cache_rate=0)", data_load_time)
    TIMINGS["data_load_s"] = data_load_time

    # Build model config for DynUNet
    model_config = ModelConfig(
        family=ModelFamily.MONAI_DYNUNET,
        name="dynunet_default",
        in_channels=1,
        out_channels=2,
        architecture_params={"filters": [32, 64, 128, 256]},
    )

    # Inference runner
    inference_runner = SlidingWindowInferenceRunner(
        roi_size=(128, 128, 16),
        num_classes=2,
        overlap=0.5,
        sw_batch_size=4,
    )

    # MLflow setup for evaluation results
    import mlflow

    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    mlflow.set_experiment(EVALUATION_EXPERIMENT)

    all_eval_results: dict[str, dict[str, Any]] = {}
    eval_issues: list[str] = []

    for pt_result in post_training_results:
        ckpt_path = Path(pt_result["output_path"])
        loss_fn = pt_result["loss_function"]
        method = pt_result["post_training_method"]
        condition_name = f"dynunet__{loss_fn}__fold0__{method}"

        logger.info("--- Evaluating: %s ---", condition_name)

        if not ckpt_path.exists():
            msg = f"Checkpoint missing: {ckpt_path}"
            logger.error(msg)
            eval_issues.append(msg)
            continue

        # Build model and load weights
        try:
            adapter = build_adapter(model_config)
            adapter.load_checkpoint(ckpt_path)
            adapter.eval()
            adapter.to(device)
            logger.info("Model loaded: %s", condition_name)
        except Exception as e:
            msg = f"Failed to load model {condition_name}: {e}"
            logger.error(msg)
            eval_issues.append(msg)
            continue

        # Run inference on all validation volumes
        volume_metrics: list[dict[str, float]] = []
        vol_times: list[float] = []

        for batch_idx, batch in enumerate(val_loader):
            vol_start = time.perf_counter()
            image = batch["image"]  # (1, C, H, W, D) MONAI convention
            label = batch["label"]

            try:
                pred = inference_runner.predict_volume(adapter, image, device=device)
                label_np = label.squeeze(0).squeeze(0).cpu().numpy().astype(np.int64)

                # Compute metrics
                metrics = _compute_volume_metrics(pred, label_np)
                volume_metrics.append(metrics)

                vol_elapsed = time.perf_counter() - vol_start
                vol_times.append(vol_elapsed)

                if batch_idx < 3 or batch_idx % 5 == 0:
                    logger.info(
                        "  Vol %d/%d: DSC=%.4f, clDice=%.4f (%.1fs)",
                        batch_idx + 1,
                        len(fold0_val),
                        metrics.get("dsc", 0.0),
                        metrics.get("cldice", 0.0),
                        vol_elapsed,
                    )
            except Exception as e:
                msg = f"Inference failed on vol {batch_idx} for {condition_name}: {e}"
                logger.error(msg)
                eval_issues.append(msg)
                continue

        if not volume_metrics:
            logger.error("No volumes evaluated for %s", condition_name)
            continue

        # Aggregate metrics
        mean_metrics: dict[str, float] = {}
        for key in volume_metrics[0]:
            values = [m[key] for m in volume_metrics]
            mean_metrics[key] = float(np.mean(values))
            mean_metrics[f"{key}_std"] = float(np.std(values))

        mean_inference_time = float(np.mean(vol_times))
        total_inference_time = float(np.sum(vol_times))

        logger.info(
            "%s: mean DSC=%.4f±%.4f, mean clDice=%.4f±%.4f, %.1fs/vol, %.1fs total",
            condition_name,
            mean_metrics.get("dsc", 0),
            mean_metrics.get("dsc_std", 0),
            mean_metrics.get("cldice", 0),
            mean_metrics.get("cldice_std", 0),
            mean_inference_time,
            total_inference_time,
        )

        # Log to MLflow
        upstream_tags = pt_result.get("upstream_tags", {})
        run_tags = {
            "model_family": upstream_tags.get("model_family", "dynunet"),
            "loss_function": loss_fn,
            "fold_id": upstream_tags.get("fold_id", "0"),
            "with_aux_calib": upstream_tags.get("with_aux_calib", "false"),
            "post_training_method": method,
            "recalibration": "none",
            "ensemble_strategy": "none",
            "flow_name": "analysis-flow",
        }

        try:
            with mlflow.start_run(run_name=condition_name, tags=run_tags) as run:
                # Log aggregate metrics
                for key, val in mean_metrics.items():
                    mlflow.log_metric(key, val)
                mlflow.log_metric("inference_time_per_vol_s", mean_inference_time)
                mlflow.log_metric("total_inference_time_s", total_inference_time)
                mlflow.log_metric("n_volumes_evaluated", float(len(volume_metrics)))

                # Log per-volume metrics for biostatistics
                for vol_idx, vol_metrics in enumerate(volume_metrics):
                    for metric_name, metric_val in vol_metrics.items():
                        mlflow.log_metric(
                            f"eval/fold_0/vol/{vol_idx}/{metric_name}",
                            metric_val,
                        )

                all_eval_results[condition_name] = {
                    "run_id": run.info.run_id,
                    "mean_metrics": mean_metrics,
                    "n_volumes": len(volume_metrics),
                    "inference_time_s": total_inference_time,
                    "tags": run_tags,
                }
        except Exception as e:
            msg = f"MLflow logging failed for {condition_name}: {e}"
            logger.error(msg)
            eval_issues.append(msg)

        # Free GPU memory
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    phase_elapsed = time.perf_counter() - phase_start
    TIMINGS["phase2_total_s"] = phase_elapsed

    logger.info(
        "Phase 2 COMPLETE: %d conditions evaluated in %.1fs",
        len(all_eval_results),
        phase_elapsed,
    )
    if eval_issues:
        logger.warning("Phase 2 issues (%d):", len(eval_issues))
        for issue in eval_issues:
            logger.warning("  - %s", issue)

    return {
        "eval_results": all_eval_results,
        "issues": eval_issues,
        "n_conditions": len(all_eval_results),
    }


def _compute_volume_metrics(
    pred: Any,
    label: Any,
) -> dict[str, float]:
    """Compute segmentation metrics for a single volume.

    Uses MONAI metrics where available, with scipy fallback for clDice.
    """
    import numpy as np

    pred = np.asarray(pred)
    label = np.asarray(label)

    # Binary masks (foreground = class 1)
    pred_bin = (pred > 0).astype(np.float32)
    label_bin = (label > 0).astype(np.float32)

    metrics: dict[str, float] = {}

    # Dice Score Coefficient
    intersection = np.sum(pred_bin * label_bin)
    denom = np.sum(pred_bin) + np.sum(label_bin)
    metrics["dsc"] = float(2.0 * intersection / max(denom, 1e-8))

    # clDice (centerline Dice) — use skimage skeletonize
    try:
        from skimage.morphology import skeletonize

        pred_skel = skeletonize(pred_bin.astype(bool)).astype(np.float32)
        label_skel = skeletonize(label_bin.astype(bool)).astype(np.float32)

        # Topology precision: fraction of pred skeleton inside label
        tprec = float(np.sum(pred_skel * label_bin) / max(np.sum(pred_skel), 1e-8))
        # Topology sensitivity: fraction of label skeleton inside pred
        tsens = float(np.sum(label_skel * pred_bin) / max(np.sum(label_skel), 1e-8))
        metrics["cldice"] = float(2.0 * tprec * tsens / max(tprec + tsens, 1e-8))
    except ImportError:
        metrics["cldice"] = 0.0

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Biostatistics Flow
# ═══════════════════════════════════════════════════════════════════════════


def phase_3_biostatistics() -> dict[str, Any]:
    """Run biostatistics flow on evaluation results."""
    logger.info("=" * 70)
    logger.info("PHASE 3: Biostatistics Flow")
    logger.info("=" * 70)

    phase_start = time.perf_counter()
    results: dict[str, Any] = {}
    issues: list[str] = []

    try:
        from prefect.testing.utilities import prefect_test_harness

        with prefect_test_harness():
            from minivess.orchestration.flows.biostatistics_flow import (
                run_biostatistics_flow,
            )

            biostats_result = run_biostatistics_flow(
                config_path=str(BIOSTATS_CONFIG),
                trigger_source="local_debug_3flow",
            )
            results["biostats_result"] = str(type(biostats_result).__name__)
            logger.info(
                "Biostatistics flow completed: %s", type(biostats_result).__name__
            )

            # Extract key results
            if hasattr(biostats_result, "anova_results"):
                results["n_anova_results"] = len(biostats_result.anova_results)
            if hasattr(biostats_result, "pairwise_results"):
                results["n_pairwise_results"] = len(biostats_result.pairwise_results)

    except Exception as e:
        msg = f"Biostatistics flow failed: {e}"
        logger.error(msg, exc_info=True)
        issues.append(msg)

    phase_elapsed = time.perf_counter() - phase_start
    TIMINGS["phase3_total_s"] = phase_elapsed
    results["issues"] = issues

    logger.info("Phase 3 COMPLETE in %.1fs", phase_elapsed)
    if issues:
        logger.warning("Phase 3 issues (%d):", len(issues))
        for issue in issues:
            logger.warning("  - %s", issue)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Execute the full 3-flow local debug pipeline."""
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║  Local 3-Flow Debug: Post-Training → Eval → Biostatistics  ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")

    total_start = time.perf_counter()
    all_issues: list[str] = []

    # Phase 0
    _time("phase0_start")
    preflight = phase_0_preflight()
    _time("phase0_end")
    TIMINGS["phase0_total_s"] = TIMINGS["phase0_end"] - TIMINGS["phase0_start"]

    # Phase 1
    _time("phase1_start")
    post_training_results = phase_1_post_training(preflight)
    _time("phase1_end")

    # Phase 2
    _time("phase2_start")
    eval_results = phase_2_evaluation(post_training_results, preflight)
    _time("phase2_end")
    all_issues.extend(eval_results.get("issues", []))

    # Phase 3
    _time("phase3_start")
    biostats_results = phase_3_biostatistics()
    _time("phase3_end")
    all_issues.extend(biostats_results.get("issues", []))

    total_elapsed = time.perf_counter() - total_start
    TIMINGS["total_s"] = total_elapsed

    # Summary
    logger.info("=" * 70)
    logger.info("EXECUTION COMPLETE")
    logger.info("=" * 70)
    logger.info(
        "Total wall-clock time: %.1fs (%.1f min)", total_elapsed, total_elapsed / 60
    )
    logger.info("Phase 0 (pre-flight):     %.1fs", TIMINGS.get("phase0_total_s", 0))
    logger.info("Phase 1 (post-training):  %.1fs", TIMINGS.get("phase1_total_s", 0))
    logger.info("Phase 2 (evaluation):     %.1fs", TIMINGS.get("phase2_total_s", 0))
    logger.info("Phase 3 (biostatistics):  %.1fs", TIMINGS.get("phase3_total_s", 0))

    if all_issues:
        logger.warning("ISSUES FOUND (%d):", len(all_issues))
        for issue in all_issues:
            logger.warning("  - %s", issue)
    else:
        logger.info("NO ISSUES FOUND — all 3 flows completed successfully")

    # Write timing data for report
    timing_path = OUTPUT_DIR / "debug_3flow_timings.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(
        json.dumps(
            {
                "timings": TIMINGS,
                "eval_results": {
                    k: {
                        "mean_metrics": v["mean_metrics"],
                        "n_volumes": v["n_volumes"],
                        "inference_time_s": v["inference_time_s"],
                    }
                    for k, v in eval_results.get("eval_results", {}).items()
                },
                "issues": all_issues,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Timing data written to %s", timing_path)

    if all_issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
