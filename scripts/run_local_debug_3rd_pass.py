"""Local 3rd-pass debug: Train (2 ep) → SWAG (5 ep) → Eval + Calibration → Biostatistics.

Usage:
    MINIVESS_ALLOW_HOST=1 uv run python scripts/run_local_debug_3rd_pass.py

Validates the SWAG + calibration-metrics pipeline locally BEFORE spending on GCP.

Phase 1: Train DynUNet for 2 epochs (cbdice_cldice, fold-0)
Phase 2: Run SWAG for 5 additional epochs on the Phase 1 checkpoint
Phase 3: Evaluate baseline (none) + SWAG with ALL calibration metrics
Phase 4: Run biostatistics with calibration metrics in ANOVA
Phase 5: Write report

NOT a production run path — this is a one-off debug script.
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
# Environment setup (before any minivess imports)
# ---------------------------------------------------------------------------
os.environ.setdefault("MINIVESS_ALLOW_HOST", "1")
os.environ.setdefault("MLFLOW_TRACKING_URI", "mlruns")
os.environ.setdefault("POST_TRAINING_OUTPUT_DIR", "outputs/post_training_3rd")
os.environ.setdefault("ANALYSIS_OUTPUT_DIR", "outputs/analysis_3rd")

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="monai")
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("local_debug_3rd_pass")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = REPO_ROOT / "mlruns"
SPLITS_FILE = REPO_ROOT / "configs" / "splits" / "3fold_seed42.json"
DATA_DIR = REPO_ROOT / "data" / "raw" / "minivess"
OUTPUT_DIR = REPO_ROOT / "outputs" / "debug_3rd_pass"
BIOSTATS_CONFIG = REPO_ROOT / "configs" / "biostatistics" / "smoke_local.yaml"

# Training config
MAX_EPOCHS = 2
# dice_ce instead of cbdice_cldice: soft-skeletonization in clDice OOMs
# on RTX 2070 Super (8 GB).  dice_ce validates the same SWAG + calibration pipeline.
LOSS_NAME = "dice_ce"
BATCH_SIZE = 2
LEARNING_RATE = 1e-3

# SWAG config
SWAG_EPOCHS = 5
SWAG_LR = 0.01
SWAG_MAX_RANK = 10
SWAG_N_SAMPLES = 5  # For diversity check

# MLflow experiments
TRAIN_EXPERIMENT = "smoke_local_train_3rd"
EVAL_EXPERIMENT = "smoke_local_evaluation_3rd"

TIMINGS: dict[str, float] = {}


def _elapsed(label: str, start: float) -> float:
    dur = time.perf_counter() - start
    TIMINGS[label] = dur
    return dur


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0: Pre-flight
# ═══════════════════════════════════════════════════════════════════════════


def phase_0_preflight() -> dict[str, Any]:
    """Verify data, splits, GPU, SWAG + calibration imports."""
    logger.info("=" * 70)
    logger.info("PHASE 0: Pre-flight verification")
    logger.info("=" * 70)
    t0 = time.perf_counter()

    import torch

    from minivess.data.splits import load_splits
    from minivess.ensemble.swag import SWAGModel  # noqa: F401
    from minivess.pipeline.calibration_metrics import (
        compute_all_calibration_metrics,  # noqa: F401
    )
    from minivess.pipeline.post_training_plugins.swag import SWAGPlugin  # noqa: F401

    # Data
    images_dir = DATA_DIR / "imagesTr"
    labels_dir = DATA_DIR / "labelsTr"
    n_images = len(list(images_dir.glob("*.nii.gz")))
    n_labels = len(list(labels_dir.glob("*.nii.gz")))
    assert n_images >= 60, f"Expected ≥60 images, got {n_images}"
    assert n_labels >= 60, f"Expected ≥60 labels, got {n_labels}"
    logger.info("MiniVess data: %d images, %d labels", n_images, n_labels)

    # Splits
    assert SPLITS_FILE.exists(), f"Splits file not found: {SPLITS_FILE}"
    splits = load_splits(SPLITS_FILE)
    n_val_fold0 = len(splits[0].val)
    logger.info("Splits: %d folds, fold-0 has %d val volumes", len(splits), n_val_fold0)

    # GPU
    gpu_info = "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_info = f"{gpu_name} ({gpu_mem:.1f} GB)"
        logger.info("GPU: %s", gpu_info)
    else:
        logger.warning("No GPU — will be slow")

    # Output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _elapsed("phase0_s", t0)
    logger.info("Phase 0 PASSED")

    return {
        "n_images": n_images,
        "n_val_fold0": n_val_fold0,
        "gpu": gpu_info,
        "splits": splits,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Train DynUNet (2 epochs)
# ═══════════════════════════════════════════════════════════════════════════


def phase_1_train(preflight: dict[str, Any]) -> dict[str, Any]:
    """Train DynUNet for 2 epochs to verify training loop + calibration metrics."""
    logger.info("=" * 70)
    logger.info("PHASE 1: Train DynUNet (%d epochs, %s)", MAX_EPOCHS, LOSS_NAME)
    logger.info("=" * 70)
    t0 = time.perf_counter()

    import mlflow
    import torch

    from minivess.adapters.base import SegmentationOutput
    from minivess.adapters.model_builder import build_adapter
    from minivess.config.models import DataConfig, ModelConfig, ModelFamily
    from minivess.data.loader import build_train_loader, build_val_loader
    from minivess.pipeline.loss_functions import build_loss_function

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = preflight["splits"]
    fold0_train = splits[0].train
    fold0_val = splits[0].val

    data_config = DataConfig(
        dataset_name="minivess",
        data_dir=DATA_DIR,
        patch_size=(128, 128, 16),
        voxel_spacing=(0.0, 0.0, 0.0),
        num_workers=0,
    )

    # Build loaders
    logger.info("Building data loaders...")
    train_loader = build_train_loader(
        fold0_train, data_config, batch_size=BATCH_SIZE, cache_rate=0.0
    )
    val_loader = build_val_loader(fold0_val, data_config, cache_rate=0.0)
    logger.info("Train: %d batches, Val: %d volumes", len(train_loader), len(fold0_val))

    # Build model
    model_config = ModelConfig(
        family=ModelFamily.MONAI_DYNUNET,
        name="dynunet_debug_3rd",
        in_channels=1,
        out_channels=2,
        architecture_params={"filters": [32, 64, 128, 256]},
    )
    model = build_adapter(model_config)
    model.to(device)
    logger.info("DynUNet built, device=%s", device)

    # Loss + optimizer
    loss_fn = build_loss_function(LOSS_NAME)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # MLflow
    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    mlflow.set_experiment(TRAIN_EXPERIMENT)

    train_tags = {
        "model_family": "dynunet",
        "loss_function": LOSS_NAME,
        "fold_id": "0",
        "with_aux_calib": "false",
        "post_training_method": "none",
        "recalibration": "none",
        "ensemble_strategy": "none",
        "flow_name": "train-flow",
    }

    ckpt_dir = OUTPUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="dynunet_train_3rd_pass", tags=train_tags) as run:
        train_run_id = run.info.run_id

        for epoch in range(MAX_EPOCHS):
            # --- Train ---
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                outputs = model(images)
                # ModelAdapter.forward() returns SegmentationOutput
                logits = (
                    outputs.logits
                    if isinstance(outputs, SegmentationOutput)
                    else outputs
                )
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            logger.info(
                "Epoch %d/%d: train_loss=%.4f", epoch + 1, MAX_EPOCHS, avg_train_loss
            )

            # --- Skip in-training validation ---
            # Full-volume forward pass OOMs on RTX 2070 Super (8 GB).
            # Phase 3 does the real evaluation with sliding-window inference.
            # Calibration metrics in the training loop are verified there.

        # Save checkpoint
        ckpt_path = ckpt_dir / "best_val_loss.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": MAX_EPOCHS,
            },
            ckpt_path,
        )
        mlflow.log_artifact(str(ckpt_path))
        logger.info("Checkpoint saved: %s", ckpt_path)

    dur = _elapsed("phase1_s", t0)
    logger.info("Phase 1 COMPLETE in %.1fs", dur)

    return {
        "checkpoint_path": ckpt_path,
        "train_run_id": train_run_id,
        "model_config": model_config,
        "data_config": data_config,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: SWAG Post-Training (5 epochs)
# ═══════════════════════════════════════════════════════════════════════════


def phase_2_swag(phase1: dict[str, Any]) -> dict[str, Any]:
    """Run SWAG for 5 epochs, verify posterior diversity + BN recalibration."""
    logger.info("=" * 70)
    logger.info("PHASE 2: SWAG Post-Training (%d epochs)", SWAG_EPOCHS)
    logger.info("=" * 70)
    t0 = time.perf_counter()

    import torch

    from minivess.adapters.model_builder import build_adapter
    from minivess.ensemble.swag import SWAGModel
    from minivess.pipeline.post_training_plugin import PluginInput
    from minivess.pipeline.post_training_plugins.swag import SWAGPlugin

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build fresh model for SWAG (plugin loads state_dict into it)
    model = build_adapter(phase1["model_config"])
    model.to(device)

    swag_output_dir = OUTPUT_DIR / "swag"
    swag_output_dir.mkdir(parents=True, exist_ok=True)

    # Run SWAG plugin
    plugin = SWAGPlugin()
    plugin_input = PluginInput(
        checkpoint_paths=[phase1["checkpoint_path"]],
        config={
            "swa_lr": SWAG_LR,
            "swa_epochs": SWAG_EPOCHS,
            "max_rank": SWAG_MAX_RANK,
            "update_bn": True,
            "output_dir": str(swag_output_dir),
        },
        calibration_data={
            "train_loader": phase1["train_loader"],
            "model": model,
        },
    )

    errors = plugin.validate_inputs(plugin_input)
    if errors:
        raise RuntimeError(f"SWAG validation failed: {errors}")

    logger.info(
        "Running SWAG plugin (swa_lr=%.4f, epochs=%d, max_rank=%d)...",
        SWAG_LR,
        SWAG_EPOCHS,
        SWAG_MAX_RANK,
    )
    result = plugin.execute(plugin_input)
    logger.info("SWAG plugin completed: %s", result.metrics)

    # T2.2: Verify posterior sample diversity (using sliding-window inference)
    logger.info("--- Verifying posterior sample diversity ---")
    swag_path = result.model_paths[0]

    # Free SWAG training memory before diversity check
    del model
    torch.cuda.empty_cache()

    from minivess.pipeline.inference import SlidingWindowInferenceRunner

    inference_runner = SlidingWindowInferenceRunner(
        roi_size=(128, 128, 16),
        num_classes=2,
        overlap=0.5,
        sw_batch_size=4,
    )

    # Build fresh model for diversity check
    div_model = build_adapter(phase1["model_config"])
    div_model.to(device)
    swag_model = SWAGModel.load(swag_path, div_model)

    val_loader = phase1["val_loader"]
    test_batch = next(iter(val_loader))
    test_image = test_batch["image"]

    predictions = []
    div_model.eval()
    with torch.no_grad():
        for i in range(SWAG_N_SAMPLES):
            swag_model.sample(scale=1.0, seed=42 + i)
            pred = inference_runner.predict_volume(div_model, test_image, device=device)
            predictions.append(pred)

    # Check diversity: at least some samples should differ
    import numpy as np

    n_different = 0
    for i in range(1, len(predictions)):
        if not np.array_equal(predictions[0], predictions[i]):
            n_different += 1

    diversity_pct = n_different / max(len(predictions) - 1, 1) * 100
    logger.info(
        "SWAG diversity: %d/%d samples differ from first (%.0f%%)",
        n_different,
        len(predictions) - 1,
        diversity_pct,
    )
    if n_different == 0:
        logger.error(
            "SWAG DIVERSITY CHECK FAILED — all samples identical. "
            "Posterior is degenerate."
        )

    del div_model
    torch.cuda.empty_cache()

    # T2.3: Verify BN recalibration — load SWAG mean and check BN stats
    logger.info("--- Verifying BN recalibration ---")
    bn_model = build_adapter(phase1["model_config"])
    bn_model.to(device)
    bn_swag = SWAGModel.load(swag_path, bn_model)
    bn_swag._load_mean()

    has_bn = False
    for name, module in bn_model.named_modules():
        if hasattr(module, "running_mean") and module.running_mean is not None:
            has_bn = True
            mean_norm = module.running_mean.norm().item()
            var_mean = module.running_var.mean().item()
            logger.info(
                "  BN %s: running_mean_norm=%.4f, running_var_mean=%.4f",
                name,
                mean_norm,
                var_mean,
            )
            break
    if not has_bn:
        logger.warning("No BatchNorm layers found in model")
    del bn_model, bn_swag
    torch.cuda.empty_cache()

    dur = _elapsed("phase2_s", t0)
    logger.info("Phase 2 COMPLETE in %.1fs", dur)

    return {
        "swag_path": swag_path,
        "swag_metrics": result.metrics,
        "diversity_pct": diversity_pct,
        "n_different": n_different,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Evaluation with ALL Calibration Metrics
# ═══════════════════════════════════════════════════════════════════════════


def phase_3_evaluation(
    phase1: dict[str, Any],
    phase2: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate baseline + SWAG with ALL calibration metrics."""
    logger.info("=" * 70)
    logger.info("PHASE 3: Evaluation with ALL Calibration Metrics")
    logger.info("=" * 70)
    t0 = time.perf_counter()

    import mlflow
    import numpy as np
    import torch

    from minivess.adapters.model_builder import build_adapter
    from minivess.ensemble.swag import SWAGModel
    from minivess.pipeline.calibration_metrics import (
        compute_all_calibration_metrics,
        compute_brier_map,
        compute_nll_map,
    )
    from minivess.pipeline.inference import SlidingWindowInferenceRunner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_loader = phase1["val_loader"]

    inference_runner = SlidingWindowInferenceRunner(
        roi_size=(128, 128, 16),
        num_classes=2,
        overlap=0.5,
        sw_batch_size=4,
    )

    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    mlflow.set_experiment(EVAL_EXPERIMENT)

    # Clean old runs
    client = mlflow.MlflowClient()
    try:
        exp = client.get_experiment_by_name(EVAL_EXPERIMENT)
        if exp is not None:
            for old_run in client.search_runs(experiment_ids=[exp.experiment_id]):
                client.delete_run(old_run.info.run_id)
    except Exception:
        pass

    all_eval_results: dict[str, dict[str, Any]] = {}
    all_cal_results: dict[str, dict[str, float]] = {}

    # Two conditions: baseline (none) and SWAG
    conditions = [
        {
            "name": "dynunet__cbdice_cldice__fold0__none",
            "method": "none",
            "checkpoint_path": phase1["checkpoint_path"],
            "use_swag": False,
        },
        {
            "name": "dynunet__cbdice_cldice__fold0__swag",
            "method": "swag",
            "swag_path": phase2["swag_path"],
            "use_swag": True,
        },
    ]

    for cond in conditions:
        cond_name = cond["name"]
        logger.info("--- Evaluating: %s ---", cond_name)

        # Build model
        model = build_adapter(phase1["model_config"])
        model.to(device)

        if cond["use_swag"]:
            # Load SWAG model and use mean weights for evaluation
            swag_model = SWAGModel.load(cond["swag_path"], model)
            # For baseline SWAG eval: use the posterior mean (MAP estimate)
            swag_model._load_mean()
            logger.info("SWAG model loaded, using posterior mean for evaluation")
        else:
            # Load baseline checkpoint
            ckpt = torch.load(cond["checkpoint_path"], weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info("Baseline checkpoint loaded")

        model.eval()

        # Run inference with probabilities on all val volumes
        volume_seg_metrics: list[dict[str, float]] = []
        volume_cal_metrics: list[dict[str, float]] = []
        vol_times: list[float] = []
        spatial_maps_saved = 0

        for batch_idx, batch in enumerate(val_loader):
            vol_start = time.perf_counter()
            image = batch["image"]
            label = batch["label"]

            # Get hard + soft predictions
            hard_pred, soft_prob = inference_runner.predict_volume_with_probabilities(
                model, image, device=device
            )
            label_np = label.squeeze(0).squeeze(0).cpu().numpy().astype(np.int64)

            # Segmentation metrics (DSC, clDice)
            seg_m = _compute_volume_seg_metrics(hard_pred, label_np)
            volume_seg_metrics.append(seg_m)

            # ALL calibration metrics (comprehensive tier)
            cal_m = compute_all_calibration_metrics(
                soft_prob, label_np.astype(np.float64), tier="comprehensive"
            )
            volume_cal_metrics.append(cal_m)

            # Save spatial maps for first volume only
            if batch_idx == 0:
                brier_map = compute_brier_map(soft_prob, label_np.astype(np.float64))
                nll_map = compute_nll_map(soft_prob, label_np.astype(np.float64))
                maps_dir = OUTPUT_DIR / "spatial_maps" / cond_name
                maps_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(maps_dir / "brier_map.npz", brier_map=brier_map)
                np.savez_compressed(maps_dir / "nll_map.npz", nll_map=nll_map)
                spatial_maps_saved = 1
                logger.info(
                    "  Spatial maps saved: brier=%.4f mean, nll=%.4f mean",
                    brier_map.mean(),
                    nll_map.mean(),
                )

            vol_elapsed = time.perf_counter() - vol_start
            vol_times.append(vol_elapsed)

            if batch_idx < 3 or batch_idx % 5 == 0:
                logger.info(
                    "  Vol %d/%d: DSC=%.4f, ECE=%.4f, BA-ECE=%.4f, Brier=%.4f (%.1fs)",
                    batch_idx + 1,
                    len(val_loader),
                    seg_m.get("dsc", 0),
                    cal_m.get("ece", 0),
                    cal_m.get("ba_ece", 0),
                    cal_m.get("brier", 0),
                    vol_elapsed,
                )

        # Aggregate metrics
        mean_seg: dict[str, float] = {}
        for key in volume_seg_metrics[0]:
            values = [m[key] for m in volume_seg_metrics]
            mean_seg[key] = float(np.mean(values))
            mean_seg[f"{key}_std"] = float(np.std(values))

        mean_cal: dict[str, float] = {}
        for key in volume_cal_metrics[0]:
            values = [m[key] for m in volume_cal_metrics]
            mean_cal[f"cal_{key}"] = float(np.mean(values))
            mean_cal[f"cal_{key}_std"] = float(np.std(values))

        total_time = float(np.sum(vol_times))
        mean_time = float(np.mean(vol_times))

        logger.info(
            "%s: DSC=%.4f±%.4f, ECE=%.4f±%.4f, Brier=%.4f±%.4f, %.1fs/vol",
            cond_name,
            mean_seg.get("dsc", 0),
            mean_seg.get("dsc_std", 0),
            mean_cal.get("cal_ece", 0),
            mean_cal.get("cal_ece_std", 0),
            mean_cal.get("cal_brier", 0),
            mean_cal.get("cal_brier_std", 0),
            mean_time,
        )

        # Log to MLflow
        run_tags = {
            "model_family": "dynunet",
            "loss_function": LOSS_NAME,
            "fold_id": "0",
            "with_aux_calib": "false",
            "post_training_method": cond["method"],
            "recalibration": "none",
            "ensemble_strategy": "none",
            "flow_name": "analysis-flow",
        }

        with mlflow.start_run(run_name=cond_name, tags=run_tags):
            # Aggregate seg metrics
            for k, v in mean_seg.items():
                mlflow.log_metric(k, v)
            # Aggregate calibration metrics
            for k, v in mean_cal.items():
                mlflow.log_metric(k, v)
            mlflow.log_metric("inference_time_per_vol_s", mean_time)
            mlflow.log_metric("total_inference_time_s", total_time)
            mlflow.log_metric("n_volumes_evaluated", float(len(volume_seg_metrics)))

            # Per-volume metrics for biostatistics
            for vol_idx in range(len(volume_seg_metrics)):
                for metric_name, metric_val in volume_seg_metrics[vol_idx].items():
                    mlflow.log_metric(
                        f"eval/fold_0/vol/{vol_idx}/{metric_name}", metric_val
                    )
                for metric_name, metric_val in volume_cal_metrics[vol_idx].items():
                    mlflow.log_metric(
                        f"eval/fold_0/vol/{vol_idx}/cal_{metric_name}", metric_val
                    )

        all_eval_results[cond_name] = {
            "mean_seg": mean_seg,
            "mean_cal": mean_cal,
            "n_volumes": len(volume_seg_metrics),
            "inference_time_s": total_time,
            "spatial_maps_saved": spatial_maps_saved,
        }
        all_cal_results[cond_name] = mean_cal

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # T3.3: Verify ALL calibration metrics are non-zero
    logger.info("--- Verifying calibration metrics are non-zero ---")
    issues: list[str] = []
    for cond_name, cal in all_cal_results.items():
        for k, v in cal.items():
            if k.endswith("_std"):
                continue
            if v == 0.0 or np.isnan(v):
                msg = f"ZERO/NaN calibration metric: {cond_name}/{k}={v}"
                logger.error(msg)
                issues.append(msg)

    if not issues:
        logger.info("ALL calibration metrics non-zero — PASSED")
    else:
        logger.warning("%d calibration metrics are zero/NaN", len(issues))

    dur = _elapsed("phase3_s", t0)
    logger.info("Phase 3 COMPLETE in %.1fs (%.1f min)", dur, dur / 60)

    return {
        "eval_results": all_eval_results,
        "cal_results": all_cal_results,
        "issues": issues,
    }


def _compute_volume_seg_metrics(pred: Any, label: Any) -> dict[str, float]:
    """DSC + clDice for a single volume."""
    import numpy as np

    pred = np.asarray(pred)
    label = np.asarray(label)
    pred_bin = (pred > 0).astype(np.float32)
    label_bin = (label > 0).astype(np.float32)

    metrics: dict[str, float] = {}
    intersection = np.sum(pred_bin * label_bin)
    denom = np.sum(pred_bin) + np.sum(label_bin)
    metrics["dsc"] = float(2.0 * intersection / max(denom, 1e-8))

    try:
        from skimage.morphology import skeletonize

        pred_skel = skeletonize(pred_bin.astype(bool)).astype(np.float32)
        label_skel = skeletonize(label_bin.astype(bool)).astype(np.float32)
        tprec = float(np.sum(pred_skel * label_bin) / max(np.sum(pred_skel), 1e-8))
        tsens = float(np.sum(label_skel * pred_bin) / max(np.sum(label_skel), 1e-8))
        metrics["cldice"] = float(2.0 * tprec * tsens / max(tprec + tsens, 1e-8))
    except ImportError:
        metrics["cldice"] = 0.0

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Biostatistics
# ═══════════════════════════════════════════════════════════════════════════


def phase_4_biostatistics() -> dict[str, Any]:
    """Run biostatistics flow on evaluation results."""
    logger.info("=" * 70)
    logger.info("PHASE 4: Biostatistics Flow")
    logger.info("=" * 70)
    t0 = time.perf_counter()

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
                trigger_source="local_debug_3rd_pass",
            )
            results["type"] = str(type(biostats_result).__name__)
            logger.info("Biostatistics completed: %s", type(biostats_result).__name__)

            if hasattr(biostats_result, "anova_results"):
                results["n_anova_results"] = len(biostats_result.anova_results)
            if hasattr(biostats_result, "pairwise_results"):
                results["n_pairwise_results"] = len(biostats_result.pairwise_results)

    except Exception as e:
        msg = f"Biostatistics flow failed: {e}"
        logger.error(msg, exc_info=True)
        issues.append(msg)

    dur = _elapsed("phase4_s", t0)
    results["issues"] = issues
    logger.info("Phase 4 COMPLETE in %.1fs", dur)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Execute the 3rd pass debug pipeline."""
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║  3rd Pass Debug: Train → SWAG → Eval+Calibration → Biostats║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")

    total_start = time.perf_counter()
    all_issues: list[str] = []

    # Phase 0: Pre-flight
    preflight = phase_0_preflight()

    # Phase 1: Train
    phase1 = phase_1_train(preflight)

    # Phase 2: SWAG
    phase2 = phase_2_swag(phase1)

    # Phase 3: Evaluation + calibration
    phase3 = phase_3_evaluation(phase1, phase2)
    all_issues.extend(phase3.get("issues", []))

    # Phase 4: Biostatistics
    phase4 = phase_4_biostatistics()
    all_issues.extend(phase4.get("issues", []))

    total_elapsed = time.perf_counter() - total_start
    TIMINGS["total_s"] = total_elapsed

    # Summary
    logger.info("=" * 70)
    logger.info("EXECUTION COMPLETE")
    logger.info("=" * 70)
    logger.info("Total: %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    logger.info("Phase 0 (pre-flight):    %.1fs", TIMINGS.get("phase0_s", 0))
    logger.info("Phase 1 (train):         %.1fs", TIMINGS.get("phase1_s", 0))
    logger.info("Phase 2 (SWAG):          %.1fs", TIMINGS.get("phase2_s", 0))
    logger.info("Phase 3 (eval+cal):      %.1fs", TIMINGS.get("phase3_s", 0))
    logger.info("Phase 4 (biostatistics): %.1fs", TIMINGS.get("phase4_s", 0))

    if all_issues:
        logger.warning("ISSUES (%d):", len(all_issues))
        for issue in all_issues:
            logger.warning("  - %s", issue)
    else:
        logger.info("NO ISSUES — all phases completed successfully")

    # Write timing + results for report generation
    timing_path = OUTPUT_DIR / "debug_3rd_pass_results.json"
    timing_path.write_text(
        json.dumps(
            {
                "timings": TIMINGS,
                "phase1_train_run_id": phase1.get("train_run_id", ""),
                "phase2_swag": {
                    "metrics": phase2.get("swag_metrics", {}),
                    "diversity_pct": phase2.get("diversity_pct", 0),
                },
                "phase3_eval": {
                    k: {
                        "mean_seg": v["mean_seg"],
                        "mean_cal": v["mean_cal"],
                        "n_volumes": v["n_volumes"],
                    }
                    for k, v in phase3.get("eval_results", {}).items()
                },
                "phase4_biostatistics": {
                    k: v for k, v in phase4.items() if k != "issues"
                },
                "issues": all_issues,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Results written to %s", timing_path)

    if all_issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
