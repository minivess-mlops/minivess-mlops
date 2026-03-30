# Local DynUNet Mechanics Test — Live Report

**Branch**: `test/local-dynunet-run`
**Started**: 2026-03-30
**Plan**: `local-dynunet-mechanics-debug-plan.xml` v1.1
**MLflow**: https://dagshub.com/petteriTeikari/vascadia.mlflow

## Pre-Launch Validation

| Gate | Status | Result |
|------|--------|--------|
| Staging tests | ✓ | 6948 passed, 0 failed |
| Docker images | ✓ | base, train, analyze built |
| CUDA in container | ✓ | PyTorch 2.10.0+cu128, GPU=True |
| Data volumes | ✓ | MiniVess 70 + DeepVess 1 |
| DagsHub MLflow | ✓ | URI configured |
| Experiment configs | ✓ | 4 configs in image |
| Dry run | ✓ | Preflight passed |

## Factorial Design

| Factor | Values | Count |
|--------|--------|-------|
| model_family | dynunet | 1 |
| loss_name | dice_ce, cbdice_cldice | 2 |
| aux_calibration | false, true | 2 |
| post_training | none, swag | 2 |
| ensemble_strategy | per_loss_single_best, all_loss_single_best | 2 |
| **Total conditions** | | **16** |
| Folds | 2 | |
| Epochs | 15 | |
| Training jobs | 4 (2 losses × 2 aux_calib) | |

## Training Status Matrix

| Experiment | Folds | Status | Duration | val_dice |
|------------|-------|--------|----------|----------|
| local_dynunet_dice_ce | 2 | PENDING | | |
| local_dynunet_dice_ce_auxcalib | 2 | PENDING | | |
| local_dynunet_cbdice_cldice | 2 | PENDING | | |
| local_dynunet_cbdice_cldice_auxcalib | 2 | PENDING | | |

## Analysis Status

| Component | Status |
|-----------|--------|
| Ensemble evaluation | PENDING |
| DeepVess external test | PENDING |
| Per-volume metrics | PENDING |

## Biostatistics Status

| Component | Status |
|-----------|--------|
| DuckDB materialization | PENDING |
| Pairwise comparisons | PENDING |
| R/ggplot2 figures | PENDING |
| LaTeX tables | PENDING |

## Observations

(Updated during execution)

## Watchlist

1. cbdice_cldice NaN at random init — sanity check downgraded to warning
2. DeepVess has only 1 labeled volume (not 7 as KG claims) — statistics limited
3. SWAG post-training may need more than 15 epochs for meaningful weight collection
4. aux_calibration effect on DynUNet is unknown — may not improve on this architecture

## Cost

| Component | GPU Hours | Estimated |
|-----------|-----------|-----------|
| Training (4 jobs × ~1h) | ~4h | $0 (local) |
| Analysis | ~0.5h | $0 (local) |
| Biostatistics | ~0.1h | $0 (local) |
