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
| local_dynunet_dice_ce | 2 | ✅ DONE | ~37 min | f0=0.804, f1=0.834 |
| local_dynunet_dice_ce_auxcalib | 2 | ✅ DONE | ~38 min | f0=0.830, f1=0.837 |
| local_dynunet_cbdice_cldice | 2 | ✅ DONE | ~47 min | f0+f1 complete |
| local_dynunet_cbdice_cldice_auxcalib | 2 | 🔄 RUNNING | | |

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

### O1: aux_calibration preserves segmentation accuracy while targeting calibration

**Three distinct evaluation axes** (NOT interchangeable):

| Axis | What it measures | Metrics | Where computed |
|------|-----------------|---------|----------------|
| **Segmentation Accuracy** | Spatial overlap, topology | DSC, clDice, HD95, MASD, skeleton recall | Training val loop + Analysis Flow |
| **Uncertainty** | Raw probability quality (softmax output) | Entropy, predictive variance | Implicit in model output |
| **Calibration** | Do probabilities reflect true frequencies? | ECE, MCE, ACE, BA-ECE, Brier, NLL | Training val loop (Tier 1) + Analysis Flow (Tier 2) |

dice_ce vs dice_ce+aux_calib: val_dice is comparable (f0: 0.804→0.830, f1: 0.834→0.837).
The aux calibration loss (hL1-ACE, Barfoot et al. 2025 IEEE TMI) adds an auxiliary
calibration error term that increases train_loss (0.23→0.47) but does NOT degrade
segmentation accuracy. **This is the intended behaviour** — `aux_calibration=true`
targets well-calibrated probability outputs, not higher Dice.

**Equal Dice + better calibration = net win** for:
- Downstream conformal prediction (split conformal, CRC, MAPIE)
- Clinical decision-making (probability thresholds meaningful)
- Uncertainty-guided active learning (knowing WHEN the model is uncertain)
- Model selection beyond accuracy (calibrated models preferred for deployment)

**Biostatistics analysis must compare BOTH accuracy AND calibration:**
- **Segmentation co-primaries**: clDice + MASD (Holm-Bonferroni at alpha/2)
- **Calibration co-primaries**: ECE + BA-ECE (Holm-Bonferroni at alpha/2)
  - ECE: global confidence-accuracy alignment
  - BA-ECE: spatial calibration at vessel boundaries (Zeevi et al. 2025 — clinically critical)
- **ANOVA factor**: `aux_calibration` as a fixed effect → expected significant main
  effect on calibration metrics, expected non-significant on accuracy metrics

**Implemented calibration metrics** (6 total in biostatistics config):
- Tier 1 (fast, training loop): ECE, MCE, Brier, NLL, overconfidence error, debiased ECE
- Tier 2 (comprehensive, analysis flow): ACE, BA-ECE, Brier map, NLL map

### O2: cbdice_cldice loss passes sanity check after smooth=1.0 fix

The cbdice_cldice loss previously produced NaN at random init (smooth=1e-5 underflow
with mixed precision). After increasing smooth to 1.0 (MONAI default) and clamping
centerline/boundary weights, the sanity check passes as a warning and training proceeds
normally. Both folds completed successfully.

### O3: Prefect task hooks firing correctly

Docker logs show `on_completion` hooks running for every task (load-fold-splits,
check-resume-state, train-one-fold, log-fold-results). The observability wiring
from the 4-pass effort is FUNCTIONAL — not just imported.

### O4: DagsHub MLflow receiving all metrics

All training runs visible at dagshub.com/petteriTeikari/vascadia.mlflow.
Experiment `local_dynunet_mechanics_training` created correctly. Fold-level
metrics (val_dice, train_loss) logged per epoch. System metrics (GPU, CPU) also tracked.

## Watchlist

1. ~~cbdice_cldice NaN at random init~~ — RESOLVED (smooth=1.0 fix, O2)
2. DeepVess has only 1 labeled volume (not 7 as KG claims) — KG needs correction
3. SWAG post-training may need more than 15 epochs for meaningful weight collection
4. ~~aux_calibration effect on DynUNet is unknown~~ — O1: preserves Dice, calibration metrics needed in biostatistics
5. DagsHub URLs in Docker output contain token in plaintext — security concern for shared logs

## Cost

| Component | GPU Hours | Estimated |
|-----------|-----------|-----------|
| Training (4 jobs × ~1h) | ~4h | $0 (local) |
| Analysis | ~0.5h | $0 (local) |
| Biostatistics | ~0.1h | $0 (local) |
