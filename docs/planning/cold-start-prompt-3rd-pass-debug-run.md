# Cold-Start Prompt: 3rd Pass Local Debug Run (SWAG + Calibration)

**Date**: 2026-03-21
**Current branch**: `test/local-debug-3flow-execution`
**Session goal**: Execute 3rd pass debug run with SWAG + ALL calibration metrics

---

## How to Use This Prompt

```bash
claude -p "Read and execute the plan at:
/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/cold-start-prompt-3rd-pass-debug-run.md"
```

---

## STATUS: Ready for 3rd Pass Execution

Previous session (this one) completed:
- Phase A0: Renamed SWA → checkpoint averaging (~30 files)
- Phase A1+A2: Implemented real SWAG plugin (Maddox 2019) with TDD (18 tests)
- Phase B0: Created comprehensive calibration metrics module (11 functions, 22 tests)
- Phase B1: Integrated Tier 1 fast calibration metrics into training validation loop (9 tests)
- Phase B2: Integrated ALL calibration metrics into Analysis Flow evaluation runner
- Phase B3: Updated BiostatisticsConfig with calibration metrics + ROPE values (6 tests)

**Test suite**: 5678 passed, 2 skipped, 0 failed (make test-staging)

---

## What to Execute

### Plan file:
`docs/planning/run-debug-factorial-experiment-report-3rd-pass-plan.xml`

### Report output:
`docs/planning/run-debug-factorial-experiment-report-3rd-pass-local.md`

### Use `/factorial-monitor` skill for flow tracking + timing

### Execution order (5 phases):

**Phase 0: Pre-Flight Verification**
- T0.1: Verify SWA→checkpoint_averaging rename (grep returns 0 hits)
- T0.2: Verify SWAG plugin imports
- T0.3: Verify calibration metrics module imports
- T0.4: Run `make test-staging` → 0 failures
- T0.5: Verify SegmentationMetrics.compute() returns Tier 1 calibration metrics

**Phase 1: Train DynUNet (2 Epochs)**
- T1.1: Train DynUNet 2 epochs with cbdice_cldice, fold-0
- T1.2: Verify val_ece, val_mce, val_brier logged to MLflow
- Expected: ~5 min

**Phase 2: SWAG Post-Training (5 Epochs)**
- T2.1: Run SWAG plugin on 2-epoch checkpoint (swa_lr=0.01, swa_epochs=5, max_rank=10)
- T2.2: Verify posterior sample diversity (5 samples → different predictions)
- T2.3: Verify BN recalibration (running_mean/running_var change)
- Expected: ~7 min

**Phase 3: Analysis Flow with ALL Calibration Metrics**
- T3.1: Evaluate baseline (none) with ALL calibration metrics on 24 val volumes
- T3.2: Evaluate SWAG ensemble (30 posterior samples) with ALL calibration metrics
- T3.3: Verify ALL calibration metrics are non-zero (ECE, MCE, ACE, BA-ECE, Brier, NLL, etc.)
- T3.4: Verify spatial maps saved (Brier map, NLL map as .npz)
- Expected: ~15 min

**Phase 4: Biostatistics Flow**
- T4.1: Run biostatistics with calibration metrics in factorial ANOVA
- T4.2: Verify DuckDB has calibration metric columns
- T4.3: Verify pairwise comparisons on ECE, BA-ECE (expect SWAG ≤ none)
- Expected: ~1 min

**Phase 5: Report + Verify**
- T5.1: Write 3rd pass report to `docs/planning/run-debug-factorial-experiment-report-3rd-pass-local.md`
- T5.2: Run `make test-staging` → confirm no regressions

---

## New Components Since 2nd Pass

### SWAG (Maddox et al. 2019) — Post-Training Plugin
- `src/minivess/ensemble/swag.py` — SWAGModel: low-rank+diagonal Gaussian posterior
- `src/minivess/pipeline/post_training_plugins/swag.py` — SWAGPlugin: resumes training with SWALR
- `src/minivess/config/post_training_config.py` — SWAGPluginConfig (swa_lr, swa_epochs, max_rank, n_samples)

### Calibration Metrics — Comprehensive Module
- `src/minivess/pipeline/calibration_metrics.py` — 11 new functions:
  - **Tier 1 (fast)**: compute_ece, compute_mce, compute_rmsce, compute_brier_score, compute_nll, compute_overconfidence_error, compute_debiased_ece
  - **Tier 2 (comprehensive)**: compute_ace, compute_ba_ece, compute_brier_map, compute_nll_map
  - **Convenience**: compute_all_calibration_metrics(tier="fast"|"comprehensive")

### Training Loop Integration
- `src/minivess/pipeline/metrics.py` — SegmentationMetrics now computes Tier 1 calibration metrics during validation (val_ece, val_mce, val_brier, etc.)

### Analysis Flow Integration
- `src/minivess/pipeline/evaluation_runner.py` — UnifiedEvaluationRunner now computes ALL calibration metrics and adds them to FoldResult.aggregated with `cal_` prefix
- `src/minivess/pipeline/inference.py` — New `infer_dataset_with_probabilities()` returns soft predictions

### Biostatistics Integration
- `src/minivess/config/biostatistics_config.py` — 14 default metrics (8 seg + 6 cal), calibration ROPE values, calibration_co_primary_metrics field

### Renamed Components (Phase A0)
- `SWAPlugin` → `CheckpointAveragingPlugin`
- `MultiSWAPlugin` → `SubsampledEnsemblePlugin`
- `uniform_swa()` → `uniform_checkpoint_average()`
- Config: `swa:` → `checkpoint_averaging:`, `multi_swa:` → `subsampled_ensemble:`

---

## Three Co-Primary Calibration Metrics (ANOVA Response Variables)

| Metric | Dimension | Key in MLflow | Why |
|--------|-----------|---------------|-----|
| **ECE** | Global | `cal_ece` | Standard baseline, comparable to all literature |
| **BA-ECE** | Spatial | `cal_ba_ece` | Calibration WHERE it matters — vessel boundaries |
| **Brier** | Overall | `cal_brier` | Proper scoring rule, measures both calibration + sharpness |

---

## Key References (read before executing)

```
1. docs/planning/run-debug-factorial-experiment-report-3rd-pass-plan.xml (THE PLAN)
2. docs/planning/run-debug-factorial-experiment-report-2nd-pass-local.md (PREVIOUS REPORT)
3. docs/planning/calibration-and-swag-implementation-plan.xml (IMPLEMENTATION PLAN)
4. src/minivess/pipeline/calibration_metrics.py (CALIBRATION MODULE)
5. src/minivess/ensemble/swag.py (SWAG MODEL)
6. src/minivess/pipeline/post_training_plugins/swag.py (SWAG PLUGIN)
7. src/minivess/pipeline/metrics.py (TRAINING LOOP INTEGRATION)
8. src/minivess/pipeline/evaluation_runner.py (ANALYSIS FLOW INTEGRATION)
```

---

## What NOT to Do

- Do NOT skip calibration metric verification — every metric must be non-zero
- Do NOT use `subsampled_ensemble` — it was replaced by `swag` in the factorial design
- Do NOT hardcode calibration thresholds — they come from BiostatisticsConfig
- Do NOT run on more than 1 model/1 loss/1 fold — this is debug validation only
- Do NOT skip SWAG diversity check — identical samples = broken posterior
