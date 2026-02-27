# MLflow Runs Evaluation & Verification Plan

> **Date**: 2026-02-27
> **Branch**: `feat/mlruns-evaluate-verification`
> **Purpose**: Comprehensive verification that all MLflow artifacts from the
> `dynunet_loss_variation_v2` experiment are complete, correct, and sufficient
> for academic reporting (Nature-level), ensemble construction, model serving,
> and future experiments.
> **Reviewed by**: 4 specialized agents (mlruns inventory, analysis pipeline,
> TRIPOD+AI guidelines, foundation-PLR patterns)

---

## 0. Executive Summary

The `dynunet_loss_variation_v2` experiment (4 losses x 3 folds x 100 epochs,
25.5 hours on RTX 2070 Super) completed successfully. However, verification
reveals **6 critical gaps** that must be addressed before the analysis pipeline
can run and before committing to additional training experiments:

| # | Gap | Severity | Impact |
|---|-----|----------|--------|
| 1 | **No model registry** — `mlruns/models/` does not exist | CRITICAL | Cannot serve models, build ensembles from registry, or promote champion |
| 2 | **Analysis flow never executed** — all code exists but 0% end-to-end coverage | CRITICAL | No ensemble results, no statistical comparisons, no champion selection |
| 3 | **No `log_model()` during training** — checkpoints saved as artifacts only | HIGH | Models not servable via MLflow pyfunc without manual loading |
| 4 | **No validation predictions saved** — `.npz` store exists but not wired into trainer | HIGH | Cannot recompute metrics post-hoc without re-running inference |
| 5 | **Broken/orphan runs polluting experiment** — 5 non-production runs in v2 experiment | MEDIUM | Confuses `discover_training_runs()`, wastes 2.2 GB storage |
| 6 | **Missing HD95 and NSD metrics** — MetricsReloaded computes them but we don't track | LOW | Incomplete per TRIPOD+AI / MetricsReloaded recommendations |

---

## 1. Current MLflow Inventory

### 1.1 Experiment Overview

| Exp ID | Name | Runs | Useful | Size |
|--------|------|------|--------|------|
| `843896622863223169` | dynunet_loss_variation_v2 | 9 | **4** | 4.0 GB |
| `498103124621304121` | dynunet_loss_variation | 9 | 0 | 1.2 GB |
| `444377748703248440` | test_smoke | 9 | 0 | 1.2 GB |
| `309410943942924473` | minivess_evaluation | 11 | 0 | 1.4 MB |
| `0` | Default | 0 | 0 | 8 KB |

**Only the 4 full v2 runs are research-grade data.** The remaining 34 runs are
test debris, aborted attempts, or a broken run.

### 1.2 The 4 Production Runs (dynunet_loss_variation_v2)

| Run ID | Loss | Duration | Status |
|--------|------|----------|--------|
| `af4adc15...` | dice_ce | 6.08h | FINISHED |
| `3a9f3615...` | cbdice | 6.24h | FINISHED |
| `4b2451ac...` | dice_ce_cldice | 6.83h | FINISHED |
| `01d904c6...` | cbdice_cldice | 6.34h | FINISHED |

**Per-run contents (verified):**
- 46 metrics (training + validation + eval)
- 300 training data points (3 folds x 100 epochs)
- 63 extended metric data points (MetricsReloaded every 5 epochs)
- 8 artifacts: 7 best checkpoints + `metric_history.json`
- ~453 MB per run

### 1.3 Non-Production Runs to Clean Up

| Run ID | Issue | Action |
|--------|-------|--------|
| `1df139b6...` | SCHEDULED status, never completed, no `last.pth` | Delete or mark FAILED |
| `e5d8e030...` | 2-epoch aborted attempt (dice_ce) | Delete |
| `e7626e3f...` | 2-epoch aborted attempt (cbdice) | Delete |
| `9f76077e...` | 2-epoch aborted attempt (dice_ce_cldice) | Delete |
| `9c02275d...` | 2-epoch aborted attempt (cbdice_cldice) | Delete |

---

## 2. Per-Run Artifact Verification Checklist

For each of the 4 production runs, verify the following. This is the **minimum
data contract** required for the analysis pipeline.

### 2.1 Metrics (46 per run)

| Category | Metrics | Data Points | Verified? |
|----------|---------|-------------|-----------|
| Training (per-epoch) | `train_loss`, `train_dice`, `train_f1_foreground`, `learning_rate` | 300 each | |
| Validation (per-epoch) | `val_loss`, `val_dice`, `val_f1_foreground` | 300 each | |
| Validation extended (every 5 epochs) | `val_cldice`, `val_masd`, `val_compound_masd_cldice` | 63 each | |
| Post-training eval (per fold) | `eval_fold{0,1,2}_dsc` | 1 each | |
| Post-training eval (per fold) | `eval_fold{0,1,2}_centreline_dsc` | 1 each | |
| Post-training eval (per fold) | `eval_fold{0,1,2}_measured_masd` | 1 each | |
| Bootstrap CIs (per fold, per metric) | `eval_fold{0,1,2}_{metric}_ci_{level,lower,upper}` | 1 each | |

**Test**: For each run, assert `len(metric_files) == 46` and verify data point
counts match expected values.

### 2.2 Parameters

| Parameter | Expected Value | Source |
|-----------|---------------|--------|
| `loss_function` | One of `dice_ce`, `cbdice`, `dice_ce_cldice`, `cbdice_cldice` | Tag |
| `num_folds` | `3` | Tag |
| `batch_size` | `1` | Param |
| `max_epochs` | `100` | Param |
| `num_classes` | `2` | Param |
| `learning_rate` | `0.0001` | Param |
| `mixed_precision` | `True` | Param |
| `patch_size` | Recorded | Param |
| `model_name` | `dynunet` | Param |
| `seed` | `42` | Param |
| `compute_profile` | `gpu_low` | Param |

**Test**: Assert all required params/tags exist for each run.

### 2.3 Artifacts (8 per run)

| Artifact | Path | Expected Size | Purpose |
|----------|------|---------------|---------|
| `checkpoints/best_val_loss.pth` | artifacts/ | ~65 MB | Best by val_loss |
| `checkpoints/best_val_dice.pth` | artifacts/ | ~65 MB | Best by val_dice |
| `checkpoints/best_val_f1_foreground.pth` | artifacts/ | ~65 MB | Best by val_f1 |
| `checkpoints/best_val_cldice.pth` | artifacts/ | ~65 MB | Best by clDice |
| `checkpoints/best_val_masd.pth` | artifacts/ | ~65 MB | Best by MASD |
| `checkpoints/best_val_compound_masd_cldice.pth` | artifacts/ | ~65 MB | Best by compound (primary) |
| `checkpoints/last.pth` | artifacts/ | ~65 MB | Final epoch checkpoint |
| `history/metric_history.json` | artifacts/ | ~46 KB | Per-epoch metric history |

**Tests:**
1. Assert all 8 artifacts exist for each run
2. Assert checkpoint sizes are consistent (~67.7 MB)
3. Assert `last.pth` size is identical across all 4 runs (same architecture)
4. Assert `metric_history.json` is valid JSON with 100 epoch entries per fold
5. Load each checkpoint and verify it contains `model_state_dict`,
   `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`

### 2.4 Checkpoint Loadability

| Test | What It Verifies |
|------|------------------|
| Load checkpoint, instantiate DynUNet, call `load_state_dict()` | Weights are compatible with architecture |
| Run forward pass on random input `(1, 1, 64, 64, 32)` | Model produces valid output shape `(1, 2, 64, 64, 32)` |
| Check output is not all-zeros or all-same-value | Model has learned something |
| Compare `best_val_compound_masd_cldice.pth` vs `last.pth` | Different weights (not accidentally the same) |
| Load all 7 checkpoints from same run | All loadable, all different |

### 2.5 Metric Reproducibility

| Test | What It Verifies |
|------|------------------|
| Load `best_val_compound_masd_cldice.pth`, run sliding window inference on fold val set | Reproduce DSC within 0.01 of logged value |
| Run MetricsReloaded on inference output | Reproduce clDice and MASD within tolerance |
| Check metric_history.json epoch-100 values match MLflow logged metrics | History and tracking are consistent |

---

## 3. Missing Artifacts — What Must Be Added

### 3.1 CRITICAL: Model Registry

**Current state**: `mlruns/models/` does not exist. No models are registered.

**What's needed for the analysis pipeline:**

| Registration | Model Name | Alias | Source |
|-------------|------------|-------|--------|
| Per-loss champion | `minivess-dynunet-{loss}` | `champion` | Best compound metric across 3 folds |
| Per-fold best | `minivess-dynunet-{loss}-fold{i}` | `latest` | Each fold's best compound checkpoint |
| Ensemble (strategy 1) | `minivess-ensemble-per-loss-single` | `latest` | K folds, primary metric, per loss |
| Ensemble (strategy 2) | `minivess-ensemble-all-loss-single` | `latest` | All folds across all losses |
| Ensemble (strategy 3) | `minivess-ensemble-per-loss-all` | `latest` | K folds, all 6 metrics, per loss |
| Ensemble (strategy 4) | `minivess-ensemble-all-loss-all` | `latest` | Full deep ensemble (up to 72 members) |
| Overall champion | `minivess-dynunet-champion` | `champion` | Single best model overall |

**Implementation**: Use existing `src/minivess/serving/model_logger.py`:
`log_single_model()` and `log_ensemble_model()` with `MiniVessSegModel` pyfunc wrapper.

### 3.2 HIGH: log_model() Integration

During training, only raw `.pth` checkpoints were saved as artifacts. The
`log_pyfunc_model()` method exists in `ExperimentTracker` but was not called
during the v2 training run.

**Impact**: Models cannot be loaded via `mlflow.pyfunc.load_model()` without
manually instantiating the architecture and loading weights.

**Fix options**:
1. **Post-hoc registration** (preferred): Script that loads each checkpoint,
   wraps in `MiniVessSegModel`, and calls `mlflow.pyfunc.log_model()` against
   the existing run.
2. **Re-training**: Not worth 25 hours just for pyfunc logging.

### 3.3 HIGH: Validation Predictions (.npz)

`src/minivess/pipeline/prediction_store.py` exists with `save_volume_prediction()`
and `load_volume_prediction()`, but the trainer does not call it during validation.

**Impact**: Cannot recompute metrics post-hoc (e.g., HD95, NSD, new compound
metrics) without re-running inference on all 70 volumes x 4 losses x 3 folds.

**Fix**: Add prediction saving to the post-training evaluation step. For the
existing v2 runs, run inference once and save predictions as a new artifact.

### 3.4 MEDIUM: Additional MetricsReloaded Metrics

Currently tracked: `dsc`, `centreline_dsc`, `measured_masd`.

Should also track (per MetricsReloaded + TRIPOD+AI):
- `measured_hausdorff_distance_perc` (HD95) — standard boundary metric
- `normalised_surface_distance` (NSD) — boundary overlap at tolerance

The `EvaluationRunner` already computes 5 metrics including HD95 and NSD, but only
3 are logged to MLflow. This is a code fix in `evaluate_fold_and_log()`.

---

## 4. Analysis Pipeline End-to-End Verification

The analysis pipeline (9 phases, all code-complete) has **never been executed
against real training data**. Each component must be verified.

### 4.1 MLflow Query Layer

| Test | What It Verifies |
|------|------------------|
| `discover_training_runs(experiment_name="dynunet_loss_variation_v2")` returns exactly 4 runs | Filters out aborted/broken runs correctly |
| Each discovered run has correct `loss_function` tag | Tag parsing works |
| `load_checkpoint(run_id, metric="val_compound_masd_cldice")` loads successfully | Artifact download + torch.load works |
| All 28 checkpoints (4 runs x 7 metrics) are individually loadable | No corrupt files |

### 4.2 Ensemble Builder (4 Strategies)

| Strategy | Expected Members | Test |
|----------|-----------------|------|
| `per_loss_single_best` | 3 members (3 folds, primary metric, 1 loss) | Build for each of 4 losses |
| `all_loss_single_best` | 12 members (3 folds x 4 losses, primary metric) | Build once |
| `per_loss_all_best` | 18 members (3 folds x 6 metrics, 1 loss) | Build for each of 4 losses |
| `all_loss_all_best` | 72 members max (3 folds x 4 losses x 6 metrics) | Build once, may deduplicate |

**Tests per strategy:**
1. Assert correct member count
2. Assert all members produce valid output on test input
3. Assert ensemble mean prediction has valid shape and range
4. Assert uncertainty decomposition (epistemic + aleatoric) works
5. Assert ensemble DSC >= best single member DSC (expected for averaging)

### 4.3 Evaluation Reproducibility

| Test | Tolerance | What It Verifies |
|------|-----------|------------------|
| Single model DSC matches logged `eval_fold{i}_dsc` | ±0.005 | Inference pipeline produces consistent results |
| Single model clDice matches logged `eval_fold{i}_centreline_dsc` | ±0.01 | MetricsReloaded skeleton computation is deterministic |
| Single model MASD matches logged `eval_fold{i}_measured_masd` | ±0.1 | Surface distance computation is stable |
| Bootstrap CIs match logged CI values | ±0.02 | Bootstrap with same seed produces same CIs |

### 4.4 Cross-Loss Comparison

| Test | What It Verifies |
|------|------------------|
| `build_comparison_table()` produces 4-row table with all metrics | All runs queryable |
| `paired_bootstrap_test(dice_ce, cbdice_cldice, metric="dsc")` returns p-value | Statistical test runs |
| All 6 pairwise comparisons complete (4 losses → C(4,2)=6 pairs) | Full comparison matrix |
| Holm-Bonferroni correction applied to 6 p-values | Multiple comparison handling |
| `format_comparison_markdown()` produces valid markdown | Report generation works |

### 4.5 Model Registration & Promotion

| Test | What It Verifies |
|------|------------------|
| `register_model()` creates entry in `mlruns/models/` | Registry population |
| `set_registered_model_alias("champion")` works | Alias system |
| `mlflow.pyfunc.load_model(f"models:/{name}@champion")` loads successfully | Serving readiness |
| `predict()` on loaded pyfunc produces valid segmentation | End-to-end serving |
| `predict_with_uncertainty()` returns mean + epistemic + aleatoric | UQ serving |

### 4.6 New MLflow Analysis Experiment

The analysis flow should create a new experiment `minivess_analysis_v2` with
structured entries:

| Entry Type | Count | Content |
|------------|-------|---------|
| Per-fold per-loss | 12 (4 losses x 3 folds) | All 6 validation metrics for that fold |
| Per-loss CV mean | 4 (4 losses) | Mean ± std across 3 folds for all metrics |
| Per-ensemble strategy | 4+ (4 strategies x relevant losses) | Ensemble DSC, clDice, MASD, UQ metrics |
| Champion model | 1 | Best overall model metrics |

**Tests:**
1. Assert 12 per-fold entries exist with correct tags
2. Assert 4 CV-mean entries have `mean_dsc`, `std_dsc`, etc.
3. Assert ensemble entries have uncertainty metrics
4. Assert all entries are queryable by `loss_function` and `fold_id` tags

---

## 5. Academic Reporting Requirements (TRIPOD+AI Compliance)

### 5.1 Reporting Frameworks Checklist

The following must be satisfiable from MLflow data:

| Framework | Items | Status |
|-----------|-------|--------|
| **TRIPOD+AI** (Collins et al., BMJ 2024) | 27 items, 52 subitems | Primary framework |
| **CLAIM 2024** (RSNA) | 42 items | Medical imaging specific |
| **MetricsReloaded** (Nature Methods 2024) | Metric selection rationale | Already using |
| **REFORMS** (Science Advances 2024) | 32 questions, 8 modules | Reproducibility |

### 5.2 Data That Must Be Extractable from MLflow

| TRIPOD+AI Item | Data Needed | Available in MLflow? |
|----------------|-------------|---------------------|
| 12a: Data partitioning | Fold splits (train/val per fold) | Partially (split file path logged, not actual splits) |
| 12c: Model specification | Architecture, hyperparameters, training details | Yes (params) |
| 12e: Performance measures | DSC, clDice, MASD with CIs | Yes (metrics + eval CIs) |
| 15: Model output | Probability maps, threshold | Not stored (no predictions saved) |
| 21: Participant numbers | N per fold per split | Partially (num_folds logged, not N per fold) |
| 22: Full model details | Weights, code, config | Yes (artifacts) |
| 23a: Performance + CIs | All metrics with 95% bootstrap CIs | Yes for eval, not for training metrics |
| 23b: Heterogeneity | Per-fold variance, per-volume performance | Partial (per-fold eval, not per-volume) |
| 18e: Data availability | Dataset DOI, access instructions | NOT in MLflow (must add) |
| 18f: Code availability | Repository URL, version | NOT in MLflow (must add) |

### 5.3 Missing for TRIPOD+AI Compliance

| Gap | What's Needed | Priority |
|-----|---------------|----------|
| Per-volume metrics | DSC, clDice, MASD per individual volume (not just mean) | HIGH |
| Fold split contents | Actual volume IDs per fold logged to MLflow | HIGH |
| Software versions | Python, PyTorch, MONAI, CUDA versions as params | MEDIUM |
| Hardware spec | GPU model, VRAM, RAM as params | MEDIUM |
| Dataset DOI | MiniVess dataset identifier | MEDIUM |
| Repo commit hash | `git rev-parse HEAD` as tag | MEDIUM |
| Training wall time per fold | Individual fold durations | LOW (in metric_history.json) |

### 5.4 Statistical Analyses Required

| Analysis | Method | Status |
|----------|--------|--------|
| **95% Bootstrap CIs** on all metrics | BCa bootstrap, 2000+ resamples | Exists for eval; need for comparisons |
| **Paired tests** (4 losses → 6 pairs) | Wilcoxon signed-rank on per-volume metrics | Code exists (`paired_bootstrap_test`); needs per-volume data |
| **Multiple comparison correction** | Holm-Bonferroni on 6 p-values | Not implemented |
| **Effect sizes** | Cohen's d or rank-biserial | Not implemented |
| **Calibration analysis** | ECE, reliability diagrams, temperature scaling | Code exists (`ensemble/calibration.py`); not run |
| **Subgroup analysis** | By volume Z-depth, voxel spacing, fold | Needs per-volume metrics |

### 5.5 Visualizations Required

| Visualization | Data Source | Status |
|---------------|-------------|--------|
| Violin/box plots (per fold per loss) | Per-volume metrics | Needs per-volume data |
| Learning curves (loss vs epoch) | `metric_history.json` | Data exists |
| Qualitative segmentation overlays | Predictions + reference | Needs saved predictions |
| Statistical comparison matrix (heatmap) | Paired test p-values | Needs per-volume data |
| Radar/spider chart (multi-metric) | Cross-loss means | Data exists |
| Bland-Altman (predicted vs reference volumes) | Per-volume DSC | Needs per-volume data |
| Uncertainty maps | Ensemble predictions | Needs ensemble inference |
| CDF curves (fraction achieving DSC >= threshold) | Per-volume DSC | Needs per-volume data |

---

## 6. Ensemble & Serving Verification

### 6.1 Single Model Serving

| Test | What It Verifies |
|------|------------------|
| Load `.pth` checkpoint, wrap in `MiniVessSegModel` | pyfunc wrapper works |
| `model.predict(input_tensor)` returns segmentation | Forward pass works |
| Output shape matches `ModelSignature` `(-1, 2, -1, -1, -1)` | Signature correct |
| ONNX export via `torch.onnx.export()` succeeds | ONNX serving ready |
| ONNX inference matches PyTorch inference (cosine sim > 0.999) | Export fidelity |

### 6.2 Ensemble Serving

| Test | What It Verifies |
|------|------------------|
| `MiniVessEnsembleModel` loads N member checkpoints | Multi-model loading |
| `predict()` returns averaged segmentation | Mean aggregation works |
| `predict_with_uncertainty()` returns 3 maps (mean, epistemic, aleatoric) | Lakshminarayanan decomposition |
| Ensemble with 3 members is faster than 3x single inference (shared preprocessing) | Efficiency |
| Greedy soup (`strategies.greedy_soup()`) produces single merged model | Weight averaging |

### 6.3 Four Ensemble Strategies (from builder.py)

| Strategy | Members | Test |
|----------|---------|------|
| `per_loss_single_best` | 3 per loss (fold 0,1,2 x primary metric) | 4 ensembles, 3 members each |
| `all_loss_single_best` | 12 (all folds x all losses x primary metric) | 1 ensemble, 12 members |
| `per_loss_all_best` | 18 per loss (3 folds x 6 metrics) | 4 ensembles, ≤18 members each |
| `all_loss_all_best` | ≤72 (3 folds x 4 losses x 6 metrics) | 1 ensemble, deduplicated |

---

## 7. Future Experiment Readiness

### 7.1 What Must Be Verified BEFORE Running More Experiments

Before running DynUNet width variations or other experiments:

| Check | Why |
|-------|-----|
| All 4 production runs have complete metrics | Don't lose comparison baseline |
| Per-volume metrics can be computed | Needed for statistical tests |
| `metric_history.json` has wall times per epoch | Runtime comparison |
| Fold splits are deterministic (`seed=42`) | Same splits for new experiments |
| `discover_training_runs()` correctly filters by experiment name | New experiment won't confuse old |
| Config YAML fully specifies experiment | Reproducible from config alone |
| `extended_frequency` and `tracked_metrics` logged as params | Comparability with future runs |

### 7.2 Params That Should Be Logged (Currently Missing)

| Param | Value | Why Needed |
|-------|-------|-----------|
| `extended_metric_frequency` | `5` | Compare MetricsReloaded overhead across experiments |
| `python_version` | `3.12.x` | Reproducibility |
| `pytorch_version` | `2.x.x` | Reproducibility |
| `monai_version` | `1.x.x` | Reproducibility |
| `cuda_version` | `12.x` | Reproducibility |
| `gpu_model` | `RTX 2070 Super` | Hardware comparison |
| `gpu_vram_mb` | `8192` | VRAM budget context |
| `total_ram_gb` | `62.7` | Memory context |
| `num_workers` | `2` | DataLoader config |
| `cache_rate` | `1.0` | Memory trade-off |
| `git_commit` | `HEAD` sha | Code version |
| `config_yaml_hash` | SHA256 of config file | Config reproducibility |
| `fold_split_hash` | SHA256 of split file | Data reproducibility |

### 7.3 Metrics That Should Be Logged (Currently Missing)

| Metric | When | Why |
|--------|------|-----|
| `eval_fold{i}_hd95` | Post-training eval | Standard boundary metric (TRIPOD+AI) |
| `eval_fold{i}_nsd` | Post-training eval | Normalized surface overlap |
| `eval_fold{i}_vol_{name}_dsc` | Post-training eval | Per-volume metrics for statistical tests |
| `eval_fold{i}_vol_{name}_cldice` | Post-training eval | Per-volume topology |
| `eval_fold{i}_vol_{name}_masd` | Post-training eval | Per-volume surface distance |
| `fold{i}_train_time_sec` | Post-fold | Per-fold wall time |
| `fold{i}_eval_time_sec` | Post-eval | Evaluation overhead |
| `peak_gpu_mb` | Post-training | Resource utilization |
| `peak_rss_gb` | Post-training | Memory utilization |

---

## 8. Implementation Roadmap (Ordered by Priority)

### Phase A: MLflow Data Integrity (Pre-Analysis Gate)

**Must be done before running the analysis flow.**

| Task | Description | Est. Tests |
|------|-------------|-----------|
| A1. Run artifact verification tests | Assert 46 metrics, 8 artifacts, all params per production run | ~20 |
| A2. Clean up orphan runs | Delete/mark 5 non-production runs in v2 experiment | ~3 |
| A3. Verify checkpoint loadability | Load all 28 production checkpoints, check state_dict keys | ~8 |
| A4. Verify metric_history.json | Parse, validate 100 epochs x 3 folds, check consistency | ~6 |
| A5. Log missing params | Add software versions, hardware spec, git commit to existing runs | ~4 |

### Phase B: Post-Hoc Artifact Enhancement

**Adds missing data to existing runs without re-training.**

| Task | Description | Est. Tests |
|------|-------------|-----------|
| B1. Post-hoc pyfunc registration | Wrap each checkpoint in `MiniVessSegModel`, call `log_model()` | ~8 |
| B2. Save validation predictions | Run inference on all folds, save as `.npz` artifacts | ~6 |
| B3. Compute per-volume metrics | DSC, clDice, MASD, HD95, NSD per volume per fold | ~10 |
| B4. Log per-volume metrics to MLflow | `eval_fold{i}_vol_{name}_{metric}` entries | ~4 |
| B5. Compute HD95 and NSD | Add to post-training evaluation | ~4 |
| B6. Log fold split contents | Volume IDs per fold as artifact | ~2 |

### Phase C: Analysis Flow Execution

**First end-to-end run of the analysis pipeline.**

| Task | Description | Est. Tests |
|------|-------------|-----------|
| C1. `discover_training_runs()` integration test | Finds exactly 4 production runs | ~4 |
| C2. Build all 4 ensemble strategies | Construct and verify member counts | ~12 |
| C3. Evaluate single models (reproduce logged metrics) | Assert reproducibility within tolerance | ~8 |
| C4. Evaluate ensembles (new metrics) | DSC, clDice, MASD for each ensemble strategy | ~8 |
| C5. Cross-loss paired bootstrap comparison | 6 pairwise tests with Holm-Bonferroni | ~6 |
| C6. Create analysis experiment in MLflow | 12 per-fold + 4 CV-mean + ensemble entries | ~8 |
| C7. Register champion model | Promote best to registry with alias | ~4 |
| C8. Generate analysis report | Markdown with embedded results | ~2 |

### Phase D: Academic Reporting Infrastructure

**Builds the figure/table generation pipeline for the manuscript.**

| Task | Description | Est. Tests |
|------|-------------|-----------|
| D1. DuckDB extraction pipeline | MLflow runs → DuckDB for fast SQL queries | ~6 |
| D2. Metric registry (no hardcoded strings) | YAML-driven metric names, display names, directions | ~4 |
| D3. Plot config + JSON sidecars | Style, colors, save_figure() with data provenance | ~4 |
| D4. Violin plots per fold per loss | Per-volume DSC, clDice, MASD distributions | ~2 |
| D5. Learning curves | Loss and metrics vs epoch, all folds overlaid | ~2 |
| D6. Statistical comparison heatmap | P-values + effect sizes matrix | ~2 |
| D7. Radar chart (multi-metric) | Cross-loss mean comparison | ~2 |
| D8. Qualitative segmentation overlays | Best/worst/median cases per loss | ~2 |
| D9. TRIPOD+AI checklist generator | Auto-fill from MLflow data | ~4 |
| D10. Reproducibility archive script | Model weights → Zenodo, config + splits + seeds | ~2 |

**Total estimated tests: ~154**

---

## 9. Cross-Reference: Related Planning Documents

| Document | Relevance |
|----------|-----------|
| `docs/results/dynunet_loss_variation_v2_report.md` | Training results (reference values for verification) |
| `docs/planning/loss-and-metrics-double-check-report.md` | Loss/metric research (what alternatives exist) |
| `docs/planning/compound-loss-implementation-plan.md` | Why MASD is not differentiable, boundary loss research |
| `docs/planning/loss-metric-improvement-implementation.xml` | Phases 0-3 implementation plan (training pipeline) |
| `docs/planning/multi-metric-downstream-double-check.md` | 5 artifact gaps identified pre-training |
| `docs/planning/experiment-planning-and-metrics-prompt.md` | Original experiment requirements |
| `docs/planning/experiment-run-to-mlflow-plan.md` | MLflow logging plan |
| `docs/planning/evaluation-and-ensemble-execution-plan.xml` | 9-phase analysis pipeline plan |
| `docs/planning/dynunet-ablation-plan.md` | Future ablation study design |
| `docs/planning/dynunet-evaluation-plan.xml` | Evaluation pipeline design |

## 10. Cross-Reference: Related GitHub Issues

| Issue | Title | Status | Relevance |
|-------|-------|--------|-----------|
| #76 | MLflow serving wrapper (pyfunc) | Closed | Code exists, needs execution against real data |
| #78 | Model registration with champion/challenger | Closed | Code exists, registry empty |
| #79 | Cross-loss comparison with paired bootstrap | Closed | Code exists, needs real data |
| #80 | Ensemble construction from multi-metric checkpoints | Closed | Code exists, needs execution |
| #82 | BentoML serving from MLflow registry | Open | Blocked by empty registry |
| #83 | MONAI Deploy MAP packaging | Open | Blocked by empty registry |
| #88 | Deep ensemble uncertainty decomposition | Closed | Code exists, needs execution |
| #89 | Evaluation config with primary metric | Closed | Config exists |
| #91 | Ensemble builder: 4 strategies | Closed | Code exists, needs execution |
| #92 | Unified evaluation runner | Closed | Code exists |
| #93 | Analysis Prefect Flow (Flow 3) | Closed | Code exists, never run end-to-end |
| #100 | Boundary Loss (Kervadec) | Open | Future loss (v3) |
| #101 | Generalized Surface Loss (Celaya) | Open | Future loss (v3) |

## 11. Reporting Framework References

- Collins et al. (2024) — TRIPOD+AI (BMJ)
- Mongan et al. (2024) — CLAIM 2024 Update (Radiology: AI)
- Maier-Hein et al. (2024) — MetricsReloaded (Nature Methods)
- Kapoor et al. (2024) — REFORMS (Science Advances)
- Sounderajah et al. (2025) — STARD-AI (Nature Medicine)
- Varoquaux et al. (2022) — ML in Life Sciences Reproducibility (Nature Methods)
- Wortsman et al. (2022) — Model Soups (ICML)
- Lakshminarayanan et al. (2017) — Deep Ensembles (NeurIPS)
