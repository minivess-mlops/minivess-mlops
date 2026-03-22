---
title: "Intermediate Plan Synthesis — Pre-Debug Run"
status: active
created: "2026-03-20"
builds_on:
  - intermedia-plan-synthesis.md (v1, 2026-03-07)
  - intermedia-plan-synthesis-v2.md (v2, 2026-03-08)
---

# MinIVess MLOps v2 — Pre-Debug Run Synthesis

**Date:** 2026-03-20
**Branch:** test/debug-factorial-run (from main after PR #871)
**Purpose:** Eliminate all ambiguity before executing the 4-flow debug factorial run
**Scope:** COMPLETE specification of the factorial design, flow pipeline, and statistical analysis

This document supersedes v1 and v2 for the factorial experiment scope. v1 and v2
remain valid references for infrastructure decisions (Docker volumes, config pipeline
gaps) but this document is the definitive source for the factorial experiment.

---

## Part 1: The Complete Factorial Design

### 1.1 What Is the Experiment?

We are running a **multi-factor comparison** of segmentation models on multiphoton
brain vasculature data. This is a **platform paper** (Nature Protocols) — the
contribution is the MLOps platform itself, not the segmentation SOTA. The factorial
experiment demonstrates that the platform handles arbitrary model/loss/calibration
combinations without code changes.

### 1.2 The FULL Factorial — Every Factor, Every Level

The experiment has factors at **four distinct layers**. Each layer adds new
conditions on top of the previous layer's output.

#### LAYER A: Training Factors (GPU, cloud-only)

These factors are varied during model training. Each combination produces one
trained model (checkpoint). The Cartesian product defines the training grid.

| Factor | Levels | Values | Source of Truth |
|--------|--------|--------|-----------------|
| **model_family** | 4 | dynunet, mambavesselnet, sam3_topolora, sam3_hybrid | `configs/hpo/paper_factorial.yaml::factors.model_family` |
| **loss_name** | 3 | cbdice_cldice, dice_ce, dice_ce_cldice | `configs/hpo/paper_factorial.yaml::factors.loss_name` |
| **aux_calibration** | 2 | true, false | `configs/hpo/paper_factorial.yaml::factors.aux_calibration` |
| **fold_id** | 3 (prod) / 1 (debug) | 0, 1, 2 | `configs/hpo/paper_factorial.yaml::fixed.num_folds` |

**Training grid:**
- Production: 4 × 3 × 2 = 24 conditions × 3 folds = **72 training runs**
- Debug: 4 × 3 × 2 = 24 conditions × 1 fold = **24 training runs**

**Zero-shot baselines** (evaluated but NOT trained in the factorial grid):
- **sam3_vanilla**: Frozen SAM3 encoder, zero-shot eval on MiniVess + DeepVess
- **vesselfm**: Foundation model, zero-shot eval on DeepVess ONLY (data leakage on MiniVess)

**Fixed training parameters** (NOT factors — model-adaptive defaults):
- max_epochs: 50 (production) / 2 (debug)
- learning_rate: 1.0e-3
- batch_size: 2
- dataset: MiniVess (70 volumes, 3-fold CV seed=42)

#### LAYER B: Post-Training Factors (CPU, local-capable)

Applied to each trained checkpoint AFTER training. These are model-agnostic
weight-space operations. No gradients flow — these are NOT end-to-end
differentiable (cf. DiffML, Schlosser et al. 2023).

| Factor | Levels | Values | Source of Truth |
|--------|--------|--------|-----------------|
| **post_training_method** | 3-4 | none, swa, multi_swa, model_soup | `src/minivess/config/post_training_config.py` |
| **calibration_method** | 2-3 | none, temperature_scaling, conformal | `knowledge-graph/decisions/L3-technology/calibration_method.yaml` |

**Post-training operations:**
- **SWA** (Stochastic Weight Averaging): Averages checkpoints saved at different
  training stages for improved generalization (Izmailov et al., 2018)
- **Multi-SWA**: Creates N independent SWA models via checkpoint subsampling
- **Model Soup**: Weight interpolation between checkpoints (Wortsman et al., 2022)
- **Temperature Scaling**: Post-hoc calibration of softmax outputs (Guo et al., 2017)
- **Conformal Prediction**: Distribution-free coverage guarantees (MAPIE, Mondrian)

**Post-training expansion:**
Each training run can produce multiple post-training variants. For the debug run,
the minimum set is: {none, swa} × {none, temperature_scaling} = 4 variants per
training condition. This means 24 training runs → up to 96 post-training variants.

**Key principle:** Post-training creates NEW MLflow runs tagged with the method,
inheriting all upstream tags. This enables Biostatistics to treat post-training
method as a factorial factor.

#### LAYER C: Analysis Factors (CPU for DynUNet, GPU for SAM3)

The Analysis Flow takes all training + post-training runs and builds ensemble
predictions. Each ensemble strategy is a NEW factorial condition for Biostatistics.

| Factor | Levels | Values | Source of Truth |
|--------|--------|--------|-----------------|
| **ensemble_strategy** | 4 | per_loss_single_best, all_loss_single_best, per_loss_all_best, all_loss_all_best | `src/minivess/ensemble/builder.py::EnsembleStrategyName` |
| **inference_strategy** | 1-2 | standard_patch (primary), fast_patch | `src/minivess/config/evaluation_config.py::InferenceStrategyConfig` |

**Ensemble strategies:**
1. **per_loss_single_best**: For each loss, take the top fold → 3 ensembles (one per loss)
2. **all_loss_single_best**: Take the single best fold across all losses → 1 ensemble
3. **per_loss_all_best**: For each loss, take ALL folds × ALL metric checkpoints → 3 ensembles
4. **all_loss_all_best**: Full deep ensemble (all losses × folds × checkpoints) → 1 ensemble

**Critical: Ensemble averaging happens at the LOGIT level.**
The correct approach: average voxel-level probability maps → threshold → compute metrics.
The WRONG approach: compute per-member metrics → average the metrics.
Implementation: `_EnsembleInferenceWrapper` in `analysis_flow.py`.

**Analysis creates runs in a SEPARATE MLflow experiment** (`minivess_evaluation`),
not in the training experiment. This prevents polluting training metrics with
evaluation-only runs.

**Uncertainty quantification** from deep ensembles (Lakshminarayanan et al., 2017):
- Total uncertainty: H[p̄] = entropy of mean ensemble prediction
- Aleatoric: E[H[p]] = average per-member entropy
- Epistemic: MI = H[p̄] - E[H[p]] = model disagreement
- Implementation: `MiniVessEnsembleModel.predict_with_uncertainty()`

#### LAYER D: Biostatistics Factors (CPU, researcher's laptop)

These are NOT experimental factors — they are **analytical choices** (researcher
degrees of freedom). The specification curve analysis systematically varies them.

| Analytical Choice | Levels | Values |
|-------------------|--------|--------|
| **metric** | 8+ | clDice, MASD, DSC, HD95, ASSD, NSD, BE₀, BE₁ |
| **aggregation** | 2 | mean, median |
| **MCC_method** | 2 | holm_bonferroni (co-primary), bh_fdr (secondary) |
| **alpha** | configurable | from `BiostatisticsConfig.alpha` (default 0.05) |
| **rope** | per-metric | from `BiostatisticsConfig.rope_values` |
| **fold_subset** | varies | all folds, leave-one-out, single fold |
| **test_type** | 4+ | wilcoxon, friedman, bayesian_signed_rank, bootstrap |
| **effect_size** | 3 | cohens_d, cliffs_delta, vda |

**The specification curve** iterates over all combinations of these analytical
choices to demonstrate how conclusions change (Simonsohn et al., 2020).

### 1.3 The COMPLETE Factorial — All 6 Factors

**Source of truth:** `docs/planning/pre-gcp-master-plan.xml` line 16:
`4 models × 3 losses × 2 aux_calib × 3 post-training × 2 recalibration × 5 ensemble`

**The factorial design is COMPOSABLE via Hydra config groups.** Users define
their own factorial YAML in `configs/factorial/`. Our `paper_full.yaml` and
`debug.yaml` are just TWO EXAMPLE named configs. A different lab could create
`my_lab.yaml` with 2 models × 1 loss × 1 calib × 1 post-training × 1 recalib × 2 ensemble.

**The YAML structure is sectioned by layer:**
```yaml
# configs/factorial/paper_full.yaml
factors:
  training:        # Layer A (GPU, cloud)
    model_family: [dynunet, mambavesselnet, sam3_topolora, sam3_hybrid]  # 4
    loss_name: [cbdice_cldice, dice_ce, dice_ce_cldice]                  # 3
    aux_calibration: [true, false]                                        # 2
  post_training:   # Layer B (CPU, local)
    method: [none, swa, multi_swa]                                        # 3
    recalibration: [none, temperature_scaling]                            # 2
  analysis:        # Layer C (CPU/GPU)
    ensemble_strategy: [none, per_loss_single_best, all_loss_single_best,
                        per_loss_all_best, all_loss_all_best]             # 5
fixed:
  max_epochs: 50
  num_folds: 3

# Full factorial: 4 × 3 × 2 × 3 × 2 × 5 = 720 conditions × 3 folds
```

### 1.4 Named Configs (Composable via Hydra)

The platform provides named configs. Users can create their own.

#### `configs/factorial/paper_full.yaml` — Publication Gate (Nature Protocols)

```
Layer A: 4 models × 3 losses × 2 aux_calib = 24 training cells × 3 folds = 72 runs
Layer B: × 3 post-training methods × 2 recalibration = 6 variants per cell = 432 post-training
Layer C: × 5 ensemble strategies = 2160 evaluation runs + zero-shot baselines
TOTAL: 720 conditions × 3 folds = 2160 evaluations
ANOVA: Layered — core 3-way → extended 5-way → ensemble analysis
Cost: ~$150+ on GCP L4 spot (training only; post-training/analysis/biostats on local CPU)
```

#### `configs/factorial/debug.yaml` — Pipeline Validation (All 6 Dimensions, Fewer Levels)

```
Layer A: 4 models × 3 losses × 2 aux_calib = 24 training cells × 1 fold = 24 runs
Layer B: × 2 post-training methods × 2 recalibration = 4 variants per cell = 96 post-training
Layer C: × 4 ensemble strategies = 384 evaluation runs + zero-shot baselines
TOTAL: 384 conditions × 1 fold = 384 evaluations (2 epochs, half data)
ANOVA: Same layered structure as production but K=1 (per-volume replication)
Cost: ~$5-10 on GCP L4 spot
```

**Debug scope reductions (ONLY these 3 + level reductions):**
1. Epochs: 2 (not 50)
2. Data: half volumes (~23 train / ~12 val, not ~47 / ~23)
3. Folds: 1 (fold-0 only, not 3-fold CV)
4. Post-training levels: 2 (not 3) — {none, swa} instead of {none, swa, multi_swa}
5. Ensemble levels: 4 (not 5) — drop `none` for debug

**EVERYTHING ELSE is identical:** All 4 models, all 3 losses, both calibration
levels, recalibration, all remaining ensemble strategies.

**NOTHING ELSE is reduced.** All 4 models, all 3 losses, both calibration levels,
all ensemble strategies, all baselines, all statistical tests. Debug = full
production with less training, NOT less testing.

#### Full Factorial Research Design (FFRD) — For Nature Protocols Submission

```
TRAINING:    4 models × 3 losses × 2 calibrations × 3 folds × 50 epochs = 72 runs
POST-TRAIN:  72 runs × {none, swa, multi_swa, calibrated} = 288 variants
ANALYSIS:    288 variants × 4 ensemble strategies = up to 1152 evaluations
             + 2 zero-shot baselines × 3 folds = 1158 total
BIOSTATISTICS: Full N-way ANOVA with fold as random effect
               + complete spec curve + rank stability + TRIPOD mapping
```

**Estimated cost:** ~$65 on GCP L4 spot for training, <$5 for post-training/analysis

### 1.4 The Metric Hierarchy (MetricsReloaded)

| Tier | Metrics | Correction | Rationale |
|------|---------|------------|-----------|
| **Co-primary** | clDice, MASD | Holm-Bonferroni | Topology + boundary accuracy for tubular structures |
| **FOIL** | DSC | BH-FDR | Included to show misleading rankings (rank inversion IS a finding) |
| **Secondary** | HD95, ASSD, NSD, BE₀, BE₁ | BH-FDR | Supplementary analysis |
| **Compound** | compound_masd_cldice | N/A | Champion selection metric (combines MASD + clDice) |

**The DSC vs clDice rank inversion** is a key finding. DSC rewards volume-filling
predictions, while clDice rewards topology-preserving predictions. A model can score
high DSC but low clDice by filling vessel volumes while missing thin branches.
This demonstrates WHY MetricsReloaded recommends task-specific metrics.

---

## Part 2: The 4-Flow Pipeline — Exact Contracts

### 2.1 Flow Execution Order

```
┌─────────────────────────────────────────────────────────────────────┐
│ Model Training Flow (Flow 2) — GPU REQUIRED (GCP L4 via SkyPilot)  │
│                                                                     │
│ Input:  configs/hpo/paper_factorial.yaml                           │
│ Output: MLflow runs in "minivess_training" experiment              │
│         - Checkpoints: best_val_compound_masd_cldice.pth           │
│         - Tags: model_family, loss_function, fold_id, with_aux_calib│
│         - Metrics: train/loss, val/dice, val/cldice, val/masd      │
│         - Artifact: config/resolved_config.yaml                     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ MLflow tag discovery
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Post-Training Flow (Flow 2.5) — CPU CAPABLE (local laptop OK)      │
│                                                                     │
│ Input:  Upstream training run (by experiment name + flow_name tag)  │
│ Output: NEW MLflow runs in SAME experiment, tagged with method     │
│         - SWA checkpoint (averaged weights)                        │
│         - Calibration params (temperature T)                       │
│         - Tags: post_training_method, upstream_training_run_id     │
│         - Inherits: model_family, loss_function, fold_id, calib    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ MLflow tag discovery
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Analysis Flow (Flow 3) — CPU for DynUNet, GPU for SAM3 inference   │
│                                                                     │
│ Input:  All training + post-training runs from upstream experiment  │
│ Output: NEW MLflow runs in "minivess_evaluation" experiment        │
│         - Per-volume metrics: eval/{fold}/vol/{id}/{metric}        │
│         - External test: test/deepvess/all/{metric}                │
│         - Ensemble results: one run per ensemble strategy          │
│         - UQ: total/aleatoric/epistemic uncertainty maps           │
│         - Champion: model registry entry + champion tags           │
│         - Artifacts: comparison_table.md, comparison_table.tex     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ MLflow tag discovery
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Biostatistics Flow (Flow 5b) — CPU ONLY (local laptop)             │
│                                                                     │
│ Input:  All runs from "minivess_evaluation" experiment              │
│ Output: DuckDB database + Parquet + figures + LaTeX tables         │
│         - N-way ANOVA (model × loss × calib, fold random effect)   │
│         - Pairwise: Wilcoxon + Cohen's d + Cliff's δ + VDA        │
│         - Bayesian: ROPE equivalence (baycomp)                     │
│         - Spec curve: all researcher degrees of freedom            │
│         - Rank stability: Kendall's tau (DSC vs clDice inversion)  │
│         - Calibration: Brier + O/E + IPA                           │
│         - TRIPOD+AI mapping                                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Inter-Flow Discovery Mechanism

**Every flow discovers its upstream via MLflow tags**, not filesystem paths:

```python
# Post-Training discovers Training:
find_upstream_safely(
    experiment_name="minivess_training",   # or debug variant
    upstream_flow="training-flow",          # tag: flow_name
)

# Analysis discovers Training + Post-Training:
EnsembleBuilder.discover_training_runs()    # queries by experiment_name
discover_post_training_models(
    experiment_name="minivess_post_training",
)

# Biostatistics discovers Analysis:
discover_source_runs(
    mlruns_dir=...,
    experiment_names=["minivess_evaluation"],
)
```

### 2.3 MLflow Experiment Names

| Experiment | Used By | Purpose |
|-----------|---------|---------|
| `minivess_training` | Train Flow | Training runs (24 debug / 72 production) |
| `minivess_post_training` | Post-Training Flow | SWA/calibration variants |
| `minivess_evaluation` | Analysis Flow | Ensemble evaluations, external test |
| `minivess_biostatistics` | Biostatistics Flow | Statistical analysis artifacts |

**Debug variants** use `resolve_experiment_name()` which appends a debug suffix
when `MINIVESS_DEBUG_SUFFIX` env var is set.

### 2.4 MLflow Tag Schema

Every run MUST have these tags for downstream discovery:

| Tag | Set By | Values |
|-----|--------|--------|
| `flow_name` | All flows | `"training-flow"`, `"post-training-flow"`, `"analysis-flow"`, `"biostatistics-flow"` |
| `model_family` | Train Flow | `dynunet`, `mambavesselnet`, `sam3_topolora`, `sam3_hybrid` |
| `loss_function` | Train Flow | `cbdice_cldice`, `dice_ce`, `dice_ce_cldice` |
| `fold_id` | Train Flow | `0`, `1`, `2` |
| `with_aux_calib` | Train Flow | `true`, `false` |
| `post_training_method` | Post-Training | `none`, `swa`, `multi_swa`, `calibrated` |
| `upstream_training_run_id` | Post-Training, Analysis | MLflow run_id of upstream |
| `ensemble_strategy` | Analysis | `per_loss_single_best`, etc. |

**Known tag mismatch (B3 blocker):** Training logs `loss_name` but ensemble builder
reads `loss_function`. Fallback chain exists in `builder.py:229-234`.

---

## Part 3: What Each Flow Actually Does

### 3.1 Post-Training Flow — Detailed Operations

**File:** `src/minivess/orchestration/flows/post_training_flow.py` (18.7K)

The Post-Training Flow applies model-agnostic weight-space techniques to trained
checkpoints. It does NOT retrain — no gradient computation, no loss functions.

**Plugin architecture:**
```
PostTrainingPlugin (ABC)
├── SWAPlugin              # Average checkpoints for better generalization
├── MultiSWAPlugin         # N independent SWA sub-models
├── ModelMergingPlugin     # Weight interpolation (linear, SLERP)
├── CalibrationPlugin      # Temperature scaling, isotonic regression
├── CRCConformalPlugin     # Conformalized Risk Control
└── ConSeCoPlugin          # Conservative-Selective FP control
```

**Execution order:**
1. Discover upstream training run checkpoint paths
2. Run weight-based plugins (parallel): SWA, Multi-SWA, Model Merging
3. Run data-dependent plugins (parallel): Calibration, CRC, ConseCo
4. Aggregate results
5. Log to MLflow

**Best-effort principle:** Plugin failure does NOT block downstream flows.
If SWA fails, calibration still runs. If all fail, the original checkpoint
is used by Analysis Flow.

**Key question RESOLVED:** Post-training creates runs in the SAME experiment
as training (not separate) so that Analysis Flow discovers all variants in
one query. Each post-training run is distinguished by the `post_training_method`
tag.

### 3.2 Analysis Flow — Detailed Operations

**File:** `src/minivess/orchestration/flows/analysis_flow.py` (66.4K — largest flow)

The Analysis Flow is the most complex flow. It takes ALL training/post-training
runs and produces ensemble evaluations.

**10-step pipeline:**

1. **Discover post-training models** (optional) — queries `minivess_post_training`
2. **Load training artifacts** — discovers runs via `EnsembleBuilder.discover_training_runs()`
3. **Build ensembles** — creates 4 ensemble strategies from discovered runs
4. **Log models to MLflow** — registers single + ensemble models as pyfunc
5. **Extract single models** — deduplicates by run_id
6. **Evaluate ALL models** — inference on MiniVess val + DeepVess test
7. **MLflow evaluate** — custom segmentation metrics
8. **Generate comparison** — Markdown + LaTeX comparison tables
9. **Register champion** — best model to MLflow Model Registry
10. **Tag champions** — writes champion tags to mlruns/ filesystem

**Ensemble inference:** `_EnsembleInferenceWrapper` averages member logits:
```python
def forward(self, x):
    outputs = [member(x) for member in self._members]
    return torch.stack(outputs).mean(dim=0)  # LOGIT-level averaging
```

**Per-volume metrics:** Essential for Biostatistics. Logged as:
`eval/{fold_id}/vol/{volume_id}/{metric_name}` — this is what enables
Wilcoxon signed-rank tests and specification curve analysis.

### 3.3 Biostatistics Flow — Current Implementation Status

**File:** `src/minivess/orchestration/flows/biostatistics_flow.py` (20.4K)

**Implemented (as of commit a570989 + this session):**
- SourceRun with model_family + with_aux_calib fields
- DuckDB schema with factorial columns
- N-way ANOVA (compute_factorial_anova) with K=1 fallback
- Pairwise comparisons (Wilcoxon + 3 effect sizes + 2-tier MCC)
- Bayesian signed-rank with ROPE (baycomp)
- Variance decomposition (Friedman + ICC)
- **NEW:** Specification curve analysis (Simonsohn et al., 2020)
- **NEW:** Rank stability (Kendall's tau, detects DSC vs clDice inversion)
- **NEW:** Calibration metrics (Brier, O/E, IPA)
- **NEW:** K=1 debug fallback (per-volume replication)
- **NEW:** TRIPOD+AI preregistration mapping
- Wired into biostatistics_flow.py as Prefect tasks

**All statistical parameters read from BiostatisticsConfig** — never hardcoded
(CLAUDE.md Rule #29, Issue #881).

---

## Part 4: Debug Run Status — Where We Are

### 4.1 First Pass Results (2026-03-20)

26 conditions launched on GCP L4 spot. Results:

| Model | Conditions | Succeeded | Failed | Root Cause |
|-------|-----------|-----------|--------|------------|
| dynunet | 6 | 6 | 0 | All OK |
| sam3_hybrid | 6 | 5 | 1 stuck | Stuck in spot queue |
| sam3_topolora | 6 | 0 | 6 | Glitch #9: LoRA on Conv2d |
| mambavesselnet | 6 | 0 | 6 | Glitch #10: mamba-ssm not compiled |
| sam3_vanilla | 1 | 0 | 1 | Glitch #12: max_epochs=0 rejected |
| vesselfm | 1 | 0 | 1 | Glitch #12: max_epochs=0 rejected |
| **Total** | **26** | **11** | **15** | |

**CRITICAL:** All 11 succeeded runs LOST checkpoints due to Glitch #8
(Cloud Run 32MB HTTP body limit on MLflow artifact upload).

### 4.2 Fixes Applied (This Session)

| Glitch | Fix | Status |
|--------|-----|--------|
| #8 (checkpoint 413) | SkyPilot YAML: file_mounts GCS + local MLflow URI | Fixed in code, needs cloud verification |
| #9 (LoRA Conv2d) | Skip Conv2d in `_apply_lora_to_encoder()` | Fixed, committed |
| #10 (mamba-ssm) | Docker rebuild with `INSTALL_MAMBA=1` | Fixed, image pushed |
| #12 (max_epochs=0) | Change `ge=1` to `ge=0` + zero-shot handling | Fixed, committed |

### 4.3 What Needs to Happen Next

**Immediate (before 2nd pass cloud run):**
1. Verify all 4 fixes pass staging tests (**DONE** — 5474 passed, 0 failed)
2. Verify all 4 fixes pass prod tests (**DONE** — 5785 passed, 0 failed)
3. Run 2nd pass: 15 conditions (14 failed + 1 probe)
4. Verify checkpoints persist (Glitch #8 fix)

**After 2nd pass succeeds (all 26/26):**
1. Run Post-Training Flow locally on downloaded DynUNet artifacts
2. Run Analysis Flow locally (ensemble building, evaluation)
3. Run Biostatistics Flow locally (N-way ANOVA on all conditions)
4. Verify complete 4-flow pipeline end-to-end

**After local validation succeeds:**
1. Run full production factorial (72 runs on GCP)
2. Run all 4 flows on production results
3. Generate paper figures and tables

---

## Part 5: Known Gaps and Blockers

### 5.1 Config Pipeline Gap (B1)

`train_flow.py` bypasses Hydra-zen. Uses argparse + 9-key dict instead of
`compose_experiment_config()`. `log_hydra_config()` is never called.

**Impact:** MLflow runs lack `config/resolved_config.yaml` artifact. Downstream
flows cannot discover the full config used for training.

**Status:** Documented in `script-consolidation.xml` Phase 0. Not yet fixed.
Workaround: factorial SkyPilot YAML passes all args explicitly.

### 5.2 Tag Mismatch (B3)

Training logs `loss_name` tag but ensemble builder reads `loss_function`.
Fallback chain exists but is fragile.

**Impact:** Ensemble builder may fail to match runs if both tag names are absent.

### 5.3 Cloud Artifact Sync (Issue #882)

No standardized mechanism to download MLflow artifacts from cloud to local disk.
`make dev-gpu-sync` exists for RunPod but not for GCP.

**Impact:** Local post-training/analysis/biostatistics flows cannot access cloud
training artifacts without manual download.

### 5.4 Post-Training Volume Mount (B7)

`post_training_out` volume missing from `docker-compose.flows.yml`.

**Impact:** Post-training artifacts lost on container exit.

---

## Part 6: Metalearning — Recurring Failure Patterns

### Pattern 1: Factorial Design Context Amnesia

**Occurrences:** 2026-03-19 (QA round 1), 2026-03-20 (Q2 in analysis XML)
**Description:** Claude asks questions about the factorial design that are already
answered in the KG, user's prompt, or config files.
**Prevention:** Before ANY factorial question, read `paper_factorial.yaml` and
`ensemble_strategy.yaml`. If the answer is there, DO NOT ASK.
**Doc:** `.claude/metalearning/2026-03-20-factorial-design-context-amnesia.md`

### Pattern 2: Hardcoded Parameters

**Occurrences:** 2026-03-20 (alpha=0.05, seed=42 throughout)
**Description:** Claude defaults to hardcoding "obvious" values instead of
reading from config.
**Prevention:** CLAUDE.md Rule #29. All statistical params from `BiostatisticsConfig`.
Guard test: `test_no_hardcoded_alpha.py`.
**Doc:** `.claude/metalearning/2026-03-20-hardcoded-significance-level-antipattern.md`

### Pattern 3: Debug Run Scope Reduction

**Occurrences:** 2026-03-19 (tried to skip aux_calibration)
**Description:** Claude proposes reducing debug scope beyond the 3 allowed reductions.
**Prevention:** CLAUDE.md Rule #27. Debug = full production minus ONLY epochs/data/folds.
**Doc:** `.claude/metalearning/2026-03-19-debug-run-is-full-production-no-shortcuts.md`

### Pattern 4: Metric Confusion

**Occurrences:** 2026-03-19 (suggested DSC as primary)
**Description:** Claude proposes DSC as primary metric when KG clearly states
clDice + MASD are co-primary and DSC is a FOIL.
**Prevention:** Read `primary_metrics.yaml` before any metric discussion.

### Pattern 5: Confusing Training Grid with Full Factorial

**Occurrences:** 6 times across 2026-03-17 through 2026-03-20
**Description:** Claude repeatedly states "the full factorial is 4×3×2 = 24 conditions"
when the ACTUAL full factorial from `pre-gcp-master-plan.xml` line 16 is
4×3×2×3×2×5 = 720 conditions. The 24 is only the TRAINING LAYER (Layer A).
**Prevention:** Read `pre-gcp-master-plan.xml` line 16 before any factorial discussion.
The full factorial multiplies training × post-training × ensemble layers.
Only Layer A (72 runs) costs money on GPU. Layers B+C run on local CPU (free).
**Doc:** `.claude/metalearning/2026-03-20-full-factorial-is-not-24-cells.md`

### Pattern 6: Shallow KG Reading

**Occurrences:** Every session
**Description:** Claude reads file names and summaries but doesn't trace
relationships between decisions. Reads `ensemble_strategy.yaml` but doesn't
connect it to the Analysis Flow's ensemble permutations.
**Prevention:** Read navigator → domains → decisions → TRACE EDGES between nodes.

---

## Part 7: Reference Tables

### 7.1 Complete Factor Table

**The factorial design has 6 EXPERIMENTAL factors and 7 ANALYTICAL factors.**
The 6 experimental factors define the Cartesian product of conditions.
The 7 analytical factors are researcher degrees of freedom for the specification curve.

**All factors are defined in the composable factorial YAML (`configs/factorial/*.yaml`).**

#### Experimental Factors (6 — define the 720-condition grid)

| # | Factor | Layer | Full Levels | Debug Levels | Compute | Config Section |
|---|--------|-------|-------------|-------------|---------|----------------|
| 1 | model_family | A (Training) | 4 | 4 | GPU (GCP) | `factors.training` |
| 2 | loss_name | A (Training) | 3 | 3 | GPU (GCP) | `factors.training` |
| 3 | aux_calibration | A (Training) | 2 | 2 | GPU (GCP) | `factors.training` |
| 4 | post_training_method | B (Post-Training) | 3 (none, swa, multi_swa) | 2 (none, swa) | CPU (local) | `factors.post_training` |
| 5 | recalibration | B (Post-Training) | 2 (none, temperature_scaling) | 2 | CPU (local) | `factors.post_training` |
| 6 | ensemble_strategy | C (Analysis) | 5 (none + 4 strategies) | 4 | CPU (local) | `factors.analysis` |

**Full product:** 4 × 3 × 2 × 3 × 2 × 5 = **720 conditions** × 3 folds = 2160 evaluations
**Debug product:** 4 × 3 × 2 × 2 × 2 × 4 = **384 conditions** × 1 fold = 384 evaluations

**Cost:** ONLY Layer A runs on GPU (72 cloud runs ~$65). Layers B and C run on LOCAL CPU (free).

#### Analytical Factors (7 — researcher degrees of freedom for specification curve)

| # | Factor | Layer | Levels | Config Source |
|---|--------|-------|--------|---------------|
| 7 | metric | D (Biostatistics) | 8+ | `BiostatisticsConfig.metrics` |
| 8 | aggregation | D (Biostatistics) | 2 | spec curve engine |
| 9 | mcc_method | D (Biostatistics) | 2 | `BiostatisticsConfig.co_primary_metrics` |
| 10 | alpha | D (Biostatistics) | configurable | `BiostatisticsConfig.alpha` |
| 11 | rope | D (Biostatistics) | per-metric | `BiostatisticsConfig.rope_values` |
| 12 | test_type | D (Biostatistics) | 4+ | spec curve engine |
| 13 | effect_size | D (Biostatistics) | 3 | `biostatistics_statistics.py` |

#### Additional Non-Factor Variables

| Variable | Role | Values |
|----------|------|--------|
| fold_id | Random effect (not a factor) | 0, 1, 2 (3-fold CV seed=42) |
| inference_strategy | Best-effort comparison | standard_patch (primary), fast_patch |

### 7.2 Dataset Table

| Dataset | Role | N | Modality | Use In Factorial |
|---------|------|---|----------|-----------------|
| MiniVess | Train/Val | 70 | Multiphoton | 3-fold CV, 47 train / 23 val per fold |
| DeepVess | External Test | 7 | Multiphoton | All models evaluated, separate analysis |
| VesselNN | Drift ONLY | 12 | Two-photon | NOT a test set, drift simulation only |
| TubeNet | EXCLUDED | - | Mixed | Wrong organ, never re-add |

### 7.3 Model VRAM Requirements

| Model | VRAM (GB) | Local (RTX 2070 8GB) | Cloud (L4 24GB) |
|-------|-----------|---------------------|-----------------|
| DynUNet | 3.5 | YES | YES |
| MambaVesselNet++ | 4-8 | MAYBE | YES |
| SAM3 Vanilla | 2.9 | YES | YES |
| SAM3 TopoLoRA | 22.7 | NO | YES |
| SAM3 Hybrid | 7.2 | MAYBE | YES |
| VesselFM | TBD | TBD | YES |

### 7.4 File Cross-Reference

| Topic | Authoritative File |
|-------|-------------------|
| Full factorial (ALL 6 factors) | `configs/factorial/paper_full.yaml` (TO CREATE — currently `configs/hpo/paper_factorial.yaml` has only Layer A) |
| Debug factorial (ALL 6 factors, fewer levels) | `configs/factorial/debug.yaml` (TO CREATE — currently `configs/experiment/debug_factorial.yaml` has only Layer A) |
| Full factorial definition | `docs/planning/pre-gcp-master-plan.xml` line 16 — `4×3×2×3×2×5=720` |
| Model lineup | `knowledge-graph/domains/models.yaml::paper_model_comparison` |
| Metrics | `knowledge-graph/decisions/L3-technology/primary_metrics.yaml` |
| Ensemble | `knowledge-graph/decisions/L2-architecture/ensemble_strategy.yaml` |
| Calibration | `knowledge-graph/decisions/L3-technology/calibration_method.yaml` |
| Biostatistics | `src/minivess/config/biostatistics_config.py` |
| Post-training | `src/minivess/config/post_training_config.py` |
| Splits | `configs/splits/3fold_seed42.json` |
| Docker base | `deployment/docker/Dockerfile.base` |
| SkyPilot YAML | `deployment/skypilot/train_factorial.yaml` |

---

## Part 8: Open Questions (Genuinely Ambiguous)

These are the ONLY questions that cannot be answered from the KG or existing docs:

1. **Post-training experiment naming:** Should post-training runs go into the SAME
   experiment as training (`minivess_training`) or a SEPARATE experiment
   (`minivess_post_training`)? Trade-off: unified discovery vs clean separation.
   **Recommendation:** Same experiment, distinguished by `post_training_method` tag.

2. **Per-volume uncertainty maps:** Should full aleatoric/epistemic maps (5D tensors,
   ~50MB per volume) be stored as MLflow artifacts, or only summary statistics?
   **Recommendation:** Summary stats only for debug; full maps for production.

3. **Per-volume soft predictions:** Should probability maps be stored for post-hoc
   threshold sweeps? Storage vs flexibility trade-off.
   **Recommendation:** Store for production runs; skip for debug.

---

*This document was synthesized from 50+ source files across the knowledge graph,
planning documents, metalearning docs, config files, and source code. Last updated
2026-03-20.*
