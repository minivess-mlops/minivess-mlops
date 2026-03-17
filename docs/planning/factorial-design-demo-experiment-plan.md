# Factorial Design Demo Experiment Plan

**Created**: 2026-03-17
**Purpose**: Minimum Scientific Experiment for MLOps Platform Demonstration

## Original User Prompt (verbatim)

> So you could update the kg also as a potential "Minimum Scientific Experiment" to do a "Minimum Viable Demonstration" of the MLOps platform as comparing all the different models with different losses so you would get a factorial design of n models (with trainable parameters) x m losses (and compound losses). So that there would be something for the Biostatistics Model to do. To not get ridiculous with the loss space, we could choose 3 key single and compound losses that we have been using (see our initial pilot here with dynUnet: /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/loss-and-metrics-double-check-report.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/topology-loss-plan.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/compound-loss-double-check.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/compound-loss-implementation-plan.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/debug-training-all-losses-plan.md /home/petteri/Dropbox/github-personal/minivess-mlops/docs/results/dynunet_half_width_v1_report.md
> /home/petteri/Dropbox/github-personal/minivess-mlops/docs/results/dynunet_all_losses_debug_report.md Save my prompt verbatim to /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/factorial-design-demo-experiment-plan.md and see also my previous project for a similar design /home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/foundationPLR and /home/petteri/Dropbox/github-personal/foundation-PLR/foundation_PLR (don't need to be doing the exact same tests, do an open-ended multi-hypothesis decision matrix for pros and cons of different biostatistics needed for the Biostatistic Flow for this minimum experiment! Again, rather ask too many questions than too little when planning this approach before continuing with the infrastructure work as this aligns then with our model testing on live cloud infrastructure as well. Create a new branch feat/biostatistics-demo-experiment on top of the current plan and let's create the research report and an actionable plan for this factorial design experiment that we could run for the manuscript in /home/petteri/Dropbox/github-personal/sci-llm-writer/manuscripts/vasculature-mlops and remember to update and improve th kg with this biostatistics and experiment info in addition to the model info and the losses!

---

## Q&A Session Log

### Round 1 (Experiment Design)

**Q1: Factorial scope?**
- **Answer**: 4 models × 3 losses = 12 conditions × 3 folds = 36 runs + 2 zero-shot baselines
- Models: DynUNet (done), MambaVesselNet++, SAM3 TopoLoRA, SAM3 Hybrid
- Losses: dice_ce, cbdice_cldice, dice_ce_cldice
- Zero-shot: SAM3 Vanilla, VesselFM (external data only)
- New runs needed: 27 (9 conditions × 3 folds). DynUNet's 12 already done.

**Q2: Epochs per run?**
- **Answer**: 50 epochs for all models (fair comparison)

**Q3: Primary statistical question?**
- **Answer**: Variance decomposition (η² for model vs loss vs interaction)
- "Does loss function matter more than architecture?"

**Q4: Budget?**
- **Answer**: $150+ (thorough). Full factorial + ablations + buffer for preempted runs.

### Round 2 (Biostatistics + Metrics)

**Q5: Metrics for ANOVA?**
- **Clarification by user**: MetricsReloaded report (`docs/MetricsReloaded.html`) is the ground truth.
  clDice and MASD are the TRUSTED metrics. DSC (Dice) is included as a FOIL to show how
  wrong metric choice leads to wrong model rankings. Other 5 metrics are optional/supplementary.
- **Answer**: 3 metrics — clDice (trusted, topology), MASD (trusted, surface), DSC (foil, overlap)
- **KG gap found**: `primary_metrics.yaml` did not link to MetricsReloaded report or distinguish
  trusted vs foil metrics. Fixed with updated decision node + metalearning doc.

**Q6: Two statistical analyses?**
- **Answer**: Yes — (A) Factorial ANOVA for 4×3 design + (B) 6-model pairwise for all models
- Analysis A: variance decomposition (η² for model/loss/interaction) on clDice, MASD, DSC
- Analysis B: Wilcoxon + Bayesian signed-rank + Holm correction + Critical Difference diagram
  for all 6 models' best configuration

**Q7: DynUNet epoch fairness?**
- **Answer**: Extract epoch-50 metrics from existing MLflow runs (zero cost, fair comparison)

**Q8: Dice-as-foil narrative?**
- **Answer**: Yes — explicitly frame as "Dice would rank models WRONG, topology metrics matter"
  Show ranking discrepancy table in paper. Strong scientific argument for MetricsReloaded.

---

# Factorial Design Minimum Scientific Experiment -- Research Report & Plan

**Version**: 1.0
**Date**: 2026-03-17
**Branch**: `test/mambavesselnet` (merge target: `main` via `feat/biostatistics-demo-experiment`)
**Manuscript**: NEUROVEX (Nature Protocols target)

---

## 1. Experiment Design Overview

### 1.1 Scientific Question

> "In 3D multiphoton vascular segmentation, how much of model performance variance is
> explained by **architecture choice** vs **loss function choice** vs their **interaction**?"

This question directly serves the NEUROVEX manuscript's R3a (loss ablation), R3b (multi-model
comparison), and R3c (foundation model external evaluation) results sections. It also provides
the Biostatistics Flow (`biostatistics_flow.py`) with its first real-world multi-factor dataset,
validating the statistical engine end-to-end.

### 1.2 Factorial Matrix

**Full factorial design**: 4 trainable models x 3 losses x 3 folds = **36 runs**
plus 2 zero-shot baselines = **38 total evaluations**.

| # | Model | Family | Loss | Folds | Training | Status |
|---|-------|--------|------|-------|----------|--------|
| 1-9 | DynUNet | CNN baseline | dice_ce, cbdice_cldice, dice_ce_cldice | 3 | Full (100 epochs; extract epoch-50) | **COMPLETE** (only 3 of 4 losses done; dice_ce_cldice done) |
| 10-18 | MambaVesselNet++ | SSM hybrid | dice_ce, cbdice_cldice, dice_ce_cldice | 3 | Full (50 epochs) | CODE_COMPLETE, GPU_PENDING |
| 19-27 | SAM3 TopoLoRA | FM + LoRA | dice_ce, cbdice_cldice, dice_ce_cldice | 3 | LoRA fine-tune (50 epochs) | GPU_PENDING |
| 28-36 | SAM3 Hybrid | FM + DynUNet fusion | dice_ce, cbdice_cldice, dice_ce_cldice | 3 | Hybrid train (50 epochs) | PARTIALLY_VALIDATED |
| 37 | SAM3 Vanilla | FM frozen | N/A (zero-shot) | 3 | Zero-shot eval only | GPU_PENDING |
| 38 | VesselFM | FM pretrained | N/A | 3 | Zero-shot + fine-tuned (DeepVess/TubeNet ONLY) | GPU_PENDING |

**DynUNet existing runs**: The `dynunet_loss_variation_v2` experiment ran 4 losses x 3 folds x
100 epochs. Three of the four losses (`dice_ce`, `cbdice_cldice`, `cbdice`) match our factorial
design. For fair comparison at epoch 50, we extract intermediate checkpoint metrics from the
existing MLflow runs. The `cbdice` loss is excluded from the factorial design (replaced by
`dice_ce_cldice` compound loss) but remains available for supplementary analysis.

**New runs needed**: 27 (9 conditions x 3 folds for Mamba++, SAM3 TopoLoRA, SAM3 Hybrid) plus
zero-shot baselines.

### 1.3 Loss Functions

Three losses chosen to span the single-compound-topology continuum:

| Loss ID | Components | Topology Awareness | Evidence |
|---------|------------|-------------------|----------|
| `dice_ce` | Dice + CrossEntropy | None (overlap only) | Standard MONAI baseline |
| `cbdice_cldice` | CenterlineBoundary Dice + centerline Dice | Strong (soft skeleton) | Winner from `dynunet_loss_variation_v2` ([Shit et al. (2021). "clDice." *CVPR*.](https://arxiv.org/abs/2003.07311)) |
| `dice_ce_cldice` | Dice + CE + clDice | Moderate (additive topology term) | Three-term compound, tested in DynUNet experiment |

The design tests whether topology-aware losses (cbdice_cldice, dice_ce_cldice) consistently
outperform topology-unaware loss (dice_ce) **across architectures**, or whether the benefit
is architecture-dependent (significant interaction term).

### 1.4 Metrics (MetricsReloaded Ground Truth)

Metric selection follows the MetricsReloaded questionnaire output (`docs/MetricsReloaded.html`)
as defined in `knowledge-graph/decisions/L3-technology/primary_metrics.yaml`:

| Metric | Role | What It Measures | Direction | ROPE |
|--------|------|-----------------|-----------|------|
| **clDice** | TRUSTED | Vessel topology/connectivity preservation | Higher = better | +/-0.01 |
| **MASD** | TRUSTED | Mean average surface distance (boundary accuracy) | Lower = better | +/-0.5 |
| **DSC** | FOIL | Volumetric overlap (Dice coefficient) | Higher = better | +/-0.01 |

**Why DSC is a foil**: [Maier-Hein et al. (2024). "Metrics Reloaded." *Nature Methods*.](https://doi.org/10.1038/s41592-023-02151-z) demonstrated that Dice is volume-biased for tubular structures: a model can achieve high Dice by segmenting thick vessel trunks while missing all thin branches. Including Dice alongside clDice explicitly demonstrates how wrong metric choice leads to wrong model rankings (Analysis C).

---

## 2. Statistical Analysis Plan

### 2.1 Analysis A: Factorial ANOVA (4 models x 3 losses)

**Design**: Two-way factorial ANOVA with repeated measures (fold as random effect).

- **Response variables**: clDice, MASD, DSC (3 separate ANOVAs)
- **Fixed factors**: Model (4 levels: DynUNet, Mamba++, SAM3 TopoLoRA, SAM3 Hybrid), Loss (3 levels: dice_ce, cbdice_cldice, dice_ce_cldice)
- **Random factor**: Fold (3 levels, seed=42)
- **Observations per cell**: 23 per-volume scores (from 23 validation volumes per fold)
- **Total N**: 4 x 3 x 3 x 23 = 828 per-volume observations per metric

**Effect sizes**:
- eta-squared (partial) for each factor and interaction
- omega-squared (bias-corrected) as recommended by [Lakens (2013). "Calculating and reporting effect sizes." *Frontiers in Psychology*.](https://doi.org/10.3389/fpsyg.2013.00863)

**Post-hoc**: Tukey HSD for pairwise comparisons when main effects are significant.

**Interaction term**: Model x Loss interaction tests whether the best loss function depends on the architecture. A significant interaction means "the best loss for DynUNet is not necessarily the best loss for SAM3."

**Power consideration**: With 23 volumes per fold x 3 folds = 69 observations per cell, the design has adequate power for detecting medium effects (Cohen's f >= 0.25) at alpha=0.05. Small effects (f < 0.10) may not be detectable -- this is a known limitation documented in `knowledge-graph/manuscript/limitations.yaml`.

### 2.2 Analysis B: 6-Model Pairwise Comparison

After the factorial analysis, all 6 models are compared on their **best configuration**:

| Model | Configuration for Pairwise |
|-------|---------------------------|
| DynUNet | Best loss at epoch 50 (extracted from existing runs) |
| MambaVesselNet++ | Best loss at epoch 50 |
| SAM3 TopoLoRA | Best loss at epoch 50 |
| SAM3 Hybrid | Best loss at epoch 50 |
| SAM3 Vanilla | Zero-shot (frozen encoder + lightweight decoder), no training |
| VesselFM | Zero-shot + fine-tuned, on DeepVess/TubeNet only (data leakage: C9) |

**Statistical tests**:
1. **Wilcoxon signed-rank test** -- non-parametric paired comparison (per-volume scores paired across models)
2. **Holm-Bonferroni correction** -- for primary metric (clDice), controlling FWER
3. **Benjamini-Hochberg FDR** -- for secondary metrics (MASD, DSC)
4. **Bayesian signed-rank with ROPE** -- P(A>B), P(rope), P(B>A) via `baycomp.SignedRankTest`
   - ROPE widths: clDice +/-0.01, MASD +/-0.5, DSC +/-0.01
5. **Critical Difference diagram** -- Nemenyi post-hoc on average ranks ([Demsar (2006). "Statistical Comparisons of Classifiers over Multiple Data Sets." *JMLR*.](https://jmlr.org/papers/v7/demsar06a.html))

All five methods are already implemented in:
- `src/minivess/pipeline/biostatistics_statistics.py` (Wilcoxon, Bayesian signed-rank, Friedman+Nemenyi, ICC)
- `src/minivess/pipeline/biostatistics_rankings.py` (Critical Difference computation)
- `src/minivess/pipeline/comparison.py` (Cohen's d, Cliff's delta, Holm-Bonferroni)

### 2.3 Analysis C: Dice-as-Foil Narrative

This analysis explicitly frames DSC as a misleading metric for vascular segmentation:

1. **Rank all 6 models** by DSC, clDice, and MASD independently
2. **Identify rank discrepancies**: models ranked #1 by Dice but #3+ by clDice
3. **Quantify misleadingness**: Spearman rank correlation between Dice rankings and clDice rankings
4. **Expected finding** (from DynUNet pilot): `dice_ce` achieves highest DSC (0.824) but worst topology (clDice=0.832), while `cbdice_cldice` achieves best topology (clDice=0.906) but lower DSC (0.772). This 8.9% clDice gap is invisible to Dice-only evaluation.
5. **Frame in paper** as a MetricsReloaded justification: "Had we used only Dice, we would have ranked [model X] as best, missing the topology-preserving [model Y]."

This directly supports manuscript claim C2 and validates the MetricsReloaded questionnaire approach.

---

## 3. Multi-Hypothesis Decision Matrix: Biostatistics Methods

| # | Method | Pros | Cons | Implemented? | Foundation-PLR Precedent | Recommendation |
|---|--------|------|------|-------------|--------------------------|----------------|
| 1 | **Two-way ANOVA** (parametric) | Standard factorial analysis; partitions variance into Model, Loss, Model x Loss; widely understood; exact eta-squared | Assumes normality and homoscedasticity; per-volume scores may violate normality for small structures | NOT yet (Friedman is the current non-parametric alt) | Not used in foundation-PLR (different design) | **ADD** -- primary analysis for factorial design. Use `scipy.stats.f_oneway` + `pingouin.anova` for two-way with interaction. Report alongside Friedman for robustness. |
| 2 | **Friedman test** (non-parametric rank-based) | No distributional assumptions; robust to outliers; already implemented with Nemenyi post-hoc | Ranks lose magnitude information; lower power than ANOVA with normal data; no interaction term in standard formulation | **YES** (`biostatistics_statistics.py::compute_variance_decomposition`) | Not used | **KEEP** as robustness check for Analysis B. Use alongside parametric ANOVA. |
| 3 | **Mixed-effects model** (fold as random, model+loss as fixed) | Proper handling of fold as random effect; handles unbalanced designs; accounts for fold-level variance | Requires `lme4`-equivalent (statsmodels `MixedLM` or `pingouin`); more complex to interpret; convergence issues with small K | NOT yet | Not used | **ADD** -- most statistically correct approach for this design. Fold should be random, not fixed. Use `statsmodels.MixedLM` or `pingouin.mixed_anova`. |
| 4 | **Bayesian ANOVA** (posterior probability of effects) | Full posterior distributions; no p-value dichotomy; quantifies uncertainty on effect sizes; Bayes factors for model comparison | Requires BAMBI or PyMC; computationally expensive; prior specification decisions | NOT yet | Not used | **DEFER** to Phase 2. Valuable for manuscript but not blocking. Can run post-hoc on the same data. |
| 5 | **Bayesian signed-rank with ROPE** | Directly answers "is the difference practically meaningful?"; three probabilities (A>B, rope, B>A); no correction needed | Requires `baycomp`; pairwise only (no factorial decomposition); ROPE width is a researcher degree of freedom | **YES** (`biostatistics_statistics.py::compute_bayesian_comparisons`) | Not used directly | **KEEP** -- primary for Analysis B pairwise comparisons. Already battle-tested. |
| 6 | **Bootstrap ANOVA** (permutation-based) | No distributional assumptions; exact p-values; handles non-normal data gracefully | Computationally expensive (10K+ permutations); not standard in MICCAI literature; interaction testing requires custom permutation scheme | NOT yet | Not used | **DEFER** -- add as supplementary robustness check if reviewers question normality assumption. Low priority vs mixed-effects model. |
| 7 | **Specification curve analysis** | Shows how results change across all analytical configurations; visual impact; addresses "researcher degrees of freedom" | Complex to implement; requires defining the "specification space"; unfamiliar to most medical imaging reviewers | NOT yet | **YES** -- used in foundation-PLR as primary visualization | **ADD** -- high-impact visualization. Plot all 14 model x loss conditions (12 factorial + 2 zero-shot) ranked by clDice with 95% CI bands + binary indicator matrix for model/loss factors below. |
| 8 | **Critical Difference diagrams** ([Demsar (2006)](https://jmlr.org/papers/v7/demsar06a.html)) | Visual summary of rank-based pairwise differences; widely used in ML benchmarks; Nemenyi post-hoc built in | Rank-based (loses magnitude); assumes comparable sample sizes; q-values tabulated only up to K=10 | **YES** (`biostatistics_rankings.py::_critical_difference`) | Not used | **KEEP** -- standard for Analysis B (6-model comparison). Nemenyi matrix already implemented in `biostatistics_statistics.py`. |
| 9 | **Variance decomposition with ICC** | Quantifies consistency across folds; complementary to ANOVA; directly answers "how reproducible are model rankings?" | ICC interpretation depends on ICC type; small K=3 limits CI precision; power caveat always applies | **YES** (`biostatistics_statistics.py::_compute_icc`, using `pingouin.intraclass_corr`) | Not used | **KEEP** -- important for reproducibility narrative (manuscript claim C1). ICC(2,1) with 95% CI. Report power caveat for K=3 explicitly. |
| 10 | **Effect size analysis** (Cohen's d, Cliff's delta, VDA) | Practical significance beyond p-values; Cliff's delta is non-parametric; VDA gives win probability | Effect sizes alone don't test hypotheses; Cohen's d assumes normality; interpretation thresholds are field-dependent | **YES** (`comparison.py::cohens_d`, `cliffs_delta`, `vargha_delaney_a`) | Not used | **KEEP** -- essential complement to p-values. Report both Cohen's d (parametric) and Cliff's delta (non-parametric) for every pairwise comparison. Follow [Vargha & Delaney (2000). "A Critique and Improvement of the CL Common Language Effect Size Statistic." *JEBS*.](https://doi.org/10.3102/10769986025002101) |

### Summary of Actions

| Priority | Method | Action | Implementation Target |
|----------|--------|--------|----------------------|
| **P0** (blocking) | Two-way ANOVA | Add to `biostatistics_statistics.py` | Before Phase 4 |
| **P0** (blocking) | Mixed-effects model | Add to `biostatistics_statistics.py` | Before Phase 4 |
| **P0** (blocking) | Specification curve | Add to `biostatistics_figures.py` | Before Phase 4 |
| **P1** (important) | Interaction plot | Add to `biostatistics_figures.py` | Before Phase 4 |
| **P1** (important) | Rank discrepancy table | Add to `biostatistics_tables.py` | Before Phase 4 |
| **P2** (nice to have) | Bayesian ANOVA | New module or extend existing | Post Phase 4 |
| **P2** (nice to have) | Bootstrap ANOVA | New function in statistics module | Post Phase 4 |
| Keep | Friedman, Bayesian signed-rank, CD, ICC, Effect sizes | Already implemented | N/A |

---

## 4. Visualization Plan

Six publication-quality figures following [Maier-Hein et al. (2024)](https://doi.org/10.1038/s41592-023-02151-z) and specification curve precedent:

### (A) Specification Curve -- All 14 Conditions

- **X-axis**: 14 conditions (12 factorial + 2 zero-shot) ranked by mean clDice (descending)
- **Y-axis top panel**: clDice point estimate with 95% bootstrap CI error bars
- **Y-axis bottom panel**: Binary indicator matrix (rows = model, loss; columns = conditions)
- **Highlight**: Trusted metric (clDice); foil metric (DSC) as gray overlay
- **Purpose**: Shows at a glance which model+loss combinations are "in the running" and which factors drive the ranking
- **Implementation**: New function `_generate_specification_curve()` in `biostatistics_figures.py`

### (B) Critical Difference Diagram -- 6 Models

- **Layout**: Horizontal axis = average rank; models connected if difference < CD
- **Source**: Nemenyi post-hoc from `compute_variance_decomposition()`
- **K=6 models**: DynUNet, Mamba++, SAM3 TopoLoRA, SAM3 Hybrid, SAM3 Vanilla, VesselFM
- **Purpose**: Standard ML benchmark visualization for Analysis B
- **Implementation**: New function using `scikit-posthocs` or custom rendering in `biostatistics_figures.py`

### (C) Interaction Plot -- Model x Loss

- **Layout**: 2-panel figure (left: clDice, right: MASD)
- **X-axis**: Loss function (3 levels)
- **Y-axis**: Mean metric value with 95% CI
- **Lines**: One per model (4 lines), color-coded with Okabe-Ito palette
- **Purpose**: Visually shows whether lines are parallel (no interaction) or crossing (significant interaction)
- **Implementation**: New function `_generate_interaction_plot()` in `biostatistics_figures.py`

### (D) Variance Decomposition Lollipop -- eta-squared per Factor

- **Layout**: Horizontal lollipop chart
- **Y-axis**: Factors (Model, Loss, Model x Loss, Fold, Residual)
- **X-axis**: Partial eta-squared (0 to 1)
- **Separate panels**: One per metric (clDice, MASD, DSC)
- **Purpose**: Directly answers "what explains the variance?" -- the core scientific question
- **Implementation**: New function; requires eta-squared from two-way ANOVA output

### (E) Rank Discrepancy Table -- Dice vs clDice

- **Format**: LaTeX table + matplotlib heatmap
- **Columns**: Model, Best Config, DSC Rank, clDice Rank, MASD Rank, Rank Delta (DSC - clDice)
- **Highlight**: Red cells where DSC rank and clDice rank disagree by >= 2 positions
- **Purpose**: The "smoking gun" for the Dice-as-foil narrative (Analysis C)
- **Implementation**: Extend `_generate_ranking_table()` in `biostatistics_tables.py`

### (F) Box Plots -- Per-Model Distributions

- **Layout**: Standard MICCAI-style grouped box + strip plots
- **X-axis**: Models (6), grouped by metric
- **Y-axis**: Per-volume scores
- **Overlay**: Individual volume scores as jittered points (strip)
- **Purpose**: Shows distribution shape, outliers, overlap between models
- **Implementation**: Already partially implemented as `_generate_distribution_plot()` in `biostatistics_figures.py`; extend to group by model rather than condition

---

## 5. Execution Plan

### Phase 0: Extract DynUNet Epoch-50 Metrics (local, zero cost)

**Goal**: Extract intermediate metrics from existing MLflow runs at training step 50.

**Steps**:
1. Query `mlruns/` for all DynUNet runs from `dynunet_loss_variation_v2` experiment
2. For each run x fold x loss, read the MLflow metric history files
3. Extract `val_cldice`, `val_masd`, `val_dice` at `step=50`
4. If step-50 metrics are not logged per-step (only final epoch), re-evaluate the epoch-50 checkpoint on the validation fold using the evaluation pipeline
5. Store extracted metrics in a standardized format for the biostatistics flow

**Deliverable**: `outputs/factorial/dynunet_epoch50_metrics.json` with per-volume scores for 3 losses x 3 folds

**Risk**: MLflow may not have logged per-step metrics at step 50. If only final-epoch metrics exist, we either (a) accept epoch 100 vs epoch 50 mismatch, or (b) re-run DynUNet for 50 epochs (low cost, ~2 hours local).

### Phase 1: 1-Fold Pilot ($10-15)

**Goal**: Validate that all 9 new conditions (3 models x 3 losses) train successfully on fold-0 before committing to the full 3-fold run.

**Steps**:
1. Launch fold-0 for MambaVesselNet++ x {dice_ce, cbdice_cldice, dice_ce_cldice}
2. Launch fold-0 for SAM3 Hybrid x {dice_ce, cbdice_cldice, dice_ce_cldice}
3. Launch fold-0 for SAM3 TopoLoRA x {dice_ce, cbdice_cldice, dice_ce_cldice}
4. Verify metrics are finite and in reasonable ranges (clDice > 0.01, MASD < 100)
5. Check VRAM usage -- SAM3 TopoLoRA needs ~16 GB (L4 has 24 GB, should fit)

**Compute**: GCP L4 spot ($0.22/hr approx, europe-north1)
**Estimated time**: ~2 hours per run x 9 runs = ~18 GPU-hours
**Estimated cost**: ~$4 (L4 spot rate) + buffer for preemptions = ~$10-15

**Go/No-Go**: If any condition produces NaN metrics, debug before Phase 2.

### Phase 2: Full 3-Fold Factorial ($30-50)

**Goal**: Complete the remaining 2 folds for all 9 conditions (fold-1, fold-2).

**Steps**:
1. Run fold-1 and fold-2 for all 9 conditions (18 total runs)
2. Sequential by model family to share cached weights:
   - MambaVesselNet++ (6 runs: 3 losses x 2 folds)
   - SAM3 Hybrid (6 runs: 3 losses x 2 folds)
   - SAM3 TopoLoRA (6 runs: 3 losses x 2 folds)
3. Use `sky jobs launch` with spot recovery for automatic preemption handling
4. Tag all runs with `experiment_name=factorial_design_v1` in MLflow

**Compute**: 18 runs x ~2 hrs = ~36 GPU-hours
**Estimated cost**: ~$8 (L4 spot) + 50% buffer for preemptions = ~$12

### Phase 3: Zero-Shot Baselines ($5-10)

**Goal**: Evaluate the two zero-shot/pretrained baselines.

**Steps**:
1. **SAM3 Vanilla**: Zero-shot inference on MiniVess validation folds (3 runs)
   - Frozen SAM3 ViT-32L encoder + lightweight decoder
   - Expected DSC: 0.05-0.55 (large domain gap: natural images -> 2PM microvasculature)
2. **VesselFM**: Zero-shot + fine-tuned on DeepVess/TubeNet (NOT MiniVess -- data leakage C9)
   - Zero-shot evaluation on DeepVess (Cornell eCommons) and TubeNet (UCL Figshare)
   - Optional: fine-tune on DeepVess, evaluate on TubeNet (or vice versa)

**Note**: VesselFM results are NOT directly comparable to the factorial design (different dataset). They appear in R3c (external evaluation), not R3b (MiniVess comparison). The 6-model pairwise comparison (Analysis B) uses each model's best result on its respective valid dataset.

**Compute**: ~10 GPU-hours
**Estimated cost**: ~$2-5

### Phase 4: Biostatistics Flow ($0, local)

**Goal**: Run the full biostatistics pipeline on all 38 evaluations.

**Steps**:
1. Sync all MLflow runs to local `mlruns/` directory
2. Configure biostatistics flow for the factorial experiment:
   ```yaml
   experiment_names: [factorial_design_v1, dynunet_loss_variation_v2]
   metrics: [cldice, masd, dsc]
   primary_metric: cldice
   rope_values: {cldice: 0.01, masd: 0.5, dsc: 0.01}
   min_folds_per_condition: 3
   min_conditions: 12  # 4 models x 3 losses
   ```
3. Run via Docker: `docker compose -f deployment/docker-compose.flows.yml run biostatistics`
4. Generated artifacts:
   - ANOVA tables (eta-squared, omega-squared per factor)
   - Specification curve figure (14 conditions ranked by clDice)
   - Critical Difference diagram (6 models)
   - Interaction plots (Model x Loss for clDice, MASD)
   - Variance decomposition lollipop chart
   - Rank discrepancy table (Dice vs clDice rankings)
   - Box/strip distribution plots
   - LaTeX tables (pairwise comparisons, effect sizes, rankings)
   - DuckDB analytics database
   - JSON sidecar files for all figures

**Compute**: CPU only (Tier B Docker image: `minivess-base-cpu`)
**Estimated cost**: $0

### Phase 5: Manuscript Integration ($0, local)

**Goal**: Populate the manuscript with real experimental numbers.

**Steps**:
1. Update `docs/manuscript/latent-methods-results/results/results-03-models.tex`:
   - R3a: DynUNet loss ablation (already has data; update to epoch-50 fair comparison)
   - R3b: Full 4-model x 3-loss factorial + 6-model pairwise comparison
   - R3c: VesselFM external evaluation (DeepVess/TubeNet)
2. Export specification curve and CD diagram as PDF/SVG for LaTeX inclusion
3. Generate the "Dice-as-foil" narrative paragraph with exact rank discrepancy numbers
4. Update `knowledge-graph/manuscript/results.yaml` status: R3b BLOCKED -> data_available
5. Sync KG snapshot to `sci-llm-writer/manuscripts/vasculature-mlops/kg-snapshot.yaml`

---

## 6. Cost Estimate

| Phase | Runs | GPU Hours (est.) | Cost (L4 spot ~$0.22/hr) |
|-------|------|-----------------|--------------------------|
| Phase 0 (local) | 0 (MLflow query) | 0 | $0 |
| Phase 1 (1-fold pilot) | 9 | ~18 | ~$4 |
| Phase 2 (remaining 2 folds) | 18 | ~36 | ~$8 |
| Phase 3 (zero-shot baselines) | 2-6 | ~10 | ~$2 |
| Buffer (spot preemption retries) | -- | ~20 | ~$4 |
| **Total** | **29-33** | **~84** | **~$18** |

**Worst case** (if SAM3 TopoLoRA needs A100 due to VRAM): $40-50 total.

**Notes**:
- SAM3 TopoLoRA VRAM estimate: ~16 GB (L4 has 24 GB, should fit with AMP). If not, fall back to A100 40 GB at ~$0.60-1.20/hr spot.
- MambaVesselNet++ VRAM: TBD from profiling, estimated 4-8 GB based on DynUNet baseline (~3.5 GB) + Mamba overhead.
- All runs use BF16 mixed precision (L4 Ada Lovelace supports BF16; T4 Turing is banned per CLAUDE.md).
- Well under the $150+ budget authorized in Q4.

---

## 7. SkyPilot YAML Configurations Needed

### 7.1 Factorial Experiment Launcher

A parameterized SkyPilot YAML that accepts `MODEL_FAMILY`, `LOSS_NAME`, and `FOLD_ID` as env vars:

**File**: `deployment/skypilot/train_factorial.yaml`

Based on the existing `deployment/skypilot/train_production.yaml` template, with these modifications:
- `envs.MODEL_FAMILY`: one of {dynunet, mambavesselnet, sam3_topolora, sam3_hybrid}
- `envs.LOSS_NAME`: one of {dice_ce, cbdice_cldice, dice_ce_cldice}
- `envs.FOLD_ID`: one of {0, 1, 2}
- `envs.MAX_EPOCHS`: 50 (fixed for fair comparison)
- `envs.EXPERIMENT`: `factorial_design_v1`
- MLflow experiment name: `factorial_design_v1_{MODEL_FAMILY}_{LOSS_NAME}_fold{FOLD_ID}`

**Launch commands** (example):
```bash
# Single run
sky jobs launch deployment/skypilot/train_factorial.yaml \
  --env MODEL_FAMILY=mambavesselnet --env LOSS_NAME=cbdice_cldice --env FOLD_ID=0 \
  --env-file .env -y

# Batch launch (shell loop, sequential by model for weight caching)
for model in mambavesselnet sam3_hybrid sam3_topolora; do
  for loss in dice_ce cbdice_cldice dice_ce_cldice; do
    for fold in 0 1 2; do
      sky jobs launch deployment/skypilot/train_factorial.yaml \
        --env MODEL_FAMILY=$model --env LOSS_NAME=$loss --env FOLD_ID=$fold \
        --env-file .env -y
    done
  done
done
```

### 7.2 Zero-Shot Evaluation YAML

**File**: `deployment/skypilot/eval_zeroshot.yaml`

For SAM3 Vanilla and VesselFM zero-shot evaluation:
- No training loop -- evaluation only
- SAM3 Vanilla: iterate over 3 folds on MiniVess
- VesselFM: iterate over DeepVess and TubeNet datasets

### 7.3 Resource Specifications

| Model | GPU Requirement | SkyPilot `accelerators` |
|-------|----------------|------------------------|
| DynUNet | 4 GB VRAM | L4:1 (or RTX4090:1 on RunPod) |
| MambaVesselNet++ | TBD (~4-8 GB est.) | L4:1 |
| SAM3 Hybrid | ~6 GB VRAM | L4:1 |
| SAM3 TopoLoRA | ~16 GB VRAM | L4:1 (24 GB available) |
| SAM3 Vanilla | ~3 GB VRAM | L4:1 |
| VesselFM | TBD | L4:1 (fallback: A100:1) |

All YAML configs use `image_id: docker:ghcr.io/petteriteikari/minivess-base:latest` (bare VM is banned).

---

## 8. KG Updates Required

### 8.1 New Decision Node

**File**: `knowledge-graph/decisions/L3-technology/factorial_experiment_design.yaml`

```yaml
decision_id: factorial_experiment_design
title: "Factorial Experiment Design for Model x Loss Comparison"
level: L3
status: planned
description: >
  4 models x 3 losses x 3 folds = 36 factorial runs + 2 zero-shot baselines.
  Primary analysis: Two-way ANOVA for variance decomposition.
  Secondary: 6-model pairwise comparison (Bayesian signed-rank + CD diagram).
  Dice-as-foil narrative demonstrates MetricsReloaded justification.
conditional_on:
  - {parent_decision_id: paper_model_comparison, strength: strong}
  - {parent_decision_id: loss_function, strength: strong}
  - {parent_decision_id: primary_metrics, strength: strong}
```

### 8.2 Existing Files to Update

| File | Update Needed |
|------|--------------|
| `knowledge-graph/decisions/L3-technology/paper_model_comparison.yaml` | Add `factorial_experiment` reference field; update R3b blocking status to reference factorial plan |
| `knowledge-graph/decisions/L3-technology/primary_metrics.yaml` | Already updated with trusted/foil distinction -- no changes needed |
| `knowledge-graph/domains/training.yaml` | Add `factorial_experiment_design` under `decisions` section; add planning doc reference |
| `knowledge-graph/manuscript/results.yaml` | Update R3b `blocked_by` to reference factorial plan; add Phase timeline |
| `knowledge-graph/manuscript/methods.yaml` | Update M8 (Evaluation) key_points to reference factorial ANOVA design |
| `knowledge-graph/domains/manuscript.yaml` | Update `open_issues` to reference factorial experiment timeline |
| `knowledge-graph/experiments/dynunet_loss_variation_v2.yaml` | Add `factorial_reference` field pointing to new experiment |
| `knowledge-graph/bibliography.yaml` | Add citations: Lakens (2013), Demsar (2006) if not present, Simmons (2011) specification curve |

### 8.3 New Experiment YAML

**File**: `knowledge-graph/experiments/factorial_design_v1.yaml`

```yaml
experiment:
  id: factorial_design_v1
  design: "4 models x 3 losses x 3 folds (factorial) + 2 zero-shot baselines"
  models: [dynunet, mambavesselnet_pp, sam3_topolora, sam3_hybrid, sam3_vanilla, vesselfm]
  losses: [dice_ce, cbdice_cldice, dice_ce_cldice]
  dataset: minivess (MiniVess primary; DeepVess/TubeNet for VesselFM)
  epochs: 50 (fair comparison)
  folds: 3 (seed=42)
  status: planned
  metrics: [cldice (trusted), masd (trusted), dsc (foil)]
  analyses: [factorial_anova, pairwise_comparison, specification_curve, dice_as_foil]
  paper_sections: [R3a, R3b, R3c]
```

---

## 9. Implementation Gaps -- What Needs to Be Built

### 9.1 Statistical Methods (biostatistics_statistics.py)

| Method | Function to Add | Library | Priority |
|--------|----------------|---------|----------|
| Two-way ANOVA | `compute_factorial_anova()` | `pingouin.anova` or `statsmodels.stats.anova` | P0 |
| Mixed-effects model | `compute_mixed_effects()` | `statsmodels.MixedLM` | P0 |
| Eta-squared / Omega-squared | Part of ANOVA output | `pingouin` (returns eta-sq automatically) | P0 |
| Tukey HSD post-hoc | `compute_tukey_posthoc()` | `pingouin.pairwise_tukey` | P1 |
| Spearman rank correlation | `compute_rank_correlation()` | `scipy.stats.spearmanr` | P1 |

### 9.2 Figures (biostatistics_figures.py)

| Figure | Function to Add | Priority |
|--------|----------------|----------|
| Specification curve | `_generate_specification_curve()` | P0 |
| Interaction plot | `_generate_interaction_plot()` | P0 |
| Variance decomposition lollipop | `_generate_variance_lollipop()` | P1 |
| Critical Difference diagram (visual) | `_generate_cd_diagram()` | P1 |

### 9.3 Tables (biostatistics_tables.py)

| Table | Function to Add | Priority |
|-------|----------------|----------|
| ANOVA summary table | `_generate_anova_table()` | P0 |
| Rank discrepancy table | `_generate_rank_discrepancy_table()` | P1 |

### 9.4 Configuration (biostatistics_config.py)

The `BiostatisticsConfig` class needs extension:

- Add `analysis_type` field: `Literal["pairwise", "factorial", "both"]` (default: "both")
- Add `factorial_factors` field: `dict[str, list[str]]` mapping factor names to levels
- Add `anova_type` field: `Literal["parametric", "nonparametric", "both"]` (default: "both")
- Add `rope_enabled` field: `bool` (default: True)

### 9.5 SkyPilot YAMLs

- `deployment/skypilot/train_factorial.yaml` (parameterized trainer)
- `deployment/skypilot/eval_zeroshot.yaml` (zero-shot evaluation)

### 9.6 Hydra Experiment Configs

- `configs/experiment/factorial_mamba_dice_ce.yaml`
- `configs/experiment/factorial_mamba_cbdice_cldice.yaml`
- `configs/experiment/factorial_mamba_dice_ce_cldice.yaml`
- `configs/experiment/factorial_sam3topolora_dice_ce.yaml`
- `configs/experiment/factorial_sam3topolora_cbdice_cldice.yaml`
- `configs/experiment/factorial_sam3topolora_dice_ce_cldice.yaml`
- `configs/experiment/factorial_sam3hybrid_dice_ce.yaml`
- `configs/experiment/factorial_sam3hybrid_cbdice_cldice.yaml`
- `configs/experiment/factorial_sam3hybrid_dice_ce_cldice.yaml`

Alternatively, a single `configs/experiment/factorial_base.yaml` with Hydra overrides:
```bash
# Override model and loss via Hydra CLI
HYDRA_OVERRIDES="model=mambavesselnet,loss_name=cbdice_cldice,max_epochs=50,fold_id=0"
```

This is the preferred approach per CLAUDE.md (zero hardcoding, config-driven dispatch).

---

## 10. Timeline

| Week | Phase | Key Deliverable |
|------|-------|----------------|
| Week 0 (current) | Planning | This document; KG updates; SkyPilot YAML templates |
| Week 1 | Phase 0 | DynUNet epoch-50 metric extraction |
| Week 1-2 | Phase 1 | 1-fold pilot (9 runs); go/no-go decision |
| Week 2-3 | Phase 2 | Full 3-fold factorial (18 runs) |
| Week 3 | Phase 3 | Zero-shot baselines (SAM3 Vanilla, VesselFM) |
| Week 3-4 | Biostatistics impl | Two-way ANOVA, mixed-effects, spec curve, interaction plots |
| Week 4 | Phase 4 | Biostatistics flow on all 38 evaluations |
| Week 4-5 | Phase 5 | Manuscript integration (R3a, R3b, R3c) |

**Critical path**: MambaVesselNet++ adapter must be fully functional on GPU (currently CODE_COMPLETE, GPU_PENDING). SAM3 TopoLoRA VRAM must fit on L4 (currently estimated, not verified).

---

## 11. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| SAM3 TopoLoRA OOM on L4 (24 GB) | Medium | High (requires A100, +$30) | Phase 1 pilot catches this; budget includes buffer |
| MambaVesselNet++ training diverges | Low | Medium (need to debug loss) | `mamba-ssm` compilation verified; architecture validated in [Xu & Chen et al. (2025)](https://doi.org/10.1145/3757324) |
| DynUNet epoch-50 metrics not in MLflow | Medium | Low (re-run 50 epochs locally, ~2 hrs) | Check MLflow metric history first |
| GCP spot preemption delays | High | Low (SkyPilot auto-recovers) | Budget includes 50% preemption buffer |
| Normality assumption violated for ANOVA | High | Low (Friedman as backup; bootstrap as supplement) | Report both parametric and non-parametric results |
| VesselFM weights unavailable (HF token) | Low | Medium (skip R3c) | HF token in `.env`; cache during SkyPilot setup |
| Interaction term non-significant | Medium | Low (report null result honestly) | Non-significant interaction is still informative: "loss choice is architecture-independent" |

---

## 12. References

All citations follow author-year format with hyperlinks per CLAUDE.md citation rules.

- [Benavoli et al. (2017). "Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis." *JMLR*.](https://jmlr.org/papers/v18/16-305.html)
- [Demsar (2006). "Statistical Comparisons of Classifiers over Multiple Data Sets." *JMLR*.](https://jmlr.org/papers/v7/demsar06a.html)
- [Koo & Li (2016). "A Guideline of Selecting and Reporting Intraclass Correlation Coefficients for Reliability Research." *Journal of Chiropractic Medicine*.](https://doi.org/10.1016/j.jcm.2016.02.012)
- [Lakens (2013). "Calculating and Reporting Effect Sizes to Facilitate Cumulative Science." *Frontiers in Psychology*.](https://doi.org/10.3389/fpsyg.2013.00863)
- [Maier-Hein et al. (2024). "Metrics Reloaded: Recommendations for Image Analysis Validation." *Nature Methods*.](https://doi.org/10.1038/s41592-023-02151-z)
- [Shit et al. (2021). "clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation." *CVPR*.](https://arxiv.org/abs/2003.07311)
- [Simmons et al. (2011). "False-Positive Psychology: Undisclosed Flexibility in Data Collection and Analysis." *Psychological Science*.](https://doi.org/10.1177/0956797611417632)
- [Steegen et al. (2016). "Increasing Transparency Through a Multiverse Analysis." *Perspectives on Psychological Science*.](https://doi.org/10.1177/1745691616658637)
- [Vargha & Delaney (2000). "A Critique and Improvement of the CL Common Language Effect Size Statistic of McGraw and Wong." *JEBS*.](https://doi.org/10.3102/10769986025002101)
- [Chen et al. (2024). "MambaVesselNet." *ACM Multimedia Asia*.](https://doi.org/10.1145/3696409.3700231)
- [Xu & Chen et al. (2025). "MambaVesselNet++." *ACM TOMM*.](https://doi.org/10.1145/3757324)
- [Ravi et al. (2025). "SAM 3." *Meta AI Research*.](https://github.com/facebookresearch/sam3)
- [Wittmann et al. (2024). "VesselFM." *arXiv*.](https://arxiv.org/abs/2411.17386)
- [Yang et al. (2023). "SkyPilot: An Intercloud Broker for Sky Computing." *NSDI*.](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng)
