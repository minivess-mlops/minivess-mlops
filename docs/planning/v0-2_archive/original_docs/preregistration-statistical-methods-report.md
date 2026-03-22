# Preregistration and Statistical Methods for ML/Biomedical Factorial Experiments

**Date**: 2026-03-20
**Purpose**: Literature review covering preregistration practices, statistical tests, data formats, and multiple comparisons correction for a 4-model x 3-loss x 2-calibration factorial design in biomedical image segmentation.

---

## 1. Preregistration for ML Benchmark Studies

### 1.1 What Preregistration Is

Preregistration is the practice of publicly specifying a research plan — hypotheses, analysis methods, metrics, and decision criteria — **before** conducting the experiment or analyzing data. The Center for Open Science (COS) defines it as a mechanism to "distinguish planned from unplanned work," creating transparency about whether hypotheses were generated before or after observing results.

Key principle from COS: **"A plan, not a prison."** Researchers can conduct unplanned analyses but must clearly label them as exploratory. Deviations from the registered protocol are expected and acceptable — they must simply be documented transparently.

**Sources:**
- [Simonsen et al. (2025). "Preregistration: A Key to Credible Real-World Evidence Generation." *Pharmacoepidemiol Drug Saf.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC12397443/)
- [Dirnagl (2020). "Preregistration of exploratory research: Learning from the golden age of discovery." *PLoS Biology.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7098547/)
- [Center for Open Science — Preregistration](https://www.cos.io/initiatives/prereg)

### 1.2 Why It Matters for ML Benchmarks

ML benchmark studies face specific integrity threats that preregistration mitigates:

1. **ML-specific p-hacking**: Adjusting problem definitions, metrics, or data splits post-hoc based on modeling results. Hofman et al. (2023) identified "overlooked contextual factors, data-dependent decision-making, and unintentional re-use of test data" as core threats.
2. **Selective reporting**: Reporting only the metric or data split where a method wins.
3. **HARKing** (Hypothesizing After Results are Known): Framing exploratory findings as confirmatory.
4. **Undisclosed variance**: Bouthillier et al. (2021) showed that variance from data sampling, parameter initialization, and hyperparameter choice "impact markedly" ML benchmark results, yet most papers report only a single run.

**Sources:**
- [Hofman et al. (2023). "Pre-registration for Predictive Modeling." *arXiv:2311.18807.*](https://arxiv.org/abs/2311.18807)
- [Bouthillier et al. (2021). "Accounting for Variance in Machine Learning Benchmarks." *MLSys.*](https://arxiv.org/abs/2103.03098)

### 1.3 What to Preregister (Hofman et al. Two-Phase Template)

**Phase A — Before Training:**
| Element | What to specify |
|---------|----------------|
| Research question | Which factor (model, loss, calibration) most affects segmentation quality? |
| Outcome variables | Primary: Dice (DSC). Secondary: clDice, NSD, HD95, ECE. |
| Data construction | Dataset splits (3-fold, seed=42), train/val/test partition. |
| Performance thresholds | Minimum DSC to consider a configuration viable. |
| Baseline comparisons | Which configurations serve as baselines (e.g., DynUNet + DiceCE + no calibration). |
| Analysis plan | Factorial ANOVA / LMM, post-hoc tests, multiple comparisons correction method. |

**Phase B — Before Testing (after training, before evaluating on held-out test set):**
| Element | What to specify |
|---------|----------------|
| Final model specifications | Architecture, hyperparameters, training details. |
| Confirmation of test set integrity | Test set not accessed during development. |
| Secondary/exploratory analyses | Any analyses not in Phase A, clearly labeled. |

### 1.4 Where to Register

| Platform | Best for | Features |
|----------|----------|----------|
| **OSF (Open Science Framework)** | General ML/science | DOI, embargo up to 4 years, version control, wiki |
| **RWE Registry (on OSF)** | Real-world evidence studies | Lightweight metadata, protocol upload, DOI |
| **arXiv + GitHub** | ML community norm | Post protocol as preprint + frozen repo tag |

**Recommendation for MinIVess**: Register on OSF with the full factorial design, analysis plan, and metric hierarchy. Freeze the experiment config YAML as a supplementary file. This aligns with Nature Protocols expectations.

---

## 2. Statistical Tests for a 4-Model x 3-Loss x 2-Calibration Factorial Design

### 2.1 Overview of the Design

The factorial experiment has:
- **Factor A**: Model (4 levels: DynUNet, SegResNet, SAM3-Vanilla, SAM3-Hybrid)
- **Factor B**: Loss (3 levels: DiceCE, cbDice+clDice, TopK+clDice)
- **Factor C**: Calibration (2 levels: none, temperature scaling)
- **Total cells**: 4 x 3 x 2 = 24 configurations
- **Replication**: 3 folds x N volumes per fold
- **Blocking**: Volumes are the observational units (repeated across configurations)

### 2.2 Recommended Statistical Framework

#### Tier 1: Primary Analysis — Linear Mixed-Effects Model (LMM)

**Why LMM, not classical ANOVA:**
- Volumes are measured repeatedly across all 24 configurations (crossed random effect).
- Folds introduce another grouping structure.
- Dice scores are bounded [0, 1] and typically non-normal — LMM with appropriate link function handles this.
- Classical repeated-measures ANOVA requires sphericity; LMM does not.
- LMM naturally handles the nested structure: volumes within folds.

**Model specification:**
```
DSC ~ Model * Loss * Calibration + (1 | Volume) + (1 | Fold)
```

- **Fixed effects**: Model, Loss, Calibration, and all 2-way and 3-way interactions.
- **Random effects**: Volume (accounts for easy/hard volumes), Fold (accounts for fold-specific train/val partition effects).
- **Software**: `statsmodels.MixedLM` (Python), `lme4::lmer` (R), or `pymer4`.

**What it answers:**
- Main effects: Does model architecture matter? Does loss function matter? Does calibration matter?
- Interactions: Does the effect of loss depend on the model? Does calibration benefit some model-loss combinations more than others?
- The three-way interaction reveals whether the full factorial space has non-additive structure.

**Sources:**
- [JCI — "Guidelines for repeated measures statistical analysis approaches with basic science research considerations" (2023).](https://pmc.ncbi.nlm.nih.gov/articles/PMC10231988/)
- [ScienceDirect — "Beyond t test and ANOVA: applications of mixed-effects models for more rigorous statistical analysis in neuroscience research."](https://www.sciencedirect.com/science/article/pii/S089662732100845X)

#### Tier 2: Non-Parametric Omnibus — Friedman Test

**When to use**: As a robustness check when LMM distributional assumptions are questionable, or for the primary report if reviewers prefer distribution-free methods.

**How it works**: Ranks the 24 configurations within each volume, then tests whether average ranks differ across configurations. This is the non-parametric analog of repeated-measures ANOVA.

**Post-hoc**: Nemenyi test for all-pairwise comparisons, or Holm-corrected Wilcoxon signed-rank tests for specific contrasts.

**Limitation**: Friedman does not decompose into factorial main effects and interactions — it treats the 24 cells as an unstructured set. It answers "do configurations differ?" but not "which factor drives the difference?"

**Sources:**
- [Demsar (2006). "Statistical Comparisons of Classifiers over Multiple Data Sets." *JMLR* 7, 1-30.](https://jmlr.org/papers/v7/demsar06a.html)
- [Garcia & Herrera (2008). "An Extension on Statistical Comparisons of Classifiers over Multiple Data Sets for all Pairwise Comparisons." *JMLR* 9, 2677-2694.](https://www.jmlr.org/papers/volume9/garcia08a/garcia08a.pdf)

#### Tier 3: Pairwise Comparisons — Wilcoxon Signed-Rank Test

**When to use**: For specific pre-planned contrasts (e.g., "Does SAM3-Hybrid beat DynUNet under the same loss and calibration?").

**How it works**: Paired test on per-volume Dice differences between two configurations. Non-parametric — no normality assumption needed.

**This is the standard in medical image segmentation**: Isensee et al. use Wilcoxon signed-rank on per-case Dice scores in nnU-Net comparisons (MICCAI 2024).

**Sources:**
- [Isensee et al. (2024). "nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation." *MICCAI 2024.*](https://arxiv.org/abs/2404.09556)

#### Tier 4: Bootstrap and Permutation Tests

**When to use**: As a complement to parametric/rank-based tests, especially for:
- Confidence intervals on pairwise Dice differences.
- Small sample sizes where asymptotic test assumptions are fragile.

**How it works**:
- **Paired bootstrap**: Resample (volume, config_A_dice, config_B_dice) tuples with replacement; compute delta distribution; derive BCa confidence interval.
- **Permutation test**: Randomly swap config_A and config_B labels within each volume; compute test statistic under null; derive exact p-value.

**Sources:**
- [arXiv:2511.19794 — "When +1% Is Not Enough: A Paired Bootstrap Protocol for Evaluating Small Improvements."](https://arxiv.org/abs/2511.19794)

#### Tier 5: Specification Curve Analysis (Multiverse Analysis)

**When to use**: To demonstrate robustness of findings across the "multiverse" of defensible analytical choices (metric choice, aggregation method, fold weighting, outlier handling).

**Three steps** (Simonsohn, Simmons, & Nelson, 2020):
1. Identify all theoretically justified, non-redundant specifications.
2. Run the analysis under each specification and display results as a sorted curve.
3. Conduct joint inference (permutation-based) to test whether the overall curve is inconsistent with the null.

**Application to the factorial design**: Each specification is a combination of choices: which metric (DSC vs clDice vs NSD), which aggregation (mean vs median), which volumes to include/exclude, which statistical test. The specification curve shows whether the conclusion "Model X > Model Y" holds across all these choices.

**Sources:**
- [FORRT — Specification Curve Analysis](https://forrt.org/glossary/english/specification_curve_analysis/)
- [Simonsohn, Simmons & Nelson (2020). "Specification curve analysis." *Nature Human Behaviour.*](https://www.semanticscholar.org/paper/0a6f39bad41608b86673b6226f2893912b27a72c)

### 2.3 Decision Tree: Which Test When

```
Start
  |
  v
[Full factorial analysis?] --YES--> Linear Mixed Model (Tier 1)
  |                                    -> Reports main effects + interactions
  |                                    -> Post-hoc: emmeans contrasts with Holm correction
  |
  NO (or robustness check)
  |
  v
[Compare all 24 configs as unstructured set?] --YES--> Friedman (Tier 2)
  |                                                      -> Post-hoc: Nemenyi
  |
  NO (specific pairwise question)
  |
  v
[Compare 2 specific configs?] --YES--> Wilcoxon signed-rank (Tier 3)
  |                                      -> + Bootstrap CI (Tier 4)
  |
  v
[Robustness across analytical choices?] --YES--> Specification Curve (Tier 5)
```

---

## 3. Which Tests Need Per-Volume Metrics (Not Just Per-Fold Aggregates)

### 3.1 The Critical Distinction

| Granularity | What you have | What you can do |
|-------------|--------------|-----------------|
| **Per-fold aggregate** (3 values per config) | Mean DSC per fold | Almost nothing — 3 data points per config is too few for any reliable test |
| **Per-volume metrics** (N values per config) | DSC for each of the ~70 volumes per config | Full statistical power — paired tests, LMM, bootstrap |

**Per-volume metrics are non-negotiable for all tests above.** With only 3 folds, per-fold aggregates give you 3 observations per configuration — insufficient for any statistical test.

### 3.2 Test-by-Test Requirements

| Test | Requires per-volume? | Minimum data granularity | Why |
|------|---------------------|-------------------------|-----|
| **LMM** | **YES** | DSC per (volume, config) | Volume is the random effect; needs per-volume observations |
| **Friedman** | **YES** | Rank per (volume, config) | Ranks are computed per volume across configs |
| **Wilcoxon signed-rank** | **YES** | DSC per (volume, config_A) and (volume, config_B) | Paired differences computed per volume |
| **Bootstrap CI** | **YES** | DSC per (volume, config) | Resamples volume-level paired tuples |
| **Permutation test** | **YES** | DSC per (volume, config) | Permutes within-volume labels |
| **Spec curve** | **YES** | Full per-volume data under each specification | Each specification re-analyzes volume-level data |

### 3.3 Isensee et al. (2024) Validation: Intra- vs Inter-Method Variance

Isensee et al. introduced a benchmarking suitability metric:
- **Intra-method SD**: Standard deviation of a single method's DSC across folds (noise).
- **Inter-method SD**: Standard deviation of mean DSC across different methods (signal).
- **Suitability score** = inter-method SD / intra-method SD.

This requires per-fold AND per-volume metrics to compute. Datasets with high suitability scores (e.g., KiTS, AMOS) can discriminate between methods; those with low scores (e.g., BraTS — saturated performance) cannot.

**Implication for MinIVess**: Before running the full factorial, compute this suitability metric on MiniVess to verify it can discriminate between configurations. If intra-method variance is high relative to inter-method variance, the dataset may need more volumes or the design needs more folds.

---

## 4. Data Format Needed for Each Test

### 4.1 Universal Format: Long-Form Per-Volume Table

All tests operate on or can be derived from a single canonical table:

```
| volume_id | fold | model      | loss          | calibration | dsc    | cldice | nsd    | hd95   | ece    |
|-----------|------|------------|---------------|-------------|--------|--------|--------|--------|--------|
| vol_001   | 0    | dynunet    | dice_ce       | none        | 0.823  | 0.714  | 0.891  | 3.42   | 0.045  |
| vol_001   | 0    | dynunet    | dice_ce       | temp_scale  | 0.825  | 0.718  | 0.893  | 3.38   | 0.021  |
| vol_001   | 0    | dynunet    | cbdice_cldice | none        | 0.841  | 0.762  | 0.902  | 3.11   | 0.051  |
| ...       | ...  | ...        | ...           | ...         | ...    | ...    | ...    | ...    | ...    |
| vol_070   | 2    | sam3hybrid | topk_cldice   | temp_scale  | 0.867  | 0.801  | 0.924  | 2.87   | 0.018  |
```

**Rows**: One row per (volume, configuration) pair.
**Expected row count**: ~70 volumes x 24 configurations = ~1,680 rows (exact count depends on fold assignments — each volume appears in validation for exactly one fold, so effectively ~23 val volumes x 24 configs x 3 folds = ~1,656 rows).

### 4.2 Format Requirements by Test

| Test | Input format | Derivation from canonical table |
|------|-------------|-------------------------------|
| **LMM** | Long-form table as-is | Direct: `DSC ~ Model * Loss * Calibration + (1|volume_id) + (1|fold)` |
| **Friedman** | Matrix: volumes (rows) x configs (columns), cells = DSC | Pivot canonical table; configs = `f"{model}_{loss}_{calibration}"` |
| **Wilcoxon** | Two paired vectors: DSC_config_A[vol_i] and DSC_config_B[vol_i] | Filter canonical table to two configs, align by volume_id |
| **Bootstrap** | Paired tuples: (volume_id, DSC_A, DSC_B) | Same as Wilcoxon, joined |
| **Spec curve** | Re-run analysis pipeline under each specification | Full canonical table + specification definitions |

### 4.3 Additional Metadata Columns (Recommended)

For subgroup analysis and covariate modeling:

```
| volume_id | fold | dataset   | voxel_spacing_um | volume_shape     | vessel_density | snr   |
|-----------|------|-----------|------------------|------------------|----------------|-------|
| vol_001   | 0    | minivess  | 0.6x0.6x1.0     | 256x256x128      | 0.034          | 12.4  |
```

These enable:
- **Subgroup analysis**: Does Model X win only on high-SNR volumes?
- **Covariate adjustment in LMM**: `DSC ~ Model * Loss * Calibration + vessel_density + (1|volume_id)`
- **DeepVess test evaluation**: Same format, `dataset = "deepvess"`, enables train/test domain shift analysis.

### 4.4 Storage Recommendation

Store as:
1. **Parquet** (primary): Typed columns, fast queries, DuckDB-compatible.
2. **CSV** (secondary): Human-readable backup.
3. **MLflow**: Log per-volume metrics as artifacts (JSON/Parquet) attached to each run.

Schema enforcement via Pandera to catch missing volumes or malformed metric values before analysis.

---

## 5. Multiple Comparisons Correction in DL Model Comparisons

### 5.1 The Problem

With 24 configurations and all-pairwise comparisons, there are C(24,2) = 276 possible pairwise tests. At alpha = 0.05 without correction, you expect ~14 false positives by chance alone. Even with pre-planned contrasts, a factorial design with 3 main effects, 3 two-way interactions, 1 three-way interaction, and pairwise post-hocs generates dozens of tests.

### 5.2 Correction Methods (Ordered by Conservatism)

| Method | Controls | Power | When to use |
|--------|----------|-------|-------------|
| **Bonferroni** | FWER | Lowest | Never preferred — always dominated by Holm |
| **Holm (step-down Bonferroni)** | FWER | Higher than Bonferroni | **Confirmatory** pairwise comparisons. Uniformly more powerful than Bonferroni, same FWER guarantee. |
| **Hochberg (step-up)** | FWER | Higher than Holm | When test statistics are independent or positively correlated |
| **Hommel** | FWER | Highest FWER method | Most powerful FWER control, but computationally expensive |
| **Benjamini-Hochberg (BH)** | FDR | Much higher | **Exploratory** analyses, large number of comparisons. Controls expected proportion of false discoveries rather than probability of any false discovery. |
| **Nemenyi** | FWER (Friedman context) | Moderate | All-pairwise post-hoc after Friedman test |

**Sources:**
- [Holm (1979). "A simple sequentially rejective multiple test procedure." *Scand J Statistics.*](https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method)
- [Garcia & Herrera (2008). Extension on statistical comparisons with Bergmann-Hommel post-hoc. *JMLR.*](https://www.jmlr.org/papers/volume9/garcia08a/garcia08a.pdf)
- [Benjamini & Hochberg (1995). "Controlling the false discovery rate." *JRSS-B.*](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x)

### 5.3 Recommended Strategy for the MinIVess Factorial Design

**Layer 1 — Omnibus tests (no correction needed):**
- LMM F-tests for main effects and interactions: 7 tests total (3 main + 3 two-way + 1 three-way). These are omnibus and do not require multiple comparisons correction beyond the model-level Type III tests.

**Layer 2 — Pre-planned contrasts (Holm correction):**
- Pre-specify a small number of scientifically motivated contrasts in the preregistration:
  - SAM3-Hybrid vs DynUNet (foundation model vs U-Net baseline)
  - cbDice+clDice vs DiceCE (topology-aware vs standard loss)
  - Calibrated vs uncalibrated (within best model-loss combination)
  - ~6-10 contrasts total
- Apply **Holm correction** across this set. This is the standard in the field (Demsar 2006, Garcia & Herrera 2008).

**Layer 3 — Exploratory pairwise comparisons (BH-FDR correction):**
- All remaining C(24,2) = 276 pairwise tests.
- Apply **Benjamini-Hochberg FDR** at q = 0.05.
- Clearly label these as exploratory in the manuscript.
- Use critical difference diagrams (Demsar 2006) for visualization.

**Layer 4 — Robustness (specification curve):**
- No correction needed at the individual specification level.
- Joint inference via permutation test across the full specification curve.

### 5.4 Practical Implementation in Python

```python
# Holm correction for pre-planned contrasts
from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(raw_pvals, method='holm')

# BH-FDR for exploratory comparisons
reject, pvals_corrected, _, _ = multipletests(raw_pvals, method='fdr_bh')

# Friedman + Nemenyi (via scikit-posthocs)
import scikit_posthocs as sp
sp.posthoc_nemenyi_friedman(data_matrix)

# Critical difference diagram
# Use Orange3 or autorank package
```

### 5.5 What to Report in the Manuscript

For each comparison, report:
1. **Raw p-value** (uncorrected)
2. **Corrected p-value** (with method identified)
3. **Effect size** (Cohen's d or rank-biserial correlation)
4. **95% confidence interval** (bootstrap BCa preferred)
5. **Whether the comparison was pre-registered** (confirmatory vs exploratory)

---

## 6. Summary of Recommendations

### For Preregistration
1. Register on OSF before running the factorial experiment.
2. Use Hofman et al.'s two-phase template: Phase A before training, Phase B before test-set evaluation.
3. Freeze the experiment YAML config, splits file, and analysis plan as supplementary materials.
4. Clearly separate confirmatory contrasts (pre-specified) from exploratory analyses.

### For Statistical Analysis
1. **Primary**: LMM with Model x Loss x Calibration as fixed effects, Volume and Fold as random effects.
2. **Robustness check**: Friedman test + Nemenyi post-hoc as non-parametric complement.
3. **Pairwise**: Wilcoxon signed-rank with bootstrap CIs for specific pre-planned contrasts.
4. **Transparency**: Specification curve analysis across metric/aggregation choices.

### For Data Collection
1. Log **per-volume** metrics for every configuration — aggregates alone are insufficient.
2. Use long-form Parquet as canonical storage format.
3. Include volume-level metadata (spacing, shape, vessel density) for covariate analysis.
4. Enforce schema with Pandera; store alongside MLflow run artifacts.

### For Multiple Comparisons
1. Pre-planned contrasts: Holm correction (FWER).
2. Exploratory pairwise: Benjamini-Hochberg FDR.
3. Always report both raw and corrected p-values.
4. Always report effect sizes and confidence intervals, not just p-values.
5. Use critical difference diagrams for visual communication.

---

## Key References

1. [Demsar (2006). "Statistical Comparisons of Classifiers over Multiple Data Sets." *JMLR* 7, 1-30.](https://jmlr.org/papers/v7/demsar06a.html)
2. [Garcia & Herrera (2008). "An Extension on Statistical Comparisons of Classifiers over Multiple Data Sets." *JMLR* 9, 2677-2694.](https://www.jmlr.org/papers/volume9/garcia08a/garcia08a.pdf)
3. [Bouthillier et al. (2021). "Accounting for Variance in Machine Learning Benchmarks." *MLSys.*](https://arxiv.org/abs/2103.03098)
4. [Isensee et al. (2024). "nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation." *MICCAI 2024.*](https://arxiv.org/abs/2404.09556)
5. [Hofman et al. (2023). "Pre-registration for Predictive Modeling." *arXiv:2311.18807.*](https://arxiv.org/abs/2311.18807)
6. [Simonsen et al. (2025). "Preregistration: A Key to Credible Real-World Evidence Generation." *Pharmacoepidemiol Drug Saf.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC12397443/)
7. [Dirnagl (2020). "Preregistration of exploratory research." *PLoS Biology.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7098547/)
8. [Benjamini & Hochberg (1995). "Controlling the false discovery rate." *JRSS-B.*](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x)
9. [Simonsohn, Simmons & Nelson (2020). "Specification curve analysis." *Nature Human Behaviour.*](https://www.semanticscholar.org/paper/0a6f39bad41608b86673b6226f2893912b27a72c)
10. [Center for Open Science — Preregistration](https://www.cos.io/initiatives/prereg)
