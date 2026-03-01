# Comprehensive Metrics Reporting Plan for MinIVess

**Date:** 2026-02-28
**Status:** Reference document for manuscript preparation
**Purpose:** Supporting document for writing the academic paper — defines ALL evaluation
metrics considered, the supplementary table plan, main paper reporting strategy, and
recommended fixes to our current compound metric.

---

## Table of Contents

1. [Why Metric Choice Matters for Vascular Segmentation](#1-why-metric-choice-matters)
2. [Classical Medical Image Segmentation Metrics (Taha & Hanbury 2015)](#2-classical-metrics)
3. [Topology-Aware Metrics for Tubular Structures](#3-topology-aware-metrics)
4. [Tubular Metric Benchmark (Decroocq et al. 2025)](#4-tubular-benchmark)
5. [Supplementary Table S1: Complete Metric Catalog](#5-supplementary-table)
6. [Main Paper Metrics — Reported Separately](#6-main-paper-metrics)
7. [Compound Metric Fix: From Range-Collapsed to Principled](#7-compound-metric-fix)
8. [Model Selection Strategy](#8-model-selection-strategy)
9. [Implementation Status in MinIVess](#9-implementation-status)
10. [References](#10-references)

---

## 1. Why Metric Choice Matters for Vascular Segmentation

### 1.1 The Downstream Pipeline Demands Topology

Vascular segmentation is not an end in itself. The segmentation mask feeds into
downstream analyses that are **catastrophically sensitive to topological errors**:

| Downstream Task | Topology Requirement | Failure Mode |
|----------------|---------------------|-------------|
| **Graph extraction** (branch points, vessel trees) | Gap-free connectivity | A single 1-voxel disconnection severs the entire subtree |
| **Computational hemodynamics** (CFD/1D models) | Watertight, loop-consistent mesh | Gaps produce unrealistic pressure drops; false loops create shunt pathways |
| **Vascular biomarkers** (diameter, tortuosity, branching angles) | Accurate centerlines with correct branching | Missing branches undercount bifurcations; shortcuts distort tortuosity |
| **Artery-vein classification** | Connected components separated by type | Merged AV loops propagate label errors through entire subtrees |
| **Extravascular distance maps** | Complete vessel coverage | Missing vessels create false "avascular zones" |
| **Fractal dimension / scaling laws** | Full tree structure (Murray's law) | Pruned branches bias the power-law exponent |
| **Functional biomarkers** (blood flow, pO2, SO2) | Correct vessel hierarchy for resistance modeling | Topology errors propagate nonlinearly through Poiseuille flow models |

**Key insight:** A segmentation with 0.90 Dice but broken topology may be **less useful**
than one with 0.85 Dice but intact connectivity. Standard volumetric metrics are blind
to this distinction. This motivates our multi-family metric reporting strategy.

### 1.2 The Three Families of Segmentation Error

Following Decroocq et al. (2025), segmentation errors for tubular structures
fall into three families:

| Family | Examples | Affects Betti Numbers? | Detected By |
|--------|----------|----------------------|-------------|
| **Topology** | False/missing components, disconnections, cycle breaks, merging, holes | Yes (changes beta_0, beta_1) | Betti errors, Betti matching, clDice, ccDice |
| **Morphology** | Missing/false branches, branch merging, self-merging | No (branch structure changes, not component count) | clDice, cbDice, skeleton recall/precision, graph metrics |
| **Geometry** | Radius dilation/erosion, deformation, shortened/extended terminals | No (shape changes only) | Dice, NSD, HD95, ASSD, cbDice |

**No single metric covers all three families.** This is the fundamental reason why
Metrics Reloaded (Maier-Hein et al. 2024) recommends reporting metrics from multiple
families separately, not combining them into a single score.

---

## 2. Classical Medical Image Segmentation Metrics (Taha & Hanbury 2015)

Taha and Hanbury catalog 20 metrics across 6 categories, validated on 4,833 segmentations
from the VISCERAL Anatomy benchmarks. The following table includes all 20 with annotations
for vascular segmentation relevance.

### 2.1 Category A: Overlap-Based (8 metrics)

| Metric | Abbr | Range | Dir | Formula | Vascular Relevance |
|--------|------|-------|-----|---------|-------------------|
| Dice Coefficient | DSC | [0,1] | Higher | 2TP/(2TP+FP+FN) | **Primary baseline.** Universal comparability. Biased toward large vessels. |
| Jaccard Index | JAC | [0,1] | Higher | TP/(TP+FP+FN) | Monotonically related to DSC — **redundant, do not report alongside DSC.** |
| True Positive Rate | TPR | [0,1] | Higher | TP/(TP+FN) | Recall. Measures "did we find all vessels?" Critical for completeness. |
| True Negative Rate | TNR | [0,1] | Higher | TN/(TN+FP) | **Avoid for vascular data** — dominated by background (>95% of voxels). |
| False Positive Rate | FPR | [0,1] | Lower | FP/(FP+TN) | Complement of TNR. Same class-imbalance problem. |
| False Negative Rate | FNR | [0,1] | Lower | FN/(FN+TP) | Complement of TPR. Same information. |
| F-Measure | FMS | [0,1] | Higher | 2*PPV*TPR/(PPV+TPR) | Identical to DSC for binary segmentation. **Redundant.** |
| Global Consistency Error | GCE | [0,1] | Lower | See Taha 2015 | Includes TN. **Avoid for vascular data.** |

**Redundancy warning (Taha & Hanbury 2015):** DSC, JAC, and FMS produce **identical
rankings** due to monotonic relationships. Never report all three.

### 2.2 Category B: Volume-Based (1 metric)

| Metric | Abbr | Range | Dir | Vascular Relevance |
|--------|------|-------|-----|-------------------|
| Volumetric Similarity | VS | [0,1] | Higher | Can be 1.0 with zero overlap — only measures size match, not spatial accuracy. Use as complement to DSC, never alone. |

### 2.3 Category C: Pair-Counting (2 metrics)

| Metric | Abbr | Range | Dir | Vascular Relevance |
|--------|------|-------|-----|-------------------|
| Rand Index | RI | [0,1] | Higher | **Avoid** — includes TN, biased by class imbalance. |
| Adjusted Rand Index | ARI | [-1,1] | Higher | Chance-corrected. Useful for multi-label segmentation (e.g., AV classification). |

### 2.4 Category D: Information-Theoretic (2 metrics)

| Metric | Abbr | Range | Dir | Vascular Relevance |
|--------|------|-------|-----|-------------------|
| Mutual Information | MI | [0,inf) | Higher | Rewards recall (penalizes FN > FP). Unbounded — difficult to compare across studies. |
| Variation of Information | VOI | [0,inf) | Lower | Includes TN component. Unbounded. Limited use for binary vascular segmentation. |

### 2.5 Category E: Probabilistic (4 metrics)

| Metric | Abbr | Range | Dir | Vascular Relevance |
|--------|------|-------|-----|-------------------|
| Interclass Correlation | ICC | [-1,1] | Higher | Measures agreement strength. Useful for inter-rater studies. |
| Probabilistic Distance | PBD | [0,inf) | Lower | Reaches infinity at zero overlap — problematic for failed cases. |
| Cohen Kappa | KAP | [-1,1] | Higher | Chance-corrected. Alternative to ARI for binary case. |
| Area Under ROC | AUC | [0,1] | Higher | Useful if soft predictions available. Single operating point = TPR. |

### 2.6 Category F: Spatial Distance-Based (3 metrics)

| Metric | Abbr | Range | Dir | Vascular Relevance |
|--------|------|-------|-----|-------------------|
| Hausdorff Distance | HD | [0,inf) mm | Lower | Extreme outlier sensitivity. **Use HD95 instead.** |
| Average Hausdorff Distance | AVD | [0,inf) mm | Lower | Stable mean boundary error. **Recommended.** Same concept as ASSD/MASD. |
| Mahalanobis Distance | MHD | [0,inf) | Lower | Compares shape ellipsoids. Insensitive to holes/density. Limited for vessels. |

### 2.7 Key Findings from Taha & Hanbury

1. **Metric correlation decreases as overlap decreases** — DSC and distance metrics
   diverge for small/thin structures (like vessels).
2. **Metrics including TN are biased** for class-imbalanced data (vessels ~2-5% of volume).
3. **Minimum recommended set:** one overlap metric + one distance metric from different
   correlation groups.
4. **For small structures:** distance-based metrics (AVD, HD95) are more discriminative
   than overlap metrics.

---

## 3. Topology-Aware Metrics for Tubular Structures

These metrics were developed after Taha & Hanbury (2015) and are essential for
vascular segmentation evaluation.

### 3.1 Skeleton-Based Metrics

| Metric | Range | Dir | Paper | What It Measures |
|--------|-------|-----|-------|-----------------|
| **clDice** | [0,1] | Higher | Shit et al., CVPR 2021 | Centerline connectivity preservation. Harmonic mean of topology precision (predicted skeleton inside GT mask) and sensitivity (GT skeleton inside predicted mask). |
| **cbDice** | [0,1] | Higher | Shi et al., MICCAI 2024 | clDice + boundary/radius awareness. Weights by local vessel diameter, ensuring equitable evaluation across vessel sizes. |
| **Skeleton Recall** (Tsens) | [0,1] | Higher | Component of clDice; Kirchhoff et al., ECCV 2024 | Fraction of GT centerline covered by prediction. Low = missing branches. |
| **Skeleton Precision** (Tprec) | [0,1] | Higher | Component of clDice | Fraction of predicted centerline inside GT mask. Low = hallucinated branches. |
| **Smooth clDice** | [0,1] | Higher | Shit et al. 2021; Morand et al. 2025 | Differentiable approximation via soft min/max pooling. Used as loss, not metric. |

### 3.2 Topological Counting Metrics

| Metric | Range | Dir | Paper | What It Measures |
|--------|-------|-----|-------|-----------------|
| **Betti error (beta_0)** | [0,inf) | Lower | Algebraic topology; Mosinska et al. 2018 | Difference in connected component count. Detects fragmentation/merging. |
| **Betti error (beta_1)** | [0,inf) | Lower | Same | Difference in loop/hole count. Detects spurious/missing cycles. |
| **Betti matching error** | [0,inf) | Lower | Stucki et al., ICML 2023 | Spatially-aware Betti error. Uses persistent homology to match topological features at corresponding locations. Resolves the cancellation problem of naive Betti counting. |

### 3.3 Component-Level Metrics

| Metric | Range | Dir | Paper | What It Measures |
|--------|-------|-----|-------|-----------------|
| **ccDice (foreground)** | [0,1] | Higher | Rouge et al., TGI3@MICCAI 2024 | Dice computed at connected-component level. Penalizes fragmentation and spurious components. |
| **ccDice (background)** | [0,1] | Higher | Same | Inverted ccDice for background topology (holes, cavities). |

### 3.4 Persistent Homology Metrics

| Metric | Range | Dir | Paper | What It Measures |
|--------|-------|-----|-------|-----------------|
| **Wasserstein distance** (W_p) | [0,inf) | Lower | Edelsbrunner & Harer 2010; Hu et al., NeurIPS 2019 | Aggregate cost of matching topological features across scales. Stability-guaranteed. |
| **Bottleneck distance** (W_inf) | [0,inf) | Lower | Same | Worst-case topological feature mismatch. |

### 3.5 Boundary Metrics (Post-2015)

| Metric | Range | Dir | Paper | What It Measures |
|--------|-------|-----|-------|-----------------|
| **NSD** (Surface Dice) | [0,1] | Higher | Nikolov et al. 2018; MSD, KiTS, FLARE | Fraction of boundary within tolerance tau. Clinically interpretable. |
| **HD95** | [0,inf) mm | Lower | BraTS challenge convention | 95th percentile Hausdorff. Robust to single-voxel outliers. |
| **ASSD/MASD** | [0,inf) mm | Lower | Various | Mean boundary distance. Stable, well-understood. |

### 3.6 Graph-Based Metrics

| Metric | Range | Dir | Paper | What It Measures |
|--------|-------|-----|-------|-----------------|
| **NetMets** | [0,1] per component | Lower (FP/FN) | Mayerich et al., BMC Bioinformatics 2012 | Separates geometric from connective errors in network segmentation. |
| **DIADEM** | [0,1] | Higher | Gillette et al., Neuroinformatics 2011 | Tree topology matching weighted by subtree importance. Designed for neuronal arbors. |

---

## 4. Tubular Metric Benchmark (Decroocq et al. 2025)

Decroocq, Poon, Schlachter, and Skibbe (ShapeMI@MICCAI 2025, **Best Paper Award**)
provide the first systematic benchmark of topology-aware metrics for tubular structures.

### 4.1 Metrics Evaluated

8 metrics producing 10 scores: Dice, clDice, cbDice, ccDice_0, ccDice_1,
Betti_error_0, Betti_error_1, Betti_matching_0, Betti_matching_1.

### 4.2 Error Taxonomy (17 Error Types, 3 Families)

**Topology errors** (change Betti numbers): false_component, missing_component,
component_merging, disconnection, cycle_disconnection, hole_merging, merging, hole.

**Morphology errors** (change branch structure): missing_branch, false_branch,
branch_merging, self_merging.

**Geometry errors** (change shape only): missing_terminal, false_terminal,
radius_dilation, radius_erosion, deformation.

### 4.3 Datasets

6 diverse tubular structure datasets: CREMI (neural EM), Roads (aerial),
NeuroMorpho (neurons), Colon_cells, LES-AV (retinal vessels), **MiniVess**
(microvascular — our dataset).

### 4.4 Key Findings

1. **No single metric captures all three error families.** Dice is sensitive to
   geometry but blind to topology; Betti errors detect topology but ignore geometry.
2. **clDice and cbDice bridge topology and morphology** via skeleton overlap, but
   are still insensitive to pure geometry errors (radius changes).
3. **Betti matching > Betti counting** for spatial topology assessment (resolves
   the cancellation problem where one extra + one missing component = 0 error).
4. **ccDice captures component-level topology** without requiring skeletonization,
   making it computationally cheaper than clDice.
5. **Metric behavior varies across datasets** — reinforcing the need for
   application-specific metric selection rather than universal recommendations.

### 4.5 Implications for MinIVess

Since MiniVess is one of the 6 benchmark datasets, the Decroocq et al. findings
apply directly to our evaluation. Their framework validates our choice of reporting
metrics from multiple families (overlap + boundary + topology).

---

## 5. Supplementary Table S1: Complete Metric Catalog

This table is designed for the paper's supplementary material. It catalogs every
metric considered, with justification for inclusion/exclusion in the main results.

### Table S1: Evaluation Metrics for Vascular Segmentation

| # | Metric | Abbr | Family | Range | Dir | In Main? | Rationale |
|---|--------|------|--------|-------|-----|----------|-----------|
| **Overlap-Based** | | | | | | | |
| 1 | Dice Coefficient | DSC | Geometry | [0,1] | Higher | **Yes** | Universal baseline. Required for cross-study comparison. |
| 2 | Jaccard Index | JAC | Geometry | [0,1] | Higher | No | Monotonically related to DSC (JAC = DSC/(2-DSC)). Redundant. |
| 3 | Sensitivity (Recall) | TPR | Geometry | [0,1] | Higher | No | Subsumed by DSC. Could report if FN/FP asymmetry is important. |
| 4 | Specificity | TNR | Geometry | [0,1] | Higher | No | Dominated by background class in vascular data (>95% TN). |
| 5 | F-Measure | FMS | Geometry | [0,1] | Higher | No | Identical to DSC for binary segmentation. |
| 6 | Volumetric Similarity | VS | Geometry | [0,1] | Higher | No | Achieves 1.0 with zero spatial overlap. Misleading alone. |
| 7 | Global Consistency Error | GCE | Geometry | [0,1] | Lower | No | Includes TN. Class-imbalance bias. |
| **Pair-Counting / Info-Theoretic** | | | | | | | |
| 8 | Rand Index | RI | — | [0,1] | Higher | No | TN-biased. Not chance-corrected. |
| 9 | Adjusted Rand Index | ARI | — | [-1,1] | Higher | No | Chance-corrected. Useful for multi-label AV classification. |
| 10 | Mutual Information | MI | — | [0,inf) | Higher | No | Unbounded. Hard to compare across studies. |
| 11 | Variation of Information | VOI | — | [0,inf) | Lower | No | Includes TN. Unbounded. |
| 12 | Cohen Kappa | KAP | — | [-1,1] | Higher | No | Alternative to ARI. Useful for inter-rater studies. |
| **Probabilistic** | | | | | | | |
| 13 | ICC | ICC | — | [-1,1] | Higher | No | For inter-rater reliability studies. |
| 14 | Probabilistic Distance | PBD | — | [0,inf) | Lower | No | Infinity at zero overlap. Fragile. |
| 15 | AUC | AUC | — | [0,1] | Higher | No | Requires soft predictions. Single-point = TPR. |
| **Spatial Distance-Based** | | | | | | | |
| 16 | Hausdorff Distance | HD100 | Geometry | [0,inf) mm | Lower | No | Extreme outlier sensitivity. Use HD95 instead. |
| 17 | 95th-percentile Hausdorff | HD95 | Geometry | [0,inf) mm | Lower | **Yes** | Robust worst-case boundary error. Standard in BraTS, TopCoW. |
| 18 | ASSD / MASD | ASSD | Geometry | [0,inf) mm | Lower | **Yes** | Mean boundary distance. Complementary to HD95. |
| 19 | NSD (Surface Dice) | NSD | Geometry | [0,1] | Higher | **Yes** | Bounded boundary metric. Standard in MSD, KiTS, FLARE, AMOS. |
| 20 | Mahalanobis Distance | MHD | Geometry | [0,inf) | Lower | No | Covariance ellipsoid comparison. Insensitive to vessel detail. |
| **Topology-Aware (Skeleton)** | | | | | | | |
| 21 | Centerline Dice | clDice | Topology | [0,1] | Higher | **Yes** | Centerline connectivity. Definitive tubular topology metric. |
| 22 | Centerline Boundary Dice | cbDice | Topology+Geometry | [0,1] | Higher | Supp | Topology + radius awareness. Newer, less established. |
| 23 | Skeleton Recall | Tsens | Topology | [0,1] | Higher | Supp | Missing branches. Decomposition of clDice. |
| 24 | Skeleton Precision | Tprec | Topology | [0,1] | Higher | Supp | False branches. Decomposition of clDice. |
| **Topology-Aware (Counting)** | | | | | | | |
| 25 | Betti error (beta_0) | BE_0 | Topology | [0,inf) | Lower | **Yes** | Component count mismatch. Detects fragmentation. |
| 26 | Betti error (beta_1) | BE_1 | Topology | [0,inf) | Lower | **Yes** | Loop count mismatch. Detects false/missing cycles. |
| 27 | Betti matching error | BME | Topology | [0,inf) | Lower | Supp | Spatially-aware Betti. Resolves cancellation problem. |
| **Topology-Aware (Component)** | | | | | | | |
| 28 | ccDice (foreground) | ccDice_0 | Topology | [0,1] | Higher | Supp | Component-level Dice. Cheap topology proxy. |
| 29 | ccDice (background) | ccDice_1 | Topology | [0,1] | Higher | No | Background topology. Less relevant for vascular. |
| **Persistent Homology** | | | | | | | |
| 30 | Wasserstein distance | W_p | Topology | [0,inf) | Lower | No | Multi-scale topology. Computationally expensive. Research-grade. |
| 31 | Bottleneck distance | W_inf | Topology | [0,inf) | Lower | No | Worst-case topology mismatch. Same scalability issues. |
| **Graph-Based** | | | | | | | |
| 32 | NetMets | NM | Topology+Geometry | [0,1] | Lower | No | Requires graph extraction preprocessing. Neuroscience-specific. |
| 33 | DIADEM | DIADEM | Topology | [0,1] | Higher | No | Tree-only (no loops). Requires SWC format. |
| **Diameter-Aware** | | | | | | | |
| 34 | Radius-stratified DSC | — | Geometry | [0,1] | Higher | Supp | Per-diameter-bin Dice. Reveals capillary vs. arteriole performance. |

**Legend:** "In Main?" — **Yes** = reported in main results table; **Supp** = reported in
supplementary; **No** = excluded with rationale.

---

## 6. Main Paper Metrics — Reported Separately

Following Metrics Reloaded (Maier-Hein et al. 2024) and the TopCoW challenge (Yang
et al. 2024), we report metrics **individually, not as a compound score**.

### 6.1 Main Results Table Design

The main results table reports 7 metrics from 3 families:

| Family | Metrics | Rationale |
|--------|---------|-----------|
| **Overlap** | DSC | Universal baseline, cross-study comparability |
| **Boundary** | HD95, ASSD, NSD(tau) | Geometric accuracy at surface level |
| **Topology** | clDice, BE_0, BE_1 | Centerline connectivity + topological feature counting |

**Table format (per loss function, 3-fold CV):**

```
Loss Function    | DSC          | clDice       | HD95 (mm)    | ASSD (mm)    | NSD(tau)     | BE_0         | BE_1
─────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼─────────────
dice_ce          | 0.90 ± 0.01  | 0.82 ± 0.02  | 2.1 ± 0.3    | 0.99 ± 0.1   | 0.91 ± 0.02  | 3 ± 1        | 2 ± 1
cbdice           | 0.87 ± 0.02  | 0.78 ± 0.03  | 2.8 ± 0.5    | 1.91 ± 0.3   | 0.88 ± 0.03  | 5 ± 2        | 3 ± 1
dice_ce_cldice   | 0.85 ± 0.01  | 0.90 ± 0.01  | 3.1 ± 0.4    | 1.89 ± 0.2   | 0.86 ± 0.02  | 2 ± 1        | 1 ± 1
cbdice_cldice    | 0.87 ± 0.01  | 0.90 ± 0.01  | 2.5 ± 0.3    | 1.83 ± 0.2   | 0.89 ± 0.02  | 2 ± 1        | 1 ± 1
```

*Note: Values above are illustrative. Actual values from completed training.*

### 6.2 Supplementary Results Table

The supplementary material extends to all computed metrics, including cbDice, skeleton
recall/precision, ccDice, Betti matching error, and radius-stratified DSC.

### 6.3 NSD Tolerance Parameter

NSD requires a tolerance `tau` defining clinically acceptable boundary deviation.
Following the Medical Segmentation Decathlon convention:

```
tau = 2 * median_voxel_spacing
```

For MiniVess (median spacing ~0.5 um): `tau = 1.0 um` (approximately 2 voxels).

We report NSD at this primary tau, with a supplementary NSD-vs-tau curve.

---

## 7. Compound Metric Fix: From Range-Collapsed to Principled

### 7.1 Current Problem (Documented in compound-loss-double-check.md)

Our `val_compound_masd_cldice = 0.5*(1 - MASD/50) + 0.5*clDice` suffers from
**range collapse**: MASD contributes 0.058 spread vs. clDice's 0.417 spread,
making the compound effectively clDice-only. The hardcoded `max_masd=50.0` is
~7x larger than our worst observed value (6.83), wasting most of the MASD range.

**Impact:** Rankings diverge under different normalization schemes. The current
compound selects `dice_ce_cldice` as champion; rank-then-aggregate selects `dice_ce`.

### 7.2 Recommended Fix: Three-Tier Approach

#### Tier 1: Academic Reporting (Main Paper)

**Report all metrics separately.** No compound score in the main results table.
This follows Metrics Reloaded and is the consensus approach in modern challenges.

#### Tier 2: Model Selection (Champion Tagging)

**Use rank-then-aggregate** for automated champion selection:

```python
# Rank each loss function on each metric independently
rank_dsc = models.sort_values("dsc", ascending=False).rank()
rank_cldice = models.sort_values("cldice", ascending=False).rank()
rank_assd = models.sort_values("assd", ascending=True).rank()

# Average ranks (equal weight per metric family)
composite_rank = (rank_dsc + rank_cldice + rank_assd) / 3
champion = composite_rank.idxmin()
```

This is what TopCoW, BraTS, KiTS, and FLARE use. It sidesteps the normalization
problem entirely by converting to ordinal ranks before combining.

#### Tier 3: Per-Epoch Training Tracking

**Replace MASD with NSD in the compound metric:**

```python
# New: val_compound_nsd_cldice = 0.5 * NSD(tau) + 0.5 * clDice
# Both components in [0, 1], higher is better. No normalization needed.
compound = 0.5 * nsd + 0.5 * cldice
```

If NSD computation is too expensive per-epoch, use **exponential decay normalization**
as a transitional fix:

```python
# Exponential decay (CellMap-inspired)
norm_masd = math.exp(-masd / median_masd)  # median_masd ~2.5 from calibration
compound = math.sqrt(norm_masd * cldice)   # geometric mean
```

### 7.3 Implementation Plan

| Component | Current | Proposed | Priority |
|-----------|---------|----------|----------|
| Main paper table | Single compound score | 7 metrics reported separately | **P0** |
| Champion tagger | Maximize compound | Rank-then-aggregate (DSC + clDice + ASSD) | P1 |
| Per-epoch metric | `0.5*(1-MASD/50) + 0.5*clDice` | `0.5*NSD(tau) + 0.5*clDice` | P2 |
| Existing results | Compound logged to MLflow | Recomputable from stored per-fold metrics | No change needed |
| Training loop | Loss functions | Unchanged — losses are separate from eval metrics | No change needed |

### 7.4 Reporting the Fix in the Paper

The paper should **transparently document** this metric evolution:

> "During development, we used a compound metric (0.5 * normalized MASD + 0.5 * clDice)
> for automated model selection. Post-hoc analysis revealed range collapse in the MASD
> component (effective spread 0.058 vs. clDice's 0.417), making the compound effectively
> clDice-only. Following Metrics Reloaded recommendations (Maier-Hein et al. 2024), we
> report all metrics separately in the final evaluation and use rank-then-aggregate for
> model comparison, consistent with TopCoW (Yang et al. 2024) and BraTS (Menze et al. 2015)
> challenge methodology."

This turns a potential weakness into a strength — demonstrating rigorous self-correction.

---

## 8. Model Selection Strategy

### 8.1 Champion Categories

Following the champion tagger design (already implemented in
`src/minivess/pipeline/champion_tagger.py`), we tag three champion categories:

| Category | Selection Method | Meaning |
|----------|-----------------|---------|
| `champion_best_single_fold` | Best individual fold by primary metric | Peak performance (optimistic) |
| `champion_best_cv_mean` | Best 3-fold CV mean by primary metric | Expected performance (realistic) |
| `champion_best_ensemble` | Best ensemble strategy | Combined model performance |

### 8.2 Primary Metric for Champion Selection

**Current:** `val_compound_masd_cldice` (range-collapsed, effectively clDice-only)

**Proposed:** Rank-then-aggregate across (DSC, clDice, ASSD), reporting:
- **Topology champion:** Best clDice (for topology-critical applications)
- **Overlap champion:** Best DSC (for volumetric accuracy applications)
- **Balanced champion:** Best mean rank across all metrics (for general use)

This acknowledges that "best" depends on the downstream application (Section 1.1).

### 8.3 Statistical Testing

For comparing loss functions, use **paired bootstrap testing** (already implemented
in `src/minivess/pipeline/comparison.py`) on each metric independently:

- Report p-values with Bonferroni correction for multiple comparisons
- Use paired tests (same folds) to account for fold-level variance
- Significance threshold: p < 0.05 after correction

---

## 9. Implementation Status in MinIVess

### 9.1 Currently Implemented

| Metric | Module | Status |
|--------|--------|--------|
| DSC (Dice) | `monai.metrics.DiceMetric` | Logged every epoch |
| F1 (foreground) | `torchmetrics.F1Score` | Logged every epoch |
| clDice | Custom (skeleton-based) | Logged every 5 epochs (MetricsReloaded frequency) |
| MASD/ASSD | Custom (surface distance) | Logged every 5 epochs |
| Compound (MASD+clDice) | `validation_metrics.py` | Logged every 5 epochs — **needs fix** |
| Paired bootstrap | `comparison.py` | Analysis flow |
| Champion tagger | `champion_tagger.py` | Analysis flow (24 tests) |

### 9.2 To Be Implemented

| Metric | Library | Priority | Effort |
|--------|---------|----------|--------|
| NSD (Surface Dice) | `monai.metrics.SurfaceDiceMetric` | **P0** | Low — MONAI built-in |
| HD95 | `monai.metrics.HausdorffDistanceMetric` | **P0** | Low — MONAI built-in |
| Betti error (beta_0, beta_1) | `skimage.measure.label` + counting | P1 | Low |
| cbDice | Custom or `PengchengShi1220/cbDice` | P2 | Medium |
| Betti matching error | `nstucki/Betti-Matching-3D` | P2 | Medium (external dep) |
| ccDice | `PierreRouge/ccDice` | P2 | Low |
| Skeleton recall/precision | Decompose existing clDice | P2 | Low |
| Rank-then-aggregate | Analysis flow update | **P0** | Low |
| Radius-stratified DSC | Custom (distance transform binning) | P3 | Medium |

### 9.3 Metric Computation Budget

| Metric Group | Frequency | Time per Epoch | Notes |
|-------------|-----------|----------------|-------|
| DSC, F1 | Every epoch | ~1 sec | Fast tensor operations |
| clDice, MASD, NSD, HD95 | Every 5 epochs | ~4 min | Skeleton computation on 24 full volumes |
| Betti errors | Every 5 epochs | ~30 sec | Connected component labeling |
| cbDice, ccDice, BME | Post-training only | ~10 min | Analysis flow, not per-epoch |
| Graph metrics (NetMets) | Not planned | — | Requires graph extraction pipeline |

---

## 10. References

### Metric Frameworks and Guidelines

1. **Maier-Hein, L. et al. (2024)** "Metrics Reloaded: Recommendations for image
   analysis validation." *Nature Methods* 21:195-212.

2. **Maier-Hein, L. et al. (2018)** "Why rankings of biomedical image analysis
   competitions should be interpreted with care." *Nature Communications* 9:5217.

3. **Reinke, A. et al. (2024)** "Understanding metric-related pitfalls in image
   analysis validation." *Nature Communications*.

4. **Taha, A.A. and Hanbury, A. (2015)** "Metrics for evaluating 3D medical image
   segmentation: analysis, selection, and tool." *BMC Medical Imaging* 15:29.

5. **Muller, P. et al. (2024)** "Pitfalls of distance metric implementations in
   medical image segmentation." *arXiv:2410.02630*.

### Topology-Aware Metrics

6. **Shit, S. et al. (2021)** "clDice — A Novel Topology-Preserving Loss Function
   for Tubular Structure Segmentation." *CVPR 2021*.

7. **Shi, P. et al. (2024)** "Centerline Boundary Dice Loss for Vascular
   Segmentation." *MICCAI 2024*. arXiv:2407.01517.

8. **Rouge, P., Merveille, O., and Passat, N. (2024)** "ccDice: A Topology-Aware
   Dice Score Based on Connected Components." *TGI3@MICCAI 2024*.

9. **Stucki, N. et al. (2023)** "Topologically Faithful Image Segmentation via
   Induced Matching of Persistence Barcodes." *ICML 2023*. PMLR 202:32698-32727.

10. **Stucki, N. et al. (2024)** "Pitfalls of Topology-Aware Image Segmentation."
    *arXiv:2412.14619*.

11. **Kirchhoff, Y. et al. (2024)** "Skeleton Recall Loss for Connectivity
    Conserving and Resource Efficient Segmentation of Thin Tubular Structures."
    *ECCV 2024*. arXiv:2404.03010.

12. **Morand, O. et al. (2025)** "Smooth clDice: A Reliable Metric for Vascular
    Segmentation Evaluation." *LRDE EPITA Technical Report*.

13. **Hu, X. et al. (2019)** "Topology-Preserving Deep Image Segmentation."
    *NeurIPS 2019*.

### Tubular Structure Benchmarks

14. **Decroocq, M., Poon, C., Schlachter, M., and Skibbe, H. (2025)** "Benchmarking
    Evaluation Metrics for Tubular Structure Segmentation in Biomedical Images."
    *ShapeMI@MICCAI 2025*. LNCS 16171:87-102. **Best Paper Award.**

### Challenge Methodology

15. **Yang, L. et al. (2024)** "TopCoW: Topology-Aware Anatomical Segmentation of
    the Circle of Willis for CTA and MRA." *arXiv:2312.17670*.

16. **Antonelli, M. et al. (2022)** "The Medical Segmentation Decathlon." *Nature
    Communications* 13:4128.

17. **Kavur, A.E. et al. (2021)** "CHAOS Challenge: Combined (CT-MR) Healthy
    Abdominal Organ Segmentation." *Medical Image Analysis* 69:101950.

18. **Menze, B.H. et al. (2015)** "The Multimodal Brain Tumor Image Segmentation
    Benchmark (BRATS)." *IEEE TMI* 34(10):1993-2024.

### Boundary Metrics

19. **Nikolov, S. et al. (2021)** "Clinically Applicable Segmentation of Head and
    Neck Anatomy for Radiotherapy: A Domain-Specific Atlas and Neural Network
    Approach." *JMIR* 23(7):e26151.

20. **Yeghiazaryan, V. and Voiculescu, I. (2018)** "Family of Boundary Overlap
    Metrics for the Evaluation of Medical Image Segmentation." *J. Med. Imaging*.

### Graph-Based Metrics

21. **Mayerich, D. et al. (2012)** "NetMets: Software for Quantifying and
    Visualizing Errors in Biological Network Segmentation." *BMC Bioinformatics*.

22. **Gillette, T.A. et al. (2011)** "The DIADEM Metric: Comparing Multiple
    Reconstructions of the Same Neuron." *Neuroinformatics*.

### Tools

23. **EvaluateSegmentation** — Taha & Hanbury's C++ tool for all 20 classical metrics.
    https://github.com/Visceral-Project/EvaluateSegmentation

24. **BenchmarkTopoSegMetrics** — Decroocq et al.'s Python benchmark code.
    https://github.com/megdec/BenchmarkTopoSegMetrics

25. **MONAI** — `monai.metrics` (DiceMetric, HausdorffDistanceMetric,
    SurfaceDiceMetric, SurfaceDistanceMetric). https://docs.monai.io/en/stable/metrics.html

26. **seg-metrics** — Fast Python package for common segmentation metrics.
    https://pypi.org/project/seg-metrics/

---

## Appendix A: Library Availability Summary

| Metric | MONAI | TorchMetrics | scikit-image | GUDHI | Custom |
|--------|-------|-------------|--------------|-------|--------|
| DSC | DiceMetric | Dice | — | — | — |
| HD95 | HausdorffDistanceMetric | — | — | — | — |
| ASSD/MASD | SurfaceDistanceMetric | — | — | — | — |
| NSD | SurfaceDiceMetric | — | — | — | — |
| clDice | SoftclDiceLoss (loss only) | — | — | — | jocpae/clDice |
| cbDice | — | — | — | — | PengchengShi1220/cbDice |
| ccDice | — | — | — | — | PierreRouge/ccDice |
| Betti error | — | — | label() for beta_0 | Betti numbers | — |
| Betti matching | — | — | — | — | nstucki/Betti-Matching-3D |
| Persistence dist. | — | — | — | wasserstein_distance | — |
| NetMets | — | — | — | — | Original C++ |
| DIADEM | — | — | — | — | PyNeval |

## Appendix B: Metric Properties Quick Reference

| Property | Metrics With This Property | Impact on Vascular Evaluation |
|----------|---------------------------|------------------------------|
| **Bounded [0,1]** | DSC, JAC, clDice, cbDice, ccDice, NSD, TPR, VS | Directly comparable, no normalization needed |
| **Unbounded [0,inf)** | HD, ASSD, MASD, BE, MI, VOI, PBD, W_p | Require normalization or rank-aggregate for comparison |
| **TN-biased** | TNR, RI, GCE, VOI | Inflated by background majority. Avoid for vessels. |
| **Chance-corrected** | ARI, KAP | Account for baseline agreement. Use for multi-label. |
| **Topology-aware** | clDice, cbDice, ccDice, BE, BME, W_p | Detect connectivity errors invisible to DSC |
| **Boundary-focused** | HD95, ASSD, NSD | Detect surface accuracy. Complement overlap metrics. |
| **Radius-aware** | cbDice, diameter-stratified DSC | Equitable across vessel sizes |
| **Outlier-sensitive** | HD100 | Single voxel can dominate. Use HD95 instead. |
| **Differentiable** | Soft clDice, BME, W_p | Usable as training loss. Not relevant for evaluation. |
