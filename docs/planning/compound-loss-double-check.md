# Compound Metric Double-Check: `val_compound_masd_cldice`

**Date:** 2026-02-28
**Status:** REVIEW REQUIRED — current implementation has defensible but improvable design choices
**Affects:** Model selection, champion tagging, academic reporting

---

## 1. What We Currently Have

```python
# src/minivess/pipeline/validation_metrics.py

def normalize_masd(masd: float, *, max_masd: float = 50.0) -> float:
    """1 - masd/max_masd, clamped to [0, 1]. Higher is better."""
    return max(0.0, min(1.0, 1.0 - masd / max_masd))

def compute_compound_masd_cldice(*, masd, cldice, w_masd=0.5, w_cldice=0.5, max_masd=50.0):
    """0.5 * normalize_masd(masd) + 0.5 * cldice"""
```

**Formula:** `compound = 0.5 * (1 - MASD/50) + 0.5 * clDice`

**Problem statement:** MASD is in `[0, +inf)` voxels (lower is better) while clDice is in
`[0, 1]` (higher is better). We need a single scalar for automated model selection and
champion tagging. How should we combine them?

---

## 2. Observed Value Ranges in Our Data

From the `dynunet_loss_variation_v2` (full-width) and `dynunet_half_width_v1` experiments:

| Metric | Min | Max | Typical best | Unit |
|--------|-----|-----|-------------|------|
| **MASD** | 0.99 | 6.83 | 1.8–2.0 | voxels |
| **clDice** | 0.064 | 0.898 | 0.85–0.90 | unitless [0,1] |
| **Dice (DSC)** | 0.440 | 0.900 | 0.85–0.90 | unitless [0,1] |

### Range Collapse Problem

With `max_masd=50.0`, our actual MASD values (0.99–6.83) map to:

| MASD (voxels) | `1 - MASD/50` | Effective range used |
|---------------|---------------|---------------------|
| 0.99 (best) | 0.980 | |
| 1.83 (good) | 0.963 | |
| 6.83 (worst) | 0.863 | |

The normalized MASD component spans only **0.863–0.980** (a 0.117 range), while clDice
spans **0.064–0.898** (a 0.834 range). After 0.5/0.5 weighting:

- MASD contributes: 0.5 * [0.863, 0.980] = **[0.432, 0.490]** — a 0.058 spread
- clDice contributes: 0.5 * [0.064, 0.898] = **[0.032, 0.449]** — a 0.417 spread

**clDice dominates the compound metric ~7:1.** The MASD component is essentially a constant
offset, contributing almost no discriminative power to model ranking. This means
`val_compound_masd_cldice ≈ 0.46 + 0.5 * clDice` in practice.

---

## 3. Literature Review: How Others Combine Heterogeneous Metrics

### 3.1 The Consensus: Do Not Create Compound Scores

**Metrics Reloaded** (Maier-Hein et al., Nature Methods 2024) — the definitive reference:

> "A single metric typically cannot cover the complex requirements of the driving
> biomedical problem."

Their recommendation: report multiple complementary metrics from different families
(overlap, boundary, topology) **separately**. They intentionally provide no guidance on
combining metrics into a single score.

**Reinke et al. (Nature Communications 2024)** — Understanding metric-related pitfalls:

> Some metrics "do not have upper or lower bounds, or the theoretical bounds may not be
> achievable in practice, rendering interpretation difficult."

### 3.2 How Major Challenges Actually Rank

No major challenge uses naive weighted averaging of raw metrics with different scales:

| Challenge | Metrics | Aggregation Method |
|-----------|---------|-------------------|
| **BraTS** (2023) | DSC + HD95, 3 regions | Rank-then-aggregate (mean of 6 ranks) |
| **KiTS23** | Dice + Surface Dice (NSD) | Rank-then-aggregate (mean of 2 ranks) |
| **FLARE** (2021–23) | DSC + NSD + efficiency | Rank-then-aggregate per case per metric |
| **TopCoW** (2023) | Dice + clDice + HD95 + Betti + F1 | Rank-then-aggregate (equal weight per rank) |
| **AMOS** (2022) | DSC + NSD, 15 organs | Rank-then-aggregate (mean of 2 ranks) |
| **MSD** (2022) | DSC + NSD, 10 tasks | Wilcoxon significance ranking |
| **CHAOS** | Dice + RAVD + ASSD + MSSD | **Linear normalization to [0,100]** with inter-rater thresholds |
| **CellMap** (Janelia) | Accuracy + HD | **Exponential decay** `1.01^(-HD/voxel_size)`, geometric mean |

**Key finding:** The dominant paradigm is **rank-then-aggregate**, which sidesteps the
normalization problem entirely by converting all metrics to ordinal ranks before combining.

### 3.3 When Scalarization Is Used

The two challenges that do normalize raw metric values use principled approaches:

1. **CHAOS** — Linear min-max normalization using thresholds derived from **inter-rater
   variability** (not hardcoded constants). The normalization bounds are empirically
   justified by how much human annotators disagree.

2. **CellMap** — Exponential decay `1.01^(-HD / ||voxel_size||)` with geometric mean.
   The decay constant is derived from voxel geometry (dataset-adaptive), and the geometric
   mean penalizes solutions where one metric is excellent but the other is poor.

### 3.4 Vascular Segmentation Specifically

- **clDice** (Shit et al., CVPR 2021): Reports Dice, clDice, precision, recall, Hausdorff,
  and graph similarity **all separately**. No compound metric proposed.
- **cbDice** (Lin et al., MICCAI 2024): Reports Dice, clDice, NSD, Betti error, Betti
  matching error **all separately**.
- **TopCoW** (2023–24): The most relevant challenge for cerebrovascular segmentation.
  Uses rank-then-aggregate across Dice + clDice + HD95 + Betti error.

**No paper in the vascular segmentation literature proposes a single compound score
combining clDice + MASD/ASSD.** Our `val_compound_masd_cldice` is a custom construct
without direct precedent.

---

## 4. Problems with Our Current Approach

### Problem 1: Hardcoded `max_masd=50.0` is Unjustified

The value 50.0 is not derived from:
- Image geometry (the diagonal of a 512x512x110 volume is ~533 voxels)
- Inter-rater variability (no multi-annotator study exists for MiniVess)
- Observed data distribution (actual MASD range is 0.99–6.83)
- Literature precedent

It is an arbitrary constant that happens to be ~7x larger than our worst observed value,
causing severe range collapse.

### Problem 2: MASD Component Has Almost No Discriminative Power

As shown in Section 2, the MASD component contributes a 0.058 spread while clDice
contributes a 0.417 spread. The compound metric is effectively a clDice proxy with a
near-constant MASD offset.

### Problem 3: Linear Normalization is Fragile

Linear `1 - d/d_max` has known problems:
- Sensitive to the choice of `d_max` (our entire analysis)
- No graceful degradation for outliers (cliff-edge at `d_max`, clamped to 0)
- Wastes discriminative range when actual values are far from `d_max`

### Problem 4: Equal Weighting is Not Justified

For vascular segmentation, **topology preservation (clDice) is arguably more important
than mean boundary distance (MASD)**. Shit et al. (CVPR 2021) emphasize that topology is
"the most important characteristic" of tubular structures. Equal 0.5/0.5 weighting
implies these are equally important, which may not reflect clinical reality.

---

## 5. Recommended Alternatives (Ranked)

### Option A: Replace MASD with NSD (Recommended for Academic Paper)

**Replace the unbounded MASD with Normalized Surface Distance (NSD)**, which is:
- Already bounded [0, 1] (higher is better)
- Used by MSD, KiTS23, FLARE, AMOS — the dominant boundary metric in modern challenges
- Available in MONAI: `monai.metrics.SurfaceDiceMetric`
- Has a clinically interpretable tolerance parameter `tau`

**New compound:** `val_compound_nsd_cldice = 0.5 * NSD(tau) + 0.5 * clDice`

Both components are in [0, 1] and higher-is-better — no normalization needed. The
tolerance `tau` should be set based on voxel spacing (e.g., `tau = 2 * median_spacing`).

**Pros:** Eliminates the normalization problem entirely. Aligns with MetricsReloaded
recommendations. Both metrics bounded and comparable.

**Cons:** Requires computing NSD (surface distance transform), adds `tau` parameter.

### Option B: Data-Adaptive Normalization (Pragmatic Fix)

Keep MASD but replace `max_masd=50.0` with a data-derived bound:

```python
# Option B1: Use observed 99th percentile
max_masd = np.percentile(all_masd_values, 99)  # e.g., ~7.0

# Option B2: Use image diagonal (physical maximum)
max_masd = np.sqrt(512**2 + 512**2 + max_z**2)  # ~533 voxels

# Option B3: Exponential decay (CellMap-inspired)
norm_masd = np.exp(-masd / median_masd)  # median ~2.5 voxels
```

**Pros:** Quick fix, no new metrics to compute, preserves existing results for comparison.

**Cons:** Still ad-hoc, still requires justifying the bound choice.

### Option C: Rank-Then-Aggregate (Most Principled)

For model selection, rank all models on each metric independently, then average ranks:

```python
rank_cldice = models.sort_values("cldice", ascending=False).rank()
rank_masd = models.sort_values("masd", ascending=True).rank()
rank_dice = models.sort_values("dice", ascending=False).rank()
composite_rank = (rank_cldice + rank_masd + rank_dice) / 3
```

**Pros:** Nonparametric. No normalization needed. Used by TopCoW, BraTS, KiTS, FLARE.
Statistically robust (Maier-Hein et al. 2018).

**Cons:** Produces ranks not scores — not directly loggable as a per-epoch metric.
Only works for comparing multiple models (not meaningful for a single model).

### Option D: Geometric Mean (CellMap-Inspired)

```python
norm_masd = np.exp(-masd / scale)  # scale = median MASD from calibration
compound = np.sqrt(norm_masd * cldice)  # geometric mean
```

**Pros:** Penalizes solutions where one metric is excellent but the other is poor (unlike
arithmetic mean which allows trading off). Exponential decay handles outliers gracefully.

**Cons:** Geometric mean = 0 if either component = 0 (harsh for failed cases).

---

## 6. Recommendation for MinIVess

### For Automated Model Selection (Champion Tagging)

Use **Option C (rank-then-aggregate)** in the analysis flow. This is what TopCoW does for
cerebrovascular segmentation — the closest challenge to our problem domain. The champion
tagger should rank models on DSC, clDice, and MASD independently, then average ranks.

### For Per-Epoch Tracking During Training

Use **Option A (NSD + clDice)** or **Option B3 (exponential decay)** as the logged metric.
This gives a meaningful per-epoch scalar for learning curve visualization.

If keeping the current linear normalization as a transitional measure, at minimum fix the
`max_masd` to use the image diagonal (`~533` voxels) or observed 99th percentile (`~7.0`
voxels) rather than the arbitrary 50.0.

### For Academic Reporting

Report **all metrics separately** (DSC, clDice, NSD, MASD) following Metrics Reloaded
guidelines. Use rank-then-aggregate for the summary ranking table. Note the compound metric
as a practical convenience for MLOps model selection, not as a primary evaluation metric.

---

## 7. Sensitivity Analysis: Rankings DIVERGE

We computed model rankings under four normalization schemes using actual experiment data.
**Rankings are NOT concordant — the normalization choice materially affects which model
wins.**

### Full-Width DynUNet (4 losses, 3 folds each, all completed)

```
Loss               DSC    clDice   MASD  | Cur(50)   Fix(7)  ExpDecay  clDice
─────────────────────────────────────────┼───────────────────────────────────
dice_ce           0.900   0.821   0.988  |  0.9006   0.8399   0.7473   0.821
cbdice            0.870   0.784   1.914  |  0.8729   0.7553   0.6245   0.784
dice_ce_cldice    0.852   0.898   1.891  |  0.9301   0.8139   0.6837   0.898
cbdice_cldice     0.874   0.895   1.831  |  0.9292   0.8167   0.6879   0.895
```

### Rankings (1 = best)

| Method | #1 | #2 | #3 | #4 |
|--------|----|----|----|----|
| **Current (max=50)** | dice_ce_cldice | cbdice_cldice | dice_ce | cbdice |
| **Fixed (max=7)** | **dice_ce** | cbdice_cldice | dice_ce_cldice | cbdice |
| **Exp decay** | **dice_ce** | cbdice_cldice | dice_ce_cldice | cbdice |
| **clDice only** | dice_ce_cldice | cbdice_cldice | dice_ce | cbdice |
| **Rank-aggregate** | **dice_ce** (1.67) | cbdice_cldice (2.00) | dice_ce_cldice (2.67) | cbdice (3.67) |

### Key Finding: The Winner Changes

- **Current compound (max=50):** `dice_ce_cldice` wins — because MASD component is nearly
  constant, so compound ≈ clDice, and `dice_ce_cldice` has the best clDice.
- **Data-adaptive normalization (max=7, exp decay):** `dice_ce` wins — because it has by
  far the best MASD (0.988 vs ~1.9), and with proper normalization this advantage becomes
  visible.
- **Rank-then-aggregate (DSC + clDice + MASD):** `dice_ce` wins (mean rank 1.67) — it
  ranks #1 on DSC, #1 on MASD, and only #3 on clDice.

**The current compound metric is effectively clDice-only due to range collapse.**
Fixing the normalization or using rank-then-aggregate would change the champion from
`dice_ce_cldice` to `dice_ce`. This is a material difference for the academic paper.

### Interpretation

`dice_ce` achieves the best DSC (0.900) and best MASD (0.988) but mediocre clDice (0.821).
`dice_ce_cldice` achieves the best clDice (0.898) but worst DSC (0.852) and poor MASD (1.891).

The "right" champion depends on what matters clinically:
- If topology preservation is paramount → `dice_ce_cldice` (clDice champion)
- If overall accuracy matters → `dice_ce` (DSC + boundary champion)
- If balanced performance matters → `cbdice_cldice` (rank #2 in nearly all schemes)

**This is exactly why Metrics Reloaded recommends reporting metrics separately.**

---

## 8. Impact Assessment

### What Changes If We Fix This

| Component | Impact |
|-----------|--------|
| `validation_metrics.py` | Update `normalize_masd` or add NSD-based compound |
| Champion tagger | Minor — already uses `maximize=True` on the compound |
| Analysis flow | Add rank-then-aggregate as alternative ranking |
| Training runs | No impact — metric is computed post-hoc from DSC, clDice, MASD |
| Existing results | Recomputable from stored per-fold metrics |
| Academic paper | Must report individual metrics separately regardless |

### What Does NOT Change

- Training loop (loss functions are separate from evaluation metrics)
- Per-epoch val_dice, val_cldice, val_masd logging (these are correct and unaffected)
- The relative ranking of loss functions is likely stable (sensitivity analysis needed)

---

## 9. Key References

1. **Maier-Hein et al. (2024)** "Metrics Reloaded: Recommendations for image analysis
   validation." *Nature Methods* 21:195-212.
   — The definitive reference. Recommends separate reporting, not compound scores.

2. **Maier-Hein et al. (2018)** "Why rankings of biomedical image analysis competitions
   should be interpreted with care." *Nature Communications* 9:5217.
   — Shows rankings are fragile to aggregation method. Rank-then-aggregate recommended.

3. **Shit et al. (2021)** "clDice — a Novel Topology-Preserving Loss Function for Tubular
   Structure Segmentation." *CVPR 2021*.
   — Original clDice paper. Reports all metrics separately. No compound metric proposed.

4. **Antonelli et al. (2022)** "The Medical Segmentation Decathlon." *Nature Communications*
   13:4128.
   — Uses DSC + NSD (both bounded [0,1]) with Wilcoxon significance ranking.

5. **Yang et al. (2024)** "TopCoW: Topology-Aware Anatomical Segmentation of the Circle of
   Willis." *arXiv:2312.17670*.
   — Most relevant challenge: cerebrovascular segmentation. Rank-then-aggregate across
   Dice + clDice + HD95 + Betti error.

6. **Kavur et al. (2021)** "CHAOS Challenge: Combined (CT-MR) Healthy Abdominal Organ
   Segmentation." *Medical Image Analysis* 69:101950.
   — Linear normalization using inter-rater-derived thresholds, not hardcoded constants.

7. **Lin et al. (2024)** "Centerline Boundary Dice Loss for Vascular Segmentation."
   *MICCAI 2024*.
   — cbDice paper. Reports 5 metrics separately for vascular segmentation evaluation.

8. **Nikolov et al. (2021)** "Clinically applicable segmentation of head and neck anatomy
   for radiotherapy." *Journal of Medical Internet Research* 23(7):e26151.
   — NSD definition and clinical tolerance parameter.

9. **Reinke et al. (2024)** "Understanding metric-related pitfalls in image analysis
   validation." *Nature Communications*.
   — Documents pitfalls of unbounded metrics and aggregation schemes.

10. **Stucki et al. (2024)** "Pitfalls of Topology-Aware Image Segmentation."
    *arXiv:2412.14619*.
    — Shows topology metric rankings can be fragile. Recommends disentangled reporting.

---

## 10. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-26 | Created `val_compound_masd_cldice` with `max_masd=50.0` | Needed single scalar for per-epoch tracking and champion selection |
| 2026-02-28 | Identified range collapse problem | MASD component spans 0.058 vs clDice 0.417 — effectively clDice-only |
| 2026-02-28 | Literature review completed | No precedent for MASD+clDice compound; challenges use rank-aggregate or NSD |
| TBD | Sensitivity analysis | Verify whether rankings actually change before modifying |
| TBD | Adopt recommendation | Based on sensitivity analysis results |
