# Loss Functions & Metrics: Double-Check Report

> **Date**: 2026-02-27
> **Context**: Mid-training review during `dynunet_loss_variation_v2` experiment
> (4 losses x 3 folds x 100 epochs on MiniVess dataset, RTX 2070 Super 8 GB)
> **Reviewed by**: Codebase explorer + web research agents, cross-verified

---

## 1. Current Experiment Configuration

### 1.1 Losses Used in v2 Training

| Loss | Implementation | Components | Rationale |
|------|---------------|------------|-----------|
| `dice_ce` | `monai.losses.DiceCELoss` | 50% Dice + 50% CE | nnU-Net standard baseline (Isensee et al., 2021) |
| `cbdice` | Vendored `CenterlineBoundaryDiceLoss` | 40% Dice + 30% centerline + 30% boundary | Diameter-aware vessel loss (Shi et al., 2024) |
| `dice_ce_cldice` | Custom `VesselCompoundLoss` | 50% DiceCE + 50% SoftclDice | Region + skeleton compound |
| `cbdice_cldice` | Custom `CbDiceClDiceLoss` | 50% cbDice + 50% VesselCompoundLoss | Dual topology (replaces failed `warp`) |

### 1.2 Tracked Validation Metrics (6 best-model checkpoints per fold)

| Metric | Source | Purpose |
|--------|--------|---------|
| `val_loss` | Training loss on val set | Standard convergence monitor |
| `val_dice` | MONAI `DiceMetric` | Volumetric overlap (MetricsReloaded) |
| `val_f1_foreground` | `torchmetrics.F1Score` | Per-voxel classification quality |
| `val_cldice` | MetricsReloaded `centreline_dsc` | Skeleton/topology preservation |
| `val_masd` | MetricsReloaded `measured_masd` | Surface distance accuracy |
| `val_compound_masd_cldice` | Custom: `0.5*(1 - masd/50) + 0.5*clDice` | **Primary**: topology + surface balance |

### 1.3 Interim Training Results (10/12 folds complete)

| Loss | Mean DSC | Mean clDice | Mean MASD | Notes |
|------|----------|-------------|-----------|-------|
| `dice_ce` | **0.824** | 0.832 | **1.677** | Best overlap and surface accuracy |
| `cbdice` | 0.767 | 0.799 | 2.125 | Middle ground, no clear win |
| `dice_ce_cldice` | 0.736 | **0.904** | 1.960 | Best topology, -8.8% DSC tradeoff |
| `cbdice_cldice` | 0.755* | 0.908* | 2.462* | *fold 0 only so far |

**Key observation**: Clear topology-accuracy tradeoff. Losses with clDice component
achieve 0.90+ clDice but sacrifice ~9% DSC. The compound primary metric
(`val_compound_masd_cldice`) is designed to find the optimal balance.

---

## 2. Why Is Pure clDice Not in the v2 Experiment?

### 2.1 It IS Implemented

Pure clDice (`SoftclDiceLoss`) is available in the loss factory:

```python
# src/minivess/pipeline/loss_functions.py, line 348-349
if loss_name == "cldice":
    return SoftclDiceLoss(smooth=1e-5, iter_=3)
```

It was explicitly included in earlier ablation planning (`docs/planning/dynunet-ablation-plan.md`,
line 23) and tested in `tests/v2/unit/test_dynunet_ablation.py`.

### 2.2 Why Excluded from v2

The v2 experiment was designed to compare **4 losses** (not all 12 available), keeping
the training matrix manageable at 4 x 3 x 100 = 1,200 epochs (~25 hours on one GPU).
The selection rationale from `compound-loss-implementation-plan.md` Section 3.1:

1. **`dice_ce`** — Standard baseline (required for comparison)
2. **`cbdice`** — Topology-aware via centerline+boundary, no clDice component (control)
3. **`dice_ce_cldice`** — Tests what adding clDice to the baseline does
4. **`cbdice_cldice`** — Replaces failed `warp` loss; maximum topology supervision

Pure clDice was **implicitly covered** by `dice_ce_cldice` (which is 50% clDice) and
the experiment was designed to test compound losses vs. baselines, not to ablate
individual components.

### 2.3 Known Risks of Standalone clDice

| Risk | Severity | Evidence |
|------|----------|----------|
| **Empty prediction collapse** | High | If model predicts all-background, skeleton is empty; clDice numerator/denominator both → 0. Smoothing prevents NaN but gradients become vanishingly weak. Recovery from empty-prediction state is difficult. |
| **Accuracy loss** | Medium | clDice optimizes skeleton overlap, not volumetric overlap. Boundaries may be inaccurate even when topology is correct. This is by design (Shit et al., 2021). |
| **No region anchor** | High | The original clDice paper (Shit et al., 2021) explicitly combines clDice with Dice or BCE "for stability reasons and to ensure good volumetric segmentation." No ablation of standalone clDice is provided. |
| **MONAI API issues** | Medium | MONAI Issue #8239: `SoftDiceclDiceLoss` reported to produce zero loss. `SoftclDiceLoss` lacks standard parameters (`sigmoid`, `softmax`, `to_onehot_y`). Our factory works around this with manual preprocessing. |
| **Diameter imbalance** | Low | clDice is skeleton-weighted, so thick and thin vessels contribute equally to the skeleton but large vessels dominate Dice. Without a region-based anchor, thin vessels may be over-represented. |

**Recommendation**: Pure clDice should be tested in a follow-up ablation study, but
always paired with a Dice/CE anchor in compound form for production use.

---

## 3. Why MASD Is Not Used as a Loss

### 3.1 Fundamental Non-Differentiability

MASD computation requires four operations that break the gradient chain:

1. **Hard thresholding**: `p > 0.5 → {0, 1}` — step function with zero gradient everywhere
   except at the threshold, where it is undefined
2. **Surface extraction**: Boolean morphological operation to find boundary voxels — discrete,
   non-differentiable
3. **Nearest-neighbor search**: `argmin` over point-to-point distances — piecewise constant,
   zero gradient almost everywhere
4. **Set cardinality**: `|S|` counts discrete boundary points — integer-valued, non-differentiable

Additionally, MASD is **undefined** when either surface is empty (common during early
training), and it is **unbounded** (measured in physical units), making it hard to balance
against bounded losses like Dice ([0, 1]).

### 3.2 Differentiable Surface Distance Proxies

These approximate MASD-like supervision in a differentiable way:

| Proxy | Paper | In MONAI? | Approach | Key Limitation |
|-------|-------|-----------|----------|----------------|
| **HausdorffDTLoss** | Karimi and Salcudean (2019) | Yes | Distance transform of predictions multiplied with ground truth boundary | GPU memory leak (Issue #7480); **not patch-safe** (needs full-volume DTM) |
| **LogHausdorffDTLoss** | Karimi and Salcudean (2019) | Yes | Log-scaled variant of above | Same limitations as HausdorffDTLoss |
| **Boundary Loss** | Kervadec et al. (2019) | **No** (milestone exists) | `sum(softmax * signed_distance_map)` with precomputed EDT | Unbounded output; sensitive to anisotropic spacing |
| **Generalized Surface Loss** | Celaya et al. (2024) | **No** | Normalized boundary loss bounded in [0, 1] | Requires precomputed DTMs per volume |
| **Regional HD Losses** | Mauget et al. (2025) | **No** | Differentiable erosion-based distance on probability maps | Very new (2025), limited validation |
| **Sub-Differentiable HD** | (2025, Scientific Reports) | **No** | Smooth differentiable HD for brain tumors | Very new, not vessel-specific |

### 3.3 What Would Actually Work for MiniVess

Given our constraints (patch-based training, 8 GB VRAM, 3D volumes):

| Proxy | Patch-Safe? | VRAM Overhead | Implementation Effort | Verdict |
|-------|-------------|---------------|----------------------|---------|
| HausdorffDTLoss | **No** (needs full DTM) | ~15-20% | Low (MONAI native) | Not viable for patch training |
| Boundary Loss | **Yes** (precomputed EDT) | ~10% | Medium (custom dataloader) | **Best candidate for v3** |
| Generalized Surface Loss | **Yes** (precomputed EDT) | ~15% | Medium | Good alternative to Boundary Loss |
| Regional HD | **Yes** (erosion-based) | ~10-15% | High (from scratch) | Too new, insufficient validation |

**Previous plan decision** (`compound-loss-implementation-plan.md` Section 2.3):
Boundary Loss (Kervadec et al., 2019) was identified as the recommended MASD proxy for
future work. It requires pre-computing signed Euclidean distance transform maps in the
dataloader, which is straightforward but was deferred to avoid scope creep in v2.

---

## 4. All Available Losses in the Factory

The loss factory (`src/minivess/pipeline/loss_functions.py`) currently supports 13 losses:

| # | Name | Type | Status | In v2? |
|---|------|------|--------|--------|
| 1 | `dice_ce` | MONAI `DiceCELoss` | Production | Yes |
| 2 | `dice` | MONAI `DiceLoss` | Available | No |
| 3 | `focal` | MONAI `FocalLoss` | Available | No |
| 4 | `cldice` | MONAI `SoftclDiceLoss` | Available | **No** (see Section 2) |
| 5 | `cb_dice` | MONAI `ClassBalancedDiceLoss` | Available | No |
| 6 | `dice_ce_cldice` | Custom `VesselCompoundLoss` | Production | Yes |
| 7 | `cbdice` | Vendored `CenterlineBoundaryDiceLoss` | Production | Yes |
| 8 | `cbdice_cldice` | Custom `CbDiceClDiceLoss` | Production | Yes |
| 9 | `betti` | Custom `BettiLoss` | Exploratory | No |
| 10 | `full_topo` | Custom `TopologyCompoundLoss` | Exploratory | No |
| 11 | `centerline_ce` | Vendored `CenterlineCrossEntropyLoss` | Available | No |
| 12 | `warp` | Vendored `WarpLoss` | **Failed** (DSC ~0.015) | No |
| 13 | `topo` | Vendored `TopoLoss` (CoLeTra) | Available | No |

### 4.1 Losses NOT Yet Implemented (Candidates for v3)

| Candidate | Paper | Why Interesting | Implementation Path |
|-----------|-------|----------------|---------------------|
| **Boundary Loss** | Kervadec et al. (2019) | Differentiable MASD proxy; patch-safe | Precompute signed EDT in dataloader; `sum(softmax * sdt)` |
| **Generalized Surface Loss** | Celaya et al. (2024) | Bounded [0,1] boundary loss | Similar to Boundary Loss but normalized |
| **dice_ce + boundary** | Compound | Region + surface compound | `alpha * DiceCE + (1-alpha) * BoundaryLoss` |
| **dice_ce_cldice + boundary** | Compound | Region + topology + surface | Triple compound for maximum vessel supervision |
| **SkelRecall** | Kirchhoff et al. (2024) | Skeleton recall metric as loss | +2% VRAM, +8% time; ECCV 2024 |

---

## 5. Metrics: What We Track vs. What Exists

### 5.1 MetricsReloaded Alignment

Our metrics align with MetricsReloaded recommendations for binary segmentation:

| Category | MetricsReloaded Recommended | We Track | Gap |
|----------|-----------------------------|----------|-----|
| Overlap | DSC (Dice) | `val_dice` | None |
| Topology | centreline DSC (clDice) | `val_cldice` | None |
| Boundary | MASD | `val_masd` | None |
| Classification | F1 (voxel-level) | `val_f1_foreground` | None |
| Composite | Custom compound | `val_compound_masd_cldice` | Novel (not in MetricsReloaded) |
| Boundary | Hausdorff Distance (HD95) | **Not tracked** | Minor gap |
| Overlap | NSD (Normalized Surface Dice) | **Not tracked** | Minor gap |

### 5.2 Metrics We Could Add

| Metric | MetricsReloaded? | Effort | Value |
|--------|-------------------|--------|-------|
| **HD95** (95th percentile Hausdorff) | Yes | Low (MetricsReloaded has it) | Standard boundary metric, complements MASD |
| **NSD** (Normalized Surface Dice) | Yes | Low | Measures boundary overlap at tolerance |
| **Betti-0 error** | No | Medium | Counts connected component difference |
| **Branch-point F1** | No | High | Skeleton topology correctness |

---

## 6. Recommendations

### 6.1 For Current v2 Experiment (no changes needed)

The 4-loss x 6-metric setup is well-designed:
- **dice_ce** provides the volumetric baseline
- **cbdice** tests diameter-aware supervision without clDice
- **dice_ce_cldice** tests adding topology to the baseline
- **cbdice_cldice** tests maximum topology supervision
- The **compound primary metric** balances the DSC-clDice tradeoff

### 6.2 For v3 Ablation Study

| Priority | Addition | Rationale |
|----------|----------|-----------|
| **P1** | Pure `cldice` (standalone) | Ablation completeness: isolate clDice contribution |
| **P1** | `dice_ce + boundary` compound | Add differentiable surface supervision |
| **P2** | `dice_ce_cldice + boundary` triple | Region + topology + surface — maximum supervision |
| **P2** | HD95 as tracked metric | Standard boundary metric alongside MASD |
| **P3** | Generalized Surface Loss | Bounded alternative to Kervadec boundary loss |
| **P3** | `centerline_ce` (already in factory) | clCE alternative to clDice (Acebes et al., 2024) |

### 6.3 Why the Current Tradeoff Pattern Is Expected

The v2 results showing `dice_ce` winning DSC while `dice_ce_cldice` winning clDice is
the **textbook topology-accuracy tradeoff** documented in Shit et al. (2021), Shi et al.
(2024), and Acebes et al. (2024). This is not a failure — it confirms the losses work
as designed. The compound primary metric exists precisely to navigate this tradeoff.

---

## 7. Key References

- Isensee et al. (2021) — nnU-Net: self-configuring framework (Nature Methods)
- Shit et al. (2021) — clDice: topology-preserving loss for tubular structures (CVPR)
- Shi et al. (2024) — cbDice: centerline boundary Dice (MICCAI)
- Acebes et al. (2024) — clCE: centerline cross-entropy (MICCAI)
- Kervadec et al. (2019) — Boundary loss for unbalanced segmentation (MIDL, runner-up best paper)
- Karimi and Salcudean (2019) — Reducing HD with CNNs (IEEE TMI)
- Celaya et al. (2024) — Generalized Surface Loss for reducing HD (arXiv)
- Mauget et al. (2025) — Regional Hausdorff Distance Losses (MLMI)
- Kirchhoff et al. (2024) — SkelRecall: skeleton recall metric (ECCV)

### MONAI Issues Referenced
- [#8239](https://github.com/Project-MONAI/MONAI/issues/8239) — `SoftDiceclDiceLoss` zero-loss bug
- [#7480](https://github.com/Project-MONAI/MONAI/issues/7480) — `HausdorffDTLoss` GPU memory leak
