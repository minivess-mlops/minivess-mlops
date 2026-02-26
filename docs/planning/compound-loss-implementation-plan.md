# Compound Loss & Multi-Metric Improvement Plan

> **Created:** 2026-02-26
> **Branch:** `feat/experiment-evaluation`
> **Status:** Research complete, implementation pending
> **Depends on:** Multi-metric tracking (Phases 1-3, DONE), DynUNet loss variation experiment (DONE)

---

## 0. User Prompts (Verbatim)

### Prompt 1 — Drop warp, add compound loss, Hausdorff research

> Well let's drop warp altogether from the list of loss functions then if it is total garbage! And let's replace that with a compound loss that is 0.5*cbdice + 0.5*dice_ce_cldice. Could you plan how to define this as production-grade as possible to /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/compound-loss-implementation-plan.md . Should we create some special CompoundLoss Class or what you prefer, is Monai using any compound losses https://github.com/Project-MONAI/MONAI https://fastmonai.no/vision_loss_functions.html https://github.com/Project-MONAI/MONAI/issues/3694 https://monai.readthedocs.io/en/1.4.0/losses.html , see especially: https://monai.readthedocs.io/en/1.4.0/losses.html#loss-wrappers? What do you think? how to implement this? After planning with reviewer agents, let's implement with the self-learning TDD skill, and let's start training the same 3-fold training run so that we have 4 loss types trained with this compound loss replacing "warp". I don't have huge trust for Dice CE as the loss as we have a micture of think and thick vasculature in the same stack (or can happen, see screenshot). and how about adding validation/test sliding window inference some Hausdorff variant at least as it measures the distance to the boundary, right? see e.g. /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/Metrics for evaluating 3D medical image segmentation_ analysis, selection, and tool _ BMC Medical Imaging _ Springer Nature Link.html (https://github.com/Visceral-Project/EvaluateSegmentation): "The HD is generally sensitive to outliers. Because noise and outliers are common in medical segmentations, it is not recommended to use the HD directly [8, 40]. However, the quantile method proposed by Huttenlocher et al. [41] is one way to handle outliers. According to the Hausdorff quantile method, the HD is defined to be the qth quantile of distances instead of the maximum, so that possible outliers are excluded, where q is selected depending on the application and the nature of the measured point sets. The Average Distance, or the Average Hausdorff Distance (AVD), is the HD averaged over all points. The AVD is known to be stable and less sensitive to outliers than the HD." . Hausdorff would be quite heavy to use as a loss, or are there some performance-optimized versions published after that rather old https://github.com/Visceral-Project/EvaluateSegmentation ? https://github.com/mavillan/py-hausdorff ? Add this as well https://arxiv.org/abs/2302.03868v3? /home/petteri/Dropbox/github-personal/sci-llm-writer/biblio/biblio-vascular/vascular-tmp/A novel sub-differentiable hausdorff loss combined with BCE for MRI brain tumor segmentation using UNet variants _ Scientific Reports.html any good? Do deep exploration of the utility, computational feasibility of using these? And how come we did not try to directly use the 2 losses recommended by Metrics Reloaded as the 2 best metrics for tubular volumes? :o Why only DICE variants? See /home/petteri/Dropbox/github-personal/minivess-mlops/docs/MetricsReloaded.html and save this prompt of mine verbatim to the created plan, and start planning and researching with subagents!

### Prompt 2 — MASD as loss?

> I mean clDice we used but not MASD which then replaces those Hausdorff variants right? "Your selected metrics Overlap-based Metric: Center line Dice Similarity Coefficient (clDice) Boundary-based Metric: Mean Average Surface Distance (MASD)" why not add MASD as loss, and then another compound loss with 0.5*clDice + 0.5*MASD?

### Prompt 3 — MASD differentiability?

> is MASD differentiable or why was not it used? Do deep literature research in academic format on the losses again and on the compound losses and go beyond this research /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/experiment-planning-and-metrics-prompt.md (output stored where?)

### Prompt 4 — Multi-metric bestness criteria & compound permutations

> Well our multi-metric validation framework (you still remember what this unorthodox construct was? getting multiple best models on a single training run) should have then 0.5*MASD+0.5*clDice as the default metric that evaluates then the best model, with then other "bestness criteria" such as MASD alone, clDice alone, cbDice alone, clDice+cbDice. In general having the compound metric and loss involving some Dice variant AND some surface variant, and figuring out multiple permutations then. The 100 epochs with this model capacity is still quite feasible to run even on my desktop GPU for the paper, 1-2 days of GPU time is totally fine without massive token use. So update the create .md plan with insights from these. And when you are done with the comprehensive background research, let's create an executable plan /home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/loss-metric-improvement-implementation.xml without the background research, only the info needed for actual implementation (the recommendation from the report)

---

## 1. Context — First Training Run Results

4 losses × 3 folds × 100 epochs on DynUNet with MiniVess (70 volumes, 3-fold CV, gpu_low profile):

| Loss | DSC (mean±std) | clDice (centreline_dsc) | MASD | Notes |
|------|---------------|------------------------|------|-------|
| **dice_ce** | **0.843** | 0.856 | — | Best overall overlap |
| **cbdice** | 0.813 | 0.876 | — | Good centerline awareness |
| **dice_ce_cldice** | 0.741 | **0.904** | — | Best centreline topology preservation |
| **warp** | 0.015 | — | — | **FAILED** — near-empty predictions |

**Key finding:** warp loss completely failed (loss converged to 0.034 but DSC ~0.015 = nearly empty masks). Must be dropped and replaced.

---

## 2. Research Findings

### 2.1 Why Only Dice Variants as Losses?

MetricsReloaded recommends two metric categories for tubular structures:
1. **Overlap-based:** Centre Line Dice (clDice) — already used as SoftclDiceLoss
2. **Boundary-based:** Mean Average Surface Distance (MASD) — only used as evaluation metric

The gap: MASD was not used as a training loss because **MASD is NOT differentiable**.

### 2.2 Why MASD is NOT Differentiable

MASD requires four operations that break the gradient chain:

1. **Hard thresholding** (p > 0.5 → {0,1}): Step function has zero gradient almost everywhere
2. **Surface extraction** (boundary voxels): Boolean set operation, not differentiable
3. **Nearest-neighbor search** (argmin over all surface points): Piecewise constant, zero gradient
4. **Discrete cardinality** (|S|): Counting discrete surface points, undefined gradient

Mathematically: `MASD(P, G) = (1/|S_P|)·Σ_{p∈S_P} min_{g∈S_G} ‖p−g‖ + (1/|S_G|)·Σ_{g∈S_G} min_{p∈S_P} ‖g−p‖`

Every component (`S_P`, `S_G`, `|·|`, `min`, boundary extraction) requires discrete/non-smooth operations.

### 2.3 Differentiable Surface Distance Proxies

| Method | Citation | Approach | Bounded? | Patch-safe? | VRAM overhead |
|--------|----------|----------|----------|------------|---------------|
| **Boundary Loss** | Kervadec et al. (2019, MIDL) | Pre-computed signed EDT as per-voxel weights | No (unbounded) | Yes | ~10% |
| **Generalized Surface Loss (GSL)** | Celaya et al. (2023) | Pre-computed EDT, sigmoid normalization | Yes [0,1] | Yes | ~15% |
| **HausdorffDTLoss** | Karimi & Salcudean (2019) | Distance transform of predictions | Yes | **No** — needs full volume DTM | ~30% |
| **Active Contour Loss** | Chen et al. (2019) | Level-set energy approximation | Yes | Yes | ~5% |
| **SkelRecall** | Kirchhoff et al. (2024, DKFZ, ECCV) | Skeleton recall via morphological ops | Yes | Yes | ~2% VRAM, ~8% time |

**Recommendation for this project:** **Boundary Loss** (Kervadec et al. 2019) is the best MASD proxy because:
- Well-established (1200+ citations), widely validated
- Patch-safe (works with our RandCropByPosNegLabeld pipeline)
- Pre-computed EDT per-sample is fast (one-time cost in transform pipeline)
- Direct proxy for MASD (average distance to boundary)
- Already available in MONAI: not built-in, but trivially implemented (~30 LOC)

**Generalized Surface Loss** (Celaya 2023) is the best bounded alternative if the unbounded nature of Boundary Loss causes training instability.

### 2.4 MONAI Compound Loss Architecture

MONAI has **no generic CompoundLoss class**. All compound losses (DiceCELoss, DiceFocalLoss, etc.) are hard-coded pairs using the lambda-weighted pattern:

```python
class DiceCELoss(nn.Module):
    def forward(self, input, target):
        dice = self.dice(input, target)
        ce = self.ce(input, target)
        return self.lambda_dice * dice + self.lambda_ce * ce
```

This is the idiomatic MONAI pattern. Our existing `VesselCompoundLoss` already follows it. **Recommendation: follow the same pattern** for new compounds — no generic wrapper needed.

### 2.5 What nnU-Net and SOTA Use

- **nnU-Net** (Isensee et al.): DiceCE only. All surface metrics are evaluation-only.
- **MICCAI 2024 winners**: DiceCE + topology-aware losses (clDice, SkelRecall). Surface losses used as secondary terms.
- **cbDice** (Shi et al. MICCAI 2024): Already in our repo — combines centerline topology with boundary distance awareness.

### 2.6 Existing Evaluation Metrics (Already Implemented)

Our `EvaluationRunner` (evaluation.py) already computes both MetricsReloaded-recommended metrics:

| Metric | Type | Already computed? |
|--------|------|-------------------|
| `centreline_dsc` (clDice) | Overlap | Yes — PRIMARY |
| `dsc` (Dice) | Overlap | Yes — PRIMARY |
| `measured_masd` (MASD) | Boundary | Yes — PRIMARY |
| `measured_hausdorff_distance_perc` (HD95) | Boundary | Yes — EXPENSIVE |
| `normalised_surface_distance` (NSD) | Boundary | Yes — EXPENSIVE |

---

## 3. Recommendations

### 3.1 New Loss Function — Replace Warp

Replace `warp` in the experiment config with **`cbdice_cldice`**: a compound loss combining cbDice and dice_ce_cldice.

```python
loss = 0.5 * cbdice(logits, labels) + 0.5 * dice_ce_cldice(logits, labels)
```

Rationale: cbDice captures boundary-aware centerline topology (diameter-sensitive), while dice_ce_cldice captures skeleton-following topology. Together they provide complementary topology supervision without needing a surface distance proxy.

### 3.2 Boundary Loss as Optional Add-On

Implement Boundary Loss (Kervadec 2019) for future experiments:
- Pre-compute signed EDT during data loading (one-time in transform pipeline)
- Use as secondary term: `λ_region * DiceCE + λ_boundary * BoundaryLoss`
- Warmup schedule: start with region-only, linearly increase boundary weight
- **Not included in the first re-training round** to keep the experiment matrix manageable

### 3.3 Multi-Metric "Bestness Criteria" for Validation

Update the multi-metric tracking YAML to include compound validation metrics. The framework already supports arbitrary metrics — we just need to compute more of them during validation.

**Default compound metric:** `0.5 * (1 - normalized_masd) + 0.5 * clDice`

Note: MASD is in distance units (lower=better), so we invert/normalize it for compound computation. clDice is in [0,1] (higher=better).

**All "bestness criteria" to track:**

| Criterion | Formula | Direction | Purpose |
|-----------|---------|-----------|---------|
| `val_compound_masd_cldice` | 0.5*(1-norm_masd) + 0.5*clDice | maximize | **Default** — overlap+boundary |
| `val_loss` | training loss value | minimize | Convergence monitoring |
| `val_dice` | DSC | maximize | Standard overlap |
| `val_cldice` | centreline_dsc | maximize | Topology-aware overlap |
| `val_masd` | measured_masd | minimize | Boundary accuracy |
| `val_cbdice_proxy` | cbDice metric (if feasible) | maximize | Diameter-aware topology |
| `val_compound_cldice_cbdice` | 0.5*clDice + 0.5*cbDice | maximize | Pure topology compound |

### 3.4 Updated Experiment Configuration

```yaml
experiment_name: dynunet_loss_variation_v2
losses:
  - dice_ce          # Baseline (nnU-Net standard)
  - cbdice           # Centerline-boundary Dice
  - dice_ce_cldice   # DiceCE + soft clDice
  - cbdice_cldice    # NEW: compound (replaces warp)
compute: gpu_low
num_folds: 3
max_epochs: 100
checkpoint:
  tracked_metrics:
    - name: val_loss
      direction: minimize
      patience: 30
    - name: val_dice
      direction: maximize
      patience: 30
    - name: val_cldice
      direction: maximize
      patience: 30
    - name: val_masd
      direction: minimize
      patience: 30
    - name: val_compound_masd_cldice
      direction: maximize
      patience: 30
  early_stopping_strategy: all
  primary_metric: val_compound_masd_cldice
  min_epochs: 10
```

### 3.5 Training Plan

4 losses × 3 folds × 100 epochs = 12 training runs on RTX 2070 Super (8GB)

Estimated time: ~35 sec/epoch × 100 epochs × 12 runs = ~11.7 hours (~0.5 days)

---

## 4. Implementation Phases

### Phase 1: Compound Loss Class (~10 tests)
- Add `CbDiceClDiceLoss` to `loss_functions.py` following MONAI lambda-weighted pattern
- Register as `"cbdice_cldice"` in `build_loss_function()` factory
- Unit tests: forward pass, gradient flow, NaN-free, weight balancing

### Phase 2: Validation Metric Expansion (~8 tests)
- Add `val_cldice` and `val_masd` computation during validation in trainer
- Add compound metric computation: `val_compound_masd_cldice`
- Wire into MultiMetricTracker via YAML config
- Unit tests: metric computation, normalization, compound formula

### Phase 3: Experiment Config Update (~4 tests)
- Update `configs/experiments/dynunet_losses.yaml` v2
- Replace `warp` with `cbdice_cldice`
- Add expanded tracked_metrics with compound
- Test: config parsing, loss factory resolution, metric tracker build

### Phase 4: Training Run
- Run via `run_experiment.py --config configs/experiments/dynunet_losses.yaml`
- Verify all 7 metric checkpoints saved per fold
- Monitor with system_monitor.py
- Estimated wall time: ~12 hours

---

## 5. Future Work (Not This Round)

- **Boundary Loss** (Kervadec 2019): Implement with EDT pre-computation in transform pipeline
- **Generalized Surface Loss** (Celaya 2023): Bounded alternative to Boundary Loss
- **SkelRecall** (Kirchhoff et al. ECCV 2024): +2% VRAM, +8% training time — promising for centerline recall
- **Ensemble construction**: Load best-by-X checkpoints, create multi-model ensembles
- **Compound loss grid**: Systematic sweep over λ weights for compound losses
