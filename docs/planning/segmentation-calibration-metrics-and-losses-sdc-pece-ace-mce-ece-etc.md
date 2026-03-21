# Segmentation Calibration Metrics and Losses: SDC, pECE, ACE, MCE, ECE, BA-ECE, SPACE, Brier, NLL

**Date**: 2026-03-21
**Context**: MinIVess MLOps v2 — methodological validation of aux calibration loss factor
**Scope**: Multi-hypothesis exploration of calibration metrics for the Analysis Flow
and training-loop evaluation

---

## User Prompt (verbatim)

> And on continuing with the methodological validation, as we are doing the aux calib loss,
> what calibration metrics are we actually computing for the validation loss and in the
> Analysis Flow to actually measure if the loss aux improves calibration? Let's create an
> open-ended multi-hypothesis type of report on different calibration metrics (let's just
> implement them all, at least for the Analysis Flow and the actual during training evaluation
> loop can have some computational efficiency optimization and we want to use only fast
> metrics then to compute). See e.g. https://arxiv.org/html/2503.05107v2
> (https://github.com/EagleAdelaide/SDC-Loss) so at least "pixel-wise ECE (pECE)" and
> "Signed Distance Calibration (SDC) Loss" (or can this LOSS be used as a metric?); and
> from https://arxiv.org/abs/2506.03942 (https://github.com/cai4cai/Average-Calibration-Losses)
> Average Calibration Error (ACE) and Maximum Calibration Error (MCE), and use also the
> ECE as baseline. Can you find other metrics from the literature?

---

## Why Calibration Metrics Matter for MinIVess

The factorial design includes `with_aux_calib: [true, false]` as a factor. To measure
whether auxiliary calibration loss ACTUALLY improves calibration, we need calibration
metrics beyond DSC/clDice. Without these metrics, the `with_aux_calib` factor is
unverifiable — we'd be claiming calibration improvement without measuring calibration.

For **vessel segmentation** specifically:
- Thin vessels produce overconfident false positives in background
- Boundary regions are where miscalibration is clinically dangerous
- Spatial calibration (not just global) matters — a model can have good global ECE
  but terrible calibration at vessel boundaries

---

## Metric Taxonomy

### Tier 1: Training-Loop Metrics (FAST — compute every N validation epochs)

| Metric | Abbrev | Time | Bins? | Spatial? | What it Measures |
|--------|--------|------|-------|----------|-----------------|
| [Expected Calibration Error](#11-ece) | ECE | O(N) | Yes (15) | No | Global confidence-accuracy gap |
| [Maximum Calibration Error](#12-mce) | MCE | O(N) | Yes | No | Worst-case bin miscalibration |
| [Brier Score](#21-brier-score) | BS | O(N) | No | Maps | Mean squared prob-vs-truth error |
| [Negative Log-Likelihood](#22-nll) | NLL | O(N) | No | Maps | Already computed as CE loss — free |
| [Boundary Uncertainty Concentration](#34-buc) | BUC | O(N)* | No | Yes | Uncertainty at boundaries vs interior |
| [Class-wise ECE](#15-cece) | CECE | O(2N) | Yes | No | Per-class calibration (C=2 for binary) |
| [Overconfidence Error](#35-oe) | OE | O(N) | Yes | No | Only penalizes overconfident bins |
| [Debiased ECE](#35-d-ece) | D-ECE | O(N) | Yes | No | Bias-corrected ECE (Kumar 2019) |
| [Smooth ECE](#35-smece) | smECE | O(N log N) | No | No | Kernel-smoothed, no bin sensitivity |

\* BUC requires precomputed distance transform (shared with MASD/HD95)

### Tier 2: Analysis-Flow Metrics (COMPREHENSIVE — compute once per model)

| Metric | Abbrev | Time | Bins? | Spatial? | What it Measures |
|--------|--------|------|-------|----------|-----------------|
| [Pixel-wise ECE](#13-pece) | pECE | O(N) | Yes | Partial | ECE + false-positive overconfidence penalty |
| [Average Calibration Error](#14-ace) | ACE | O(N log N) | Adaptive (20) | No | Equal-count bins, all-class computation |
| [Thresholded Adaptive CE](#35-tace) | TACE | O(N log N) | Adaptive | No | ACE with confidence threshold filter |
| [Static Calibration Error](#35-sce) | SCE | O(NC) | Yes | No | Full confidence vector calibration |
| [Boundary-Aware ECE](#31-ba-ece) | BA-ECE | O(N+DT) | Distance bands | Yes | Calibration stratified by distance from boundary |
| [Spatially-Aware Calibration](#32-space) | SPACE | O(N+conv) | No | Yes | Local uncertainty-error overlap |
| [CRPS](#35-crps) | CRPS | O(N) | No | No | Proper scoring rule (generalizes MAE) |
| [Uncertainty-Dice AUC](#35-udauc) | UD-AUC | O(NK) | No | No | Calibration utility (QU-BraTS style) |
| [Per-pixel Brier map](#21-brier-score) | BS-map | O(N) | No | Yes | Spatial heatmap of calibration error |
| [Calibration Shift](#35-calshift) | delta-* | varies | — | — | In-distribution vs OOD calibration gap |
| [Reliability Diagram](#81-reliability-diagram) | — | O(N) | Yes | No | Visual: accuracy vs confidence plot |

---

## 1. Binning-Based Metrics (ECE Family)

### 1.1 ECE

**Expected Calibration Error** — the standard baseline.

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \cdot |\text{acc}(B_m) - \text{conf}(B_m)|$$

Bins predictions into M equal-width intervals over [0,1]. Population-weighted average
of |accuracy - confidence| per bin.

- **Reference**: [Naeini et al. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles." AAAI.](https://ojs.aaai.org/index.php/AAAI/article/view/9602); [Guo et al. (2017). "On Calibration of Modern Neural Networks." ICML.](https://arxiv.org/abs/1706.04599)
- **Implementation**: `torchmetrics.classification.BinaryCalibrationError(n_bins=15, norm='l1')`
- **Known issues**: Bin-count sensitivity, biased toward populated bins, hides spatial patterns
- **Tier**: 1 (training-loop)

### 1.2 MCE

**Maximum Calibration Error** — worst-case bin.

$$\text{MCE} = \max_m |\text{acc}(B_m) - \text{conf}(B_m)|$$

Same binning as ECE, but takes max instead of weighted sum. Critical for safety-sensitive
applications — even one badly-calibrated confidence range is unacceptable.

- **Reference**: [Naeini et al. (2015). AAAI.](https://ojs.aaai.org/index.php/AAAI/article/view/9602)
- **Implementation**: `torchmetrics.classification.BinaryCalibrationError(norm='max')`
- **Tier**: 1 (training-loop) — fast, captures worst case

### 1.3 pECE

**Pixel-wise Expected Calibration Error** — ECE with false-positive penalty.

$$\text{pECE} = \sum_{b=1}^{B} \frac{|(p_b - a_b) + w_{fp} \cdot \text{FPConf}_b| \cdot |\Omega_b|}{|\Omega|}$$

Where FPConf_b is the mean confidence of false-positive pixels in bin b, and w_fp
is a weighting coefficient penalizing overconfident false positives.

- **Reference**: [Li et al. (2025). "We Care Each Pixel: Calibrating on Medical Segmentation Model." arXiv 2503.05107.](https://arxiv.org/abs/2503.05107)
- **Code**: [github.com/EagleAdelaide/SDC-Loss](https://github.com/EagleAdelaide/SDC-Loss) (`metrics.py`)
- **Why important for vessels**: Small vessels are where FP overconfidence is most dangerous.
  Standard ECE doesn't distinguish between FP in background (dangerous) and TP in foreground.
- **Tier**: 2 (analysis-flow) — slightly more complex than ECE

### 1.4 ACE

**Average Calibration Error** — equal-weight bins (not population-weighted).

$$\text{ACE} = \frac{1}{CM} \sum_{c=1}^{C} \sum_{m=1}^{M} |o^c_m - e^c_m|$$

Two key differences from ECE:
1. **Adaptive equal-count (equal-mass) bins** — each bin contains roughly the same number
   of predictions, unlike ECE's equal-width intervals. This is the "A" in ACE.
2. **All-class computation** — sums ECE across ALL classes, treating all bins equally
   (no population weighting).

- **Reference**: [Nixon et al. (2019). "Measuring Calibration in Deep Learning." CVPR-W.](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.pdf)
- **Key insight from arXiv 2506.03942**: Segmentation produces millions of voxel-level
  predictions per image — enough samples per bin to make hard-binned ACE directly
  **differentiable as a training loss** (no soft binning needed).
- **Implementation**: [github.com/cai4cai/Average-Calibration-Losses](https://github.com/cai4cai/Average-Calibration-Losses) (also provides MONAI-compatible handlers)
- **Note**: arXiv 2506.03942 is the journal extension of the MICCAI 2024 paper [arXiv 2403.06759](https://arxiv.org/abs/2403.06759)
- **Tier**: 2 (analysis-flow) — requires sorting for equal-count bins: O(N log N)

### 1.5 CECE

**Class-wise Expected Calibration Error** — per-class ECE averaged.

$$\text{CECE} = \frac{1}{C} \sum_{c=1}^{C} \text{ECE}_c$$

Important for binary segmentation where foreground/background class imbalance is
extreme (vessels are <5% of volume). Standard ECE is dominated by well-calibrated
background predictions.

- **Reference**: [Kull et al. (2019). "Beyond Temperature Scaling." NeurIPS.](https://arxiv.org/abs/1910.12656)
- **Implementation**: [SDC-Loss repo](https://github.com/EagleAdelaide/SDC-Loss) (`metrics.py`)
- **Tier**: 2 (analysis-flow)

---

## 2. Proper Scoring Rules (Bin-Free)

### 2.1 Brier Score

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$

Mean squared error between predicted probability and ground truth. Bin-free, strictly
proper scoring rule. Lower is better.

**Spatial Brier map**: Compute per-pixel to produce a heatmap showing WHERE calibration
is worst. This is trivial — just `(p - y)^2` per voxel.

**Decomposition**: BS = Reliability + Resolution - Uncertainty. The reliability term
is closely related to ECE.

- **Reference**: [Brier (1950). Monthly Weather Review.](https://en.wikipedia.org/wiki/Brier_score)
- **Implementation**: `sklearn.metrics.brier_score_loss` (binary); per-pixel: `(pred_prob - label)**2`
- **Tier**: 1 (training-loop) for scalar; 2 (analysis-flow) for spatial map

### 2.2 NLL

**Negative Log-Likelihood** — already computed as cross-entropy loss.

$$\text{NLL} = -\frac{1}{N} \sum_{i=1}^{N} y_i \ln(p_i) + (1-y_i) \ln(1-p_i)$$

Strictly proper scoring rule. Strongly penalizes overconfident wrong predictions.
**Zero extra cost** — already computed during training.

Per-pixel NLL map: `torch.nn.functional.binary_cross_entropy(p, y, reduction='none')`

- **Tier**: 1 (training-loop) — FREE

---

## 3. Spatial / Boundary-Aware Metrics

### 3.1 BA-ECE

**Boundary-Aware Expected Calibration Error** — the MOST relevant metric for vessels.

Groups voxels into K distance bands from the ground-truth boundary. For each band,
computes the gap between mean uncertainty and mean error. Weights bands inversely
proportional to boundary distance.

- **Reference**: [Zeevi et al. (2025). "Spatially-Aware Evaluation of Segmentation Uncertainty." arXiv 2506.16589.](https://arxiv.org/abs/2506.16589)
- **Why critical for vessels**: Boundary regions are where miscalibration matters most.
  A model with good global ECE can have terrible calibration at thin vessel boundaries.
- **Implementation**: Requires `scipy.ndimage.distance_transform_edt` or MONAI equivalent
  (already computed for MASD/HD95 metrics).
- **Tier**: 2 (analysis-flow) — needs distance transform

### 3.2 SPACE

**Spatially-Aware Calibration Error** — local uncertainty-error overlap.

$$\text{SPACE} = \text{mean}(|G_\sigma * U - G_\sigma * E|)$$

Convolves both uncertainty map U and error map E with the same Gaussian kernel,
then compares local averages. Reveals whether the model flags mispredictions in
the right locations.

- **Reference**: [Zeevi et al. (2025). arXiv 2506.16589.](https://arxiv.org/abs/2506.16589)
- **Implementation**: `torch.nn.functional.conv3d` with Gaussian kernel
- **Tier**: 2 (analysis-flow)

### 3.3 Per-pixel Brier / NLL maps

Not separate metrics — just spatial versions of Brier Score and NLL computed per-voxel.
Produce heatmaps showing calibration error distribution across the volume.

### 3.4 BUC

**Boundary Uncertainty Concentration** — simple diagnostic.

$$\text{BUC} = \frac{\bar{U}_{\text{boundary}}}{\bar{U}_{\text{boundary}} + \bar{U}_{\text{interior}}}$$

BUC > 0.5 means uncertainty concentrates at boundaries (desirable for segmentation).

- **Reference**: [Zeevi et al. (2025). arXiv 2506.16589.](https://arxiv.org/abs/2506.16589)
- **Tier**: 1 (training-loop) — fast IF distance transform is precomputed (for MASD/HD95)

---

## 3.5 Additional Metrics (from reviewer feedback)

### CRPS — Continuous Ranked Probability Score

Strictly proper scoring rule generalizing MAE to probabilistic predictions. Used by
the [CURVAS challenge (MICCAI 2024)](https://arxiv.org/abs/2505.08685) as one of three
primary calibration metrics alongside DSC and ECE. Decomposes into calibration + sharpness.
For binary segmentation: O(N), fast. Implementation: `properscoring` or `torchmetrics.regression.CRPS`.
**Tier**: 2 (analysis-flow).

### Uncertainty-Dice AUC (QU-BraTS style)

Compute DSC after removing the top-K% most uncertain voxels for K in [0, 5, 10, ..., 50].
Report AUC of the resulting curve. Measures calibration **utility** — can the uncertainty
map actually improve downstream decisions? From the [QU-BraTS challenge](https://github.com/RagMeh11/QU-BraTS).
**Tier**: 2 (analysis-flow).

### D-ECE — Debiased ECE

[Kumar et al. (2019). "Verified Uncertainty Calibration." NeurIPS (Spotlight).](https://arxiv.org/abs/1909.10155)
Jackknife-based debiased estimator correcting systematic underestimation of binned ECE.
Library: [github.com/p-lambda/verified_calibration](https://github.com/p-lambda/verified_calibration).
**Tier**: 1 (fast — same as ECE plus bias correction).

### OE — Overconfidence Error

[Thulasidasan et al. (2019). "On Mixup Training." NeurIPS.](https://arxiv.org/abs/1905.11001)
Like ECE but only penalizes bins where confidence EXCEEDS accuracy (overconfident).
For medical imaging where overconfidence is the primary safety risk, OE is more clinically
relevant than symmetric ECE.
**Tier**: 1 (fast — same as ECE with a conditional).

### smECE — Smooth ECE

[Blasiok et al. (2024). "Smooth ECE: Principled Reliability Diagrams via Kernel Smoothing." ICLR.](https://arxiv.org/abs/2309.12236)
Replaces binning with RBF kernel smoothing, eliminating bin-count sensitivity entirely.
No hyperparameter tuning needed. Library: `pip install relplot`.
**Tier**: 1 (O(N log N), no hyperparameters).

### SCE — Static Calibration Error

[Nixon et al. (2019). CVPR-W.](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.pdf)
Extends ECE by considering the FULL confidence vector (all classes, not just top-label).
Evaluates calibration across all predictions. Relevant for binary segmentation where
both foreground AND background calibration matter.
**Tier**: 2 (analysis-flow).

### TACE — Thresholded Adaptive Calibration Error

[Nixon et al. (2019). CVPR-W.](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.pdf)
Same as ACE but applies a confidence threshold to discard low-probability predictions
before binning. Filters out near-zero (confident background) predictions that dominate
segmentation. Focuses calibration assessment on the "interesting" confidence range.
**Tier**: 2 (analysis-flow — one-line modification of ACE).

### Calibration Shift (delta metrics)

Not a metric per se, but compute ALL calibration metrics on both MiniVess (in-distribution)
and DeepVess (out-of-distribution). Report delta = metric_DeepVess - metric_MiniVess.
[Miscalibration is exacerbated under distribution shift](https://pmc.ncbi.nlm.nih.gov/articles/PMC10586223/).
**Tier**: 2 (analysis-flow — runs existing metrics on external test set).

---

## 4. Can SDC Loss Be Used as a Metric?

**SDC Loss** (Li et al. 2025, arXiv 2503.05107) has three components:

$$\mathcal{L}_{\text{SDC}} = \mathcal{L}_{\text{CE}} + \alpha \cdot \mathcal{L}_{\text{conf}} + \lambda \cdot \mathcal{L}_{\text{SDF}}$$

- L_CE: standard cross-entropy
- L_conf: confidence calibration term (penalizes overconfident FPs)
- L_SDF: signed distance function alignment (boundary geometry)

**As a metric**: L_conf and L_SDF components CAN be evaluated as metrics:
- L_conf measures overconfidence of false-positive predictions → useful as a metric
- L_SDF measures alignment between predicted boundary and GT SDF → useful as a metric

However, the SDC loss was designed as a training objective, not a metric. The individual
components (L_conf, L_SDF) are more interpretable as metrics than the combined loss.

**Recommendation**: Use **pECE** (from the same paper) as the metric, not SDC loss itself.
The paper's own evaluation uses pECE, CECE, and DSC as metrics, not the loss value.

---

## 5. Calibration Losses for Training (aux_calib factor)

These are the losses that `with_aux_calib: true` should select from:

| Loss | Paper | Key Idea | Code |
|------|-------|----------|------|
| **SDC** | [Li et al. (2025)](https://arxiv.org/abs/2503.05107) | CE + confidence penalty + SDF alignment | [SDC-Loss](https://github.com/EagleAdelaide/SDC-Loss) |
| **hL1-ACE** | [Barfoot et al. (2025)](https://arxiv.org/abs/2506.03942) | Hard-binned ACE as differentiable loss | [Avg-Calib-Losses](https://github.com/cai4cai/Average-Calibration-Losses) |
| **NACL** | [Murugesan et al. (2024)](https://arxiv.org/abs/2401.14487) | Neighbor-aware calibration constraints | — |
| **MbLS** | [Liu et al. (2022)](https://arxiv.org/abs/2111.15430) | Margin-based label smoothing | — |
| **SVLS** | [Islam & Glocker (2021)](https://arxiv.org/abs/2104.05788) | Spatially varying label smoothing | — |
| **Focal Calib** | [Mukhoti et al. (2020)](http://torrvision.com/focal_calibration/) | Focal Loss + calibration term | — |

---

## 6. Implementation Plan for MinIVess

### Training loop (Tier 1 — every val_interval epochs)

Compute these FAST metrics during validation:
1. **ECE** (15 bins) — baseline, via `torchmetrics`
2. **MCE** — worst-case, via `torchmetrics`
3. **Brier Score** — bin-free proper scoring rule
4. **NLL** — free (already computed as CE loss)
5. **BUC** — boundary uncertainty diagnostic

Total extra cost: ~5% of validation time (all O(N) on GPU).

### Analysis flow (Tier 2 — once per model, all metrics)

Compute ALL metrics during the analysis flow evaluation:
1. Everything from Tier 1
2. **pECE** — false-positive penalty (from SDC-Loss repo)
3. **ACE** (20 bins) — equal-weight bins (from Avg-Calib-Losses repo)
4. **CECE** — per-class ECE (from SDC-Loss repo)
5. **BA-ECE** — boundary-aware calibration (custom, uses distance transform)
6. **SPACE** — spatial uncertainty-error overlap (custom, Gaussian conv)
7. **Per-pixel Brier map** — spatial calibration heatmap
8. **Reliability diagram** — visualization per model

### Post-hoc calibration (optional, post-publication)

After analysis, apply post-hoc calibration and re-evaluate:
1. **Temperature Scaling** — global, 1 parameter
2. **Local Temperature Scaling** — per-pixel temperature map
3. **Isotonic Regression** — non-parametric (67% ECE reduction in literature)

---

## 7. How This Validates the aux_calib Factor

The factorial design compares `with_aux_calib: [true, false]`. To demonstrate that
aux calibration loss improves calibration, we need:

1. **Primary hypothesis**: `with_aux_calib=true` reduces ECE, pECE, and ACE
2. **Spatial hypothesis**: `with_aux_calib=true` reduces BA-ECE (boundary calibration)
3. **Safety hypothesis**: `with_aux_calib=true` reduces MCE (worst-case)
4. **Proper scoring**: `with_aux_calib=true` reduces Brier Score
5. **Trade-off hypothesis**: aux_calib may slightly reduce DSC/clDice (accuracy vs calibration trade-off)

The ANOVA in the biostatistics flow should test `with_aux_calib` as a factor for each
calibration metric, producing p-values for the calibration improvement claim.

---

## References

### Primary (cited in user prompt)

1. [Li et al. (2025). "We Care Each Pixel: Calibrating on Medical Segmentation Model." arXiv 2503.05107.](https://arxiv.org/abs/2503.05107) — SDC Loss, pECE
2. [Barfoot et al. (2025). "Average Calibration Losses for Reliable Uncertainty in Medical Image Segmentation." arXiv 2506.03942.](https://arxiv.org/abs/2506.03942) — ACE as loss, hL1-ACE, sL1-ACE

### Foundational ECE

3. [Naeini et al. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles." AAAI.](https://ojs.aaai.org/index.php/AAAI/article/view/9602) — ECE, MCE
4. [Guo et al. (2017). "On Calibration of Modern Neural Networks." ICML.](https://arxiv.org/abs/1706.04599) — Temperature scaling, modern ECE analysis
5. [Nixon et al. (2019). "Measuring Calibration in Deep Learning." CVPR-W.](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.pdf) — ACE, SCE, TACE

### Spatial calibration

6. [Zeevi et al. (2025). "Spatially-Aware Evaluation of Segmentation Uncertainty." arXiv 2506.16589.](https://arxiv.org/abs/2506.16589) — BA-ECE, SPACE, BUC
7. [Mehrtash et al. (2020). "Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation." TMI.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7704933/) — Segment-level entropy

### Calibration losses

8. [Murugesan et al. (2024). "Neighbor-Aware Calibration of Segmentation Networks with Penalty-Based Constraints." arXiv 2401.14487.](https://arxiv.org/abs/2401.14487) — NACL
9. [Liu et al. (2022). "The Devil is in the Margin: Margin-based Label Smoothing for Network Calibration." CVPR.](https://arxiv.org/abs/2111.15430) — MbLS
10. [Islam & Glocker (2021). "Spatially Varying Label Smoothing." IPMI.](https://arxiv.org/abs/2104.05788) — SVLS
11. [Mukhoti et al. (2020). "Calibrating Deep Neural Networks using Focal Loss." NeurIPS.](http://torrvision.com/focal_calibration/) — Focal calibration
12. [Kull et al. (2019). "Beyond Temperature Scaling." NeurIPS.](https://arxiv.org/abs/1910.12656) — Class-wise ECE (CECE)

### Post-hoc calibration

13. [Ding et al. (2021). "Local Temperature Scaling for Semantic Segmentation." ICCV.](https://arxiv.org/abs/2008.05105) — LTS

### Proper scoring rules

14. [Brier (1950). "Verification of Forecasts Expressed in Terms of Probability." Monthly Weather Review.](https://en.wikipedia.org/wiki/Brier_score)
15. [Widmann et al. (2019). "Calibration Tests in Multi-Class Classification: A Panoramic Study." NeurIPS.](https://arxiv.org/abs/1906.10082) — SKCE

### Code repositories

16. [github.com/EagleAdelaide/SDC-Loss](https://github.com/EagleAdelaide/SDC-Loss) — pECE, CECE, SDC loss
17. [github.com/cai4cai/Average-Calibration-Losses](https://github.com/cai4cai/Average-Calibration-Losses) — ACE, hL1-ACE, sL1-ACE
18. [TorchMetrics CalibrationError](https://lightning.ai/docs/torchmetrics/stable/classification/calibration_error.html) — ECE, MCE, RMSCE
19. [netcal calibration framework](https://github.com/EFS-OpenSource/calibration-framework) — ECE, MCE, ACE, SCE, TACE
