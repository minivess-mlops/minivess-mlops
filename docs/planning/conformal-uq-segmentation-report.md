# Conformal Prediction for 3D Segmentation Uncertainty — Academic Multi-Decision Report

**Date:** 2026-02-27
**Branch:** `feat/conformal-uq`
**Issue:** [#94](https://github.com/minivess-mlops/minivess-mlops/issues/94)

## 1. Executive Summary

This report evaluates the feasibility, library landscape, and implementation strategy for
conformal prediction (CP) in 3D binary vascular segmentation (MinIVess). Our analysis covers
18 bibliography papers, 6+ Python libraries, and 5 academic repos with open-source code.

**Key finding:** No general-purpose conformal prediction library for image segmentation exists.
All implementations are bespoke. The existing MinIVess code (voxel-level split CP + MAPIE
wrapper) provides a baseline, but lacks spatial awareness — the critical gap for medical
segmentation. We propose a 5-phase implementation that builds on existing code, starting with
vectorized voxel-level CP and progressively adding morphological (ConSeMa-inspired) and
distance-transform (CLS-inspired) prediction sets.

**Decision matrix:**

| Approach | Feasibility | Academic Novelty | Implementation Cost | Recommendation |
|----------|------------|------------------|-------------------|----------------|
| Voxel-level split CP (existing) | High | Low | Already done | **Phase 0: Optimize** |
| Morphological CP (ConSeMa) | High | Medium | 2-3 tasks | **Phase 1: Implement** |
| Distance-transform CP (CLS) | Medium | High | 2-3 tasks | **Phase 2: Implement** |
| Spatially-adaptive CP (SACP) | Medium | High | 3-4 tasks | **Phase 3: P2 issue** |
| Risk-controlling CP (RCPS) | Medium | Medium | 2 tasks | **Phase 4: Implement** |
| Random-walk CP (RW-CP) | Low | Very High | 5+ tasks | **P2 issue only** |

## 2. Library Landscape Assessment

### 2.1 General-Purpose CP Libraries

| Library | Maintainer | Segmentation Support | Pixel-level CP | 3D Support | Status |
|---------|-----------|---------------------|---------------|-----------|--------|
| **MAPIE** | scikit-learn-contrib | Not yet (2026 roadmap) | No | No | Active, v1.3+ |
| **TorchCP** | ml-stat-SUSTech | No | No | No | Active |
| **PUNCC** | DEEL-AI | No | No | No | Active |
| **Crepes** | H. Bostr | No | No | No | Active |
| **nonconformist** | donlnz | No | No | No | Stale (2020) |
| **netcal** | Fabian Kuppers | Calibration only | N/A | N/A | Active |

**Critical insight:** MAPIE's 2026 roadmap explicitly lists "image segmentation" as an upcoming
feature under their "risk control" framework. Until that lands, the MinIVess MAPIE wrapper
(flattening 3D volumes to tabular) is a reasonable but spatially-naive workaround.

**MAPIE limitations for segmentation:**
- Treats each voxel independently (no spatial structure)
- Requires flattening 5D tensors → massive memory for large volumes
- No awareness of morphological structure (vessels, boundaries)
- `SplitConformalClassifier` is designed for classification, not dense prediction
- Coverage guarantee holds marginally per-voxel, not per-volume or per-structure

### 2.2 Academic Segmentation-Specific Repos

| Repo | Paper | Approach | Dimensionality | Code Quality | Reusable? |
|------|-------|----------|---------------|-------------|----------|
| [deel-ai-papers/consema](https://github.com/deel-ai-papers/consema) | Mossina & Friedrich (2025) | Morphological dilation | 2D | Jupyter notebooks | Concepts only |
| [deel-ai-papers/conseco](https://github.com/deel-ai-papers/conformal-segmentation) | Mossina et al. (2025) | Morphological erosion/dilation | 2D | Research code | Concepts only |
| [tailabTMU/SACP](https://github.com/tailabTMU/SACP) | Bereska et al. (2025) | Spatially-adaptive, vessel-proximity weighted | 2D/3D | Research code | Partial |
| (no public repo) | Tan et al. (2025) CLS | Distance-transform, FNR control | 3D | N/A | Must reimplement |
| (no public repo) | Gaillochet et al. (2026) RW-CP | Anatomical random walks | 3D | N/A | Must reimplement |
| [widecanal/WCP](https://github.com/widecanal/WCP) | WCP (2025) | Weighted CP + MONAI DynUNet | 3D | Research code | Partial |

**No production-quality library exists for conformal prediction in segmentation.**
Every implementation is paper-specific research code (Jupyter notebooks, one-off scripts).
This is the #1 gap in the field.

## 3. Taxonomy of Conformal Prediction Approaches for Segmentation

### 3.1 Level 1: Voxel-Level Conformal Prediction

**Already implemented in MinIVess.** Two variants exist:

1. **Custom split CP** (`conformal.py`): Uses nonconformity score `s = 1 - p(y_true)` per voxel.
   Calibrates quantile with finite-sample correction. Prediction set = {c : p(c) >= 1 - q}.
   - Pro: Simple, distribution-free, works with any softmax model
   - Con: Treats voxels independently; no spatial coherence; prediction sets per-voxel
     are uninformative for binary segmentation (either {0}, {1}, or {0,1})

2. **MAPIE wrapper** (`mapie_conformal.py`): Trains LogisticRegression on flattened softmax
   probabilities, wraps in `SplitConformalClassifier`.
   - Pro: Leverages MAPIE's tested infrastructure
   - Con: LogisticRegression is redundant (softmax probs are already calibrated enough);
     same spatial-independence problem; memory-heavy for large volumes

**Verdict:** Both work but produce spatially incoherent prediction sets. For binary
segmentation (2 classes), the prediction set per voxel is at most {background, vessel},
which provides no useful geometric information. The real question is: *where* is the
segmentation boundary uncertain?

### 3.2 Level 2: Morphological Conformal Prediction (ConSeMa/ConSeCo)

**Not yet implemented. HIGH PRIORITY for MinIVess.**

Mossina & Friedrich (2025) propose a fundamentally different approach: instead of per-voxel
prediction sets, produce **inner and outer contour prediction sets** via morphological
operations:

- **Outer set (ConSeMa)**: Dilate the predicted mask by lambda iterations of a structuring
  element. The calibrated lambda is the smallest dilation count such that the dilated mask
  covers the true mask with probability >= 1-alpha.
- **Inner set (ConSeCo)**: Erode the predicted mask by lambda iterations. The calibrated
  lambda is the smallest erosion count such that the eroded mask is contained in the true
  mask with probability >= 1-alpha.
- **Prediction band**: The region between inner and outer contours is the uncertain zone.

**Why this is perfect for vascular segmentation:**
- Vessels have thin, elongated morphology — boundary uncertainty is geometrically meaningful
- The structuring element can be adapted: isotropic ball for thick vessels, anisotropic
  for thin vessels, tubular for centerline-preserving dilation
- Works with binary masks directly (no softmax probabilities needed)
- The "prediction band width" is an intuitive uncertainty measure for clinicians
- Coverage guarantee is per-volume (not per-voxel), matching clinical needs

**Implementation complexity:** Medium. Requires:
1. `scipy.ndimage.binary_dilation` / `binary_erosion` with ball structuring element
2. Calibration: for each calibration volume, find minimal lambda to cover/contain GT
3. Quantile over calibration lambdas with finite-sample correction
4. 3D extension: use `scipy.ndimage.generate_binary_structure(rank=3, connectivity=1)`

### 3.3 Level 3: Distance-Transform Conformal Prediction (CLS)

**Not yet implemented. HIGH PRIORITY for academic contribution.**

Tan et al. (2025) propose Conformal Label Smoothing (CLS) for 3D medical segmentation:

- **Key idea**: Instead of dilating by fixed iterations, use the **signed distance transform**
  (SDT) from the predicted boundary as the nonconformity score.
- **FNR control**: Guarantees false negative rate (missed vessel voxels) below alpha.
  This is critical for vascular segmentation where missing a vessel branch is dangerous.
- **SDT-based prediction sets**: For each voxel, the nonconformity score is the distance
  to the nearest predicted boundary. Voxels within distance <= q of the predicted vessel
  are included in the prediction set.

**Why this is excellent for MinIVess:**
- Directly controls the clinically relevant error rate (FNR, not just coverage)
- Distance transforms are natural for tubular structures
- `scipy.ndimage.distance_transform_edt` handles 3D natively
- The calibrated distance threshold q has a physical interpretation: "we are confident
  the true vessel boundary is within q voxels of our prediction"

**Implementation complexity:** Medium. Requires:
1. Signed distance transform of predicted and ground-truth masks
2. Nonconformity score per calibration volume: max distance from GT boundary to nearest
   predicted boundary (or hausdorff-like score)
3. Quantile calibration
4. Prediction set: dilate predicted mask by q voxels (using distance transform threshold)

### 3.4 Level 4: Spatially-Adaptive Conformal Prediction (SACP)

**Not yet implemented. P2 issue for future work.**

Bereska et al. (2025) propose learning location-specific nonconformity scores:

- Train a conformal score predictor that takes local image features and predicts the
  expected nonconformity score per voxel
- Voxels near vessel bifurcations, thin branches, or imaging artifacts get wider
  prediction intervals; confident interior voxels get tight intervals
- Requires a separate calibration neural network or random forest

**Why this is interesting but complex:**
- Truly adaptive per-voxel uncertainty (not uniform dilation)
- Directly models the vascular geometry difficulty
- But: requires additional training, more calibration data, more complex calibration pipeline
- With only ~20 volumes, spatially-adaptive calibration may overfit

### 3.5 Level 5: Risk-Controlling Prediction Sets (RCPS)

**Not yet implemented. MODERATE PRIORITY.**

Angelopoulos et al. (2022) generalize conformal prediction to control arbitrary risk functions:

- Instead of coverage, control FNR (false negative rate), FPR, or Dice loss
- Uses same calibration machinery but different loss function
- MAPIE's 2026 roadmap "risk control" feature likely implements this

**Why this is useful for MinIVess:**
- Can directly guarantee "Dice >= threshold" or "FNR <= alpha"
- More clinically meaningful than coverage guarantees
- Same calibration data requirements as standard CP

### 3.6 Level 6: Advanced Research (P2+ Issues)

- **Random-Walk CP (Gaillochet et al., 2026)**: Anatomically-aware random walks for
  spatial nonconformity propagation. Fascinating but no code available, complex to implement.
- **TUNE++ (Dhor et al., 2026)**: Combines topology-aware loss with UQ for tubular
  structures. Training-time method, not post-hoc.
- **Generative Conformal (Prob-UNet + CP)**: Combine generative uncertainty samples
  with conformal calibration for distribution over segmentation masks.

## 4. Analysis of Existing MinIVess Code

### 4.1 Strengths

- **UncertaintyOutput** dataclass provides a clean shared interface
- **DeepEnsemblePredictor** with Lakshminarayanan decomposition is gold standard
- **MCDropoutPredictor** provides quick single-model UQ
- **ConformalPredictor** has correct finite-sample quantile correction
- **MapieConformalSegmentation** wraps MAPIE properly (flatten, calibrate, reshape)
- **calibration.py** with ECE/MCE + temperature scaling for post-hoc calibration
- **GenerativeUQEvaluator** with GED and Q-Dice for multi-rater comparison
- **CalibrationShiftAnalyzer** for cross-domain calibration transfer

### 4.2 Weaknesses

1. **O(N^3) nested loops** in `conformal.py:calibrate()` (lines 75-81) and
   `mapie_conformal.py:compute_coverage_metrics()` (lines 188-195). Must vectorize.

2. **No spatial awareness** — both CP implementations treat voxels independently.
   For binary segmentation, this gives trivial prediction sets.

3. **No morphological prediction sets** — the most clinically useful CP output for
   segmentation (inner/outer contours) is not implemented.

4. **MAPIE wrapper is unnecessarily complex** — fitting a LogisticRegression on top
   of softmax probabilities adds complexity without value. The custom split CP
   (`conformal.py`) is simpler and equally correct.

5. **No FNR/FPR control** — only marginal coverage is guaranteed, not the clinically
   relevant error rates.

6. **No integration with evaluation pipeline** — CP results are not logged to MLflow
   or included in the `UnifiedEvaluationRunner` output.

### 4.3 What to Keep vs. Rewrite

| Component | Action | Reason |
|-----------|--------|--------|
| `ConformalPredictor` | **Vectorize + extend** | Correct algorithm, terrible perf |
| `MapieConformalSegmentation` | **Keep as optional** | MAPIE roadmap may improve it |
| `ConformalResult` | **Extend** | Add inner/outer contours |
| `ConformalMetrics` | **Extend** | Add FNR, FPR, band width |
| `compute_coverage_metrics` | **Vectorize** | O(N^3) → vectorized numpy |
| `UncertaintyOutput` | **Keep** | Good shared interface |
| `DeepEnsemblePredictor` | **Keep** | Gold standard, well-implemented |
| `MCDropoutPredictor` | **Keep** | Good single-model UQ |

## 5. Proposed Implementation Strategy

### Phase 0: Vectorize Existing Code (Quick Win)
- Replace nested Python loops with vectorized numpy in `conformal.py` and `mapie_conformal.py`
- ~100x speedup expected
- No new features, just correctness-preserving optimization

### Phase 1: Morphological CP (ConSeMa-Inspired)
- Implement `MorphologicalConformalPredictor` with dilation/erosion calibration
- Inner contour (high-confidence vessel core) + outer contour (maximum vessel extent)
- Prediction band width as uncertainty measure
- 3D structuring element support (ball, cross, custom tubular)

### Phase 2: Distance-Transform CP (CLS-Inspired)
- Implement `DistanceTransformConformalPredictor` using signed distance transforms
- FNR control variant: guarantee missed vessel rate <= alpha
- Calibrated boundary distance as uncertainty measure

### Phase 3: Risk-Controlling Prediction Sets
- Implement `RiskControllingPredictor` with configurable loss functions
- Support Dice loss, FNR, FPR as risk functions
- Uses Learn Then Test (LTT) framework for multiple risk control

### Phase 4: Evaluation Pipeline Integration
- Add CP metrics to `UnifiedEvaluationRunner`
- Log conformal coverage, band width, FNR to MLflow
- Comparison: voxel-level vs morphological vs distance-transform CP
- Visualization: overlay prediction bands on segmentation results

## 6. Recommendations

### Must-Have (Close Issue #94)
1. Vectorize existing `ConformalPredictor` and `compute_coverage_metrics`
2. Implement morphological CP (ConSeMa approach) — this is the primary academic contribution
3. Implement distance-transform CP (CLS approach) — FNR control for safety
4. Integrate with `UnifiedEvaluationRunner` and MLflow logging

### Should-Have (P2 Issues)
5. Risk-controlling prediction sets (RCPS/LTT)
6. Comparison study: voxel vs morphological vs distance-transform vs RCPS

### Nice-to-Have (P3 Issues)
7. Spatially-adaptive CP (SACP) when dataset grows > 50 volumes
8. Generative conformal (Prob-UNet + CP) for multi-modal uncertainty
9. Random-walk CP (RW-CP) for anatomically-aware propagation

## 7. References

- Angelopoulos, A. N. & Bates, S. (2023). "Conformal Prediction: A Gentle Introduction."
- Mossina, L. & Friedrich, C. M. (2025). "ConSeMa: Conformal Semantic Image Segmentation with Mask-level Guarantees." MICCAI 2025.
- Tan, J. et al. (2025). "Conformal Label Smoothing for 3D Medical Image Segmentation."
- Bereska, L. et al. (2025). "Spatially Adaptive Conformal Prediction for Vascular Segmentation."
- Gaillochet, M. et al. (2026). "Anatomically-Aware Random Walk Conformal Prediction."
- Shah-Mohammadi, F. & Kain, A. (2025). "Conformal Prediction for Medical Image Segmentation."
- Dhor, A. et al. (2026). "TUNE++: Topology-Aware Uncertainty for 3D Tubular Structure Segmentation."
- Lakshminarayanan, B. et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles."
- Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation."
- Angelopoulos, A. N. et al. (2022). "Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control."
