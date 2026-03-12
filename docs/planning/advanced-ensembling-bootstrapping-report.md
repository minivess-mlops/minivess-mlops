---
title: "Advanced Ensembling, Bootstrapping, and Statistical Testing"
status: reference
created: "2026-03-04"
---

# Advanced Ensembling, Bootstrapping, and Statistical Testing for Medical Image Segmentation

> **Research Report for MinIVess MLOps — Biostatistics Flow (Issue #340)**
>
> Date: 2026-03-04
>
> Purpose: Literature survey on bootstrapping deep ensembles, diverse ensemble strategies,
> instability analysis, and practical statistical testing for 3D medical image segmentation.
> Informs the design of the Biostatistics Prefect Flow and identifies P3 research directions.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Critical Distinction: Bootstrap of Evaluation vs Bootstrap of Training](#2-critical-distinction-bootstrap-of-evaluation-vs-bootstrap-of-training)
3. [Bootstrapping Deep Ensembles: The Debate](#3-bootstrapping-deep-ensembles-the-debate)
4. [Deep Ensembles for Medical Image Segmentation](#4-deep-ensembles-for-medical-image-segmentation)
5. [Diverse Ensemble Methods Beyond Standard Deep Ensembles](#5-diverse-ensemble-methods-beyond-standard-deep-ensembles)
6. [Instability Analysis for Deep Learning](#6-instability-analysis-for-deep-learning)
7. [Practical Statistical Testing in Medical Image Segmentation](#7-practical-statistical-testing-in-medical-image-segmentation)
8. [Implications for MinIVess](#8-implications-for-minivess)
9. [References](#9-references)

---

## 1. Executive Summary

This report addresses a fundamental question: **how should we perform statistical testing
for a 70-volume 3D medical image segmentation experiment?** The answer requires separating
two very different uses of "bootstrapping":

1. **Bootstrap of evaluation metrics** (resampling per-volume Dice/HD95 scores to get
   confidence intervals) — computationally trivial, standard practice, and exactly what
   the Biostatistics Flow should implement.

2. **Bootstrap of training** (retraining models on resampled data to capture data
   uncertainty) — computationally prohibitive for deep networks, theoretically problematic
   (Nixon et al. 2020), and **not what Issue #340 proposes**.

The BCa bootstrap with B=10,000 iterations proposed in Issue #340 is the **evaluation
bootstrap**: resample 70 per-volume metric values, compute the aggregate statistic, repeat
10,000 times. This takes **milliseconds** in NumPy and is entirely appropriate.

Beyond the immediate biostatistics needs, this report surveys diverse ensemble methods
(snapshot ensembles, SWAG, model soups, loss-conditioned ensembles), instability analysis
as an emerging field applicable to image segmentation, and the current best-practice
statistical testing workflow from Metrics Reloaded and nnU-Net.

---

## 2. Critical Distinction: Bootstrap of Evaluation vs Bootstrap of Training

### 2.1 Bootstrap of Evaluation Metrics (What Issue #340 Proposes)

The classical statistical bootstrap (Efron and Tibshirani 1993) resamples **observed
data points** (here: per-volume metric values) to construct sampling distributions and
confidence intervals. For a dataset of N=70 volumes:

- Each bootstrap iteration: draw 70 values with replacement → compute aggregate (mean, median)
- B=10,000 iterations: 700,000 random draws + 10,000 aggregate computations
- **Wall-clock time: milliseconds** in NumPy/SciPy
- BCa (bias-corrected and accelerated) intervals add jackknife estimates (70 leave-one-out
  computations) — trivial overhead

Efron and Tibshirani (1993) recommended B=2,000 as a default; for publication-quality BCa
intervals, B=5,000–15,000 is standard practice. B=10,000 is well within this range.

El Jurdi et al. (2025) directly studied confidence intervals for brain MRI segmentation and
used B=15,000 bootstrap iterations, reporting that this takes **less than 1 second on a
standard laptop**. For 70 test subjects, they estimated CI widths of approximately 3–4 Dice
points (95% CI), which narrows as sample size increases: ~50 subjects → ~0.017 normalized
width, ~100 subjects → ~0.012, ~200+ subjects needed for <2 Dice-point CIs.

**Verdict: B=10,000 bootstrap of evaluation metrics is computationally free and the correct
approach for Issue #340.**

### 2.2 Bootstrap of Training (What Nixon et al. 2020 Critiques)

A fundamentally different approach: train M neural networks on M different bootstrap
resamples of the training data. This is the "bootstrapped deep ensemble" that Nixon et al.
(2020) found to be **worse** than standard deep ensembles. For MinIVess:

- Each model trains for ~100 epochs on 70 volumes (hours of GPU time)
- B=10,000 would require training 10,000 models — **years of GPU time**
- Even B=5 (5 bootstrapped models) takes 5x the training budget

This approach is computationally prohibitive and theoretically unsound for deep learning
(see Section 3). It is **not** what the Biostatistics Flow should implement.

---

## 3. Bootstrapping Deep Ensembles: The Debate

### 3.1 Why Bootstrapped Deep Ensembles Fail

Nixon, Lakshminarayanan, and Tran (2020) investigated why bootstrap ensemble methods
underperform in deep neural networks despite their theoretical success in classical
settings (random forests, bagging). Their key finding:

> "The number of unique datapoints in the training set is the main determinant of ensemble
> performance."

Standard bootstrap resampling draws N samples with replacement from N, yielding only
~63.2% unique data points per bootstrap sample (the "coupon collector" effect). Nixon et al.
showed that this ~37% data loss per member is more harmful than the diversity gained. An
ensemble of models trained on the **full dataset** (differing only in random initialization
and minibatch ordering) outperforms a bootstrap ensemble.

Fort, Hu, and Lakshminarayanan (2020) provided the loss landscape explanation: random
initializations cause models to converge to **entirely different modes** in the loss
landscape, producing functionally diverse predictions. They demonstrated that "the
decorrelation power of random initializations is unmatched by popular subspace sampling
methods." This means bootstrap-induced diversity is redundant — random initialization
already provides the diversity deep ensembles need.

Abe et al. (2023) corroborated this in a study of nearly 600 classification ensembles,
finding that interventions promoting predictive diversity (including data subsampling)
**harm** large neural network ensembles. For high-capacity networks, predictions
concentrate at probability simplex vertices, so decorrelating predictions moves ensemble
output away from confident (and correct) predictions.

### 3.2 The Counter-Argument: Parametric Bootstrap for Regression

Sluijterman, Cator, and Heskes (2025) present a partial rebuttal to Nixon et al., but
with important caveats:

1. **Different goal.** Their method targets **confidence interval calibration**, not
   prediction accuracy. Standard Deep Ensembles capture optimization randomness but miss
   the "classical" parametric uncertainty from finite training data.

2. **Parametric bootstrap, not standard bootstrap.** Instead of resampling training data
   (which loses ~37% of unique examples), they resample from the model's **fitted
   distribution**, avoiding the data reduction problem entirely.

3. **Regression-specific.** The method leverages the separate mean and variance estimates
   that Deep Ensembles provide in regression, which "does not translate to classification
   where only a single probability vector is available."

4. **Computationally cheap.** Described as requiring only O(1/M) additional training time
   beyond the standard Deep Ensemble.

**Verdict: The parametric bootstrap of Sluijterman et al. (2025) does NOT contradict
Nixon et al. (2020).** It uses a different bootstrap variant (parametric, not
nonparametric) for a different purpose (CI calibration, not prediction accuracy) in a
different setting (regression, not classification/segmentation). The consensus remains:
**standard data-bootstrap of deep neural networks does not help.**

### 3.3 Current Consensus

| Claim | Evidence | Status |
|-------|----------|--------|
| Standard bootstrap of training data hurts deep ensembles | Nixon et al. (2020), Abe et al. (2023) | **Strong consensus** |
| Random initialization provides sufficient diversity | Fort et al. (2020), Lakshminarayanan et al. (2017) | **Strong consensus** |
| Parametric bootstrap can improve CIs in regression | Sluijterman et al. (2025) | Promising, regression-only |
| Bootstrap of evaluation metrics is standard practice | Efron and Tibshirani (1993), El Jurdi et al. (2025) | **Uncontroversial** |

---

## 4. Deep Ensembles for Medical Image Segmentation

### 4.1 Foundation: Lakshminarayanan et al. (2017)

The Deep Ensembles method (Lakshminarayanan, Pritzel, and Blundell 2017) is
"embarrassingly simple": train M neural networks from different random initializations,
average their predictions. The method is:

- Simple to implement and readily parallelizable
- Requires minimal hyperparameter tuning
- Produces uncertainty estimates competitive with approximate Bayesian methods
- Default ensemble size: **M=5 members**

### 4.2 Benchmark Under Distribution Shift: Ovadia et al. (2019)

Ovadia et al. (2019) conducted the definitive benchmark of uncertainty methods under
dataset shift (NeurIPS 2019), evaluating across images, text, and tabular data. Key
findings:

- **Deep ensembles are the most robust method** to dataset shift, outperforming MC Dropout,
  variational inference, and temperature scaling
- **M=5 members is sufficient** — improvement quickly diminishes beyond 5 networks, with
  comparable results to larger ensembles
- Temperature scaling, while improving calibration on in-distribution data, often
  **worsened** calibration under distribution shift

### 4.3 Medical Image Segmentation: Mehrtash et al. (2020)

Mehrtash et al. (2020) published the most thorough study of ensemble calibration for
medical image segmentation in IEEE TMI, evaluating across brain, heart, and prostate:

- Tested ensemble sizes of **1, 2, 5, 10, 25, and 50** models
- **Five-model ensembles reduced negative log-likelihood by 62–66%** compared to single models
- Ensembling outperformed MC Dropout for calibration
- FCNs trained with Dice loss are poorly calibrated (overconfident); ensembling is the
  most effective remedy
- Average entropy of ensemble predictions serves as an effective test-time quality predictor

### 4.4 Brain Tumor Segmentation: Jungo et al. (2020)

Jungo, Balsiger, and Reyes (2020) compared uncertainty methods for brain tumor
segmentation (Frontiers in Neuroscience):

- Compared softmax baseline, MC Dropout (4 variants), aleatoric uncertainty, **ensembles
  (K=10 models)**, and auxiliary networks
- **Ensemble achieved overall best results** while maintaining improved segmentation
- Critical finding: voxel-wise uncertainty showed "notable miscalibrations" (only 36–38%
  of subjects adequately calibrated). Subject-level aggregation was more useful for
  failure detection.

### 4.5 Probabilistic Approaches

Kohl et al. (2018) introduced the Probabilistic U-Net (NeurIPS 2018), combining a
conditional VAE with U-Net to generate diverse plausible segmentations. Unlike standard
ensembles, this captures **multi-modal** segmentation ambiguity. Standard ensembles produce
consistent but not necessarily diverse outputs and "are typically not able to learn the rare
variants" in ambiguous segmentation tasks.

Monteiro et al. (2020) proposed Stochastic Segmentation Networks (NeurIPS 2020), modeling
joint distributions over entire label maps rather than per-pixel independence, producing
spatially coherent uncertainty maps.

Czolbe et al. (2021) asked "Is Segmentation Uncertainty Useful?" (IPMI 2021), finding
that uncertainty correlates positively with segmentation error (useful for quality
assessment) but **does not help for active learning**.

### 4.6 Optimal Ensemble Size

| Study | Domain | Sizes Tested | Recommendation |
|-------|--------|-------------|----------------|
| Lakshminarayanan et al. (2017) | General | M=5 | Original standard |
| Ovadia et al. (2019) | Multi-domain | M=5 | "Comparable to larger ensembles" |
| Mehrtash et al. (2020) | Brain/heart/prostate | 1, 2, **5**, 10, 25, 50 | 5 achieves 62–66% NLL reduction |
| Jungo et al. (2020) | Brain tumor | K=10 | Cross-validation ensemble |
| Adams and Elhabian (2023) | Organ segmentation | Multiple methods | Multi-SWAG promising |

**Practical consensus: M=5 members, with strong diminishing returns beyond this point.**

---

## 5. Diverse Ensemble Methods Beyond Standard Deep Ensembles

### 5.1 Snapshot Ensembles

Huang et al. (2017) proposed Snapshot Ensembles (ICLR 2017): use aggressive cosine
annealing to drive SGD to multiple local minima during a **single training run**, saving
snapshots at each convergence point. This achieves ensemble diversity at **zero additional
training cost**. Reported error rates of 3.44% on CIFAR-10 and 17.41% on CIFAR-100.

**Relevance to MinIVess:** Could extract multiple snapshots per fold during training at no
extra cost.

### 5.2 Fast Geometric Ensembling (FGE)

Garipov et al. (2018) discovered that optima of complex loss functions are connected by
simple curves (mode connectivity) and proposed collecting models along these high-accuracy
pathways (NeurIPS 2018). Achieves ensemble performance in the time to train a single model,
improving over Snapshot Ensembles.

### 5.3 Stochastic Weight Averaging (SWA) and SWAG

Izmailov et al. (2018) introduced SWA (UAI 2018): average multiple points along the SGD
trajectory with a cyclical or constant learning rate. Finds flatter solutions than standard
SGD with almost no computational overhead. Now integrated into PyTorch core
(`torch.optim.swa_utils`). Unlike an ensemble, SWA produces a **single model** — no
inference overhead.

Maddox et al. (2019) extended SWA to SWAG (NeurIPS 2019) by fitting a Gaussian
distribution (low-rank + diagonal covariance) over the SGD trajectory, enabling Bayesian
model averaging. Provides well-calibrated uncertainty estimates.

Wilson and Izmailov (2020) introduced **Multi-SWAG**: run SWAG from multiple random
initializations and ensemble the resulting approximate posteriors, enabling multimodal
marginalization across basins.

Adams and Elhabian (2023) benchmarked 9 epistemic UQ methods on organ segmentation (UNSURE
Workshop, MICCAI 2023), finding that Multi-SWAG "reinforced the notion that ensembling
Bayesian methods improves approximate inference fidelity via multimodal marginalization."
Deep Ensemble achieved highest accuracy (92.77 DSC on spleen) but at highest cost.

### 5.4 Hyperparameter Ensembles

Wenzel et al. (2020) proposed "hyper-deep ensembles" (NeurIPS 2020): ensemble over both
weights **and** hyperparameters (learning rate, weight decay, data augmentation). Random
search over different hyperparameters yields substantial improvements in uncertainty
quantification compared to standard deep ensembles alone.

**Relevance to MinIVess:** The 4 losses × 3 folds ensemble is inherently a hyperparameter
ensemble, where the "hyperparameter" varied is the loss function.

### 5.5 Loss-Conditioned Ensembles

Ma et al. (2021) conducted the largest benchmarking study of loss functions for medical
image segmentation ("Loss Odyssey," Medical Image Analysis): 20 loss functions, 4 tasks,
6 public datasets. **No single loss consistently achieves best performance.** Compound
losses (Dice + TopK, Dice + Focal, Dice + HD) are the most robust.

Li et al. (2025) found that ensembling models trained with different losses yielded 2–7%
DSC improvement in CT image segmentation, with learnable ensemble approaches outperforming
static methods.

**Relevance to MinIVess:** Directly validates the MinIVess approach of training with 4
different losses and ensembling, since no single loss dominates across all metrics.

### 5.6 BatchEnsemble

Wen, Tran, and Ba (2020) introduced BatchEnsemble (ICLR 2020): each weight matrix is the
Hadamard product of a shared weight and a rank-1 matrix per ensemble member. Memory and
compute cost are roughly 1/M of a standard M-member ensemble. 3× speedup and 3× memory
reduction at ensemble size 4, with competitive accuracy and uncertainty estimates.

### 5.7 MC Dropout

Gal and Ghahramani (2016) proved that dropout training is mathematically equivalent to
approximate Bayesian inference in deep Gaussian processes (ICML 2016). At test time,
running T stochastic forward passes with dropout enabled provides uncertainty estimates at
zero implementation cost. Kendall and Gal (2017) extended this to distinguish aleatoric
from epistemic uncertainty in per-pixel semantic segmentation (NeurIPS 2017).

However, multiple studies (Ovadia et al. 2019, Mehrtash et al. 2020) show MC Dropout
produces less calibrated uncertainty than proper deep ensembles.

### 5.8 Model Soups

Wortsman et al. (2022) proposed Model Soups (ICML 2022): instead of selecting the best
model from a hyperparameter sweep, **average their weights** in weight space. Works because
fine-tuned models from the same pre-trained checkpoint lie in a single low-error basin.
Provides improvements with **zero additional inference or memory cost**.

**Relevance to MinIVess:** Could average fold-specific checkpoints within the same loss
function, combined with SWA for further flattening.

### 5.9 Ensemble Selection from Libraries

Caruana et al. (2004) proposed greedy forward stepwise selection from large model libraries
(ICML 2004), optimizing for arbitrary metrics. Could be used to select the optimal subset
from MinIVess's 4 losses × 3 folds = 12 model pool.

### 5.10 Mixture of Experts (MoE)

Shazeer et al. (2017) introduced the Sparsely-Gated MoE layer (ICLR 2017). In the medical
imaging context, SAM-Med3D-MoE (MICCAI 2024) integrates task-specific fine-tuned models
with the SAM foundation model using a gating network. Rahman et al. (2025) applied
personalized MoE for multi-site medical image segmentation (WACV 2025).

### 5.11 Taxonomy of Ensemble Diversity Sources

| Method | Diversity Source | Extra Training Cost | Extra Inference Cost |
|--------|-----------------|--------------------|--------------------|
| Deep Ensembles | Random initialization | M× | M× |
| Snapshot Ensembles | Cosine annealing snapshots | 0× (same training) | M× |
| FGE | Mode-connecting curves | 0× | M× |
| SWA | Trajectory averaging | ~0× | 0× (single model) |
| SWAG/Multi-SWAG | Gaussian fit to trajectory | ~0× | T× (T samples) |
| Hyperparameter Ensembles | Learning rate, loss, augmentation | M× | M× |
| Loss-Conditioned | Different loss functions | M× | M× |
| BatchEnsemble | Rank-1 perturbations | ~1/M× | ~1/M× |
| MC Dropout | Stochastic forward passes | 0× | T× |
| Model Soups | Weight-space averaging | 0× (from existing) | 0× |

---

## 6. Instability Analysis for Deep Learning

### 6.1 The Core Framework

Instability analysis examines whether a **single model specification** (same architecture,
hyperparameters, training data) produces consistent predictions across independent training
runs. This is distinct from ensemble disagreement, which compares intentionally different
models.

Riley and Collins (2023) defined four hierarchical stability levels for clinical prediction
models:

1. **Population-level:** Stability of mean predicted risk
2. **Distribution-level:** Stability of the distribution of predicted risks
3. **Subgroup-level:** Stability of predictions for clinical subgroups
4. **Individual-level:** Stability of predictions for specific patients

Using bootstrap resampling (≥200 iterations), they found that even models meeting
minimum sample size criteria demonstrate "quite remarkable" instability, with individual
risk estimates spanning 0 to 1 across bootstrap samples.

### 6.2 Individual-Level Prediction Instability

Miller and Blume (2026) proposed two complementary diagnostic metrics:

- **Empirical Prediction Interval Width (ePIW):** Captures variability in continuous risk
  estimates across retraining runs
- **Empirical Decision Flip Rate (eDFR):** Measures instability in threshold-based
  clinical decisions

Key findings:
- Neural networks demonstrate **substantially greater instability** than logistic regression
- **Optimization-induced randomness** (initialization, batch ordering) produces individual-
  level variability **comparable to that produced by resampling the entire training dataset**
- **Aggregate metrics mask the problem:** standard performance metrics (accuracy, log-loss)
  fail to detect individual-level instability
- Risk estimate instability near clinical decision thresholds can **alter treatment
  recommendations**

### 6.3 Prediction Churn

Lopez-Martinez et al. (2022) demonstrated at ML4H that deep learning models trained
multiple times on identical data produce "significantly different outcomes at a patient
level" despite stable aggregate performance metrics (a Google Health study).

Bhojanapalli et al. (2021) studied "churn" — disagreements between predictions of two
models independently trained by the same algorithm — finding it significant even for
standard classification tasks.

### 6.4 Application to Image Segmentation: A Research Gap

Based on thorough literature search, **formal instability analysis (with metrics like ePIW,
eDFR, MAPE) has primarily been developed for clinical risk prediction (tabular/EHR data),
not image segmentation.** In segmentation, related concepts are studied under different names:

| Instability Analysis Concept | Segmentation Equivalent |
|-----------------------------|------------------------|
| Individual-level prediction instability | Per-voxel prediction variance across seeds |
| Decision Flip Rate | Per-voxel foreground/background flips across seeds |
| Prediction Interval Width | Per-volume Dice variance across seeds |
| Population-level stability | Mean Dice stability across seeds |

**This represents an underexplored research direction.** The MinIVess project's 4 losses ×
3 folds setup provides a natural framework for studying segmentation instability:

- Train K independent runs per configuration (same loss, same fold, different seeds)
- Compute per-voxel prediction agreement across K runs
- Compute per-volume "Decision Flip Rate" — fraction of voxels that change
  foreground/background classification
- Compare instability across loss functions and folds

### 6.5 Instability Analysis vs Ensemble Disagreement

| Aspect | Instability Analysis | Ensemble Disagreement |
|--------|--------------------|-----------------------|
| **What varies** | Training randomness (same config) | Different models (different configs) |
| **Purpose** | Assess prediction reliability | Quantify predictive uncertainty |
| **Key metric** | Per-instance variability (ePIW, eDFR) | Entropy, mutual information |
| **Clinical use** | "Can we trust this prediction?" | "How uncertain is this prediction?" |
| **Action** | If unstable → more data/regularization | Disagreement IS the signal |

---

## 7. Practical Statistical Testing in Medical Image Segmentation

### 7.1 The Metrics Reloaded Framework

Reinke, Tizabi, et al. (2024) published the definitive metric pitfall taxonomy in Nature
Methods, developed through a multi-stage Delphi process by 70+ international experts.
Key recommendations:

- **Compute metrics per-image first, then aggregate** (respecting hierarchical data
  structure where pixels within an image are highly correlated)
- **Never rely on a single metric** — each has task-dependent pitfalls
- The "problem fingerprint" concept guides problem-aware metric selection

The companion paper (Maier-Hein, Reinke, et al. 2024) provides concrete recommendations
via the Metrics Reloaded framework.

### 7.2 How nnU-Net Handles Comparisons

Isensee et al. (2021) use 5-fold cross-validation as the standard evaluation protocol
in nnU-Net (Nature Methods). The paper demonstrates superiority across 23 Medical
Segmentation Decathlon datasets, primarily using rank-based aggregation across datasets.

Critically, Isensee et al. (2024) in "nnU-Net Revisited" (MICCAI 2024) propose the
**inter-method vs intra-method standard deviation ratio** as a dataset suitability check:
if intra-method variation (across folds/seeds) is comparable to inter-method variation
(across algorithms), the dataset **cannot distinguish methods**. This is directly relevant
to MinIVess's 70-volume dataset.

### 7.3 The Wilcoxon Signed-Rank Test

The Wilcoxon signed-rank test is the **de facto standard** non-parametric paired test in
medical image segmentation. The typical workflow:

1. Compute per-volume metrics (Dice, HD95, etc.) for each method
2. Compute paired differences (Method A − Method B) per volume
3. Apply Wilcoxon signed-rank test to paired differences
4. Report p-value and effect size

Evidence of widespread use:
- Maier-Hein et al. (2018) used Wilcoxon signed-rank to compare ranking stability across
  150+ biomedical image analysis challenges (Nature Communications)
- A large-scale 3D multi-organ segmentation benchmark used "the one-sided Wilcoxon signed
  rank test with Holm's adjustment for multiplicity at 5% significance level"
- Prostate, stroke, and brain tumor segmentation studies routinely use Wilcoxon on
  per-volume metrics

### 7.4 Bootstrap CIs for Segmentation

El Jurdi et al. (2025) directly studied confidence intervals for segmentation (Medical
Image Analysis), finding:

- **Parametric CIs are reasonable approximations** of bootstrap CIs for Dice scores
- **>50% of segmentation papers do not assess performance variability at all**, and only
  **0.5% report confidence intervals** — making CI reporting a significant contribution
- For 70 test subjects, expect 95% CI widths of ~3–4 Dice points
- CIs narrower than 2 Dice points require ~200+ test subjects
- B=15,000 bootstrap iterations: <1 second on a standard laptop

### 7.5 Effect Sizes

Effect sizes remain **underreported** in medical image segmentation despite being
recommended. The appropriate choices:

- **Cohen's d** (parametric): difference in means / pooled SD. Thresholds: <0.2 negligible,
  <0.5 small, <0.8 medium, ≥0.8 large.
- **Cliff's delta** (non-parametric): probability that a random observation from Group A
  exceeds one from Group B. Thresholds: ≥0.11 small, ≥0.28 medium, ≥0.43 large. Does not
  assume normality. **Appropriate companion to Wilcoxon signed-rank test.**

### 7.6 Multiple Comparison Correction

Holm-Bonferroni correction is used in the best benchmarking papers but **not universally
adopted**. With 4 losses and 6 pairwise comparisons:

- **Holm (step-down):** Uniformly more powerful than Bonferroni. Requires no independence
  assumptions. Orders p-values and adjusts thresholds progressively.
- **BH-FDR:** Controls false discovery rate instead of family-wise error rate. More
  powerful for exploratory analyses with many comparisons.

Many segmentation papers report uncorrected p-values or perform no statistical testing at
all (El Jurdi et al. 2025), making proper correction a significant contribution.

### 7.7 Emerging Best Practice Workflow

Based on the literature reviewed, the recommended workflow for statistical comparison in
medical image segmentation is:

1. **Per-volume metric computation** — Dice, HD95, NSD, clDice, etc. per test volume
2. **Paired non-parametric testing** — Wilcoxon signed-rank (paired by volume_id)
3. **Multiple comparison correction** — Holm-Bonferroni for pairwise comparisons
4. **Effect sizes** — Cliff's delta alongside p-values
5. **Confidence intervals** — BCa bootstrap CIs on aggregate metrics (B≥5,000)
6. **Dataset suitability check** — inter-method vs intra-method SD ratio
7. **Multiple metrics** — never rely on a single metric (Metrics Reloaded)

---

## 8. Implications for MinIVess

### 8.1 Biostatistics Flow (Issue #340) — Validated Design

The proposed statistical methods in Issue #340 are well-aligned with best practices:

| Method | Status | Notes |
|--------|--------|-------|
| BCa bootstrap CIs (B=10,000) | **Correct** | Evaluation bootstrap, computationally free |
| Paired bootstrap tests | **Correct** | Paired by volume_id |
| Holm-Bonferroni correction | **Best practice** | Used in top benchmarks |
| Cohen's d + Cliff's delta | **Best practice** | Both parametric + non-parametric |
| Friedman test + Nemenyi | **Good addition** | Multi-algorithm omnibus test |
| Critical Difference diagrams | **Good addition** | Standard visualization |

### 8.2 Existing Ensemble Design — Already Strong

The MinIVess 4 losses × 3 folds ensemble is inherently a **loss-conditioned hyperparameter
ensemble** (Wenzel et al. 2020). The Loss Odyssey study (Ma et al. 2021) directly validates
this approach: no single loss dominates, so ensembling across losses captures complementary
strengths.

### 8.3 P3 Research Directions

#### P3-A: Instability Analysis for Segmentation (Novel)

Adapt the Riley/Collins (2023) and Miller/Blume (2026) instability frameworks to image
segmentation. This is a **research gap** — instability analysis has been developed for
tabular clinical prediction but not formally applied to segmentation.

Proposed approach:
- Train K=5 independent runs per (loss, fold) configuration (different seeds only)
- Compute per-voxel prediction agreement across K runs
- Define Segmentation Decision Flip Rate (sDFR): fraction of voxels changing class
- Compare instability across loss functions, folds, and anatomical regions
- Test whether aggregate Dice stability masks individual-voxel instability

**References to cite:**
- Riley and Collins (2023) — Stability framework for clinical prediction models
- Miller and Blume (2026) — ePIW and eDFR diagnostic metrics
- Lopez-Martinez et al. (2022) — Prediction churn in clinical ML
- Isensee et al. (2024) — Inter-method vs intra-method SD ratio

#### P3-B: Advanced Ensemble Diversity (Engineering)

Beyond the current loss-conditioned ensemble, evaluate:
- **Snapshot Ensembles** (Huang et al. 2017): extract multiple models per training run at
  zero cost
- **SWA/SWAG** (Izmailov et al. 2018, Maddox et al. 2019): weight-space averaging for
  better generalization and uncertainty, built into PyTorch
- **Model Soups** (Wortsman et al. 2022): average fold-specific weights within same loss
- **Ensemble Selection** (Caruana et al. 2004): greedy selection from the 12-model pool

#### P3-C: Bootstrapped Deep Ensembles — Probably Not Worth It

Given Nixon et al. (2020) and Abe et al. (2023), training models on bootstrapped data
resamples is **not recommended** for MinIVess. The one exception would be exploring the
parametric bootstrap of Sluijterman et al. (2025) for regression-type tasks (e.g., SDF
prediction heads), but this is highly specialized.

**Key citation for why not:**
> "Bootstrap ensembles of deep neural networks yield no benefit over simpler baselines."
> — Nixon, Lakshminarayanan, and Tran (2020)

### 8.4 Self-Reflection: Issue #340 Bootstrap Design

The B=10,000 BCa bootstrap in Issue #340 is the **evaluation bootstrap** — resampling
per-volume metric vectors to construct confidence intervals. This is:
- Computationally free (milliseconds)
- Statistically standard (Efron and Tibshirani 1993)
- Well-validated for segmentation (El Jurdi et al. 2025)
- **Not** the same as bootstrap training of deep networks (which would be prohibitive)

The issue text is correct as-is. The paired bootstrap test (resampling paired differences
between methods) is the appropriate approach for comparing loss functions on the same 70
volumes.

---

## 9. References

Abe, T., Buchanan, E.K., Pleiss, G., and Cunningham, J.P. 2023. "Pathologies of
Predictive Diversity in Deep Ensembles." *Transactions on Machine Learning Research*.

Adams, J.K. and Elhabian, S. 2023. "Benchmarking Scalable Epistemic Uncertainty
Quantification in Organ Segmentation." *UNSURE Workshop, MICCAI 2023*.

Bhojanapalli, S., Wilber, K., Veit, A., Rawat, A.S., Kim, Y., Menon, A., and Kumar, S.
2021. "On the Reproducibility of Neural Network Predictions." *arXiv:2102.03349*.

Buddenkotte, T., Escudero Sanchez, L., Crispin-Ortuzar, M., Woitek, R., McCague, C.,
Brenton, J.D., Öktem, O., Sala, E., and Rundo, L. 2023. "Calibrating Ensembles for
Scalable Uncertainty Quantification in Deep Learning-based Medical Image Segmentation."
*Computers in Biology and Medicine*.

Caruana, R., Niculescu-Mizil, A., Crew, G., and Ksikes, A. 2004. "Ensemble Selection
from Libraries of Models." *ICML 2004*.

Czolbe, S., Arnavaz, K., Krause, O., and Feragen, A. 2021. "Is Segmentation Uncertainty
Useful?" *IPMI 2021*, pp. 715–726.

D'Angelo, F. and Fortuin, V. 2021. "Repulsive Deep Ensembles are Bayesian." *NeurIPS 2021*.

Efron, B. and Tibshirani, R.J. 1993. *An Introduction to the Bootstrap.* Chapman and
Hall/CRC.

El Jurdi, R., Varoquaux, G., and Colliot, O. 2025. "Confidence Intervals for Performance
Estimates in Brain MRI Segmentation." *Medical Image Analysis*. arXiv:2307.10926.

Fort, S., Hu, H., and Lakshminarayanan, B. 2020. "Deep Ensembles: A Loss Landscape
Perspective." *arXiv:1912.02757*.

Gal, Y. and Ghahramani, Z. 2016. "Dropout as a Bayesian Approximation: Representing Model
Uncertainty in Deep Learning." *ICML 2016*.

Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D., and Wilson, A.G. 2018. "Loss
Surfaces, Mode Connectivity, and Fast Ensembling of DNNs." *NeurIPS 2018*.

Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J.E., and Weinberger, K.Q. 2017.
"Snapshot Ensembles: Train 1, Get M for Free." *ICLR 2017*.

Isensee, F., Jaeger, P.F., Kohl, S.A.A., Petersen, J., and Maier-Hein, K.H. 2021.
"nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation."
*Nature Methods*, 18, 203–211.

Isensee, F., Wald, T., Ulrich, C., Baumgartner, M., Roy, S., Maier-Hein, K., and Jaeger,
P.F. 2024. "nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image
Segmentation." *MICCAI 2024*.

Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., and Wilson, A.G. 2018. "Averaging
Weights Leads to Wider Optima and Better Generalization." *UAI 2018*.

Jin, R., Wu, Y., et al. 2024. "Boosting Deep Ensembles with Learning Rate Tuning."
*arXiv:2410.07564*.

Jungo, A., Balsiger, F., and Reyes, M. 2020. "Analyzing the Quality and Challenges of
Uncertainty Estimations for Brain Tumor Segmentation." *Frontiers in Neuroscience*, 14:282.

Kendall, A. and Gal, Y. 2017. "What Uncertainties Do We Need in Bayesian Deep Learning
for Computer Vision?" *NeurIPS 2017*.

Kohl, S.A.A., Romera-Paredes, B., Meyer, C., De Fauw, J., Ledsam, J.R., Maier-Hein, K.H.,
Eslami, S.M.A., Rezende, D.J., and Ronneberger, O. 2018. "A Probabilistic U-Net for
Segmentation of Ambiguous Images." *NeurIPS 2018*.

Lakshminarayanan, B., Pritzel, A., and Blundell, C. 2017. "Simple and Scalable Predictive
Uncertainty Estimation using Deep Ensembles." *NeurIPS 2017*.

Li, X., et al. 2025. "Enhancing CT image segmentation accuracy through ensemble loss
function optimization." *Medical Physics*.

Lopez-Martinez, D., Yakubovich, S., Seneviratne, M., et al. 2022. "Instability in clinical
risk stratification models using deep learning." *ML4H 2022*.

Ma, J., Chen, J., Ng, M., Huang, R., Li, Y., Li, C., Yang, X., and Martel, A.L. 2021.
"Loss Odyssey in Medical Image Segmentation." *Medical Image Analysis*, July 2021.

Maddox, W.J., Garipov, T., Izmailov, P., Vetrov, D., and Wilson, A.G. 2019. "A Simple
Baseline for Bayesian Uncertainty in Deep Learning (SWAG)." *NeurIPS 2019*.

Maier-Hein, L., Eisenmann, M., Reinke, A., et al. 2018. "Why rankings of biomedical image
analysis competitions should be interpreted with care." *Nature Communications*, 10, 588.

Maier-Hein, L., Reinke, A., et al. 2024. "Metrics Reloaded: Recommendations for image
analysis validation." *Nature Methods*, 21(2), 195–212.

Mehrtash, A., Wells, W.M. III, Tempany, C.M., Abolmaesumi, P., and Kapur, T. 2020.
"Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image
Segmentation." *IEEE Transactions on Medical Imaging*, 39(12):3868–3878.

Miller, E.W. and Blume, J.D. 2026. "Diagnostics for Individual-Level Prediction Instability
in Machine Learning for Healthcare." *arXiv:2603.00192*.

Monteiro, M. et al. 2020. "Stochastic Segmentation Networks: Modelling Spatially Correlated
Aleatoric Uncertainty." *NeurIPS 2020*.

Nixon, J., Lakshminarayanan, B., and Tran, D. 2020. "Why Are Bootstrapped Deep Ensembles
Not Better?" *NeurIPS 2020 Workshop "I Can't Believe It's Not Better!"*

Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J.,
Lakshminarayanan, B., and Snoek, J. 2019. "Can You Trust Your Model's Uncertainty?
Evaluating Predictive Uncertainty Under Dataset Shift." *NeurIPS 2019*.

Rahaman, R. and Thiery, A. 2021. "Uncertainty Quantification and Deep Ensembles."
*NeurIPS 2021*.

Rahman, M.A., et al. 2025. "Personalized Mixture of Experts for Multi-Site Medical Image
Segmentation." *WACV 2025*.

Reinke, A., Tizabi, M.D., et al. 2024. "Understanding metric-related pitfalls in image
analysis validation." *Nature Methods*, 21(2), 182–194.

Riley, R.D. and Collins, G.S. 2023. "Stability of clinical prediction models developed
using statistical or machine learning methods." *Biometrical Journal*, 65(8), e2200302.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J.
2017. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer."
*ICLR 2017*.

Sluijterman, L., Cator, E., and Heskes, T. 2025. "Confident Neural Network Regression
with Bootstrapped Deep Ensembles." *Neurocomputing*, 656, 131500.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. 2014.
"Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *JMLR*, 15, 1929–1958.

Wen, Y., Tran, D., and Ba, J. 2020. "BatchEnsemble: An Alternative Approach to Efficient
Ensemble and Lifelong Learning." *ICLR 2020*.

Wenzel, F., Snoek, J., Tran, D., and Jenatton, R. 2020. "Hyperparameter Ensembles for
Robustness and Uncertainty Quantification." *NeurIPS 2020*.

Wilson, A.G. and Izmailov, P. 2020. "Bayesian Deep Learning and a Probabilistic Perspective
of Generalization." *NeurIPS 2020*.

Wortsman, M., Ilharco, G., Gadre, S.Y., et al. 2022. "Model Soups: Averaging Weights of
Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time." *ICML 2022*.
