# Post-Training Methodologies for Medical Segmentation

**Literature Research Report R1** | 31 papers | 2026-03-18
**Manuscript section**: Methods: Post-Training Pipeline (R2b)
**KG domains**: training, models
**Quality target**: MINOR_REVISION

---

## 1. Introduction: Why Post-Training Matters More Than Training

The dominant paradigm in medical image segmentation treats training as the endpoint and
post-training as an afterthought — a temperature scalar applied before deployment, if at
all. This report argues the opposite: for small-dataset biomedical segmentation (N=70
volumes), the post-training pipeline may contribute more to deployment-ready performance
than any single training hyperparameter choice.

Three converging lines of evidence support this claim. First, modern neural networks are
systematically miscalibrated — they assign confidence scores that bear little relation to
actual correctness rates, a phenomenon first quantified by [Guo et al. (2017). "On
Calibration of Modern Neural Networks." *ICML*.](https://arxiv.org/abs/1706.04599) and
shown to be particularly severe in medical segmentation by [Mehrtash et al. (2020).
"Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image
Segmentation." *IEEE TMI*.](https://doi.org/10.1109/TMI.2020.3006437). Second, weight
averaging methods like SWA find wider optima that generalize better
([Izmailov et al. (2018). "Averaging Weights Leads to Wider Optima and Better
Generalization." *UAI*.](https://arxiv.org/abs/1803.05407)), an effect that compounds
with the high variance inherent to small-dataset training. Third, conformal prediction
provides distribution-free coverage guarantees that no amount of training can replicate
([Angelopoulos et al. (2022). "Conformal Risk Control."
*ICLR 2024*.](https://arxiv.org/abs/2208.02814)).

The critical insight is that these three families — weight averaging, calibration, and
conformal prediction — are not independent. A poorly calibrated model produces unreliable
nonconformity scores; SWA changes the loss landscape geometry in ways that interact with
calibration; and the choice of training loss (Dice vs. cross-entropy vs. clDice) determines
the baseline calibration that post-training methods must correct. This report maps these
interactions and identifies which factorial design factors are scientifically justified.

**Scope**: SWA variants, model soup/merging, temperature/Platt/isotonic calibration,
conformal prediction for segmentation, calibration-aware losses, and factorial design
considerations. **Excluded**: deep ensembles (Report R2), federated learning (R3),
regulatory compliance (R4).

---

## 2. Weight Averaging: From SWA to Model Stock

### 2.1 The SWA Family: Flat Minima as a Post-Training Objective

Stochastic Weight Averaging traverses the loss landscape with a cyclical or high constant
learning rate, then averages the collected weights to find a point in a flat basin that
generalizes better than any single SGD solution. [Izmailov et al. (2018)](https://arxiv.org/abs/1803.05407)
demonstrated 0.5–1.5% accuracy gains on CIFAR and ImageNet, but the mechanism remained
unclear: was the benefit from the averaging itself or from the cyclical exploration?

[Guo, H. et al. (2023). "Stochastic Weight Averaging Revisited." *Applied Sciences*,
13(5), 2935.](https://arxiv.org/abs/2201.00519) disentangled these contributions through
experiments across 12 architectures and 12 datasets. Their finding is decisive: averaging
provides variance reduction while the cyclical schedule provides exploration. Periodic SWA
(PSWA), which applies repeated averaging phases, discovers global geometric structures in
the loss landscape that a single SWA pass misses. This has direct implications for the
MinIVess factorial design — the SWA learning rate schedule is not a nuisance parameter
but a factor that interacts with model architecture.

The domain generalization variant SWAD ([Cha et al. (2021). "SWAD: Domain Generalization
by Seeking Flat Minima." *NeurIPS*.](https://arxiv.org/abs/2102.08604)) goes further by
averaging only checkpoints from a dense region of the training trajectory where
overfit-to-domain has not yet occurred, using a validation loss criterion to select the
averaging window. For multi-site biomedical data where domain shift is the primary
failure mode, SWAD's selective averaging could outperform uniform SWA.

SWAG ([Maddox et al. (2019). "A Simple Baseline for Bayesian Uncertainty in Deep Learning."
*NeurIPS*.](https://arxiv.org/abs/1902.02476)) extends SWA by additionally collecting
second-moment statistics during training to fit a Gaussian approximate posterior over
weights. This enables Bayesian model averaging at inference time. However, SWAG requires
training-time instrumentation — it cannot be applied post-hoc to existing checkpoints —
making it fundamentally different from the other methods in this section. For the MinIVess
pipeline, which treats post-training as a separate flow operating on saved checkpoints,
SWAG is architecturally incompatible without retrofitting the training loop.

### 2.2 Model Soup: Weight-Space Ensembling Without Inference Cost

Model soups ([Wortsman et al. (2022). "Model Soups." *ICML*.](https://arxiv.org/abs/2203.05482))
average the weights of multiple independently fine-tuned models rather than their
predictions. The result is a single model with the generalization benefit of an ensemble
but no inference-time overhead. The key insight is that fine-tuned models from the same
pretrained initialization tend to lie in a connected low-loss basin in weight space.

[Sanjeev et al. (2024). "FissionFusion: Fast Geometric Generation and Hierarchical
Souping for Medical Image Analysis." *MICCAI 2024*.](https://arxiv.org/abs/2403.13341)
adapted this idea to medical imaging with two innovations. First, Fast Geometric
Generation uses cyclical learning rates to cheaply produce diverse model variants from a
single training run, avoiding the cost of independent fine-tuning. Second, Hierarchical
Souping merges models at local and global levels based on error surface smoothness,
achieving ~6% improvement over vanilla model soups on HAM10000 and CheXpert. The
hierarchical approach is particularly relevant for small datasets where the weight space
may not be as smoothly connected as in large-scale vision tasks.

An even more resource-efficient approach comes from [Jang et al. (2024). "Model Stock:
All we need is just a few fine-tuned models." *ECCV 2024*.](https://arxiv.org/abs/2403.19522),
which demonstrated that layer-wise averaging of just two fine-tuned models can surpass
model soups requiring many models. By exploiting geometric insights about fine-tuned model
positioning relative to the pretrained initialization, Model Stock approximates
center-close weights with minimal compute. For the MinIVess platform, where training even
a single model on 70 volumes takes meaningful GPU time, requiring only two fine-tuned
variants is a substantial practical advantage.

[Ajroldi et al. (2025). "When, Where and Why to Average Weights?" *arXiv preprint
arXiv:2502.06761*.](https://arxiv.org/abs/2502.06761) provides the most comprehensive
empirical evaluation to date, testing weight averaging across seven architectures using the
AlgoPerf benchmark. Their finding that averaging interacts strongly with learning rate
annealing schedules means the factorial design must cross SWA schedule with training
learning rate — they are not independent factors.

### 2.3 So What? Weight Averaging Design Implications

The convergence of these results reveals a practical hierarchy for small-dataset medical
segmentation:

1. **Minimum viable**: Uniform SWA over last-K checkpoints (already implemented in MinIVess)
2. **Better**: Model Stock with two fine-tuned variants (minimal additional cost)
3. **Best**: Hierarchical souping with cyclical LR generation (FissionFusion-style)
4. **Incompatible**: SWAG (requires training-time instrumentation)

The factorial design should treat **SWA schedule** (uniform vs. selective vs. none) and
**merging strategy** (none vs. linear vs. SLERP vs. hierarchical) as crossed factors,
because their interaction with learning rate and loss function is empirically confirmed.

---

## 3. Post-Hoc Calibration: Fixing What Training Gets Wrong

### 3.1 The Calibration Crisis in Medical Segmentation

Standard temperature scaling learns a single scalar T that divides all logits before
softmax, minimizing negative log-likelihood on a held-out calibration set. For natural
image classification, this simple approach is remarkably effective. For medical
segmentation, it fails in at least two ways that recent work has exposed.

First, the spatial structure of segmentation means that calibration needs differ
dramatically between foreground and background voxels. [Zhang et al. (2024). "Mask-TS
Net: Mask Temperature Scaling Uncertainty Calibration for Polyp Segmentation." *arXiv
preprint arXiv:2405.05830*.](https://arxiv.org/abs/2405.05830) demonstrated that a model
can appear well-calibrated on aggregate ECE while being severely miscalibrated on the
lesion region — precisely the region that matters clinically. Their Mask-TS approach
selectively scales logits within potential lesion regions, addressing this spatial
calibration mismatch. For vessel segmentation, where vessels occupy <5% of the volume,
this spatial asymmetry is extreme.

Second, standard temperature scaling degrades on small calibration sets. [Balanya et al.
(2024). "Adaptive Temperature Scaling for Robust Calibration of Deep Neural Networks."
*Neural Computing and Applications*, 36, 8073-8095.](https://arxiv.org/abs/2208.00461)
showed that complex post-hoc calibration models fail catastrophically when calibration
data is limited — a common scenario in medical imaging where validation sets of 23
volumes (as in MinIVess 3-fold CV) are the norm, not the exception. Their Entropy-based
Temperature Scaling adapts the temperature per-sample based on prediction entropy,
providing robustness against data scarcity.

### 3.2 Beyond Temperature: Platt, Isotonic, and Beta Calibration

[Rousseau et al. (2025). "Post hoc calibration of medical segmentation models."
*Discover Applied Sciences*, 7, 180.](https://doi.org/10.1007/s42452-025-06587-0)
provides the most comprehensive comparison of post-hoc calibration methods for medical
segmentation to date. Testing temperature scaling, Platt scaling, isotonic regression,
and novel spatially-aware extensions of beta calibration on BraTS, ISLES, and QUBIQ
datasets, they achieved ECE reductions of up to 67.6%. The key finding: isotonic
regression and beta calibration consistently outperformed temperature scaling,
particularly on datasets with complex multi-class calibration requirements.

This result challenges the conventional wisdom that temperature scaling is "good enough"
for medical segmentation. The MinIVess platform already implements all three methods
(temperature, Platt, isotonic) as post-training plugins. The factorial design should
include calibration method as a factor crossed with loss function, because the baseline
miscalibration pattern differs between Dice-based and cross-entropy-based losses.

### 3.3 The SWA-Calibration Interaction

A largely unexplored interaction connects weight averaging to calibration.
[Cao et al. (2024). "Deep Neural Network Confidence Calibration from Stochastic Weight
Averaging." *Electronics*, 13(3), 503.](https://doi.org/10.3390/electronics13030503)
demonstrated that SWA itself improves calibration — the wider optima found by weight
averaging produce predictions that are inherently less overconfident. This raises a
design question: if SWA already improves calibration, does subsequent temperature
scaling help or hurt?

The answer likely depends on the training loss. Dice-based losses produce models that are
miscalibrated in fundamentally different ways than cross-entropy losses — Dice optimizes
overlap directly without penalizing confident-but-wrong predictions. [Yeung et al. (2023).
"Calibrating the Dice Loss to Handle Neural Network Overconfidence for Biomedical Image
Segmentation." *Journal of Digital Imaging*, 36, 739-752.](https://doi.org/10.1007/s10278-022-00735-3)
introduced DSC++, which extends Dice loss to selectively penalize overconfident incorrect
predictions. The interaction chain is: loss function → baseline calibration →
SWA effect → post-hoc calibration residual. The factorial design must respect this
causal ordering.

---

## 4. Calibration-Aware Training: Shifting Calibration Left

### 4.1 Auxiliary Calibration Losses

Rather than fixing calibration post-hoc, a parallel research direction integrates
calibration objectives directly into the training loss. [Barfoot et al. (2024). "Average
Calibration Error: A Differentiable Loss for Improved Reliability in Image Segmentation."
*MICCAI 2024*, LNCS 15009.](https://doi.org/10.1007/978-3-031-72114-4_14) introduced differentiable
marginal L1 Average Calibration Error (mL1-ACE) as an auxiliary loss, achieving 45%
reduction in average calibration error on BraTS 2021 while maintaining 87% Dice. Their
extended evaluation across four datasets ([Barfoot et al. (2025). "Average Calibration
Losses for Reliable Uncertainty in Medical Image Segmentation." *IEEE TMI*.](https://arxiv.org/abs/2506.03942))
confirmed these gains generalize, with practitioner-controllable calibration-accuracy
trade-offs.

[Larrazabal et al. (2023). "Maximum Entropy on Erroneous Predictions (MEEP)." *MICCAI
2023*.](https://arxiv.org/abs/2112.12218) takes a different approach: instead of adding a
global calibration loss, MEEP selectively maximizes entropy only on misclassified voxels.
This targeted penalization of overconfident errors is architecture-agnostic and compatible
with any base loss function, making it composable with the MinIVess loss registry
(cbdice_cldice, dice_ce, etc.).

### 4.2 So What? Training-Time vs Post-Hoc Calibration

The existence of effective training-time calibration methods creates a design decision:
should calibration be "shifted left" into training, or kept as a separate post-training
step? The answer depends on the experimental design philosophy.

For the NEUROVEX factorial experiment, training-time calibration (DSC++, ACE loss, MEEP)
adds factors to the training phase, increasing the combinatorial space. Post-hoc
calibration (temperature, Platt, isotonic) adds factors to the post-training phase but
leaves training unchanged. Given the platform's existing architecture where post-training
is a separate Prefect flow operating on saved checkpoints, post-hoc methods are
architecturally natural. Training-time methods require modifying the loss composition,
which is config-driven but adds to the training factor count.

The pragmatic recommendation: keep post-hoc calibration as the primary approach (aligned
with existing MinIVess architecture), but include one training-time calibration method
(DSC++ or ACE loss) as a factorial factor to measure the interaction. This tests whether
"shift-left" calibration renders post-hoc calibration unnecessary.

---

## 5. Conformal Prediction: Distribution-Free Guarantees for Segmentation

### 5.1 From Classification to Voxel-Level Coverage

Split conformal prediction provides finite-sample coverage guarantees: for a user-chosen
error rate alpha, the prediction set contains the true label with probability at least
1 - alpha, regardless of the underlying distribution. Extending this from classification
to dense segmentation introduces unique challenges — the number of "test points" (voxels)
per image is enormous, and spatial correlation between adjacent voxels violates the
exchangeability assumption.

[Brunekreef et al. (2024). "Kandinsky Conformal Prediction: Efficient Calibration of
Image Segmentation Algorithms." *CVPR 2024*.](https://openaccess.thecvf.com/content/CVPR2024/papers/Brunekreef_Kandinsky_Conformal_Prediction_Efficient_Calibration_of_Image_Segmentation_Algorithms_CVPR_2024_paper.pdf)
addresses this by aggregating nonconformity scores over similar image regions rather than
treating each pixel independently. This "Kandinsky calibration" occupies a middle ground
between marginal (image-level) and pixelwise calibration, making it practical when labeled
calibration data is scarce — precisely the MinIVess scenario with 23 validation volumes.

### 5.2 Conformal Prediction for Volumetric Medical Segmentation

Three recent works push conformal prediction into 3D medical imaging territory. [Gade
et al. (2024). "Impact of uncertainty quantification through conformal prediction on
volume assessment from deep learning-based MRI prostate segmentation." *Insights into
Imaging*, 15, 297.](https://doi.org/10.1186/s13244-024-01863-w) demonstrated clinical
impact: at 85% confidence, conformal prediction reduced relative volume difference from
-8.01% to -2.81% (ICC: 0.97) by flagging and excluding unreliable voxel predictions.
This is a concrete, clinically meaningful improvement achieved entirely post-training.

[Lambert et al. (2024). "Robust Conformal Volume Estimation in 3D Medical Images."
*MICCAI 2024*.](https://arxiv.org/abs/2407.19938) tackled the distribution shift problem:
when test data comes from a different scanner or protocol than calibration data, standard
conformal guarantees break. Their weighted conformal prediction uses density ratio
estimation on compressed model representations to maintain coverage under covariate shift.
For MinIVess, where future multi-site deployment is planned (see Report R3), this
shift-robust variant is essential.

[Mossina & Friedrich (2025). "Conformal Prediction for Image Segmentation Using
Morphological Prediction Sets." *MICCAI 2025*.](https://arxiv.org/abs/2503.05618)
introduced a model-agnostic approach using mathematical morphology: dilation creates
margins around predicted boundaries, with the margin width quantifying uncertainty. This
requires no internal model access — it works on binary masks alone — making it the most
plug-and-play conformal method available. The MinIVess platform already implements this
as the morphological conformal plugin.

### 5.3 Feature-Space and Risk-Controlled Conformal Methods

[Cheung et al. (2026). "COMPASS: Robust Feature Conformal Prediction for Medical
Segmentation Metrics." *ICLR 2026*.](https://arxiv.org/abs/2509.22240) represents the
frontier: rather than calibrating on output probabilities, COMPASS performs conformal
calibration in the model's learned feature space, perturbing intermediate representations
along metric-relevant subspaces. The result is tighter prediction intervals that are
directly tied to clinically relevant metrics (Dice, Hausdorff).

[Kasa et al. (2025). "Adapting Prediction Sets to Distribution Shifts Without Labels."
*UAI 2025*.](https://arxiv.org/abs/2406.01416) addresses the practical deployment
scenario where test data is unlabeled: their ECP/EACP methods recalibrate conformal
bounds using only the model's own uncertainty signals. This is particularly relevant for
post-deployment monitoring where labeled data is unavailable.

The MinIVess platform already implements split conformal, CRC conformal
([Angelopoulos et al. (2022)](https://arxiv.org/abs/2208.02814)), MAPIE-based conformal,
and morphological conformal. The gaps are: Kandinsky-style region aggregation, weighted
conformal for shift robustness, and feature-space conformal (COMPASS).

### 5.4 So What? Conformal Methods in the Factorial Design

Conformal prediction is a post-training step that depends on the calibration quality of
the underlying model. The factorial design should treat conformal method (none vs. split
vs. CRC vs. morphological) as a post-training factor, but must account for the dependency:
conformal prediction operates on calibrated probabilities, so calibration method is a
prerequisite factor. The interaction chain is:

```
training loss → SWA → calibration method → conformal method → final prediction set
```

Each step in this chain consumes the output of the previous step. The factorial design
must respect this ordering — conformal method cannot be crossed with calibration method
as if they were independent.

---

## 6. The Small-Dataset Challenge: When 70 Volumes Is All You Have

### 6.1 Calibration Data Scarcity

The standard post-hoc calibration recipe assumes a dedicated calibration set separate from
training and test data. With 70 volumes in 3-fold CV (47 train, 23 validation), allocating
further data for calibration is problematic. [Buddenkotte et al. (2023). "Calibrating
ensembles for scalable uncertainty quantification in deep learning-based medical image
segmentation." *Computers in Biology and Medicine*, 163, 107096.](https://doi.org/10.1016/j.compbiomed.2023.107096)
proposed using k-fold cross-validation itself as the calibration mechanism, eliminating the
need for a separate calibration set entirely. The out-of-fold predictions serve double
duty: validation during training and calibration data for post-hoc methods.

This is architecturally significant for MinIVess. The existing 3-fold CV split
(`configs/splits/3fold_seed42.json`) already produces out-of-fold predictions as a
byproduct of training. These predictions can be pooled across folds to create a 70-volume
calibration dataset without any additional data collection. The post-training flow should
implement this pooling as a pre-step before calibration.

### 6.2 Overconfidence Amplification on Small Datasets

Small datasets exacerbate the overconfidence problem. [Balanya et al. (2024)](https://arxiv.org/abs/2208.00461)
demonstrated that post-hoc calibration methods themselves become unreliable when fitted on
small calibration sets — a meta-calibration problem. Their entropy-based temperature
scaling addresses this by making the temperature adaptive rather than global, reducing
sensitivity to calibration set size.

[Mukhoti et al. (2023). "Deep Deterministic Uncertainty: A New Simple Baseline." *CVPR 2023*.](https://arxiv.org/abs/2102.11582)
proposed a feature-space approach: uncertainty is estimated from the distance between test
features and the training feature distribution, using a single deterministic model with
regularized feature spaces. This avoids the calibration data problem entirely — the
uncertainty signal comes from the training data distribution, not a separate calibration
set.

### 6.3 Metrics That Matter

The Expected Calibration Error (ECE) that dominates calibration literature has known
problems: it depends on binning, is biased, and can be gamed. [Kofler et al. (2023). "Are
we using appropriate segmentation metrics?" *JMLBI*.](https://arxiv.org/abs/2103.06205)
argued for metric selection based on the clinical use case, not convention.

For the NEUROVEX manuscript, the MinIVess calibration metrics module already implements
a more robust set: Brier score, O:E ratio, IPA (Index of Prediction Accuracy), and
calibration slope. These should be the primary calibration metrics reported, with ECE
included only for comparability with existing literature.

---

## 7. Factorial Experiment Design Implications

### 7.1 Post-Training Factors for the Factorial Design

Based on the evidence in this report, the post-training pipeline introduces three factor
groups:

| Factor Group | Levels | Rationale |
|-------------|--------|-----------|
| Weight averaging | None, Uniform SWA, Model Stock (2 variants) | Guo H. (2023), Jang (2024) |
| Calibration method | None, Temperature, Isotonic, Platt, Adaptive TS | Rousseau (2025), Balanya (2024) |
| Conformal method | None, Split, CRC, Morphological | Mossina (2025), Angelopoulos (2022) |

### 7.2 Factor Interactions That Must Be Measured

The literature reveals three non-obvious interactions:

1. **Loss × SWA**: Weight averaging interacts with learning rate schedule, which interacts
   with loss function choice. The Dice-family losses produce different loss landscape
   geometry than cross-entropy ([Yeung et al. (2023)](https://doi.org/10.1007/s10278-022-00735-3)).
2. **SWA × Calibration**: SWA itself improves calibration
   ([Cao et al. (2024)](https://doi.org/10.3390/electronics13030503)), so the marginal
   benefit of post-hoc calibration depends on whether SWA was applied.
3. **Calibration × Conformal**: Conformal methods require calibrated probabilities as
   input. Poorly calibrated models produce unreliable nonconformity scores, degrading
   coverage guarantees.

### 7.3 Design Efficiency

Full factorial crossing of all factors is infeasible for an academic lab. [Shi et al.
(2023). "Evaluating Designs for Hyperparameter Tuning in Deep Neural Networks." *NEJSDS*,
1(3), 334-341.](https://doi.org/10.51387/23-NEJSDS26) showed that strong orthogonal
arrays outperform all other design strategies for neural network hyperparameter tuning.
[Fostiropoulos & Itti (2023). "ABLATOR: Robust Horizontal-Scaling of Machine Learning
Ablation Experiments." *AutoML 2023*.](https://proceedings.mlr.press/v224/fostiropoulos23a.html)
provides the infrastructure tooling for running such experiments with automatic error
handling and artifact consolidation.

For MinIVess, the recommended design is a fractional factorial that crosses training
factors (model architecture × loss function × optimizer) with a subset of post-training
factors, using the causal ordering (SWA → calibration → conformal) to reduce the
crossing. Specifically:

- **Training phase**: Full factorial on {architecture, loss, LR schedule}
- **Post-training phase**: Sequential application of best SWA, then best calibration,
  then best conformal — determined by nested holdout within the cross-validation folds

---

## 8. MinIVess Implementation Status

The local inventory (Phase 1) reveals that the MinIVess platform has already implemented
a comprehensive post-training pipeline:

### Implemented (14 methods)
- **Weight averaging**: Uniform SWA, Multi-SWA (random subsampling), Model Merging
  (linear, SLERP, layer-wise)
- **Post-hoc calibration**: Temperature scaling, Isotonic regression, Spatial Platt
  scaling, Calibration metrics (Brier, O:E, IPA, slope)
- **Conformal prediction**: Split conformal, CRC conformal, MAPIE-based, Morphological
  conformal, ConSeCo (FP control)
- **Uncertainty**: MC Dropout, Calibration under shift evaluation
- **Architecture**: PostTrainingPlugin protocol, PluginRegistry, Prefect Flow 2.5

### Gaps Identified
1. **Adaptive temperature scaling** (Balanya 2024) — per-sample entropy-based T
2. **Mask-TS** (Zhang 2024) — spatially-aware temperature scaling for sparse structures
3. **Hierarchical souping** (Sanjeev 2024, FissionFusion) — beyond uniform weight averaging
4. **Model Stock** (Jang 2024) — two-model efficient weight averaging
5. **Kandinsky conformal** (Brunekreef 2024) — region-aggregated nonconformity scores
6. **Weighted conformal for shift** (Lambert 2024) — density-ratio correction
7. **Feature-space conformal** (Cheung 2026, COMPASS) — metric-focused prediction intervals
8. **Cross-fold calibration pooling** — using out-of-fold predictions as calibration data
9. **DSC++ or ACE auxiliary loss** — training-time calibration factor
10. **SWAD-style selective averaging** — validation-guided checkpoint selection

### Priority Recommendations

| Priority | Gap | Effort | Impact | Justification |
|----------|-----|--------|--------|---------------|
| P0 | Cross-fold calibration pooling | Low | High | Eliminates calibration data scarcity problem |
| P0 | Adaptive TS (entropy-based) | Medium | High | Robust on small calibration sets |
| P1 | Model Stock (2-model averaging) | Low | Medium | Minimal compute, proven effective |
| P1 | Mask-TS (spatial calibration) | Medium | High | Critical for sparse vessel segmentation |
| P2 | SWAD selective averaging | Medium | Medium | Better than uniform for domain shift |
| P2 | Hierarchical souping | High | Medium | Complex, unclear benefit on 70 volumes |
| P3 | Feature-space conformal | High | Low | Frontier research, not yet mature |

---

## 9. Discussion: What the Field Has Not Yet Answered

### 9.1 Specific Research Gaps

1. **Loss-calibration interaction on small 3D datasets**: No published study crosses
   {Dice, CE, clDice, cbDice} × {temperature, Platt, isotonic} on datasets smaller than
   200 volumes. The MinIVess factorial experiment would be the first.

2. **SWA schedule sensitivity in medical segmentation**: The weight averaging literature
   (Guo H. 2023, Ajroldi 2025) uses computer vision benchmarks. Whether the optimal SWA
   schedule transfers to small 3D medical datasets is untested.

3. **Conformal prediction with topological losses**: All conformal methods in the
   literature use pixel/voxel-level nonconformity scores. How topology-preserving losses
   (clDice, cbDice) affect the spatial structure of conformal prediction sets is unknown.

4. **Post-training pipeline ordering effects**: The sequential ordering (SWA → calibration
   → conformal) is assumed but not empirically validated. Whether calibration before or
   after SWA produces better conformal coverage is an open question.

5. **Calibration stability across cross-validation folds**: With only 3 folds and 23
   validation volumes each, calibration parameter estimates may have high variance across
   folds. No study quantifies this instability for medical segmentation.

### 9.2 Recommendations for the NEUROVEX Manuscript

The post-training pipeline section should:
1. Report calibration metrics (Brier, IPA, slope) alongside segmentation metrics (Dice,
   clDice, Hausdorff) — never segmentation alone
2. Include reliability diagrams per model architecture
3. Report conformal prediction coverage at multiple alpha levels (0.05, 0.10, 0.20)
4. Test the SWA × calibration interaction explicitly (not just report the final pipeline)
5. Use cross-fold calibration pooling to maximize calibration data

---

## References

### Seed Papers (9)

1. [Izmailov, P. et al. (2018). "Averaging Weights Leads to Wider Optima and Better Generalization." *UAI*.](https://arxiv.org/abs/1803.05407)
2. [Wortsman, M. et al. (2022). "Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy without Increasing Inference Time." *ICML*.](https://arxiv.org/abs/2203.05482)
3. [Guo, C. et al. (2017). "On Calibration of Modern Neural Networks." *ICML*.](https://arxiv.org/abs/1706.04599)
4. [Mehrtash, A. et al. (2020). "Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation." *IEEE TMI*.](https://doi.org/10.1109/TMI.2020.3006437)
5. [Angelopoulos, A.N. et al. (2022). "Conformal Risk Control." *ICLR 2024*.](https://arxiv.org/abs/2208.02814)
6. [Kofler, F. et al. (2023). "Are we using appropriate segmentation metrics?" *JMLBI*.](https://arxiv.org/abs/2103.06205)
7. [Mukhoti, J. et al. (2023). "Deep Deterministic Uncertainty: A New Simple Baseline." *CVPR 2023*.](https://arxiv.org/abs/2102.11582)
8. [Cha, J. et al. (2021). "SWAD: Domain Generalization by Seeking Flat Minima." *NeurIPS*.](https://arxiv.org/abs/2102.08604)
9. [Maddox, W.J. et al. (2019). "A Simple Baseline for Bayesian Uncertainty in Deep Learning." *NeurIPS*.](https://arxiv.org/abs/1902.02476)
### Discovered Papers (22)

11. [Rousseau, A.-J. et al. (2025). "Post hoc calibration of medical segmentation models." *Discover Applied Sciences*, 7, 180.](https://doi.org/10.1007/s42452-025-06587-0)
12. [Zhang, Y. et al. (2024). "Mask-TS Net: Mask Temperature Scaling Uncertainty Calibration for Polyp Segmentation." *arXiv preprint arXiv:2405.05830*.](https://arxiv.org/abs/2405.05830)
13. [Barfoot, T. et al. (2024). "Average Calibration Error: A Differentiable Loss for Improved Reliability in Image Segmentation." *MICCAI 2024*, LNCS 15009.](https://doi.org/10.1007/978-3-031-72114-4_14)
14. [Barfoot, T. et al. (2025). "Average Calibration Losses for Reliable Uncertainty in Medical Image Segmentation." *IEEE TMI*.](https://arxiv.org/abs/2506.03942)
15. [Yeung, M. et al. (2023). "Calibrating the Dice Loss to Handle Neural Network Overconfidence for Biomedical Image Segmentation." *Journal of Digital Imaging*, 36, 739-752.](https://doi.org/10.1007/s10278-022-00735-3)
16. [Brunekreef, J. et al. (2024). "Kandinsky Conformal Prediction: Efficient Calibration of Image Segmentation Algorithms." *CVPR 2024*.](https://openaccess.thecvf.com/content/CVPR2024/papers/Brunekreef_Kandinsky_Conformal_Prediction_Efficient_Calibration_of_Image_Segmentation_Algorithms_CVPR_2024_paper.pdf)
17. [Gade, M. et al. (2024). "Impact of uncertainty quantification through conformal prediction on volume assessment from deep learning-based MRI prostate segmentation." *Insights into Imaging*, 15, 297.](https://doi.org/10.1186/s13244-024-01863-w)
18. [Lambert, B. et al. (2024). "Robust Conformal Volume Estimation in 3D Medical Images." *MICCAI 2024*, LNCS 15009.](https://arxiv.org/abs/2407.19938)
19. [Mossina, L. & Friedrich, C. (2025). "Conformal Prediction for Image Segmentation Using Morphological Prediction Sets." *MICCAI 2025*.](https://arxiv.org/abs/2503.05618)
20. [Cheung, M.Y. et al. (2026). "COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics." *ICLR 2026*.](https://arxiv.org/abs/2509.22240)
21. [Sanjeev, S. et al. (2024). "FissionFusion: Fast Geometric Generation and Hierarchical Souping for Medical Image Analysis." *MICCAI 2024*, LNCS 15010.](https://arxiv.org/abs/2403.13341)
22. [Jang, D.-H. et al. (2024). "Model Stock: All we need is just a few fine-tuned models." *ECCV 2024*.](https://arxiv.org/abs/2403.19522)
23. [Larrazabal, A. et al. (2023). "Maximum Entropy on Erroneous Predictions (MEEP)." *MICCAI 2023*, LNCS 14222.](https://arxiv.org/abs/2112.12218)
24. [Buddenkotte, T. et al. (2023). "Calibrating ensembles for scalable uncertainty quantification in deep learning-based medical image segmentation." *Computers in Biology and Medicine*, 163, 107096.](https://doi.org/10.1016/j.compbiomed.2023.107096)
25. [Balanya, S.A. et al. (2024). "Adaptive Temperature Scaling for Robust Calibration of Deep Neural Networks." *Neural Computing and Applications*, 36, 8073-8095.](https://arxiv.org/abs/2208.00461)
26. [Cao, Z. et al. (2024). "Deep Neural Network Confidence Calibration from Stochastic Weight Averaging." *Electronics*, 13(3), 503.](https://doi.org/10.3390/electronics13030503)
27. [Ajroldi, N. et al. (2025). "When, Where and Why to Average Weights?" *arXiv preprint arXiv:2502.06761*.](https://arxiv.org/abs/2502.06761)
28. [Meronen, L. et al. (2024). "Fixing Overconfidence in Dynamic Neural Networks." *WACV 2024*.](https://arxiv.org/abs/2302.06359)
29. [Kasa, K. et al. (2025). "Adapting Prediction Sets to Distribution Shifts Without Labels." *UAI 2025*, PMLR 286.](https://arxiv.org/abs/2406.01416)
30. [Guo, H. et al. (2023). "Stochastic Weight Averaging Revisited." *Applied Sciences*, 13(5), 2935.](https://arxiv.org/abs/2201.00519)
31. [Fostiropoulos, I. & Itti, L. (2023). "ABLATOR: Robust Horizontal-Scaling of Machine Learning Ablation Experiments." *AutoML 2023*, PMLR 224.](https://proceedings.mlr.press/v224/fostiropoulos23a.html)
32. [Shi, C. et al. (2023). "Evaluating Designs for Hyperparameter Tuning in Deep Neural Networks." *NEJSDS*, 1(3), 334-341.](https://doi.org/10.51387/23-NEJSDS26)
