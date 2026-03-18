# Ensemble Strategies and Uncertainty Quantification for Biomedical Segmentation

**Literature Research Report R2** | 28 papers | 2026-03-18
**Manuscript sections**: Methods: Evaluation Pipeline (R2c), Results: Multi-Model Comparison (R3b)
**KG domains**: training, architecture
**Quality target**: MINOR_REVISION
**Dedup**: Papers already cited in R1 (post-training methods) are excluded.

---

## 1. Introduction: The Ensemble Paradox in Small-Dataset Segmentation

Deep ensembles remain the gold standard for uncertainty quantification in neural networks
([Lakshminarayanan et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation
using Deep Ensembles." *NeurIPS*.](https://arxiv.org/abs/1612.01474)). Yet for a platform
like MinIVess with 70 volumes and 3-fold cross-validation, the standard recipe of training
M independent models multiplies compute by M without proportional gains — an ensemble of
3 models trained on 47 volumes each may be less diverse than desired.

This report examines three questions the NEUROVEX manuscript must answer: (1) which
ensemble strategy is scientifically justified for small-dataset 3D segmentation, (2) how
to select the champion model from a factorial experiment, and (3) what calibration and
evaluation methodology makes the comparison rigorous. The answers inform the platform's
five ensemble strategies (per_loss_single_best, all_loss_single_best, per_loss_all_best,
all_loss_all_best, cv_average) and the champion selection logic in the deploy flow.

**Scope**: Deep ensembles, MC Dropout limitations, evidential deep learning, ensemble
diversity, champion selection, evaluation methodology, calibration metrics. **Excluded**:
SWA/model soup (R1), federated learning (R3), regulatory compliance (R4).

---

## 2. Do Deep Ensembles Actually Help on Small Datasets?

### 2.1 The Diversity Problem

The theoretical justification for deep ensembles rests on functional diversity —
independent training runs explore different modes of the loss landscape
([Fort et al. (2019). "Deep Ensembles: A Loss Landscape Perspective."
*arXiv*.](https://arxiv.org/abs/1912.02757)). But [Abe et al. (2022). "Deep Ensembles
Work, But Are They Necessary?" *NeurIPS*.](https://arxiv.org/abs/2202.06985)
demonstrated that much of the ensemble benefit comes from implicit regularization
rather than mode diversity, and that single well-calibrated models can match ensemble
performance in many settings.

For the MinIVess factorial design, the question is sharper: with only 47 training volumes
per fold, do independently initialized models actually explore different modes? Or do they
converge to similar solutions due to the small data constraint? The existing ensemble
strategies (4 losses × 3 folds = 12 models in ALL_LOSS_ALL_BEST) introduce diversity
through loss function variation rather than random initialization, which may be more
effective than traditional deep ensembles on small datasets.

### 2.2 Hardware-Efficient Alternatives

[Blorstad et al. (2026). "Evaluating Prediction Uncertainty Estimates from BatchEnsemble."
*arXiv:2601.21581*.](https://arxiv.org/abs/2601.21581) demonstrated that BatchEnsemble
— which shares most weights across ensemble members with only rank-1 perturbations — matches
deep ensemble uncertainty quality while requiring a fraction of the parameters.
For MinIVess running on an 8 GB RTX 2070 Super locally, BatchEnsemble is the most
practical path to ensemble UQ without cloud compute.

This creates a design choice: invest GPU hours in traditional deep ensembles (high
diversity, high cost) or use hardware-efficient alternatives (lower diversity, deployable
locally). The factorial design should compare both approaches on the same validation set
to quantify the diversity-efficiency tradeoff.

---

## 3. MC Dropout: Limitations Exposed

### 3.1 The Boundary Correlation Failure

MC Dropout has been the default uncertainty method in medical segmentation due to its
simplicity — enable dropout at test time, run N stochastic passes, compute variance.
However, [Saumya (2025). "An Empirical Study on MC Dropout-Based Uncertainty-Error
Correlation in 2D Brain Tumor Segmentation." *arXiv:2510.15541*.](https://arxiv.org/abs/2510.15541)
exposed a critical limitation: global correlation between MC Dropout uncertainty and
segmentation error is weak (r = 0.30–0.38), and **boundary correlation is negligible**
(|r| < 0.05). For vessel segmentation where errors concentrate at vessel walls, this
means MC Dropout provides almost no useful signal where it matters most.

This negative result is important for MinIVess. The platform implements MC Dropout
(`mc_dropout.py`) but should not rely on it as the primary uncertainty method for
boundary-sensitive applications. Deep ensembles or conformal prediction (R1) provide
stronger guarantees.

### 3.2 Frequency-Domain Alternatives

[Zeevi et al. (2025). "Enhancing Uncertainty Estimation in Semantic Segmentation via
Monte-Carlo Frequency Dropout." *IEEE ISBI 2025*.](https://arxiv.org/abs/2501.11258)
proposed an alternative: instead of dropping neurons, stochastically attenuate signal
frequencies during inference. MC-Frequency Dropout creates textural variations that
preserve structural integrity, improving calibration on prostate, liver, and lung
segmentation. For multiphoton vascular imaging where texture carries diagnostic
information, this frequency-domain approach may capture uncertainty that spatial dropout
misses.

### 3.3 Comprehensive UQ Method Comparison

[Scalco et al. (2024). "Uncertainty Quantification in Multi-Class Segmentation:
Comparison Between Bayesian and Non-Bayesian Approaches in a Clinical Perspective."
*Medical Physics*, 51(9):6090-6102.](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.17189)
compared four UQ methods (Bayesian Dropout, Bayesian DropConnect, ensemble, TTA) on
kidney CT segmentation and found distinct **risk profiles**: Bayesian Dropout achieves
highest accuracy but poorest uncertainty detection, while TTA provides the most
conservative assessments. The implication: method choice should depend on clinical risk
tolerance, not just accuracy.

[Sikha et al. (2025). "Uncertainty-Aware Segmentation Quality Prediction." *Computerized
Medical Imaging and Graphics*, 123:102547.](https://pubmed.ncbi.nlm.nih.gov/40250215/)
extended this comparison across 2D (skin lesion) and 3D (liver) modalities, finding that
TTA with entropy achieves best cross-modality robustness. Their Grad-CAM interpretation
layer provides explainability alongside uncertainty — a feature the MinIVess dashboard
should adopt.

---

## 4. Evidential Deep Learning: A Single-Forward-Pass Alternative

### 4.1 Dirichlet-Based Uncertainty

Rather than running multiple forward passes, evidential deep learning places a Dirichlet
prior over class probabilities, estimating all uncertainty types from a single prediction.
[Li et al. (2025). "Uncertainty-Supervised Interpretable and Robust Evidential
Segmentation." *MICCAI 2025*.](https://papers.miccai.org/miccai-2025/0970-Paper3770.html)
introduced three human-intuition-inspired loss functions for evidential segmentation,
plus novel evaluation metrics (UCC — Uncertainty-Calibration Curve, and UR — Uncertainty
Ratio) that measure how well uncertainty correlates with actual errors. Validated on
cardiac MRI and fundus photography.

[Huang et al. (2024). "Deep Evidential Fusion with Uncertainty Quantification and
Reliability Learning for Multimodal Medical Image Segmentation." *Information Fusion*,
113:102648.](https://www.sciencedirect.com/science/article/abs/pii/S1566253524004263)
extended this to multi-modal fusion using Dempster-Shafer theory: per-model evidence is
combined with learned reliability coefficients. For MinIVess's multi-architecture
ensemble (DynUNet, SegResNet, SAM3), this provides a principled way to weight model
contributions based on estimated reliability rather than naive averaging.

### 4.2 So What? Single-Pass vs Multi-Pass UQ

The trade-off is clear: evidential methods eliminate the N-forward-pass cost of ensembles
and MC Dropout, but require modifying the model's output head and loss function. For the
MinIVess architecture where models are trained with task-agnostic loss registries, adding
an evidential head means adding another training factor. The recommendation: implement
evidential deep learning as a separate model variant (not as a replacement for the
existing UQ stack) and compare it against ensembles in the factorial experiment.

---

## 5. Selective Prediction: When to Abstain

### 5.1 Image-Level Abstention

Not all predictions are worth making. [Borges et al. (2024). "Soft Dice Confidence: A
Near-Optimal Confidence Estimator for Selective Prediction in Semantic Segmentation."
*arXiv:2402.10665*.](https://arxiv.org/abs/2402.10665) derived a theoretically optimal
image-level confidence estimator and proposed Soft Dice Confidence (SDC) as a practical
approximation. SDC outperforms all previous abstention methods without requiring
additional tuning data, validated on six medical imaging tasks including
out-of-distribution scenarios.

For MinIVess, SDC provides a principled quality gate: segmentations below the SDC
threshold are flagged for expert review in the annotation dashboard rather than being
silently deployed. This integrates naturally with the Prefect flow architecture —
the evaluation flow computes SDC and routes low-confidence predictions to the human
review queue.

### 5.2 Expert Disagreement as Ground Truth for UQ

[Abutalip et al. (2024). "EDUE: Expert Disagreement-Guided One-Pass Uncertainty
Estimation for Medical Image Segmentation." *arXiv:2403.16594*.](https://arxiv.org/abs/2403.16594)
leverages multi-rater annotation variability to train uncertainty-aware models in a single
forward pass, achieving 55% improvement in correlation with expert disagreement compared
to standard UQ methods. The key insight: for datasets with multiple annotators, the
inter-rater variability IS the aleatoric uncertainty ground truth. MinIVess currently
uses single-rater annotations for MiniVess, but QUBIQ-style multi-rater data would
enable this approach.

---

## 6. Foundation Model UQ: Ensembles Without Training

[Shen et al. (2024). "FastSAM-3DSlicer: A 3D-Slicer Extension for 3D Volumetric Segment
Anything Model with Uncertainty Quantification."](https://pmc.ncbi.nlm.nih.gov/articles/PMC12292515/)
demonstrated that prompt sampling creates pseudo-ensembles from a single foundation model:
generate multiple prompts from an initial mask, run inference for each, and compute
voxel-level variance. This achieves ensemble-quality UQ from a single model without
training multiple copies. For SAM3 integration in MinIVess, this is directly applicable —
instead of training 3 SAM3 models (prohibitive given 648M parameters), sample 10
different prompts and aggregate.

---

## 7. Evaluation Methodology: Getting the Comparison Right

### 7.1 Rigorous Validation Standards

[Isensee et al. (2024). "nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical
Image Segmentation." *MICCAI 2024*.](https://arxiv.org/abs/2404.09556) demonstrated that
most recent architecture claims fail under rigorous validation — properly configured CNNs
(ResNet, ConvNeXt) still match or exceed Transformer and Mamba variants when controlled
for compute. This validates MinIVess's decision to include CNN baselines (DynUNet,
SegResNet) alongside foundation models, and sets the methodological bar: **every model
comparison must control for training compute, data augmentation, and hyperparameter
budget**.

### 7.2 Cross-Validation Best Practices

[Bradshaw et al. (2023). "A Guide to Cross-Validation for Artificial Intelligence in
Medical Imaging." *Radiology: AI*,
5(4):e220232.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10388213/) provides the
definitive practical guide, identifying common pitfalls: data leakage from insufficient
patient-level splitting, repeated test-set tuning, and non-representative test sets. The
MinIVess 3-fold CV setup (`configs/splits/3fold_seed42.json`) follows patient-level
splitting, but the manuscript must explicitly document this and reference Bradshaw's
checklist.

### 7.3 Nested Cross-Validation for HPO

[Calle et al. (2025). "Integration of nested cross-validation, automated hyperparameter
optimization, high-performance computing." *arXiv:2503.08589*.](https://arxiv.org/abs/2503.08589)
exposed a gap in current practice: when HPO (Optuna, ASHA) operates on the same CV folds
used for evaluation, performance estimates are optimistically biased. Their NACHOS/DACHOS
frameworks nest HPO inside CV folds to eliminate this bias. For MinIVess, this means the
current Optuna+ASHA pipeline should ideally be nested inside the 3-fold CV — or at
minimum, the manuscript should acknowledge this limitation and quantify the potential bias.

### 7.4 Metrics Selection Framework

[Maier-Hein et al. (2024). "Metrics Reloaded: Recommendations for image analysis
validation." *Nature Methods*,
21:195-212.](https://doi.org/10.1038/s41592-023-02151-z) establishes the problem
fingerprint concept: structured representation of all aspects relevant to metric
selection. For vessel segmentation, the fingerprint includes topology-sensitive metrics
(clDice), boundary metrics (Hausdorff, MASD), overlap metrics (Dice), and calibration
metrics (Brier, IPA). The MONAI-integrated implementations can be used directly.

---

## 8. Test-Time Augmentation as Cheap Uncertainty

### 8.1 Classical TTA

[Wang et al. (2019). "Aleatoric uncertainty estimation with test-time augmentation for
medical image segmentation with convolutional neural networks." *Neurocomputing*,
338:34-45.](https://arxiv.org/abs/1807.07356) established the theoretical formulation of
TTA as Monte Carlo simulation with prior distributions of acquisition parameters. TTA
provides aleatoric uncertainty estimates that complement MC Dropout's epistemic estimates,
and is already implementable via MONAI's TTA utilities.

### 8.2 Generative TTA

[Ma et al. (2024). "Test-Time Generative Augmentation for Medical Image Segmentation."
*arXiv:2406.17608*.](https://arxiv.org/html/2406.17608v1) proposed TTGA, using Stable
Diffusion to generate realistic content-aware variations rather than simple geometric
transforms. While the generative model dependency is heavy for the MinIVess pipeline,
the evaluation methodology — comparing TTA vs MC Dropout vs deep ensembles on the same
test set — provides a template for the NEUROVEX multi-UQ comparison.

---

## 9. Ensemble Strategies for Class-Imbalanced Segmentation

### 9.1 Diversity Through Data Resampling

[Roshan et al. (2024). "A deep ensemble medical image segmentation with novel sampling
method and loss function." *Computers in Biology and Medicine*,
172:108305.](https://pubmed.ncbi.nlm.nih.gov/38503087/) proposed a dual-model ensemble
where one model trains on standard data and another on resampled data with class balance
correction. For vessel segmentation where vessels occupy <5% of volume, this data-driven
diversity injection is more natural than random initialization diversity.

### 9.2 Stacked Ensembles

[Dang et al. (2024). "Two-layer Ensemble of Deep Learning Models for Medical Image
Segmentation." *Cognitive Computation*,
16:1141-1160.](https://link.springer.com/article/10.1007/s12559-024-10257-5) proposed a
two-layer architecture where first-layer predictions become training data for the second
layer, with linear regression weighting. This learned stacking is more principled than
naive averaging and lightweight enough for small datasets.

### 9.3 Few-Shot Uncertainty

[Hu et al. (2026). "Uncertainty-guided Prototype Reliability Enhancement Network for
Few-Shot Medical Image Segmentation." *IEEE TMI*,
45(3):1279-1290.](https://pubmed.ncbi.nlm.nih.gov/41086067/) addressed UQ under extreme
data scarcity through uncertainty-guided prototype selection with dual augmentation
branches. While MinIVess's 70 volumes is not technically few-shot, the prototype
selection mechanism could benefit cross-dataset transfer (MiniVess → DeepVess → VesselNN).

---

## 10. Bayesian UQ: A Unified Framework

[Valiuddin et al. (2024). "A Review of Bayesian Uncertainty Quantification in Deep
Probabilistic Image Segmentation." *TMLR*.](https://arxiv.org/abs/2411.16370) provides
the most comprehensive taxonomy of Bayesian UQ methods for segmentation, establishing a
unified framework that standardizes theory, notation, and terminology. Their four-task
framework — Observer Variability, Active Learning, Model Introspection, Model
Generalization — directly structures how MinIVess should organize its UQ outputs. The
identified gaps (uncertainty type distinction, spatial aggregation, benchmark
standardization) are exactly the contribution the NEUROVEX platform can make as a
reproducible UQ evaluation testbed.

---

## 11. MinIVess Implementation Status and Gaps

### Implemented
- **Deep ensembles**: Full Lakshminarayanan-style uncertainty decomposition (total,
  aleatoric, epistemic) via `deep_ensembles.py`
- **Ensemble builder**: MLflow-based with 4 strategies (per_loss_single_best,
  all_loss_single_best, per_loss_all_best, all_loss_all_best)
- **MC Dropout**: N stochastic passes with entropy-based uncertainty maps
- **Calibration metrics**: ECE, MCE, Brier, O:E, IPA, calibration slope
- **Champion selection**: Primary metric configurable (default: val_compound_masd_cldice)
- **Generative UQ**: GED, Q-Dice for probabilistic models (QUBIQ benchmark)
- **Risk control**: Learn-Then-Test (LTT) with dice_loss_risk, fnr_risk, fpr_risk
- **WeightWatcher**: Spectral analysis for model quality gate

### Gaps Identified

| Priority | Gap | Effort | Impact | Reference |
|----------|-----|--------|--------|-----------|
| P0 | Nested CV for HPO | Medium | High | Calle et al. (2025) |
| P0 | Soft Dice Confidence quality gate | Low | High | Borges et al. (2024) |
| P1 | BatchEnsemble for local GPU | Medium | High | Blorstad et al. (2026) |
| P1 | MC Frequency Dropout | Medium | Medium | Zeevi et al. (2025) |
| P1 | Ensemble diversity metrics | Low | Medium | No existing implementation |
| P2 | Evidential deep learning head | High | Medium | Li et al. (2025), Huang et al. (2025) |
| P2 | Prompt-sampling pseudo-ensemble for SAM3 | Medium | Medium | Shen et al. (2024) |
| P2 | Stacked ensemble (learned weighting) | Low | Medium | Dang et al. (2024) |
| P3 | Generative TTA | High | Low | Ma et al. (2024) |

---

## 12. Discussion: Research Gaps and Factorial Design

### 12.1 Specific Research Gaps

1. **Ensemble diversity quantification on small 3D datasets**: No published study
   measures ensemble diversity (Q-statistic, correlation, double-fault) for <100 volume
   medical segmentation. The MinIVess factorial experiment, with 4 loss × 3 fold × N
   architecture combinations, can uniquely measure how loss function diversity compares
   to initialization diversity.

2. **MC Dropout boundary failure for tubular structures**: Saumya (2025) showed boundary
   correlation failure for brain tumors. Whether this extends to thin tubular structures
   (vessels) is untested — the geometry is fundamentally different.

3. **Champion selection stability**: With 3 folds and 23 validation volumes each, the
   "best model" may change depending on which fold is held out. No study quantifies
   champion selection instability for small medical segmentation datasets.

4. **Ensemble strategy × post-training interaction**: How does ensemble strategy
   (single best vs. all best) interact with post-training methods (SWA, calibration)?
   The R1 report identified the pipeline ordering; this report adds the ensemble
   dimension.

### 12.2 Recommendations for the NEUROVEX Manuscript

1. Report ensemble diversity alongside ensemble accuracy — diversity without accuracy
   gain is wasted compute
2. Include MC Dropout boundary analysis (negative result if confirmed)
3. Use Soft Dice Confidence as quality gate, report abstention rate at various thresholds
4. Compare at least 3 UQ methods (ensemble, MC Dropout, conformal) on the same test set
5. Adopt the Metrics Reloaded problem fingerprint for metric justification
6. Acknowledge nested CV gap and quantify potential HPO optimism bias

---

## References

### Seed Papers (9)

1. [Lakshminarayanan, B. et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS*.](https://arxiv.org/abs/1612.01474)
2. [Fort, S. et al. (2019). "Deep Ensembles: A Loss Landscape Perspective." *arXiv*.](https://arxiv.org/abs/1912.02757)
3. [Jungo, A. & Reyes, M. (2019). "Assessing Reliability and Challenges of Uncertainty Estimations for Medical Image Segmentation." *MICCAI*.](https://doi.org/10.1007/978-3-030-32245-8_6)
4. [Mehrtash, A. et al. (2020). "Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation." *IEEE TMI*.](https://doi.org/10.1109/TMI.2020.3006437)
5. [Maier-Hein, L. et al. (2024). "Metrics Reloaded: Recommendations for image analysis validation." *Nature Methods*, 21:195-212.](https://doi.org/10.1038/s41592-023-02151-z)
6. [Ovadia, Y. et al. (2019). "Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift." *NeurIPS*.](https://arxiv.org/abs/1906.02530)
7. [Naeini, M.P. et al. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles." *AAAI*.](https://doi.org/10.1609/aaai.v29i1.9602)
8. [Rahaman, R. & Thiery, A.H. (2021). "Uncertainty Quantification and Deep Ensembles." *NeurIPS*.](https://arxiv.org/abs/2007.08792)
9. [Abe, T. et al. (2022). "Deep Ensembles Work, But Are They Necessary?" *NeurIPS*.](https://arxiv.org/abs/2202.06985)

### Discovered Papers (19)

11. [Li, Y. et al. (2025). "Uncertainty-Supervised Interpretable and Robust Evidential Segmentation." *MICCAI 2025*.](https://papers.miccai.org/miccai-2025/0970-Paper3770.html)
12. [Saumya, B. (2025). "An Empirical Study on MC Dropout-Based Uncertainty-Error Correlation in 2D Brain Tumor Segmentation." *arXiv:2510.15541*.](https://arxiv.org/abs/2510.15541)
13. [Zeevi, T. et al. (2025). "Enhancing Uncertainty Estimation in Semantic Segmentation via Monte-Carlo Frequency Dropout." *IEEE ISBI 2025*.](https://arxiv.org/abs/2501.11258)
14. [Borges, B.L.C. et al. (2024). "Soft Dice Confidence: A Near-Optimal Confidence Estimator for Selective Prediction in Semantic Segmentation." *arXiv:2402.10665*.](https://arxiv.org/abs/2402.10665)
15. [Abutalip, K. et al. (2024). "EDUE: Expert Disagreement-Guided One-Pass Uncertainty Estimation for Medical Image Segmentation." *arXiv:2403.16594*.](https://arxiv.org/abs/2403.16594)
16. [Shen, Y. et al. (2024). "FastSAM-3DSlicer: A 3D-Slicer Extension for 3D Volumetric Segment Anything Model with Uncertainty Quantification."](https://pmc.ncbi.nlm.nih.gov/articles/PMC12292515/)
17. [Scalco, E. et al. (2024). "Uncertainty Quantification in Multi-Class Segmentation: Comparison Between Bayesian and Non-Bayesian Approaches in a Clinical Perspective." *Medical Physics*, 51(9):6090-6102.](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.17189)
18. [Valiuddin, M.M.A. et al. (2024). "A Review of Bayesian Uncertainty Quantification in Deep Probabilistic Image Segmentation." *TMLR*.](https://arxiv.org/abs/2411.16370)
19. [Huang, L. et al. (2024). "Deep Evidential Fusion with Uncertainty Quantification and Reliability Learning for Multimodal Medical Image Segmentation." *Information Fusion*, 113:102648.](https://www.sciencedirect.com/science/article/abs/pii/S1566253524004263)
20. [Isensee, F. et al. (2024). "nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation." *MICCAI 2024*.](https://arxiv.org/abs/2404.09556)
21. [Sikha, O.K. et al. (2025). "Uncertainty-Aware Segmentation Quality Prediction." *Computerized Medical Imaging and Graphics*, 123:102547.](https://pubmed.ncbi.nlm.nih.gov/40250215/)
22. [Roshan, S.E. et al. (2024). "A deep ensemble medical image segmentation with novel sampling method and loss function." *Computers in Biology and Medicine*, 172:108305.](https://pubmed.ncbi.nlm.nih.gov/38503087/)
23. [Bradshaw, T.J. et al. (2023). "A Guide to Cross-Validation for Artificial Intelligence in Medical Imaging." *Radiology: AI*, 5(4):e220232.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10388213/)
24. [Calle, P. et al. (2025). "Integration of nested cross-validation, automated hyperparameter optimization, high-performance computing." *arXiv:2503.08589*.](https://arxiv.org/abs/2503.08589)
25. [Blorstad, M. et al. (2026). "Evaluating Prediction Uncertainty Estimates from BatchEnsemble." *arXiv:2601.21581*.](https://arxiv.org/abs/2601.21581)
26. [Ma, X. et al. (2024). "Test-Time Generative Augmentation for Medical Image Segmentation." *arXiv:2406.17608*.](https://arxiv.org/html/2406.17608v1)
27. [Wang, G. et al. (2019). "Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks." *Neurocomputing*, 338:34-45.](https://arxiv.org/abs/1807.07356)
28. [Dang, T. et al. (2024). "Two-layer Ensemble of Deep Learning Models for Medical Image Segmentation." *Cognitive Computation*, 16:1141-1160.](https://link.springer.com/article/10.1007/s12559-024-10257-5)
29. [Hu, J. et al. (2026). "Uncertainty-guided Prototype Reliability Enhancement Network for Few-Shot Medical Image Segmentation." *IEEE TMI*, 45(3):1279-1290.](https://pubmed.ncbi.nlm.nih.gov/41086067/)
