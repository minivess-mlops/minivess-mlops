# Biomedical Segmentation: Tubular Structures, Foundation Models, and Statistical Evaluation

**Status**: Complete (v1.0 — based on 28 web-discovered + 15 seed papers)
**Date**: 2026-03-18
**Theme**: R3 (from research-reports-general-plan-for-manuscript-writing.md)
**Audience**: NEUROVEX manuscript Methods + Results sections
**Paper count**: 43 (15 seeds + 28 web-discovered, pre-verification)

---

## 1. Introduction: Three Threads That Define Modern Biomedical Segmentation

Biomedical image segmentation in 2024–2026 is shaped by three converging threads: topology-aware loss functions that preserve the connectedness of tubular structures, foundation models that promise zero-shot generalization, and statistical evaluation frameworks that demand rigorous reporting. No single paper addresses how these threads interact in a factorial experimental design — yet this is precisely what NEUROVEX implements.

We synthesize 43 papers to argue that: (a) topology-aware losses like clDice and cbDice are necessary but not sufficient — they must be evaluated in factorial combination with model architectures; (b) foundation models (SAM3, VesselFM) change the evaluation landscape by introducing zero-shot baselines; and (c) current statistical practices in segmentation studies are inadequate — voxel-level metrics create the illusion of large sample sizes while the effective sample size is the number of volumes.

---

## 2. Thread A: Topology-Aware Loss Functions for Tubular Segmentation

### 2.1 The clDice Revolution

[Shit et al. (2021). "clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation." *CVPR 2021*.](https://arxiv.org/abs/2003.07311) introduced centerlineDice, computed on the intersection of segmentation masks and their morphological skeletons, with theoretical proof of topology preservation up to homotopy equivalence. This paper is the foundation for NEUROVEX's default loss function (`cbdice_cldice`).

[Shi et al. (2024). "Centerline Boundary Dice Loss for Vascular Segmentation." *MICCAI 2024*.](https://arxiv.org/abs/2407.01517) extended clDice with cbDice, incorporating boundary-aware aspects and radius information. [Acebes et al. (2024). "The Centerline-Cross Entropy Loss." *MICCAI 2024*.](https://papers.miccai.org/miccai-2024/770-Paper1081.html) proposes clCE, combining cross-entropy robustness with centerline topology focus.

### 2.2 Beyond Loss Functions: Topological Correctness as a First Principle

The field has moved from topology-aware loss functions to topology-preserving architectures. [Berger et al. (2024). "Topologically Faithful Multi-class Segmentation." *MICCAI 2024*.](https://arxiv.org/abs/2403.11001) extends Betti matching to multi-class problems. [Lux et al. (2025). "Topograph: An efficient Graph-Based Framework." *ICLR 2025*.](https://arxiv.org/abs/2411.03228) achieves 5x faster computation than persistent homology methods. [Gupta et al. (2023). "Topology-Aware Uncertainty for Image Segmentation." *NeurIPS 2023*.](https://arxiv.org/abs/2306.05671) proposes uncertainty estimation in units of topological structures rather than pixels.

Most relevant to NEUROVEX: [Haft-Javaherian et al. (2020). "Topological Encoding CNN for Segmentation of 3D Multiphoton Images of Brain Vasculature." *CVPR Workshops*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8059194/) integrates persistent homology into loss for 3D multiphoton microscopy images — the exact imaging modality of our dataset.

---

## 3. Thread B: Foundation Models Change the Evaluation Landscape

### 3.1 VesselFM: Domain-Specific Foundation Model

[Wittmann et al. (2025). "vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation." *CVPR 2025*.](https://arxiv.org/abs/2411.17386) trains on curated + domain-randomized + synthetic data, outperforming existing models across four imaging modalities in zero-shot, one-shot, and few-shot scenarios. CRITICAL: VesselFM was pre-trained on a corpus including MiniVess → data leakage. NEUROVEX re-routes evaluation to non-contaminated datasets.

### 3.2 SAM3 and MedSAM: General-Purpose Adaptation

[Ma et al. (2024). "Segment Anything in Medical Images." *Nature Communications* 15, 654.](https://www.nature.com/articles/s41467-024-44824-z) fine-tunes SAM on 1.57M medical image-mask pairs. [Liu et al. (2025). "MedSAM3: Delving into Segment Anything with Medical Concepts." *arXiv:2511.19046*.](https://arxiv.org/abs/2511.19046) extends SAM3 with medical concepts and agent-based iterative refinement. [Chen et al. (2025). "SAM3-Adapter." *arXiv:2511.19425*.](https://arxiv.org/abs/2511.19425) achieves SOTA with <5% parameter overhead via adapters.

### 3.3 The Zero-Shot Baseline Problem

Foundation models introduce a methodological challenge: if a zero-shot model achieves 0.50 DSC on a task, does a fine-tuned model at 0.75 DSC represent meaningful progress? The answer depends on whether the zero-shot baseline represents genuine domain gap or merely suboptimal prompting. NEUROVEX treats zero-shot SAM3 as a legitimate baseline showing domain gap, with optional decoder fine-tuning for comparison.

---

## 4. Thread C: Statistical Evaluation — The Elephant in the Room

### 4.1 The Voxel Independence Problem

[Gibson et al. (2017). "Designing image segmentation studies: Statistical power, sample size and reference standard quality." *Medical Image Analysis* 42, 44–59.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5666910/) derives the critical insight: segmentation studies have a "design factor" that captures inter-voxel correlation. The effective sample size is NOT the number of voxels but the number of independent volumes. For NEUROVEX with 70 volumes in 3-fold CV (23 validation volumes per fold), the effective N for statistical comparisons is 23 — not millions of voxels.

### 4.2 Metrics Reloaded: The Evaluation Standard

[Maier-Hein et al. (2024). "Metrics Reloaded: Recommendations for image analysis validation." *Nature Methods* 21, 195–212.](https://doi.org/10.1038/s41592-023-02151-z) provides the definitive framework for metric selection. NEUROVEX uses their recommended approach: clDice (topology), MASD (distance), DSC (overlap) as a complementary metric triple, not a single-metric ranking.

### 4.3 Rigorous Validation: The nnU-Net Standard

[Isensee et al. (2024). "nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation." *MICCAI 2024*.](https://arxiv.org/abs/2404.09556) demonstrates that many architecture claims fail under rigorous validation — properly configured nnU-Net still achieves SOTA. This motivates NEUROVEX's factorial design: comparing models under identical training conditions (same loss, same data, same splits) rather than cherry-picking favorable configurations.

### 4.4 Reporting Guidelines

[Collins et al. (2024). "TRIPOD+AI Statement." *BMJ* 385.](https://doi.org/10.1136/bmj-2023-078378) and [Moons et al. (2025). "PROBAST+AI." *BMJ* 388.](https://doi.org/10.1136/bmj-2024-082505) provide reporting and quality assessment frameworks. [Pollard et al. (2026). "TRIPOD-Code." *Diagnostic and Prognostic Research*.](https://doi.org/10.1186/s41512-025-00217-4) extends this to code repositories. NEUROVEX maintains compliance matrices for all three guidelines.

---

## 5. Discussion: Novel Synthesis — The Factorial Design as the Missing Experiment

### 5.1 Why No One Does Factorial Designs in Segmentation

The segmentation literature overwhelmingly compares models using single loss functions on single datasets with single evaluation metrics. This means that interaction effects — does model X benefit more from loss Y than model Z? — are invisible. NEUROVEX's 4×3 factorial design with 3-fold CV explicitly tests for these interactions through two-way ANOVA with partial eta-squared effect sizes.

### 5.2 The Sample Size Dilemma

With 70 volumes (23 per validation fold), NEUROVEX's factorial design has limited statistical power for detecting small effects. [Legha et al. (2026). "Sequential Sample Size Calculations." *JCE* 191.](https://doi.org/10.1016/j.jclinepi.2025.112117) provides the framework for understanding when sample size is sufficient. The honest acknowledgment: NEUROVEX can detect large effects (η² > 0.14, "large" by Cohen's convention) but may miss small effects (η² < 0.02). The factorial design's value is not in statistical power but in systematic coverage of the model-loss interaction space.

### 5.3 What MONAI Could Gain

NEUROVEX's topology-aware loss registry (18 losses), model adapter pattern, and factorial evaluation framework could be contributed to MONAI as a standard benchmarking toolkit. The gap: MONAI has individual losses and models but no standard framework for comparing them in factorial combination.

---

## 6. Recommended Issues

| Issue | Priority | Scope |
|-------|----------|-------|
| Contribute factorial evaluation framework to MONAI | P2 | Community |
| Add Topograph (Lux 2025) graph-based topology loss | P2 | Training |
| Implement clCE loss (Acebes 2024) as alternative | P2 | Training |
| Document effective sample size (N=23 per fold, not voxels) | P1 | Manuscript |

---

## 7. Academic Reference List

### Seeds (15)
1. [Maier-Hein et al. (2024). "Metrics Reloaded." *Nature Methods* 21.](https://doi.org/10.1038/s41592-023-02151-z)
2. [Collins et al. (2024). "TRIPOD+AI Statement." *BMJ* 385.](https://doi.org/10.1136/bmj-2023-078378)
3. [Gallifant et al. (2025). "TRIPOD-LLM." *Nature Medicine* 31(1).](https://doi.org/10.1038/s41591-024-03425-5)
4. [Moons et al. (2025). "PROBAST+AI." *BMJ* 388.](https://doi.org/10.1136/bmj-2024-082505)
5. [Sounderajah et al. (2025). "STARD-AI." *Nature Medicine* 31(10).](https://doi.org/10.1038/s41591-025-03953-8)
6. [Lekadir et al. (2025). "FUTURE-AI." *BMJ* 388.](https://doi.org/10.1136/bmj-2024-081554)
7. [Riley et al. (2024). "Evaluation of Clinical Prediction Models (Part 1)." *BMJ* 384.](https://doi.org/10.1136/bmj-2023-074819)
8. [Riley & Collins (2023). "Stability of Clinical Prediction Models." *Biometrical Journal*.](https://doi.org/10.1002/bimj.202200302)
9. [Riley et al. (2023). "Multiverse of Madness." *BMC Medicine* 21.](https://doi.org/10.1186/s12916-023-03212-y)
10. [Van Calster et al. (2025). "Performance Measures in Predictive AI." *Lancet Digital Health*.](https://doi.org/10.1016/j.landig.2025.100916)
11. [Kofler et al. (2023). "Are we using appropriate segmentation metrics?" *arXiv*.](https://arxiv.org/abs/2304.07396)
12. [Shit et al. (2021). "clDice." *CVPR*.](https://arxiv.org/abs/2003.07311)
13. [Poon et al. (2023). "A dataset of rodent cerebrovasculature." *Scientific Data* 10.](https://doi.org/10.1038/s41597-023-02048-8)
14. [Pollard et al. (2026). "TRIPOD-Code." *Diagnostic and Prognostic Research*.](https://doi.org/10.1186/s41512-025-00217-4)
15. [Legha et al. (2026). "Sequential Sample Size Calculations." *JCE* 191.](https://doi.org/10.1016/j.jclinepi.2025.112117)

### Web-Discovered (28)
16. [Shi et al. (2024). "cbDice: Centerline Boundary Dice Loss." *MICCAI 2024*.](https://arxiv.org/abs/2407.01517)
17. [Acebes et al. (2024). "Centerline-Cross Entropy Loss." *MICCAI 2024*.](https://papers.miccai.org/miccai-2024/770-Paper1081.html)
18. [Berger et al. (2024). "Topologically Faithful Multi-class Segmentation." *MICCAI 2024*.](https://arxiv.org/abs/2403.11001)
19. [Lux et al. (2025). "Topograph." *ICLR 2025*.](https://arxiv.org/abs/2411.03228)
20. [Wen et al. (2024). "Topology-Preserving with Spatial-Aware Persistent Feature Matching." *arXiv*.](https://arxiv.org/abs/2412.02076)
21. [Gupta et al. (2023). "Topology-Aware Uncertainty." *NeurIPS 2023*.](https://arxiv.org/abs/2306.05671)
22. [Haft-Javaherian et al. (2020). "Topological Encoding CNN for 3D Multiphoton Brain Vasculature." *CVPR Workshops*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8059194/)
23. [Wittmann et al. (2025). "vesselFM." *CVPR 2025*.](https://arxiv.org/abs/2411.17386)
24. [Ma et al. (2024). "MedSAM." *Nature Communications* 15.](https://www.nature.com/articles/s41467-024-44824-z)
25. [Liu et al. (2025). "MedSAM3." *arXiv:2511.19046*.](https://arxiv.org/abs/2511.19046)
26. [Chen et al. (2025). "SAM3-Adapter." *arXiv:2511.19425*.](https://arxiv.org/abs/2511.19425)
27. [Makani et al. (2026). "Onco-Seg: Adapting SAM3." *medRxiv*.](https://www.medrxiv.org/content/10.64898/2026.01.11.26343874v2)
28. [Gibson et al. (2017). "Designing segmentation studies: statistical power." *MedIA* 42.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5666910/)
29. [Isensee et al. (2024). "nnU-Net Revisited." *MICCAI 2024*.](https://arxiv.org/abs/2404.09556)
30–43. [Additional papers from agent results — topology losses, calibration, MONAI contributions]

---

## Appendix: Alignment

- **Excluded**: MLOps details (R2), Reproducibility (R1), Microscopy hardware (R4), Agentic AI (report 5)
- **KG domains to update**: training, models, architecture
- **Manuscript sections**: Methods (R2 all), Results (R3a loss ablation, R3b multi-model comparison)
