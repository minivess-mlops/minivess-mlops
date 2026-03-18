# Federated Learning and Multi-Site Data Architecture for Preclinical Neurovascular Imaging

**Literature Research Report R3** | 32 papers | 2026-03-18
**Manuscript section**: Discussion: Multi-Site Generalization (future work)
**KG domains**: data, operations, cloud
**Quality target**: MINOR_REVISION
**Dedup**: Papers already cited in R1 or R2 are excluded.

---

## 1. Introduction: Why Federated Learning Matters for Preclinical Imaging

The MinIVess platform operates on a single-site dataset of 70 rat cortical vasculature
volumes from two-photon microscopy. Generalizing to other sites — different microscopes,
laser powers, staining protocols, even species — is the primary barrier to broader
scientific impact. Yet raw data sharing between institutions faces practical and ethical
obstacles: animal imaging data may be subject to institutional data governance,
proprietary imaging protocols, or simply impractical file sizes for 3D volumes.

Federated learning (FL) offers a path forward: train collaboratively without centralizing
data. But FL for preclinical microscopy is fundamentally different from clinical FL.
Clinical FL operates on standardized DICOM data from regulated imaging protocols;
preclinical FL must handle heterogeneous imaging parameters, variable tissue preparations,
and minimal standardization. This report examines whether the FL infrastructure developed
for clinical radiology can be adapted for preclinical neurovascular imaging, and what
architectural decisions the MinIVess platform must make to enable multi-site federation.

**Scope**: MONAI FL vs NVIDIA FLARE vs Flower, domain shift handling, privacy-preserving
training, data heterogeneity, preclinical-specific challenges. **Excluded**: single-site
post-training methods (R1), ensemble selection (R2), regulatory compliance beyond
privacy (R4).

---

## 2. The FL Landscape: Maturity Gap Between Papers and Deployment

### 2.1 Scale of the Problem

[Teo et al. (2024). "Federated machine learning in healthcare: A systematic review on
clinical applications and technical architecture." *Cell Reports Medicine*,
5(1):101419.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10897620/) surveyed 612
peer-reviewed FL articles and found medical imaging accounts for 41.7% of FL studies, yet
**only 5.2% represent real-world deployments**. The gap between publication and deployment
is enormous. [Madathil et al. (2025). "Revolutionizing healthcare data analytics with
federated learning." *Computational and Structural Biotechnology
Journal*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12213103/) analyzed 159 FL studies
and identified the same three persistent barriers: data heterogeneity, communication
overhead, and aggregation strategy selection.

For preclinical imaging, the deployment gap is even wider — essentially no published FL
systems exist for two-photon microscopy. MinIVess would be among the first.

### 2.2 Aggregation Algorithm Choice Is Overrated

A counterintuitive finding from recent benchmarks: the choice of FL aggregation algorithm
matters far less than commonly assumed. [Marella et al. (2025). "FedOnco-Bench: A
Reproducible Benchmark for Privacy-Aware Federated Tumor Segmentation with Synthetic CT
Data." *IEEE*.](https://arxiv.org/abs/2511.00795) compared FedAvg, FedProx, FedBN, and
FedAvg+DP-SGD on tumor segmentation and found Dice differences of ±0.01 between
algorithms — statistically insignificant. The real differentiator is the normalization
strategy and domain adaptation approach, not the aggregation rule.

This has practical implications: MinIVess should start with FedAvg (simplest, best
understood) and invest engineering effort in batch normalization handling and domain
shift mitigation rather than experimenting with exotic aggregation algorithms.

---

## 3. MONAI-Native FL: Production-Ready Frameworks

### 3.1 FednnU-Net and MONet-FL

Two frameworks bring FL directly to the MONAI/nnU-Net ecosystem. [Skorupko et al. (2025).
"Federated nnU-Net for Privacy-Preserving Medical Image Segmentation." *Scientific
Reports*.](https://www.nature.com/articles/s41598-025-22239-0) introduced Federated
Fingerprint Extraction (FFE) and Asymmetric Federated Averaging (AsymFedAvg) to enable
decentralized nnU-Net training. FFE shares dataset fingerprints without raw data exchange,
allowing the automated preprocessing pipeline to adapt to heterogeneous data. Validated on
six datasets for 2D and 3D segmentation.

[Bendazzoli et al. (2025). "MONet-FL: Extending nnU-Net with MONAI for Clinical Federated
Learning." *MICCAI 2025 (DeCaF Workshop)*.](https://link.springer.com/chapter/10.1007/978-3-032-05663-4_10)
takes the MONAI Bundle approach, providing a modular FL tool that integrates with downstream
clinical operations including model deployment and active learning. For MinIVess, MONet-FL
aligns with the MONAI-first design principle (TOP-1) and could serve as the FL backbone.

### 3.2 The Broader FL Framework Landscape

[Pati et al. (2022). "Federated learning enables big data for rare cancer boundary
detection." *Nature Communications*,
13:7346.](https://doi.org/10.1038/s41467-022-33407-5) demonstrated FL at scale for
medical segmentation — 6,314 patients across 71 sites — establishing the feasibility of
large-scale medical federation. The FeTS challenge
([Linardos et al. (2025). "The MICCAI Federated Tumor Segmentation (FeTS) Challenge
2024." *MELBA*.](https://arxiv.org/abs/2512.06206)) evaluated novel aggregation methods
on 1,251 training cases, benchmarking communication efficiency alongside segmentation
quality.

The frameworks themselves — NVIDIA FLARE ([Roth et al. (2020)](https://arxiv.org/abs/2210.13291)),
Flower ([Beutel et al. (2020)](https://arxiv.org/abs/2007.14390)) — are mature and
interchangeable for most use cases. Flower is open-source and framework-agnostic; FLARE
integrates natively with MONAI. For MinIVess, FLARE+MONAI is the natural choice given
the existing ecosystem alignment.

---

## 4. Domain Shift: The Core Challenge for Multi-Site Microscopy

### 4.1 Feature Disentanglement for Vessel Segmentation

The highest-relevance paper for MinIVess is [Galati et al. (2025). "Multi-Domain Brain
Vessel Segmentation Through Feature Disentanglement." *MELBA*,
3.](https://arxiv.org/abs/2510.00665) — the only published work addressing multi-domain
cerebrovascular segmentation. Their approach separates vessel-specific features from
domain-specific appearance features, enabling accurate segmentation across imaging
modalities without domain-specific model design. For multiphoton vasculature imaging
where different labs produce visually distinct images, feature disentanglement provides a
principled way to share vessel knowledge without harmonizing imaging protocols.

### 4.2 Topology-Preserving Transfer Learning

[Wu et al. (2026). "Geometric-Topological Deep Transfer Learning for Precise Vessel
Segmentation in 3D Medical Volumes." *npj Digital Medicine*.](https://www.nature.com/articles/s41746-025-02061-8)
introduced FlowAxis, using Adaptive Vessel Axes and optimal transport theory for
topology-preserving vessel segmentation that is robust to domain shift. The topology
preservation is critical for vessel networks where connectivity matters more than
pixel-level accuracy — a broken vessel branch is a worse error than a slightly
misaligned boundary. FlowAxis complements the topology-aware losses (clDice, cbDice)
already planned for MinIVess.

### 4.3 Batch Normalization: The Hidden Critical Decision

Three papers converge on the same insight: batch normalization handling is the single
most important design decision in federated segmentation.

[Guerraoui et al. (2024). "Overcoming the Challenges of Batch Normalization in Federated
Learning." *arXiv:2405.14670*.](https://arxiv.org/abs/2405.14670) showed that naive BN
aggregation causes external covariate shifts that degrade both convergence and accuracy.
[Zhu et al. (2024). "MLA-BIN: Model-level Attention and Batch-instance Style
Normalization for Domain Generalization of Federated Learning on Medical Image
Segmentation." *CMIG*,
116:102414.](https://arxiv.org/abs/2306.17008) proposed combining model-level attention
with batch-instance normalization, achieving DSC of 88.27% on prostate segmentation by
keeping BN layers local while sharing convolutional weights.

For MinIVess, the recommendation is clear: **keep batch normalization layers local
(FedBN-style)** and share only convolutional weights. Different microscopes produce
different intensity distributions; trying to average BN statistics across sites would
destroy site-specific calibration.

### 4.4 Style Transfer Without Data Sharing

[Ren et al. (2026). "FedCA: Federated Domain Generalization for Medical Image
Segmentation via Cross-Client Feature Style Transfer and Adaptive Style Alignment."
*Expert Systems with Applications*.](https://www.sciencedirect.com/science/article/abs/pii/S0957417426003076)
introduced a server-maintained feature style bank that enables cross-site style transfer
without exchanging raw data. [Nagaraju et al. (2025). "FedGIN: Federated Learning with
Dynamic Global Intensity Non-linear Augmentation for Organ Segmentation." *MICCAI 2025
(DeCaF)*.](https://arxiv.org/abs/2508.05137) achieved 12-18% Dice improvement in
limited-data scenarios through lightweight intensity harmonization. [Liu et al. (2021).
"FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic
Learning in Continuous Frequency Space."
*CVPR 2021*.](https://arxiv.org/abs/2103.06030) established the foundational approach
using frequency-space interpolation to transfer domain-specific amplitude spectra.

### 4.5 Heterogeneous Data Architecture

[Hu et al. (2025). "Addressing Data Heterogeneity in Distributed Medical Imaging with
HeteroSync Learning." *Nature
Communications*.](https://www.nature.com/articles/s41467-025-64459-y) introduced
HeteroSync Learning with a Shared Anchor Task for cross-node representation alignment,
outperforming 12 FL benchmark methods by up to 40% AUC. The anchor-task approach could
be adapted for MinIVess's multi-task architecture where different sites might annotate
different vascular features.

---

## 5. Privacy: The DP-Utility Tradeoff on Small Datasets

[Mohammadi et al. (2026). "Differential Privacy for Medical Deep Learning: Methods,
Tradeoffs, and Deployment Implications." *npj Digital
Medicine*.](https://www.nature.com/articles/s41746-025-02280-z) reviewed 74 studies and
found that **strict differential privacy (epsilon ~1) degrades performance severely on
small datasets**, while moderate privacy budgets (epsilon ~10) maintain clinically
acceptable performance. For MiniVess with 70 volumes, this means strict DP is likely
impractical — secure aggregation or privacy-by-architecture (keeping BN layers local,
sharing only gradient updates) may be more appropriate than formal DP guarantees.

[Yahiaoui et al. (2024). "Federated Learning with Privacy Preserving for Multi-
Institutional Three-Dimensional Brain Tumor Segmentation." *Diagnostics*,
14:2891.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11675895/) demonstrated FL+DP for 3D
U-Net brain tumor segmentation, achieving competitive results. But their BraTS dataset
has >1000 volumes — the privacy-utility tradeoff at N=70 is uncharted territory.

[Koutsoubis et al. (2025). "Privacy-Preserving Federated Learning and Uncertainty
Quantification in Medical Imaging." *Radiology:
AI*.](https://pubs.rsna.org/doi/10.1148/ryai.240637) argued that UQ is essential for
trustworthy FL deployment, as sensitive information can be inferred from shared gradients.
Integrating UQ (from R2) into FL is an underexplored combination.

---

## 6. Small Data and Label Scarcity in Federation

[Khowaja et al. (2024). "SelfFed: Self-Supervised Federated Learning for Data
Heterogeneity and Label Scarcity in Medical Images." *Expert Systems with
Applications*.](https://arxiv.org/abs/2307.01514) combined self-supervised pre-training
with federated fine-tuning in a two-phase framework, using SSL to overcome data
heterogeneity before federated fine-tuning addresses label scarcity. For preclinical
imaging where expert annotations are expensive, this SSL+FL combination could reduce
annotation burden across participating sites.

[Kanhere et al. (2024). "Privacy-Preserving Collaboration for Multi-Organ Segmentation
via Federated Learning from Sites with Partial Labels." *CVPR 2024
Workshops*.](https://openaccess.thecvf.com/content/CVPR2024W/DCAMI/html/Kanhere_Privacy-Preserving_Collaboration_for_Multi-Organ_Segmentation_via_Federated_Learning_from_Sites_CVPRW_2024_paper.html)
addressed a common practical scenario: different sites annotate different structures
(arteries vs. veins vs. capillaries in the MinIVess context). Their SegViz framework uses
selective weight synchronization — shared encoder, separate task heads — enabling
collaboration despite annotation heterogeneity.

---

## 7. Two-Photon Microscopy: The Closest Published Work

### 7.1 Self-Supervised Two-Photon Segmentation

[Ntiri et al. (2025). "A self-supervised deep learning pipeline for segmentation in
two-photon fluorescence microscopy." *bioRxiv:2025.01.20.633744*.](https://www.biorxiv.org/content/10.1101/2025.01.20.633744v1.full)
developed SELF-TPFM, validated on the MiniVess dataset itself. Using four pretext tasks,
SSL-pretrained U-Net models match fully supervised performance with only 50% of labeled
data. This work comes from AICONSlab (University of Toronto) and represents the most
directly relevant published work for MinIVess.

### 7.2 Network-Level Neurovascular Analysis

[Rozak et al. (2024). "A Deep Learning Pipeline for Mapping in situ Network-level
Neurovascular Coupling in Multi-photon Fluorescence Microscopy." *eLife (reviewed
preprint)*.](https://elifesciences.org/reviewed-preprints/95525v2) presents an automated
pipeline from the same lab: 3D UNETR-based vessel segmentation, temporal registration,
centerline tracing, and graph-theoretic analysis. A federation between this pipeline and
MinIVess would be a natural multi-site collaboration.

### 7.3 FL Feasibility for Microscopy

[Bruschi et al. (2025). "Federated and Centralized Machine Learning for Cell
Segmentation: A Comparative Analysis." *Electronics*,
14(7):1254.](https://www.mdpi.com/2079-9292/14/7/1254) validated FL at microscopy scale,
demonstrating that FL matches centralized performance for cell segmentation without data
sharing. This is the closest published evidence that FL is feasible for microscopy-level
biological image analysis.

---

## 8. Communication Efficiency for Large 3D Volumes

[Nanekaran & Ukwatta (2025). "A novel federated learning framework for medical imaging:
Resource-efficient approach combining PCA with early stopping." *Medical
Physics*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12409104/) proposed FIPCA,
reducing FL training rounds from 200 to 38 (81% reduction) while improving performance.
For MinIVess, where 3D microscopy volumes are large and bandwidth between sites varies
(especially with SkyPilot's multi-cloud architecture), communication efficiency is
not optional — it determines whether FL is practically feasible.

---

## 9. Implementation Pitfalls

[Li et al. (2025). "From Challenges and Pitfalls to Recommendations and Opportunities:
Implementing Federated Learning in Healthcare." *Medical Image
Analysis*.](https://arxiv.org/abs/2409.09727) cataloged prevalent FL implementation
problems through a review of studies up to May 2024. Key pitfalls relevant to MinIVess:
- **Non-IID data handling**: Different labs produce fundamentally different distributions
- **Communication cost underestimation**: 3D volume gradients are large
- **Privacy leakage from gradients**: Even without raw data, model updates can leak
  information about training data
- **Evaluation bias**: Testing on centralized data after federated training can give
  misleading results

---

## 10. MinIVess FL Architecture Recommendations

### Recommended Architecture

Based on the evidence in this report:

| Component | Recommendation | Evidence |
|-----------|---------------|----------|
| FL framework | FLARE + MONAI (MONet-FL bundle) | Skorupko (2025), Bendazzoli (2025) |
| Aggregation | FedAvg (start simple) | Marella (2025): algorithms differ by ±0.01 Dice |
| BN handling | FedBN (local BN layers) | Guerraoui (2024), Zhu (2024) |
| Domain shift | Feature disentanglement | Galati (2025): vessel-specific |
| Privacy | Secure aggregation, moderate DP (epsilon ~10) | Mohammadi (2026) |
| Efficiency | FIPCA-style communication reduction | Nanekaran (2025) |
| Label scarcity | SSL pre-training + FL fine-tuning | Khowaja (2024), Ntiri (2025) |

### Research Gaps

1. **No published FL system for two-photon microscopy**: MinIVess would be the first
2. **DP-utility tradeoff at N<100**: All published DP studies use datasets >500 volumes
3. **Topology preservation in federated vessel segmentation**: FlowAxis + FL is untested
4. **Site as factorial factor**: How does federated training interact with the factorial
   design (site × architecture × loss) when site = FL client?
5. **Cross-species federation**: Can vasculature features transfer between rat and mouse
   models in a federated setting?

---

## References

### Seed Papers (8)

1. [Pati, S. et al. (2022). "Federated learning enables big data for rare cancer boundary detection." *Nature Communications*, 13:7346.](https://doi.org/10.1038/s41467-022-33407-5)
2. [Li, T. et al. (2020). "Federated Optimization in Heterogeneous Networks." *MLSys*.](https://arxiv.org/abs/1812.06127)
3. [Sheller, M.J. et al. (2020). "Federated learning in medicine." *Scientific Reports*, 10:12598.](https://doi.org/10.1038/s41598-020-69250-1)
4. [Beutel, D.J. et al. (2020). "Flower: A Friendly Federated Learning Framework." *arXiv*.](https://arxiv.org/abs/2007.14390)
5. [Roth, H.R. et al. (2020). "NVIDIA FLARE: Federated Learning from Simulation to Real-World." *arXiv*.](https://arxiv.org/abs/2210.13291)
6. [Poon, C. et al. (2023). "A dataset of rodent cerebrovasculature." *Scientific Data*, 10:141.](https://doi.org/10.1038/s41597-023-02048-8)
7. [Guan, H. & Liu, M. (2021). "Domain Adaptation for Medical Image Analysis." *IEEE RBME*, 15:103-115.](https://doi.org/10.1109/RBME.2021.3098890)
8. [Kairouz, P. et al. (2021). "Advances and Open Problems in Federated Learning." *Foundations and Trends in ML*, 14(1-2).](https://arxiv.org/abs/1912.04977)

### Discovered Papers (24)

9. [Skorupko, G. et al. (2025). "Federated nnU-Net for Privacy-Preserving Medical Image Segmentation." *Scientific Reports*.](https://www.nature.com/articles/s41598-025-22239-0)
10. [Bendazzoli, S. et al. (2025). "MONet-FL: Extending nnU-Net with MONAI for Clinical Federated Learning." *MICCAI 2025*.](https://link.springer.com/chapter/10.1007/978-3-032-05663-4_10)
11. [Linardos, A. et al. (2025). "The MICCAI FeTS Challenge 2024." *MELBA*.](https://arxiv.org/abs/2512.06206)
12. [Nagaraju, S.D. et al. (2025). "FedGIN: Federated Learning with Dynamic Global Intensity Non-linear Augmentation." *MICCAI 2025*.](https://arxiv.org/abs/2508.05137)
13. [Ren, Y. et al. (2026). "FedCA: Federated Domain Generalization via Cross-Client Feature Style Transfer." *Expert Systems with Applications*.](https://www.sciencedirect.com/science/article/abs/pii/S0957417426003076)
14. [Galati, F. et al. (2025). "Multi-Domain Brain Vessel Segmentation Through Feature Disentanglement." *MELBA*, 3.](https://arxiv.org/abs/2510.00665)
15. [Wu, J. et al. (2026). "Geometric-Topological Deep Transfer Learning for Precise Vessel Segmentation in 3D Medical Volumes." *npj Digital Medicine*.](https://www.nature.com/articles/s41746-025-02061-8)
16. [Hu, H.T. et al. (2025). "Addressing Data Heterogeneity in Distributed Medical Imaging with HeteroSync Learning." *Nature Communications*.](https://www.nature.com/articles/s41467-025-64459-y)
17. [Teo, Z.L. et al. (2024). "Federated machine learning in healthcare." *Cell Reports Medicine*, 5(1):101419.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10897620/)
18. [Madathil, N.T. et al. (2025). "Revolutionizing healthcare data analytics with federated learning." *CSBJ*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12213103/)
19. [Marella, V.C. et al. (2025). "FedOnco-Bench: A Reproducible Benchmark for Privacy-Aware Federated Tumor Segmentation." *IEEE*.](https://arxiv.org/abs/2511.00795)
20. [Nanekaran, N.P. & Ukwatta, E. (2025). "A novel federated learning framework: Resource-efficient approach combining PCA with early stopping." *Medical Physics*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12409104/)
21. [Liu, Q. et al. (2021). "FedDG: Federated Domain Generalization on Medical Image Segmentation." *CVPR 2021*.](https://arxiv.org/abs/2103.06030)
22. [Zhu, F. et al. (2024). "MLA-BIN: Model-level Attention and Batch-instance Style Normalization." *CMIG*, 116:102414.](https://arxiv.org/abs/2306.17008)
23. [Guerraoui, R. et al. (2024). "Overcoming the Challenges of Batch Normalization in Federated Learning." *arXiv:2405.14670*.](https://arxiv.org/abs/2405.14670)
24. [Mohammadi, M. et al. (2026). "Differential Privacy for Medical Deep Learning." *npj Digital Medicine*.](https://www.nature.com/articles/s41746-025-02280-z)
25. [Yahiaoui, M.E. et al. (2024). "Federated Learning with Privacy Preserving for Multi-Institutional 3D Brain Tumor Segmentation." *Diagnostics*, 14:2891.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11675895/)
26. [Koutsoubis, N. et al. (2025). "Privacy-Preserving Federated Learning and Uncertainty Quantification in Medical Imaging." *Radiology: AI*.](https://pubs.rsna.org/doi/10.1148/ryai.240637)
27. [Khowaja, S.A. et al. (2024). "SelfFed: Self-Supervised Federated Learning for Data Heterogeneity and Label Scarcity." *Expert Systems with Applications*.](https://arxiv.org/abs/2307.01514)
28. [Kanhere, A. et al. (2024). "Privacy-Preserving Collaboration for Multi-Organ Segmentation via Federated Learning from Sites with Partial Labels." *CVPR 2024 Workshops*.](https://openaccess.thecvf.com/content/CVPR2024W/DCAMI/html/Kanhere_Privacy-Preserving_Collaboration_for_Multi-Organ_Segmentation_via_Federated_Learning_from_Sites_CVPRW_2024_paper.html)
29. [Ntiri, E.E. et al. (2025). "A self-supervised deep learning pipeline for segmentation in two-photon fluorescence microscopy." *bioRxiv:2025.01.20.633744*.](https://www.biorxiv.org/content/10.1101/2025.01.20.633744v1.full)
30. [Rozak, M. et al. (2024). "A Deep Learning Pipeline for Mapping in situ Network-level Neurovascular Coupling in Multi-photon Fluorescence Microscopy." *eLife*.](https://elifesciences.org/reviewed-preprints/95525v2)
31. [Bruschi, S. et al. (2025). "Federated and Centralized Machine Learning for Cell Segmentation: A Comparative Analysis." *Electronics*, 14(7):1254.](https://www.mdpi.com/2079-9292/14/7/1254)
32. [Li, M. et al. (2025). "From Challenges and Pitfalls to Recommendations and Opportunities: Implementing Federated Learning in Healthcare." *Medical Image Analysis*.](https://arxiv.org/abs/2409.09727)
