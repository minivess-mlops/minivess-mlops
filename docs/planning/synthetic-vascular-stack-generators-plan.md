---
title: "Synthetic 3D Vascular Stack Generators Roadmap"
status: planned
created: "2026-03-12"
---

# Synthetic 3D Vascular Stack Generators: Roadmap & Literature Synthesis

**Date:** 2026-03-12
**Scope:** Methods for synthesizing realistic 3D two-photon microscopy vascular volumes
**Constraints:** 8 GB VRAM (RTX 2070 Super), 70 MiniVess + 12 VesselNN training volumes
**Dual purpose:** (1) Drift simulation for monitoring, (2) Augmentation for segmentation models

---

## 1. Relationship to Existing Plans

This plan synthesizes and supersedes synthetic generation aspects from:

| Prior Document | What It Covered | Status |
|---|---|---|
| `drift-monitoring-plan.md` (Issue #38) | Multi-tier OOD pipeline: TorchIO → Bezier → D_drand | Vision only |
| `drift-monitoring-implementation-plan.xml` (#574) | Implemented Tier 1/2 drift detection; T4 VesselNN deferred | 8/9 tasks DONE |
| `vesselnn-dataset-implementation.xml` (#602) | VesselNN download, DVC, cross-dataset drift, growth sim | Plan ready |
| `synthetic-data-qa-engineering-drifts-knowledge-agentic-systems-report.md` | 33-ref literature review on drift + synthetic data + QA | Complete |
| `synthicl-plan.md` | Domain randomization pipeline (SynthICL-style) | Plan only, NOT coded |
| `monitoring-research-report.md` | Multi-tier OOD: corruption → Bezier → D_drand | Vision only |
| `generative-uq-plan.md` | Generative UQ + multi-annotator VAE | Plan only |
| `calibration-shift-plan.md` | Calibration under distribution shift | Plan only |

**What's already implemented:**
- `drift_synthetic.py` — Statistical perturbations (intensity, noise, blur, erosion)
- `acquisition_simulator.py` — Temporal drift stream generator with DriftSchedule
- `debug_dataset.py` — Random-walk tubes for test fixtures

**What's NOT implemented:** Any method that generates **morphologically realistic** vasculature.

---

## 2. The Problem: Why We Need Synthetic Generators

### 2.1 Drift Simulation (Monitoring)

The current `drift_synthetic.py` applies statistical perturbations (brightness shift, Gaussian noise, resolution degradation, erosion) to existing volumes. These are useful for validating drift detectors but do **not** simulate realistic covariate shifts that occur when:

- A microscope degrades over months (gradual PSF broadening, laser power drift)
- A new lab joins the study (different objective, different fluorophore, different mouse strain)
- Tissue preparation changes (perfusion vs immersion fixation, clearing protocol)

**Gap:** We need generators that produce volumes with **plausible morphological variation** while controlling the degree of distribution shift.

### 2.2 Augmentation (Segmentation)

With only 70 MiniVess + 12 VesselNN volumes, overfitting is a constant risk. Standard augmentations (flip, rotate, crop, intensity jitter) preserve the morphological statistics of the training set. Synthetic generators that produce novel vessel topologies expand the effective training distribution.

**Gap:** No current augmentation creates new branching patterns, novel vessel diameters, or unseen tortuosity profiles.

---

## 3. Literature: Organized by Category

### 3a. 3D Vascular Generation (Exact Use Case)

#### 3a-i. Focused on 2-Photon / Nonlinear Microscopy

| Paper | Method | 3D? | 8GB? | Relevance |
|---|---|:---:|:---:|---|
| [Zhou et al. (2024). "A Deep Learning Approach for Improving Two-Photon Vascular Imaging Speeds." *Bioengineering* 11(2):111.](https://doi.org/10.3390/bioengineering11020111) | Semi-synthetic degradation (downsample + noise) + PSSR Res-U-Net super-resolution for 2PM brain vasculature | YES | YES | **VERY HIGH** — exact modality, semi-synthetic approach immediately actionable |
| [Teikari et al. (2016). "Deep Learning Denoising of Two-Photon Fluorescence Images." *bioRxiv preprint*.](https://github.com/petteriTeikari/vesselNN) | VesselNN: 12 denoised 2PM stacks with manual labels from 3 labs | YES | N/A | **HIGH** — real data baseline for validating synthetic generators |

**Assessment:** Surprisingly thin literature on synthetic *generation* specifically for 2-photon vascular stacks. The Zhou et al. semi-synthetic degradation approach is the closest published work. Most methods in this sub-domain use real data augmentation rather than de novo synthesis.

#### 3a-ii. General 3D Vascular Generation

| Paper | Method | 3D? | 8GB? | Relevance |
|---|---|:---:|:---:|---|
| [Wang et al. (2025). "VasTSD: Learning 3D Vascular Tree-state Space Diffusion." *CVPR 2025 preprint*.](https://arxiv.org/abs/2503.12758) | State-space tree serialization + diffusion for topology-preserving angiography synthesis | YES | UNKNOWN | **HIGH** — topology-aware, but code pending, designed for MRA not microscopy |
| [Feldman et al. (2025). "VesselVAE: Recursive Variational Autoencoders for 3D Blood Vessel Synthesis." *arXiv:2506.14914*.](https://arxiv.org/abs/2506.14914) | Recursive VAE on hierarchical tree structure; learns branch connectivity + geometry (radius, length, tortuosity) | YES (graph) | YES | **HIGH** — lightweight graph VAE, needs volumetric rendering step |
| [Hamarneh & Jassi (2010). "VascuSynth: Simulating Vascular Trees for Generating Volumetric Image Data." *CMBBE: Imaging & Vis*.](https://vascusynth.cs.sfu.ca/) | Procedural growth from oxygen demand maps; renders to 3D volumes with GT segmentation | YES | YES (CPU) | **HIGH** — classic, mature, zero GPU needed; procedural textures don't match 2PM |
| [Nader et al. (2024). "A Vascular Synthetic Model for Improved Aneurysm Segmentation." *arXiv:2403.18734*.](https://arxiv.org/abs/2403.18734) | 3D splines + spherical kernel convolution + elastic deformation; 64^3 patches | YES | YES | **HIGH** — directly transferable pipeline, proven on MRA-TOF |
| [Comin & Galvao (2025). "VessShape: Few-Shot 2D Blood Vessel Segmentation by Leveraging Shape Priors from Synthetic Images." *arXiv:2510.27646*.](https://arxiv.org/abs/2510.27646) | Procedural 2D tubular geometries with varied textures for shape-bias pretraining | 2D only | YES | **MEDIUM** — shape-bias concept transferable; 3D extension needed |
| VesselFM D_drand (Wittmann et al. 2024) | Corrosion-cast graph structures + Gaussian tube profiles + intensity randomization; 500k synthetic pairs | YES | NO (8xA100 to train) | **DEFERRED** — checkpoint not public |

### 3b. Biomedical Volumetric Image Generation (Adjacent)

| Paper | Method | 3D? | 8GB? | Relevance |
|---|---|:---:|:---:|---|
| [Chakrabarty et al. (2026). "Synthetic Volumetric Data Generation Enables Zero-Shot Generalization." *arXiv:2601.12297*.](https://arxiv.org/abs/2601.12297) | SynthFM-3D: mathematical modeling of anatomy, contrast, boundary, noise; fine-tunes SAM 2 on 10k synthetic volumes | YES | LIKELY | **MEDIUM-HIGH** — analytical generation approach transferable to 2PM |
| [Friedrich et al. (2024). "Deep Generative Models for 3D Medical Image Synthesis." *arXiv:2410.17664*.](https://arxiv.org/abs/2410.17664) | Survey: VAEs, GANs, diffusion for 3D medical images | YES | Varies | **HIGH** as reference — comprehensive taxonomy |
| [Seyfarth et al. (2025). "MedLoRD: Medical Low-Resource Diffusion for 3D CT." *arXiv:2503.13211*.](https://arxiv.org/abs/2503.13211) | VQ-VAE GAN (16k codebook) + 3D U-Net + flash attention + lightweight ControlNet | YES | NO (24GB) | **LOW** — too heavy for 8GB |
| [Wang et al. (2024). "3D MedDiffusion: Controllable 3D Medical Image Generation." *arXiv:2412.13059*.](https://arxiv.org/abs/2412.13059) | Patch-Volume AE + latent diffusion; up to 512^3 | YES | NO (24GB+) | **LOW** — architecture ideas transferable but too heavy |
| BYOC (Naidoo et al. 2024, ICLR withdrawn) | VQ + diffusion for multichannel 3D microscopy cell synthesis | YES | UNKNOWN | **MEDIUM** — 3D microscopy synthesis precedent |

### 3c. General / Natural Image Generation Methods

| Paper | Method | 3D? | 8GB? | Relevance |
|---|---|:---:|:---:|---|
| [Sordo et al. (2025). "Synthetic Scientific Image Generation with VAE, GAN, and Diffusion." *J. Imaging* 11(8).](https://doi.org/10.3390/jimaging11080252) | Comparative study: VAE, DCGAN, StyleGAN, DDPM, Stable Diffusion, ControlNet on scientific images (microCT, fibers, roots) | 2D | Varies | **MEDIUM** — StyleGAN outperforms diffusion for structural scientific images |
| [Bench & Thomas (2025). "Quantifying Uncertainty of Synthetic Image Quality Metrics." *arXiv:2504.03623*.](https://arxiv.org/abs/2504.03623) | MC-dropout on autoencoder embeddings → Frechet Autoencoder Distance with uncertainty | 2D | YES | **LOW** — meta-evaluation of synthetic quality, not generation |
| [Loiseau et al. (2025). "Reliability in Semantic Segmentation: Can We Use Synthetic Data?" *ECCV 2024*.](https://doi.org/10.1007/978-3-031-73337-6_25) | Fine-tuned Stable Diffusion for OOD scene generation + segmentation reliability eval | 2D | NO (SD) | **LOW-MEDIUM** — methodology transferable |
| [Zhao et al. (2025). "Enhancing Medical Imaging OOD Detection with Synthetic Outliers." *J. Math. Imaging & Vision*.](https://doi.org/10.1007/s10851-025-01278-2) | Hybrid local+global transforms for pseudo-outlier synthesis → OOD detection | 2D? | YES | **LOW-MEDIUM** — OOD outlier generation concept useful |
| [Gupta et al. (2026). "Physics-Based Benchmarking Metrics for Multimodal Synthetic Images." *arXiv:2511.15204*.](https://arxiv.org/abs/2511.15204) | PCMDE: VLM + detection + physics-guided LLM for synthetic quality eval | 2D | NO | **LOW** — evaluation, not generation |

### 3d. Not Relevant to Our Use Case

| Paper | Why Not |
|---|---|
| [Long et al. (2025). "Leveraging Synthetic Data in Supply Chains." *IJPR*.](https://doi.org/10.1080/00207543.2024.2447927) | Supply chain domain, no imaging |
| [Mianroodi et al. (2025). "MedSynth: Synthetic Medical Dialogue-Note Pairs." *arXiv:2508.01401*.](https://arxiv.org/abs/2508.01401) | Medical NLP, no imaging |

---

## 4. Architecture: Three-Tier Synthetic Generation

Based on the literature, we propose a three-tier architecture ordered by realism and computational cost:

```
Tier 1: Procedural + Statistical (CPU, no training)
  ├─ VascuSynth procedural tree growth → volumetric rendering
  ├─ Spline-based centerlines + Gaussian tube profiles (Nader et al.)
  ├─ 2PM-calibrated noise/PSF models (Zhou et al. semi-synthetic)
  └─ Existing drift_synthetic.py perturbations

Tier 2: Lightweight Learned Models (8GB VRAM, trainable on our data)
  ├─ VesselVAE recursive graph VAE → volumetric rendering
  ├─ VQ-VAE on 32^3-64^3 patches (small codebook)
  ├─ Domain randomization pipeline (SynthICL-style)
  └─ Shape-bias pretraining (VessShape 3D extension)

Tier 3: Heavy Generative Models (deferred — 24GB+ or cloud)
  ├─ VasTSD topology-aware diffusion (when code released)
  ├─ 3D latent diffusion on patches (MedDiffusion-style)
  ├─ VesselFM D_drand (when checkpoint public)
  └─ Normalizing flows on 3D patches (3D Glow-style)
```

### 4.1 Tier 1: Procedural + Statistical (Immediate, CPU-only)

**Goal:** Generate structurally plausible vessel volumes with domain-appropriate noise, without any training.

**Implementation path:**
1. **VascuSynth integration** — Use VascuSynth (C++ library, BSD license) to generate vascular tree geometry from oxygen demand maps. Configure for capillary-scale parameters matching MiniVess (diameter 2-15 um, branching angle 60-120 deg).
2. **Volumetric rendering** — Rasterize VascuSynth tree into 3D volume with Gaussian tube profiles. Apply voxel spacing matching MiniVess (0.31-4.97 um XY, variable Z).
3. **2PM-calibrated noise model** — Fit noise parameters from real MiniVess volumes: shot noise (Poisson), detector noise (Gaussian), background fluorescence (spatially varying). Apply to rendered volumes.
4. **PSF model** — Convolve with estimated 2PM point spread function (elongated in Z due to axial resolution). Parameters from metadata or estimated from bright point sources in real data.

**Augmentation use:** Training on VascuSynth + noise-model volumes → shape-bias pretraining before fine-tuning on real MiniVess.

**Drift simulation use:** Vary noise parameters, PSF width, background level to simulate microscope degradation over time.

### 4.2 Tier 2: Lightweight Learned Models (8GB VRAM)

**Goal:** Learn the distribution of real 2PM vascular morphology and generate novel samples.

**Candidate architectures (ordered by feasibility):**

#### A. VesselVAE (Recursive Graph VAE)
- **What:** Encode vascular tree as graph; VAE learns latent space of branch geometry
- **Why:** Operates on compact graph representation (not voxels), fits 8GB trivially
- **Gap:** Needs volumetric rendering step + 2PM noise model (from Tier 1)
- **Training data:** 70 MiniVess trees extracted via skeletonization + 12 VesselNN trees = 82 graphs

#### B. Patch VQ-VAE (32^3 or 64^3 patches)
- **What:** Vector-quantized VAE on small 3D patches cropped from real volumes
- **Why:** VQ-VAE with 512-2048 codebook on 32^3 patches fits 8GB comfortably
- **Gap:** Generates patches, not full volumes; needs stitching strategy
- **Training data:** ~5000 patches extractable from 70 MiniVess volumes
- **Reference:** MedLoRD uses 16k codebook on 512^3 (too heavy), but scaled-down version is feasible

#### C. Domain Randomization Pipeline (SynthICL-style)
- **What:** MONAI transforms + parametric randomization of intensity, contrast, noise, spacing
- **Why:** No training needed; extends existing augmentation with more diversity
- **Gap:** Does not generate new morphology — only varies appearance of existing vessels
- **Training data:** N/A (transforms applied to existing volumes)

#### D. 3D StyleGAN on Small Patches
- **What:** StyleGAN2-ADA adapted for 32^3 or 64^3 patches
- **Why:** Sordo et al. (2025) show StyleGAN outperforms diffusion for structural scientific images
- **Gap:** Limited training data (70 volumes → ~5000 patches may be marginal for GAN training)
- **Feasibility:** StyleGAN2-ADA designed for limited data; 8GB fits with small patch size

### 4.3 Tier 3: Heavy Generative Models (Deferred)

Reserved for when GPU budget increases (cloud instances or lab GPU upgrade):
- **VasTSD** — Monitor [github.com/JefferyZhifeng](https://jefferyzhifeng.github.io/projects/VasTSD/) for code release
- **3D Latent Diffusion** — Patch-Volume AE (MedDiffusion) adapted for 2PM
- **VesselFM D_drand** — Monitor for checkpoint release
- **Normalizing Flows** — Memory-prohibitive for meaningful 3D volumes on 8GB

---

## 5. Recommended Implementation Roadmap

### Phase 0: Foundation (already done)
- [x] `drift_synthetic.py` — Statistical perturbations
- [x] `acquisition_simulator.py` — Temporal drift streams
- [x] `debug_dataset.py` — Random-walk tubes for tests
- [x] Drift detection suite (Tier 1 Evidently + Tier 2 MMD) — PR #608

### Phase 1: VesselNN Integration (next — Issue #602)
- [ ] `vesselnn-dataset-implementation.xml` T1-T14
- [ ] Cross-dataset drift detection (MiniVess vs VesselNN = natural drift ground truth)
- [ ] DatasetGrowthSimulator with 4 scenarios

### Phase 2: Procedural Vascular Generation (Tier 1)
- [ ] Integrate VascuSynth or implement spline-based centerline generator
- [ ] 2PM noise model calibrated from real MiniVess statistics
- [ ] PSF model for synthetic volumes
- [ ] Validate: synthetic features should overlap real feature distribution for non-morphological features (noise, contrast), but differ for morphological features (branching pattern)

### Phase 3: Lightweight Learned Models (Tier 2)
- [ ] Extract vascular skeletons from MiniVess + VesselNN (82 trees)
- [ ] Train VesselVAE or patch VQ-VAE
- [ ] Evaluate: FID/SSIM/LPIPS on held-out real volumes
- [ ] Use as augmentation: train segmentor on real + synthetic, measure DSC/clDice improvement

### Phase 4: Drift Simulation with Generators
- [ ] Replace `drift_synthetic.py` perturbations with generator-based drift:
  - Vary VascuSynth parameters → morphological drift
  - Vary noise model parameters → acquisition drift
  - Interpolate VesselVAE latent space → smooth distribution shift
- [ ] Validate drift detectors against generator-based drift (more realistic than statistical drift)

### Phase 5: Heavy Models (when resources available)
- [ ] VasTSD integration (if code released)
- [ ] 3D latent diffusion on patch mosaics
- [ ] Fine-tune on combined MiniVess + VesselNN + DeepVess + TubeNet

---

## 6. Key Architectural Decisions

### Q1: Procedural vs Learned?
**Both.** Procedural generation (VascuSynth, spline+kernel) provides unlimited volume with controllable parameters but unrealistic textures. Learned models (VesselVAE, VQ-VAE) capture real texture statistics but are limited by training data. The pipeline is: procedural geometry → learned texture/noise → realistic synthetic volume.

### Q2: How does this connect to drift monitoring?
Synthetic generators produce **parameterized distributions**. By varying generator parameters (noise level, branching density, vessel diameter), we create controlled distribution shifts that are more realistic than the current `drift_synthetic.py` perturbations. This enables:
- Calibrating drift detector sensitivity against known shift magnitudes
- Simulating scenarios that haven't been observed yet (new microscope, new tissue prep)
- Generating OOD test cases for model robustness evaluation

### Q3: How does this connect to segmentation augmentation?
Synthetic volumes with ground-truth labels (from the generator) expand the effective training set. The most impactful augmentation adds **morphological diversity** (new branching patterns, vessel diameters, tortuosity) rather than just appearance diversity (intensity, noise). This is why VascuSynth and VesselVAE are higher priority than domain randomization alone.

### Q4: What about fine-tuning for DeepVess and TubeNet?
The synthetic generators should be **calibrated** to each external dataset's characteristics:
- **DeepVess:** Multi-photon, larger vessels, BBB disruption patterns
- **TubeNet 2PM:** Two-photon, different spatial resolution (0.20 x 0.46 x 5.20 um)
- **VesselNN:** Three different acquisition campaigns with distinct noise profiles

Per-dataset calibration means fitting the 2PM noise model separately for each lab's acquisition parameters, then generating synthetic volumes matching each lab's distribution. This tests whether our segmentor generalizes to unseen labs.

### Q5: 8GB VRAM — what actually fits?
| Method | VRAM | Feasible? |
|---|---|---|
| VascuSynth (CPU) | 0 GB | YES |
| Spline + kernel (CPU) | 0 GB | YES |
| Noise model fitting (scipy) | 0 GB | YES |
| VesselVAE (graph VAE) | <1 GB | YES |
| Patch VQ-VAE 32^3 (512 codebook) | ~2 GB | YES |
| Patch VQ-VAE 64^3 (2048 codebook) | ~4-6 GB | MARGINAL |
| StyleGAN2-ADA 32^3 | ~3-4 GB | YES |
| Latent diffusion 3D (any) | >16 GB | NO |
| Normalizing flows 3D (any) | >24 GB | NO |

---

## 7. Existing Code to Reuse

| File | Reusable For |
|---|---|
| `src/minivess/data/drift_synthetic.py` | Base perturbation engine (Tier 1 statistical) |
| `src/minivess/data/acquisition_simulator.py` | Temporal drift scheduling framework |
| `src/minivess/data/feature_extraction.py` | 9-feature extraction for validating synthetic realism |
| `src/minivess/data/format_conversion.py` | TIFF/NIfTI I/O for synthetic volume export |
| `src/minivess/observability/drift.py` | FeatureDriftDetector for measuring synthetic-vs-real distribution gap |
| `src/minivess/data/debug_dataset.py` | Pattern for NIfTI volume generation (random-walk tubes) |

---

## 8. References (Organized)

### 3D Vascular (Primary)

- [Hamarneh, G. & Jassi, P. (2010). "VascuSynth: Simulating Vascular Trees for Generating Volumetric Image Data." *CMBBE: Imaging & Visualization*.](https://vascusynth.cs.sfu.ca/)
- [Wang, Z. et al. (2025). "VasTSD: Learning 3D Vascular Tree-state Space Diffusion." *CVPR 2025 preprint*, arXiv:2503.12758.](https://arxiv.org/abs/2503.12758)
- [Feldman, A. et al. (2025). "VesselVAE: Recursive Variational Autoencoders for 3D Blood Vessel Synthesis." *arXiv:2506.14914*.](https://arxiv.org/abs/2506.14914)
- [Nader, C. et al. (2024). "A Vascular Synthetic Model for Improved Aneurysm Segmentation." *arXiv:2403.18734*.](https://arxiv.org/abs/2403.18734)
- [Comin, C.H. & Galvao, W.N. (2025). "VessShape: Few-Shot 2D Blood Vessel Segmentation by Leveraging Shape Priors." *arXiv:2510.27646*.](https://arxiv.org/abs/2510.27646)

### 2-Photon Microscopy (Exact Modality)

- [Zhou, D. et al. (2024). "A Deep Learning Approach for Improving Two-Photon Vascular Imaging Speeds." *Bioengineering* 11(2):111.](https://doi.org/10.3390/bioengineering11020111)
- [Teikari, P. et al. (2016). "Deep Learning Denoising of Two-Photon Fluorescence Images." VesselNN dataset.](https://github.com/petteriTeikari/vesselNN)

### Biomedical 3D Generation

- [Chakrabarty, S. et al. (2026). "Synthetic Volumetric Data Generation Enables Zero-Shot Generalization." *arXiv:2601.12297*.](https://arxiv.org/abs/2601.12297)
- [Seyfarth, J. et al. (2025). "MedLoRD: Medical Low-Resource Diffusion for 3D CT." *arXiv:2503.13211*.](https://arxiv.org/abs/2503.13211)
- [Wang, H. et al. (2024). "3D MedDiffusion: Controllable 3D Medical Image Generation." *arXiv:2412.13059*.](https://arxiv.org/abs/2412.13059)
- [Friedrich, P. et al. (2024). "Deep Generative Models for 3D Medical Image Synthesis." *arXiv:2410.17664*.](https://arxiv.org/abs/2410.17664)

### Synthetic Quality & Evaluation

- [Sordo, Z. et al. (2025). "Synthetic Scientific Image Generation with VAE, GAN, and Diffusion." *J. Imaging* 11(8).](https://doi.org/10.3390/jimaging11080252)
- [Bench, C. & Thomas, S.A. (2025). "Quantifying Uncertainty of Synthetic Image Quality Metrics." *arXiv:2504.03623*.](https://arxiv.org/abs/2504.03623)
- [Gupta, K.D. et al. (2026). "Physics-Based Benchmarking Metrics for Multimodal Synthetic Images." *arXiv:2511.15204*.](https://arxiv.org/abs/2511.15204)

### Reliability & OOD Detection

- [Loiseau, T. et al. (2025). "Reliability in Semantic Segmentation: Can We Use Synthetic Data?" *ECCV 2024*.](https://doi.org/10.1007/978-3-031-73337-6_25)
- [Zhao, Y. et al. (2025). "Enhancing Medical Imaging OOD Detection with Synthetic Outliers." *J. Math. Imaging & Vision*.](https://doi.org/10.1007/s10851-025-01278-2)
