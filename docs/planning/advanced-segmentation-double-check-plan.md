# Advanced Segmentation Double-Check: Literature Report

**Date:** 2026-03-04
**Branch:** `feat/advanced-segmentation-double-check`
**Scope:** 44+ papers across 8 research areas, focused on what an MLOps platform
(not a model-SOTA repo) should implement to accelerate researcher productivity.

**Key Principle:** This platform does NOT chase model SOTA. It builds infrastructure
that helps researchers come up with new SOTA faster. Every recommendation below is
evaluated through this lens: does it reduce researcher friction, improve experiment
reproducibility, or provide better quality signals?

---

## Table of Contents

1. [Uncertainty Methods for Foundation Models](#1-uncertainty-methods-for-foundation-models)
2. [Conformal Prediction + Distribution Shifts](#2-conformal-prediction--distribution-shifts)
3. [Model Merging Methods](#3-model-merging-methods)
4. [3D Mamba Models with Uncertainty](#4-3d-mamba-models-with-uncertainty)
5. [Synthetic Data Generation](#5-synthetic-data-generation)
6. [Data Quality & Annotator Pipeline](#6-data-quality--annotator-pipeline)
7. [Latent Diffusion for Segmentation & Synthetic Data](#7-latent-diffusion-for-segmentation--synthetic-data)
8. [VLMs/MLLMs for 3D Medical Segmentation](#8-vlmsmllms-for-3d-medical-segmentation)
9. [Loss Functions & Training Optimization](#9-loss-functions--training-optimization)
10. [Calibration & Trustworthy AI](#10-calibration--trustworthy-ai)
11. [Orchestration & Infrastructure](#11-orchestration--infrastructure)
12. [Prioritized Implementation Roadmap](#12-prioritized-implementation-roadmap)

---

## 1. Uncertainty Methods for Foundation Models

### 1.1 Soft Dice Confidence (SDC) — Deployment Quality Gate

**Paper:** Borges, Pacheco & Silva (2024/2025), arXiv:2402.10665v4, UFSC + UdeM.

**Key Finding:** Image-level selective prediction for segmentation. SDC is a near-optimal
confidence estimator computable in O(n): `SDC = 2*sum(p*y_hat) / sum(p + y_hat)` where p
is softmax probability and y_hat is the hard prediction. Outperforms MSP, entropy, and
lesion load baselines across 6 medical imaging tasks including OOD scenarios.

**Why it matters for the platform:** SDC is a post-hoc, model-agnostic, zero-cost quality
gate. At inference, if SDC < threshold, route to human expert. Trivially implementable
(~5 lines of PyTorch). The coverage-at-selective-risk metric is directly useful for
deployment dashboards.

**Implementation:** Add `compute_sdc()` to `src/minivess/pipeline/validation_metrics.py`.
Wire into BentoML service as a response quality field. Add to dashboard flow metrics.

**Effort:** Very low (< 1 hour). Library: none needed.

### 1.2 Probabilistic SAM Prompting (P2SAM / A-SAM)

**Papers:**
- Huang et al. (2024), ACM MM 2024 — P2SAM: Gaussian distribution over prompt embeddings
- Li et al. (2024), NeurIPS 2024 — A-SAM: Dual prompt + granularity uncertainty

**Key Finding:** SAM's prompt sensitivity is a *feature*, not a bug. Modeling prompt
distributions enables diverse segmentation hypotheses with built-in uncertainty. P2SAM
achieves +12% D_max improvement with only 5.5% of training data.

**Platform relevance:** Directly applicable to the SAM3 adapter. The probabilistic prompt
modeling could provide calibrated uncertainty without ensemble overhead. However, both are
2D methods — 3D extension needed for volumetric vessel segmentation.

**Implementation:** P2 priority. Would extend the existing `SAM3Adapter` with a prompt
generation network. Requires custom code.

### 1.3 Calibrated Stochastic Refinement (CARSSS)

**Paper:** Kassapis et al. (2024), IJCV. University of Amsterdam + TomTom.

**Key Finding:** Two-stage cascaded approach: Stage 1 = any standard segmentation model,
Stage 2 = GAN-based refinement network that adds calibrated stochastic variation. A novel
calibration loss matches sample expectations to pixel-wise probabilities.

**Platform relevance:** The two-stage design means it can be bolted onto existing DynUNet
or SAM3 models as a post-processing module. No retraining of the base model needed.

**Implementation:** P2 priority. Add as optional post-processing in the deploy flow.
Code available: https://github.com/EliasKassapis/CARSSS

### 1.4 Reliability Benchmark for Segmentation

**Paper:** Volpi et al. (2023), CVPR 2023. NAVER LABS Europe + Oxford.

**Key Finding:** More accurate models are NOT more reliable. Calibration error (ECE)
increases dramatically for ALL modern architectures under domain shift. Content-dependent
calibration (pixel-wise temperature scaling) helps more than standard temperature scaling.

**Platform implication:** Validates that the platform's calibration infrastructure (MAPIE,
netcal, Local Temperature Scaling) and multi-metric evaluation (not just DSC) are
essential, not optional. OOD detection is inversely correlated with model performance —
smaller models detect OOD better.

---

## 2. Conformal Prediction + Distribution Shifts

### 2.1 COMPASS — Robust Feature Conformal Prediction (2026)

**Source:** Web search — COMPASS framework for medical segmentation metrics.

**Key Finding:** Conformal prediction applied to segmentation metric prediction (not just
pixel classification). Uses model features to construct prediction sets for Dice, HD95,
and other quality metrics. Provides formal coverage guarantees under distribution shift.

**Platform relevance:** Extends the existing CP infrastructure in
`src/minivess/pipeline/conformal.py` from pixel-level to metric-level prediction sets.
Could answer: "with 95% confidence, this volume's Dice score is in [0.82, 0.91]."

### 2.2 Anatomically-Aware CP with Random Walks (2026)

**Source:** Web search — 35.4% improvement in prediction set efficiency.

**Key Finding:** Random walk-based nonconformity scores that respect anatomical structure.
Standard CP treats pixels independently; this method uses spatial dependencies to produce
tighter prediction sets while maintaining coverage guarantees.

**Platform relevance:** Directly extends `MorphologicalConformalPredictor` and
`DistanceTransformConformalPredictor` already in the codebase. The random walk approach
is particularly relevant for vessel segmentation where spatial connectivity matters.

### 2.3 Conformal Semantic Segmentation (CRC)

**Paper:** Mossina et al. (2024), CVPR 2024 Workshop. IRT Saint Exupery.

**Key Finding:** Post-hoc Conformalized Risk Control (CRC) for multi-class segmentation.
LAC thresholding: include all classes with softmax >= 1-lambda, where lambda is calibrated
to guarantee expected risk <= alpha. Introduces "varisco" heatmaps (prediction set size
per pixel) for uncertainty visualization.

**Platform relevance:** Post-hoc, model-agnostic, formal guarantees. Extends existing CP
infrastructure. The varisco heatmaps are directly useful for the dashboard flow.
Code: https://github.com/deel-ai-papers/conformal-segmentation

**Implementation:** P0-P1 priority. Low effort, extends existing ConformalEvaluator.

### 2.4 Class-Adaptive Conformal Training (CaCT)

**Paper:** Vakalopoulou et al. (2026), arXiv:2601.09522. CentraleSupelec + ETS Montreal.

**Key Finding:** Formulates conformal training as augmented Lagrangian optimization with
class-wise multipliers. Produces smaller, more informative prediction sets for
underrepresented classes without distributional assumptions.

**Platform relevance:** Addresses a key limitation: standard CP produces uninformative
sets for rare vessel substructures (thin branches, bifurcations). Class-adaptive approach
is directly applicable to imbalanced medical segmentation.

**Implementation:** P1 priority. Integrates into training loop alongside existing losses.

### 2.5 AdaConG — CP-Guided Training Weights

**Paper:** Liu et al. (2025), arXiv:2502.16736. UMD + NC State + Adobe.

**Key Finding:** Embed split CP into training loop to adaptively weight guidance signals:
`w(x) = exp(-gamma * u(x))`. Applicable to knowledge distillation and semi-supervised
learning under domain shift.

**Platform relevance:** The principle of "trust guidance proportional to calibrated
confidence" is applicable to pseudo-label self-training and teacher-student setups.

---

## 3. Model Merging Methods

### 3.1 MedSAMix — Training-Free Model Merging

**Paper:** Yang et al. (2025), arXiv:2508.11032. U Tubingen + MPI.

**Key Finding:** Zero-order optimization (SMAC) discovers optimal layer-wise merging
configurations for SAM-based models. Two regimes: single-task optimization (+6.67%) and
multi-objective Pareto optimization (+4.37% on 25 tasks). Training-free, uses only a few
calibration samples.

**Platform relevance:** Directly maps to the ensemble module. The platform already has
champion categories (balanced/topology/overlap) — MedSAMix could combine champion models
from different categories into a single merged model without retraining. The SMAC-based
search aligns with the Optuna HPO infrastructure.

**Implementation:** P1 priority. Add `MergedModelAdapter` using SMAC + existing model
weights. Low effort — merging operations are tensor arithmetic.

### 3.2 SWA / Model Soup

**Source:** Knowledge base analysis (LLM - Ensembles and Merging).

**Key Finding:** Stochastic Weight Averaging across fold checkpoints produces a single
model with no inference overhead and wider optima. Already partially implemented in the
codebase's "soup" ensemble strategy.

**Platform relevance:** The cheapest ensemble win. Complete the existing model soup
implementation by averaging DynUNet checkpoints across folds trained with the same loss.

**Implementation:** P0 priority. Existing code, needs completion and testing.

### 3.3 SLERP Between Loss-Specialized Models

**Source:** Knowledge base (mergekit patterns from LLM merging).

**Key Finding:** SLERP (Spherical Linear Interpolation) between models trained with
different losses could produce a single model balancing overlap and topology metrics.
E.g., interpolate dice_ce model (best DSC) with cbdice_cldice model (best clDice).

**Platform relevance:** Novel opportunity specific to the platform's multi-loss training.
Instead of ensembling at inference, merge in weight space for zero inference overhead.
Requires models to share architecture (same DynUNet config) and be in the same loss basin
(fold-level models from the same experiment typically are).

**Implementation:** P1 priority. ~30 lines: `(1-t)*theta_1 + t*theta_2` for linear,
slightly more for SLERP. Add to ensemble module.

### 3.4 Non-Monotone Ensemble Scaling

**Source:** Chen et al. (2024) — voting inference scaling laws.

**Key Finding:** More ensemble members can HURT on hard queries. For vessel segmentation,
"hard" means thin distal vessels and bifurcations. Optimal ensemble size should be tuned
per difficulty tier.

**Platform implication:** The per-difficulty performance tracking should be added to the
analysis flow. Don't assume more models = better — validate on vessel subcategories.

---

## 4. 3D Mamba Models with Uncertainty

### 4.1 Comprehensive Mamba vs Transformer Benchmark

**Paper:** Wang et al. (2025), arXiv:2503.19308. U Adelaide + MBZUAI + UC Santa Cruz.

**Key Finding:** Systematic comparison showing UlikeMamba outperforms UlikeTrans in accuracy
and efficiency for 3D medical segmentation. Key architectural decisions:
- Replace 1D DWConv with **3D DWConv** in Mamba blocks for volumetric data
- Multi-scale Mamba blocks capture fine-grained details + global context better than Transformer
- **Tri-scan** strategy (left-right, up-down, front-back) best for complex anatomies
- Mamba avoids OOM issues that plague Transformer networks
- Evaluated on AMOS (15 organs), TotalSegmentator (117 structures), BraTS

**Platform relevance:** Very high. Mamba is the most promising alternative to DynUNet for
3D volumetric segmentation. O(n) complexity vs O(n^2) for attention. A `MambaAdapter`
could sit alongside `DynUNetAdapter` and `SAM3Adapter` in the model-agnostic architecture.

**Implementation:** P1-P2 priority. The `mamba-ssm` library provides core operations. The
U-shaped architecture needs custom implementation but individual blocks use standard
PyTorch/MONAI ops. Medium effort.

### 4.2 UD-Mamba — Uncertainty-Driven Pixel Scanning

**Source:** Web search — arXiv:2502.02024.

**Key Finding:** Uses uncertainty estimation to determine pixel scanning order for Mamba.
High-uncertainty regions are scanned first (or with more context), improving segmentation
quality where it matters most. Combines the efficiency of Mamba with targeted uncertainty
allocation.

**Platform relevance:** An elegant uncertainty-aware Mamba variant. If implementing a
MambaAdapter, UD-Mamba's uncertainty-driven scanning is more principled than fixed
raster-scan patterns.

### 4.3 Mamba3D-MedSeg (nnUNet v2 Framework)

**Source:** Web research — GitHub mamba3d-medseg.

**Key Finding:** UlikeMamba_3dMT variant built on nnUNet v2 framework. Supports AMOS,
TotalSeg, BraTS. The nnUNet v2 integration means it inherits nnUNet's automatic
configuration pipeline.

**Platform relevance:** If nnUNet-compatible, could be integrated via the model-agnostic
adapter pattern with minimal friction.

### 4.4 Hybrid SSM-Attention Architectures

**Source:** Knowledge base (LLM - State Space Models).

**Key Finding:** Community consensus: hybrid SSM + full attention blocks are "very
promising." Use Mamba in early (high-resolution) layers where sequence length is
prohibitive for attention, and attention in deeper (low-resolution) layers where
contextual reasoning about anatomy matters (cf. Hymba architecture).

**Platform relevance:** The optimal architecture for 3D vessel segmentation may be a
hybrid: Mamba for spatial resolution, attention for semantic reasoning at the bottleneck.
This would be a custom ModelAdapter.

---

## 5. Synthetic Data Generation

### 5.1 VasTSD — Vascular Tree-State Space Diffusion

**Paper:** Wang et al. (2025), CVPR 2025. NUDT.

**Key Finding:** Synthesizes 3D angiography from non-angiographic volumes by modeling
vascular tree topologies. Key innovation: vascular tree-state space serialization that
dynamically constructs tree topologies and integrates with diffusion. A pre-trained vision
embedder constructs vascular state space representations, ensuring anatomically continuous
vasculature across modalities.

**Platform relevance:** Two applications: (1) **Data augmentation** — generate synthetic
vascular training data to supplement MiniVess's 70 volumes; (2) **Topology-aware
serialization** — the vascular tree serialization order could inform Mamba scanning
strategies for vessel-specific models.

**Implementation:** P2 priority. Code not yet released ("soon" at github.com/zfw-cv/VasTSD).
The concept is valuable even without the code — the serialization approach could be
implemented independently using the platform's existing vessel graph pipeline
(`src/minivess/pipeline/vessel_graph.py`).

### 5.2 Existing Platform Capabilities

The platform already has `src/minivess/data/acquisition_simulator.py` with
`SyntheticAcquisitionSimulator` for drift simulation. VasTSD would complement this by
generating anatomically plausible 3D vessel structures, while the existing simulator
handles acquisition-level drift (noise, resolution changes).

---

## 6. Data Quality & Annotator Pipeline

### 6.1 SegQC — Segmentation Quality Control

**Paper:** Specktor-Fadida et al. (2025), Medical Image Analysis Vol. 103.

**Key Finding:** SegQCNet inputs scan + segmentation mask and outputs per-voxel error
probabilities, from which it derives estimated Dice, IoU, and ARVD without ground truth.
Error region detection in individual slices enables human-in-the-loop workflows.

**Platform relevance:** Directly maps to Flow 6 (QA). SegQC enables runtime QC during
inference deployment — predict segmentation quality without ground truth, flag low-quality
predictions for human review.

**Implementation:** P1 priority. Add to the QA flow as an optional QC module.

### 6.2 AmbiSSL — Annotation Ambiguity in Semi-Supervised Settings

**Paper:** Kumari & Singh (2025), CVPR 2025. IIT Roorkee.

**Key Finding:** First method handling annotation ambiguity in semi-supervised segmentation.
Uses randomized decoder pruning for diverse pseudo-label generation, Laplace distributions
for unlabeled data (avoiding overconfidence), and cross-decoder supervision.

**Platform relevance:** MiniVess deals with inherent boundary ambiguity in microvascular
segmentation. The Laplace distribution choice for pseudo-labels is a practical insight for
self-training. The diversity-through-pruning approach is simple and effective.

### 6.3 ProSona — Prompt-Guided Multi-Annotator Modeling

**Paper:** Albarqouni et al. (2025), arXiv:2511.08046. LMU/TU Munich.

**Key Finding:** Natural language descriptions of annotator styles (via CLIP text encoder)
control segmentation output style. "Conservative radiologist" vs "inclusive radiologist"
produce different valid segmentations from the same image.

**Platform relevance:** Could integrate with Label Studio for modeling inter-annotator
variability. However, 2D only — 3D extension needed.

### 6.4 C2B — Automatic Bias Detection

**Paper:** Guimard et al. (2025), U Trento.

**Key Finding:** Training-free bias detection using LLM-generated bias proposals + CLIP
retrieval. Discovers biases beyond annotated datasets.

**Platform relevance:** Could be integrated into Flow 3 (Analysis) or Flow 6 (QA) for
automatic fairness auditing. Does the model perform worse on certain imaging devices or
patient subgroups?

### 6.5 CLIP Acquisition Traces

**Paper:** Ramos et al. (2025), ICCV 2025.

**Key Finding:** Visual encoders (CLIP, DINOv2) systematically encode acquisition
parameters (camera model, exposure, compression) in their representations. These traces
can overshadow semantic content and cause biased predictions.

**Platform relevance:** For microscopy imaging, different acquisition parameters could bias
foundation model features. Validates importance of robust data augmentation when using
CLIP/DINOv2-based models. Design decision: ensure the data pipeline applies sufficient
augmentation to mitigate acquisition parameter encoding.

---

## 7. Latent Diffusion for Segmentation & Synthetic Data

### 7.1 Flow-SSN — Flow Stochastic Segmentation Networks

**Paper:** Ribeiro et al. (2025), arXiv:2507.18838. Imperial College London (Glocker lab).

**Key Finding:** The most rigorous approach to stochastic segmentation. Proves fundamental
limitations of low-rank Gaussian SSNs (effective rank grows only sublinearly). Flow-SSNs
estimate arbitrarily high-rank pixel-wise covariances without storing distributional
parameters. Both discrete-time (IAF) and continuous-time (flow matching) variants.
Evaluated on LIDC-IDRI and a retinal vessel dataset.

**Platform relevance:** Very high — addresses vessel segmentation directly with full
uncertainty quantification. The flow-matching variant is fast at inference. Most model
capacity in the learned base distribution (not the flow), making sampling efficient.
Code: https://github.com/biomedia-mira/flow-ssn

**Implementation:** P1-P2 priority. Could be integrated as a new ModelAdapter wrapping
Flow-SSN. Medium effort.

### 7.2 HSRDiff — Hierarchical Self-Regulation Diffusion

**Paper:** Yang et al. (2025), ICT Chinese Academy of Sciences.

**Key Finding:** Unified diffusion framework for stochastic segmentation with hierarchical
multi-scale condition priors. Validated on LIDC-IDRI, Cityscapes, and **RITE retinal
vessel dataset**. Code: https://github.com/yanghan-yh/HSRDiff.git

**Platform relevance:** The vessel evaluation makes this directly relevant. The
hierarchical multi-scale design suits thin vessel structures.

### 7.3 CCDM — Conditional Categorical Diffusion

**Paper:** Zbinden et al. (2023), ICCV 2023. University of Bern.

**Key Finding:** Discrete diffusion operating directly in categorical label space. Avoids
continuous-to-discrete conversion issues. Models inter-annotator disagreement via
categorical noise transitions.

**Platform relevance:** Principled approach to modeling annotation ambiguity.

### 7.4 LatentFM — Latent Flow Matching

**Paper:** Huynh et al. (2025), arXiv:2512.04821. Same group as paper 7.1's latent
diffusion variant but using flow matching (faster, fewer sampling steps).

**Platform relevance:** Flow matching is faster than diffusion at inference, making it more
practical for deployment. Latent-space operation reduces computational cost further.

---

## 8. VLMs/MLLMs for 3D Medical Segmentation

### 8.1 MedCLIPSeg — Probabilistic Vision-Language Segmentation

**Paper:** Koleilat et al. (2026), arXiv:2602.04023. Concordia University.

**Key Finding:** Adapts CLIP for medical segmentation via Probabilistic Vision-Language
(PVL) adapter. Keys and Values modeled as distributions (mean + variance), enabling
confidence-weighted attention. Monte Carlo sampling produces uncertainty maps. Works with
just 10% of training data across 16 datasets, 5 modalities, 6 organs.

**Platform relevance:** Three critical capabilities: (1) data efficiency for small
datasets like MiniVess, (2) built-in uncertainty via probabilistic attention, (3)
text-driven segmentation enabling natural language interaction. Aligns with the platform's
"zero-config start" goal — text prompts as a more intuitive interface than geometric prompts.
Code: https://tahakoleilat.github.io/MedCLIPSeg

**Implementation:** P2 priority. Requires CLIP backbone + custom PVL adapter layers.

### 8.2 CRISP-SAM2 — Cross-Modal Interaction and Semantic Prompting

**Paper:** Yu et al. (2025), ACM MM 2025. Hangzhou Dianzi University.

**Key Finding:** Extends SAM2 for text-guided 3D multi-organ segmentation. Eliminates
geometric prompts entirely via Semantic Prompt Projector. Similarity-sorting self-updating
memory (replacing FIFO) for better 3D spatial coherence.

**Platform relevance:** Eliminates the need for manual point/box annotation, making it
suitable for automated pipelines. The similarity-sorting memory strategy is relevant for
the platform's slice-by-slice 3D processing approach.
Code: https://github.com/YU-deep/CRISP_SAM2.git

### 8.3 LMM Open-World Classification Benchmark

**Paper:** Conti et al. (2025), arXiv:2503.21851. U Trento.

**Key Finding:** LMMs outperform contrastive baselines in open-world classification but lag
behind closed-world models. The evaluation methodology (LLM-as-judge, semantic similarity)
could inspire evaluation for text-driven segmentation models.

**Platform relevance:** Low for segmentation directly; useful for evaluation methodology.

### 8.4 Foundation Model Segmentation Survey

**Paper:** Zhou et al. (2024), arXiv:2408.12957. 300+ methods surveyed.

**Key Finding:** Segmentation knowledge emerges from FMs not designed for segmentation
(CLIP, DINO, diffusion models). Taxonomy: CLIP-based, DM-based, DINO-based, SAM-based,
composition of FMs.

**Platform relevance:** Validates the model-agnostic ModelAdapter design. The platform
should support multiple FM-based approaches interchangeably.
Reference: https://github.com/stanley-313/ImageSegFM-Survey

---

## 9. Loss Functions & Training Optimization

### 9.1 SPW Loss — Steerable Pyramid Weighted Loss

**Paper:** Lu (2025), arXiv:2503.06604. Cornell University.

**Key Finding:** Multi-scale adaptive weighting for cross-entropy using steerable pyramid
decomposition. Unlike boundary-aware losses using static distance transforms, SPW
dynamically decomposes both GT and predictions into multi-scale, multi-orientation
subbands. Evaluated on SNEMI3D (neurites), GlaS (glands), **DRIVE (retinal vessels)**
against 11 SOTA losses. Outperforms with minimal computational overhead.

**Platform relevance:** Directly targets thin tubular structures (vessels, neurites) — the
exact MiniVess use case. Could be integrated as loss #19 alongside clDice, cbDice, CAPE.
Code: https://anonymous.4open.science/r/SPW-0884

**Implementation:** P1 priority. Steerable pyramid via `pyrtools` or scipy FFT. Custom
PyTorch loss wrapper. Medium effort.

### 9.2 Dice Loss Gradient Analysis

**Paper:** Kervadec & de Bruijne (2023), arXiv:2304.03191. Erasmus MC + U Copenhagen.

**Key Finding:** The Dice loss gradient is a dynamically weighted negative of the ground
truth — a two-valued gradient map. Counter-intuitively, Dice can "reward" misclassified
voxels and "punish" correct ones. This explains why Dice + CE compound losses outperform
pure Dice.

**Platform relevance:** Theoretical validation for the platform's default `cbdice_cldice`
compound loss. Informs loss function documentation and researcher guidance.

### 9.3 Domain Division for Training Optimization

**Paper:** Xu et al. (2020), ACM MM. USTC.

**Key Finding:** Dynamic pixel decomposition into meta-train (uncertain, informative) and
meta-test (confident) domains during training. Meta-optimization ensures updates on
meta-train benefit meta-test. Tested on BRATS and chest X-rays.

**Platform relevance:** Could enhance training by focusing on hard-to-segment vessel
regions. Implementable as a training wrapper.

---

## 10. Calibration & Trustworthy AI

### 10.1 DA-Cal — Cross-Domain Calibration

**Paper:** Li et al. (2026), arXiv:2602.08060. USTC.

**Key Finding:** Meta Temperature Network (MTN) generates pixel-level calibration
temperatures. Under perfect calibration, soft pseudo-labels equal hard pseudo-labels.
Bi-level optimization prevents overfitting. No inference overhead.

**Platform relevance:** Directly extends the calibration stack (MAPIE, netcal, Local
Temperature Scaling). Pixel-level calibration is especially important for medical
segmentation with domain shift (different microscopes, staining protocols).

**Implementation:** P1 priority. Custom MTN + existing calibration infrastructure.

### 10.2 Trustworthy AI Framework

**Paper:** Zuluaga et al. (2026), Current Opinion in Biomedical Engineering.

**Key Finding:** Three-tier framework: (1) Core — data quality, model design for
heterogeneity, label quality; (2) Feedback — human oversight via input QC, UQ, output QC;
(3) Explainable — input/model/output-level explanations.

**Platform relevance:** Validates the platform's architecture across all three tiers:
- Core: data profiling (whylogs), adaptive compute, model profiles
- Feedback: conformal UQ, calibration, SegQC, Deepchecks, WeightWatcher
- Explainable: Captum, SHAP, Quantus

The framework provides a principled way to organize trustworthiness properties for the
paper.

### 10.3 MedConf — Medical LLM Confidence

**Paper:** Ren et al. (2026), arXiv:2601.15645. NTU + Wuhan University.

**Key Finding:** Evidence-grounded confidence for medical LLMs via RAG-based symptom
profiling. Outperforms 27 baseline confidence methods.

**Platform relevance:** Relevant to LLM/agent observability (Langfuse, LangGraph). Could
enhance reliability of LLM-driven analysis in the pipeline.

### 10.4 CP-Router / ConfAgents — CP for Agent Orchestration

**Papers:**
- Su et al. (2025) — CP-Router: route between cheap/expensive LLMs based on CP uncertainty
- Zhao et al. (2025) — ConfAgents: CP-based triage for multi-agent medical diagnosis

**Platform relevance:** Both applicable to the LangGraph agent orchestration layer.
CP-based routing could optimize cost/performance in the LLM stack.

---

## 11. Orchestration & Infrastructure

### 11.1 Prefect + SLURM Integration

**Paper:** Giron (2026), Medium blog post.

**Key Finding:** Prefect 3.x + `dask-jobqueue` + `DaskTaskRunner` bridges Prefect flows
and HPC SLURM clusters. Patterns: (1) modular subflows for pipeline composition, (2)
dynamic runtime task generation, (3) per-stage resource allocation via `cluster.scale()`.

**Platform relevance:** Bridges the gap between SkyPilot (cloud) and local (Docker Compose)
compute tiers. The dynamic workflow pattern matches HPO sweep use cases.

**Implementation:** P2 priority. All via library calls (`prefect`, `dask-jobqueue`,
`prefect-dask`).

### 11.2 WS-ICL — Weak Supervision for Annotation Efficiency

**Paper:** Hu et al. (2025), arXiv:2510.05899.

**Key Finding:** Bounding box or point prompts achieve 83% of full-annotation Dice at
<10% annotation cost. Built on Neuroverse3D.

**Platform relevance:** Could reduce annotation burden when expanding to new anatomical
structures. Relevant for the data engineering flow's annotation tools.

---

## 12. Prioritized Implementation Roadmap

### P0 — Immediate Value, Minimal Effort

| Item | Source | Effort | Platform Component |
|------|--------|--------|--------------------|
| SDC quality gate | Borges (2025) | Very low (~5 lines) | Deploy flow, BentoML |
| Complete model soup/SWA | KB merging + existing code | Low | Ensemble module |
| CRC conformal segmentation | Mossina (2024) | Low (code available) | Conformal evaluator |
| Varisco uncertainty heatmaps | Mossina (2024) | Low | Dashboard flow |

### P1 — High Value, Moderate Effort

| Item | Source | Effort | Platform Component |
|------|--------|--------|--------------------|
| MedSAMix model merging | Yang (2025) | Low-Medium | Ensemble module |
| SLERP loss-specialized merge | KB mergekit | Low | Ensemble module |
| SPW vessel loss function | Lu (2025) | Medium | Loss registry (#19) |
| DA-Cal pixel calibration | Li (2026) | Medium | Calibration stack |
| CaCT class-adaptive CP | Vakalopoulou (2026) | Medium | CP training |
| SegQC automated QC | Specktor-Fadida (2025) | Medium | QA flow |
| Flow-SSN stochastic seg | Ribeiro (2025) | Medium | New ModelAdapter |
| Anatomically-aware CP | Web (2026) | Medium | Conformal predictor |

### P2 — Strategic, Higher Effort

| Item | Source | Effort | Platform Component |
|------|--------|--------|--------------------|
| MambaAdapter (3D) | Wang (2025) | Medium-High | New ModelAdapter |
| MedCLIPSeg VLM adapter | Koleilat (2026) | Medium-High | New ModelAdapter |
| HSRDiff stochastic seg | Yang (2025) | Medium | New ModelAdapter |
| CRISP-SAM2 text prompts | Yu (2025) | Medium | SAM adapter |
| VasTSD synthetic data | Wang (2025) | High (code pending) | Data augmentation |
| Prefect+SLURM compute | Giron (2026) | Medium | Orchestration |
| Probabilistic SAM prompts | Huang/Li (2024) | Medium | SAM3 adapter |
| CARSSS refinement | Kassapis (2024) | Medium | Post-processing |
| C2B bias detection | Guimard (2025) | Medium | QA flow |

### P3 — Research Exploration

| Item | Source | Effort | Notes |
|------|--------|--------|-------|
| CCDM categorical diffusion | Zbinden (2023) | High | Multi-annotator modeling |
| ProSona multi-annotator | Albarqouni (2025) | High | 2D only, needs 3D |
| UD-Mamba uncertainty scan | Web (2025) | High | Needs MambaAdapter first |
| AmbiSSL semi-supervised | Kumari (2025) | Medium | CVPR 2025 |
| ConOVS continual learning | Hwang (2025) | High | 2D OVS, needs adaptation |
| MoSE multi-annotator | Gao (2023) | High | Needs multi-annotator data |
| AdaConG CP training | Liu (2025) | Low-Medium | General framework |

---

## Key Takeaways for the Platform Paper

1. **Model-agnostic architecture is validated** — Zhou's survey shows segmentation knowledge
   emerges from diverse FM types. The ModelAdapter ABC pattern is the right abstraction.

2. **Calibration is NOT optional** — Volpi (2023) proves robust models are NOT reliable
   models. The platform's calibration stack is essential infrastructure.

3. **Conformal prediction is maturing rapidly** — From pixel-level (existing) to metric-level
   (COMPASS), class-adaptive (CaCT), and training-integrated (AdaConG). The platform is
   well-positioned to be a reference implementation.

4. **Model merging is the cheapest ensemble win** — MedSAMix and SLERP provide single-model
   inference with multi-model diversity. Zero inference overhead.

5. **Mamba is the leading alternative architecture for 3D** — O(n) complexity, no OOM issues,
   outperforms Transformers on large 3D volumes. A MambaAdapter is strategic.

6. **SDC is the simplest deployment quality gate** — 5 lines of code, near-optimal
   confidence for selective prediction. Should be standard in all deployed models.

7. **Trustworthy AI framework** — Zuluaga's three-tier framework validates the platform's
   architecture and provides a principled paper contribution framing.

---

## Bibliography

All papers referenced are in the project bibliography at
`docs/planning/prd/bibliography.yaml`. Key new citations to add:

- borges_2025_sdc: Soft Dice Confidence
- mossina_2024_conformal_seg: Conformal Semantic Segmentation
- volpi_2023_reliability: Reliability in Semantic Segmentation
- yang_2025_medsamix: MedSAMix Model Merging
- wang_2025_mamba_analysis: Mamba vs Transformer 3D Analysis
- lu_2025_spw_loss: Steerable Pyramid Weighted Loss
- li_2026_dacal: DA-Cal Cross-Domain Calibration
- vakalopoulou_2026_cact: Class Adaptive Conformal Training
- specktor_fadida_2025_segqc: SegQC Quality Control
- zuluaga_2026_trustworthy: Trustworthy AI Framework
- ribeiro_2025_flow_ssn: Flow Stochastic Segmentation Networks
- koleilat_2026_medclipseg: MedCLIPSeg
- yu_2025_crisp_sam2: CRISP-SAM2
- kervadec_2023_dice_gradient: Dice Loss Gradient Analysis
- wang_2025_vastsd: VasTSD Vascular Tree Diffusion
- huynh_2025_latentfm: Latent Flow Matching
- yang_2025_hsrdiff: HSRDiff
- zbinden_2023_ccdm: Conditional Categorical Diffusion
- kassapis_2024_carsss: Calibrated Adversarial Refinement
- albarqouni_2025_prosona: ProSona Multi-Annotator
- giron_2026_prefect_slurm: Prefect + SLURM Integration
- kumari_2025_ambissl: AmbiSSL
