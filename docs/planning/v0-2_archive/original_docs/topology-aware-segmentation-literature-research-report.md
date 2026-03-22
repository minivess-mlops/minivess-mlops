---
title: "Topology-Aware 3D Microvascular Segmentation Literature Report"
status: reference
created: "2026-03-02"
---

# Topology-Aware 3D Microvascular Segmentation: A Multi-Hypothesis Literature Exploration

**Project:** MinIVess MLOps v2 — `feat/advanced-topological-segmentation`
**Date:** 2026-03-02
**Status:** Draft v2 — Post reviewer iteration (3 parallel reviewers: factual accuracy, Markov mandate, coverage gaps)
**Scope:** Architecture, loss, representation, and post-processing advances beyond vanilla DynUNet + cbdice_cldice

---

## Abstract

This report synthesizes evidence from recent publications (2020–2026) across vascular
segmentation, computational topology, foundation models, and uncertainty quantification
to evaluate advanced topology-aware approaches for 3D microvascular networks imaged
by two-photon fluorescence microscopy. We organize the literature around **five competing
hypotheses** for advancing beyond the current MinIVess baseline (DynUNet + cbdice_cldice,
0.906 clDice / 0.824 DSC on 70 volumes). Each hypothesis synthesizes evidence from
multiple research threads, identifies specific gaps at previously unexplored intersections,
and proposes concrete experiments with falsifiable predictions. We conclude with ranked
recommendations informed by implementation feasibility on an 8 GB VRAM budget.

**Key synthesis insights:** Two principal findings emerge from this synthesis:

1. **Representation over loss.** The field is converging on a consensus that *what the
network predicts* matters more than *how it is penalized* for topological fidelity.
Continuous centerlines (Zhao et al., 2025), signed distance fields (Wu et al., 2025),
and graph-structured outputs (Liu et al., 2025) produce larger topology gains than any
loss-function modification atop binary mask supervision. This challenges the MinIVess
status quo of iterating on compound losses over a fixed mask representation.

2. **The Width-Topology-Representation Triangle.** Three independently developed research
threads — width-aware persistent homology (Li et al., 2026), SDF-based pre-training
(Wu et al., 2025), and deformable centerline learning (Zhao et al., 2025) — converge
on a single insight: topologically correct segmentations must be simultaneously connected,
geometrically accurate (correct vessel caliber), and resolution-independent. No published
work, to our knowledge, combines all three.

---

## 1. Context: The Topology Crisis in Vascular Segmentation

### 1.1 Why topology, not just overlap, defines segmentation quality

Standard segmentation metrics (Dice, IoU) treat each voxel independently, making them
blind to the connected structure that defines a vascular network's function. A single
disconnected capillary — scoring 0.999 Dice — can render downstream hemodynamic
analysis meaningless. This fundamental observation has driven an explosion of
topology-aware methods, yet Berger et al. (2026, "Pitfalls of Topology-Aware Image
Segmentation") and Valverde et al. (2025, BMVC — TopoMortar) provide the field's most
sobering reality check: **most topology losses are not generally advantageous** and
their benefits are often confounded with dataset-specific challenges. Berger et al.
enumerate specific pitfalls: benefits that disappear with strong data augmentation,
gains attributable to small training set size rather than topology preservation, and
metric improvements that reflect label noise correction rather than true topological
learning. For MiniVess — with 70 volumes, relatively clean 2PM labels, and a strong
DynUNet baseline — this caution applies directly.

The TopoMortar benchmark (Valverde et al., 2025) compared 8 loss functions (6
topology-aware + 2 baselines including CEDice) under controlled conditions (10 random
seeds, nnUNet-level architecture, strong data augmentation) and found that **only clDice
consistently improves topology** across settings. Skeleton Recall (Kirchhoff et al.,
2024, ECCV) ranked second but was less robust. The most expensive PH-based losses
(TopoLoss: Hu et al., 2019; Warping: Hu, 2022) required 49.6–152.1 hours vs. 3.1 hours
for clDice, and did not reliably outperform simpler alternatives. Betti Matching (Stucki
et al., 2023, ICML) was not evaluable in TopoMortar (described as requiring "centuries"
to train). This validates the MinIVess default of cbdice_cldice but raises the question:
**if loss engineering has reached diminishing returns, where should we invest next?**

### 1.2 The MiniVess baseline and its limitations

The current MinIVess pipeline comprises:
- **Architecture:** DynUNet (MONAI) — a 3D encoder-decoder with adaptive kernel sizes
- **Loss:** cbdice_cldice = 0.5 * cbDice + 0.5 * dice_ce_cldice (Shi et al., 2024; Shit et al., 2021)
- **Metrics:** 8 primary metrics (DSC, HD95, ASSD, NSD, clDice, BE_0, BE_1, Junction F1)
- **Topology modules:** 18 loss functions (4 LIBRARY, 2 LIBRARY-COMPOUND, 3 HYBRID, 9 EXPERIMENTAL), centreline extraction, vessel graph pipeline, GAT modules, TFFM wrapper
- **Hardware constraint:** 8 GB VRAM (RTX 2070 Super)

Verified experimental results (4 losses x 3 folds x 100 epochs) show cbdice_cldice
achieves 0.906 clDice with only -5.3% DSC penalty vs. dice_ce. The 18-loss debug sweep
revealed that standalone topology losses (topo, betti) produce near-zero DSC (0.012–0.016),
and betti_matching OOMs on real volumes without aggressive feature capping. The system
can *detect* topology errors (comprehensive metric suite) but cannot *correct* them
post-inference — no topology-preserving post-processing exists.

### 1.3 The evaluation crisis

Before considering new methods, we must address the evaluation infrastructure.
Podobnik and Vrtovec (2025, MeshMetrics) demonstrated that existing implementations of
distance-based metrics (including Google DeepMind's surface-distance library and MONAI)
produce **HD95 discrepancies exceeding 100 mm and NSD discrepancies of 30 percentage
points** for the same segmentation pair, caused by different boundary extraction
algorithms, non-equivalent mathematical definitions, and inconsistent edge-case handling.
For MiniVess, where NSD is computed at tau = 2 * median_voxel_spacing (~1.0 um), this
quantization artifact zone is exactly where our measurements operate. Decroocq et al.
(2025) additionally benchmarked tubular-structure-specific metrics and found systematic
discrepancies across tools. These findings align with the Metrics Reloaded framework (Maier-Hein et al., 2024,
Nature Methods, 73 authors), which recommends task-specific metric selection based on
what matters clinically. For vascular segmentation, Metrics Reloaded classifies boundary
accuracy and topology as primary concerns, overlap as secondary — validating the MinIVess
8-metric suite (DSC, HD95, ASSD, NSD, clDice, BE_0, BE_1, Junction F1) but raising the
question of whether rank-aggregated champion selection adequately reflects this priority
ordering. **Recommendation:** Adopt MeshMetrics as validation backend, validate the
8-metric suite against Metrics Reloaded's tubular structure recommendations, and report
metric implementation details in all publications.

---

## 2. Five Hypotheses for Advancing Beyond the Baseline

We organize the literature around five hypotheses, ranked by the specificity and
strength of supporting evidence. Each hypothesis represents a distinct research
direction; they are not mutually exclusive.

### Hypothesis 1: Multi-Task Representation Learning (Centerline + Distance Field + Mask)

**Thesis:** Predicting multiple complementary representations simultaneously — binary
mask, signed distance field (SDF), and centerline distance map — forces the network
to learn geometry-aware features that implicitly encode topology, yielding larger gains
than any single-task loss modification.

**Supporting Evidence:**

The Deep Distance Transform (DDT, Wang et al., CVPR 2020; arXiv 2019) demonstrated
that jointly predicting segmentation masks and distance transform maps improves DSC by
+13% on pancreatic ducts over a 3D-UNet baseline, and achieves #1 on the MSD hepatic
vessel leaderboard (63.43% DSC, marginally ahead of nnU-Net at 63.00%) — adding a
distance regression head to existing architectures without topology-specific losses.
DDT's geometry-aware refinement step
reconstructs segmentation from predicted cross-sectional radii via Gaussian-weighted
maximal ball envelopes, naturally preventing the single-voxel-width artifacts that
plague pure mask predictions.

SDF-TopoNet (Wu et al., 2025) formalizes this intuition into a two-stage framework:
Stage 1 pre-trains on SDF prediction (which implicitly encodes topological structure
via level-set connectivity), Stage 2 fine-tunes with a dynamic topology-aware loss
and adapter. The authors demonstrate that SDF pre-training alone outperforms direct
topology loss training on DRIVE, CREMI, Roads, and Elegans datasets — both in
topological accuracy and pixel-level metrics — while **reducing computational cost**
relative to persistent homology losses.

The MinIVess codebase already has a CentrelineHeadAdapter that predicts centreline
distance maps as an auxiliary task, but this head has never been jointly trained with
SDF regression or integrated into the evaluation pipeline.

Li et al. (2026) provide the theoretical foundation: their width-aware persistent
homology framework proves that standard PH allows topologically correct but
anatomically meaningless single-pixel connections. By smoothing critical points via
morphological gradients (PDE-based), they guarantee connectivity, correct genus counts,
**and** minimum vessel width simultaneously — the first formal treatment of width
constraints in topological segmentation. This mathematical result directly motivates
SDF prediction as a training target: the signed distance field naturally encodes width
information that pure masks and even centerline maps lack.

**Gap at the intersection:** To our knowledge, no published work combines DDT-style multi-task learning
(mask + SDF + centerline) with the Li et al. (2026) width-aware topological guarantees.
DDT uses distance transforms for refinement but not for topological constraints; Li et
al. prove width guarantees but test only on 2D datasets (ISICDM bladder MRI, ISBI
neuron EM, Massachusetts Roads) with UNet++, DeepLabV3+, and SegFormer. A 3D
multi-task architecture that predicts mask + SDF + centerline distance map, supervised
with width-aware topological constraints, would be genuinely novel.

**Pros:**
- Strongest empirical evidence (DDT: +13% DSC over 3D-UNet on pancreatic ducts; SDF-TopoNet: outperforms PH losses)
- Architecturally simple (add prediction heads, no new backbone)
- Compatible with existing DynUNet and cbdice_cldice loss
- SDF and centerline predictions are independently useful (morphometry, vessel radius estimation)
- Li et al. (2026) provide formal guarantees

**Cons:**
- Multi-task balancing adds hyperparameters (task weights)
- SDF computation for 3D volumes at training time has non-trivial cost
- Li et al. width-aware PH tested only on 2D datasets (bladder MRI, neuron EM, roads)
- No evidence of SDF-based approaches on two-photon microscopy data

**Feasibility on 8 GB VRAM:** High. Adding 1–2 prediction heads increases memory by ~10–15%. DDT operates on the same feature maps already computed by the encoder-decoder.

---

### Hypothesis 2: Continuous Centerline Representation (Graph-First Segmentation)

**Thesis:** Instead of predicting a binary mask and extracting topology post-hoc
(skeletonize → graph), directly predict centerline coordinates and connectivity as the
primary output. This inverts the standard pipeline and eliminates the information
bottleneck of binary discretization.

**Supporting Evidence:**

DeformCL (Zhao et al., 2025) demonstrates that learning deformable centerline points
as a continuous representation achieves +3–8% clDice improvement over nnU-Net and DynUNet
on four 3D vessel segmentation datasets (coronary, cerebral, hepatic, pulmonary). The
method uses a cascaded pipeline: (1) coarse segmentation, (2) centerline point
detection, (3) deformable refinement via learned offsets, (4) edge connectivity
prediction. The key insight is that **centerlines are inherently connected by edges**,
unlike binary masks that may fragment — topology preservation becomes a structural
property of the representation rather than a learned constraint.

FlowAxis (Wu et al., 2026, npj Digital Medicine) extends this paradigm with Adaptive
Vessel Axes (AVA): adaptive keypoints functioning as interconnected vertices that
encapsulate intrinsic spatial topologies. The displacement convexity property of the
energy functional provides formal topological coherence guarantees — the first such
guarantee for continuous vessel parameterization.

The SE(3)-BBSCformerGCN framework (Wang et al., 2025, NeurIPS) provides the
mathematical machinery for operating on continuous curve representations: Ball B-Spline
Curves (BBSC) with SE(3)-equivariant attention. While tested on branch classification
rather than segmentation, the BBSC manifold achieves **266x FLOPs reduction** over
point cloud methods with superior accuracy (AUC-ROC 99.99% on TopCoW).

SLAVV (Mihelic et al., 2021) demonstrates that segmentation-less vectorization —
going directly from raw 2PM images to vector representations with connectivity — is
viable for the exact imaging modality used in MiniVess. SLAVV uses multi-scale
Laplacian-of-Gaussian filtering without any neural network, yet produces robust
morphometric statistics from large volumes (1.6 x 10^8 voxels).

**Gap at the intersection:** DeformCL and FlowAxis both assume a coarse segmentation as
input — they are refinement methods, not end-to-end solutions. No published work
combines end-to-end centerline prediction with DynUNet-quality coarse segmentation in
a unified training loop. Furthermore, neither DeformCL nor FlowAxis has been tested
on microvasculature (capillary networks have much higher branching density and more
uniform caliber than coronary or hepatic vessels), and neither uses topology-aware
losses (clDice) during training — an obvious combination.

**Pros:**
- Topology preservation is structural, not just supervised
- Strong empirical gains on 3D vessel benchmarks (+3–8% clDice)
- Natural output for downstream analysis (morphometry, hemodynamics, graph metrics)
- SLAVV validates the paradigm on two-photon microscopy

**Cons:**
- Cascaded pipeline adds training complexity
- Point-to-voxel reconstruction needed for traditional metrics (Dice, NSD)
- No evidence on datasets with >10,000 branches (microvascular density challenge)
- Deformable point learning may struggle with densely packed parallel capillaries

**Feasibility on 8 GB VRAM:** Medium. DeformCL's cascaded design allows stage-wise
training. The point detection head is lightweight. The main cost is the coarse
segmentation stage (same as current DynUNet).

---

### Hypothesis 3: Foundation Model Adaptation with Topology Priors

**Thesis:** Pre-trained foundation models (vesselFM, SAM3) provide rich feature
representations that, when combined with topology-aware adaptation (LoRA + clDice),
outperform training-from-scratch approaches — especially in the low-data regime
(MiniVess: 70 volumes).

**Supporting Evidence:**

vesselFM (Wittmann et al., 2025, CVPR) is the first foundation model designed
specifically for 3D blood vessel segmentation. Trained on heterogeneous data (curated
annotations + domain randomization + flow matching generative model), vesselFM
outperforms all medical image segmentation foundation models across four imaging
modalities in zero-, one-, and few-shot scenarios. Critically, vesselFM uses a MONAI
UNet architecture — directly compatible with the MinIVess pipeline. However, vesselFM
has never been compared against nnUNet/DynUNet and has never been evaluated on
two-photon microscopy data.

The empirical evidence on foundation models for vascular segmentation is
contradictory, creating what we term the **Foundation-Topology Paradox**: vesselFM
achieves SOTA without any topology-aware training, yet the benefits of topology losses
on strong baselines are marginal (TopoMortar). Do pretrained features already encode
topology? The answer likely differs by structural complexity.

TopoLoRA-SAM (Khazem, 2026) provides the adaptation blueprint: LoRA (rank 16) +
lightweight spatial adapter + topology-aware loss (L_BCE + L_Dice + 0.5*L_clDice) adapts
SAM ViT-B to thin structures with only 4.9M trainable parameters (5.2% of total).
On the rank ablation study (DRIVE), clDice reaches 0.678 at rank 16; main benchmark
Dice scores are 0.690 (DRIVE) and 0.569 (CHASE_DB1). The ablation reveals that LoRA is
the primary driver (+4.2 Dice, +5.1 clDice over decoder-only fine-tuning), while clDice
adds a modest but consistent +0.8 clDice. However, the Taheri et al. (2025) multi-encoder
nnU-Net study directly challenges the premise: **well-configured U-Net architectures
outperform transformer models even with large-scale SSL pretraining** (93.72% DSC vs.
Swin UNETR on BraTS 2021), and Li et al. (2025, Medical Physics — nnSAM) found "SAM/MedSAM
did not outperform standard UNet even in extremely limited data settings." These results
suggest the paradox resolves differently for macrovascular structures (where pretrained
features suffice) vs. microvasculature (where topology is too fine-grained for generic
pretraining).

The SAM3 literature research report (MinIVess internal, 2026) concluded that **vesselFM
is the primary foundation model candidate** (not SAM3), with SAM3 reframed as
complementary (annotation accelerator, ensemble member). SAM's fundamental geometric
mismatch with tubular structures is proven architectural (Zhang et al., 2024).

**Gap at the intersection:** No published work combines vesselFM's vessel-specific
pretraining with topology-aware PEFT (LoRA + clDice) for 3D microvasculature. vesselFM
was trained with standard losses; TopoLoRA-SAM was applied to SAM (2D only). A
vesselFM + TopoLoRA adaptation with the MinIVess cbdice_cldice compound loss on
two-photon microscopy data would test whether foundation model features + topology
supervision synergize or merely duplicate information already captured by the
pretrained features.

**Pros:**
- Leverages pretrained vessel-specific features (vesselFM covers 4 modalities)
- Parameter-efficient (TopoLoRA: 5.2% of parameters)
- Strong theoretical motivation for low-data regime (70 volumes)
- vesselFM uses MONAI UNet — architecture compatibility

**Cons:**
- vesselFM has never been tested on two-photon microscopy
- Foundation model advantages diminish with sufficient task-specific data
- SAM's architectural limitation for tubular structures is proven fundamental
- vesselFM weights may be too large for 8 GB VRAM without model parallelism
- Unclear whether pretrained features already encode topology (making topology losses redundant)

**Feasibility on 8 GB VRAM:** Medium-Low. vesselFM's MONAI UNet is likely feasible
with gradient checkpointing; SAM3-Base is infeasible (demonstrated in SAM3 report).
LoRA adaptation dramatically reduces trainable parameters but full forward pass still
requires full model memory.

---

### Hypothesis 4: Topology-Preserving Test-Time Adaptation and Post-Processing

**Thesis:** Rather than improving the segmentation model itself, correct topological
errors at inference time through test-time adaptation (TTA), conformal prediction, and
learned post-processing — a strategy that is model-agnostic and complementary to any
architecture/loss improvements.

**Supporting Evidence:**

TopoTTA (Zhou et al., 2025) is the first TTA framework designed specifically for
tubular structures. Its two-stage approach — Topological Meta Difference Convolutions
(TopoMDCs) that enhance topological representations without modifying pretrained
weights, followed by Topology Hard sample Generation (TopoHG) that simulates break
points and forces continuity — achieves +2–5% clDice and +1–3% Dice improvement over
vanilla TTA across 4 scenarios and 10 datasets. The TopoHG strategy of deliberately
creating pseudo-breaks is also directly usable as a training-time data augmentation.

ConformalRefiner (Pang et al., 2025) applies conformal risk control to vessel topology
reconstruction: given a segmentation with broken connections, it adaptively thresholds
the prediction to guarantee topological correctness with a user-specified confidence
level. TUNE++ (Dhor et al., 2025) extends this to 3D, providing topology-guided
uncertainty estimates that identify topologically critical regions.

Valverde et al. (2025, "Disconnect to Connect") demonstrate a surprisingly effective
data augmentation: deliberately disconnecting connected components in training labels,
then training the model to reconnect them, improves topology accuracy across multiple
datasets. This is computationally free at inference time.

The Curvi-Tracker (Heng, 2025 PhD thesis) deploys learned agent-based tracking on
foreground pixels to repair segmentation gaps, using Direction-Net and Forward-Net that
mimic human annotation behavior. This represents a fundamentally different paradigm
from voxel-level post-processing: an agent that follows vessels and repairs breaks.

The MinIVess codebase has **no topology-correcting post-processing**. The system
detects topology errors (Betti numbers, junction F1, connectivity components) but
cannot fix them. However, the conformal UQ system (69 tests, 5 implementation phases)
provides substantial infrastructure to build on: MorphologicalConformalPredictor
(ConSeMa-inspired dilation/erosion bands — directly related to ConformalRefiner's
approach), DistanceTransformConformalPredictor (CLS-inspired FNR control via EDT —
related to TUNE++'s distance-based topology guidance), and RiskControllingPredictor
(LTT framework — the statistical machinery for bounded risk). What is missing is the
**graph-level extension**: applying these conformal predictors not to individual voxels
but to graph edges (vessel connections), branches (vessel segments), and topological
features (Betti numbers).

**Gap at the intersection:** No published work combines conformal risk control with
learned reconnection agents. ConformalRefiner provides statistical guarantees on
thresholding but uses a simple global operation; Curvi-Tracker provides intelligent
local repair but has no coverage guarantees. A conformal-guided reconnection agent —
where conformal prediction identifies candidate break points with calibrated confidence,
and a learned tracker attempts reconnection with bounded risk — would provide both
statistical rigor and topological intelligence. Furthermore, none of these methods have
been tested on 3D microvasculature.

**Pros:**
- Model-agnostic — improves any baseline without retraining
- TTA methods require no labeled data at test time
- Conformal methods provide statistical coverage guarantees
- Data augmentation approaches ("Disconnect to Connect") are free at inference
- Multiple complementary mechanisms (adaptation + augmentation + post-processing)

**Cons:**
- Post-processing adds inference latency
- TTA requires multiple forward passes
- Conformal calibration requires held-out calibration set (from 70 volumes)
- No evidence of these approaches on 3D microvasculature
- Agent-based trackers (Curvi-Tracker) are 2D only

**Feasibility on 8 GB VRAM:** High. TTA is a single model with multiple augmented
views. Post-processing operates on CPU. Augmentation is training-time only.

---

### Hypothesis 5: Graph Neural Network Architectures for End-to-End Topology

**Thesis:** Incorporating graph neural networks (GNNs) into the segmentation
architecture — either as feature refinement modules or as primary decoders — enables
the network to reason about connectivity patterns that are invisible to convolutional
operations on regular grids.

**Supporting Evidence:**

ViG3D-UNet (Liu et al., 2025, IEEE JBHI) demonstrates that a dual-branch architecture
— convolutional encoder for local features + 3D vision GNN for volumetric connectivity —
surpasses competing methods on coronary artery segmentation (ASOCA, ImageCAS datasets)
in both segmentation accuracy and vascular connectivity. The ViG3D module automatically
constructs vascular graphs during training and aggregates spatial graph features within
an end-to-end framework.

The MinIVess codebase already includes the TFFMBlock3D (adapted from Ahmed et al., WACV
2026) — a topology feature fusion module that constructs kNN graphs from
adaptively-pooled 3D features, processes them with 2-layer GAT, and upsample back to
volume space. The TFFMWrapper applies this via forward hooks on the bottleneck layer.
However, the TFFM module has never been empirically validated — no training runs exist
with TFFMWrapper enabled.

The SE(3)-BBSCformerGCN (Wang et al., 2025, NeurIPS) provides the formal framework
for GNNs on continuous tubular manifolds. While applied to classification, the
manifold construction pipeline (BBSC + SE(3) attention) operates at 266x fewer FLOPs
than point cloud methods, suggesting practical feasibility for integration into
segmentation pipelines.

The MinIVess vessel graph pipeline (`vessel_graph.py`) already extracts NetworkX graphs
from segmentation masks via `skan.Skeleton`, providing the infrastructure to evaluate
graph-level predictions. This means GNN-based architectures can be validated against
existing graph metrics (APLS, BDR, Junction F1) without building new evaluation tools.

**Gap at the intersection:** ViG3D-UNet has only been tested on macrovascular structures
(coronary arteries with 10s of branches). Microvascular networks (1000s of branches,
uniform caliber, dense packing) present fundamentally different graph properties. The
TFFMBlock3D in MinIVess is ready for validation but has never been tested. No published
work combines graph-based feature refinement with topology-aware losses (clDice)
and multi-task prediction (mask + SDF). A ViG3D-style dual-branch architecture
with cbdice_cldice loss on MiniVess would test whether GNN-based connectivity
reasoning provides additional value atop strong topology supervision.

**Pros:**
- Explicit connectivity reasoning (GNNs operate on graph structure)
- TFFMBlock3D already implemented in MinIVess — ready for testing
- Complementary to all other hypotheses
- Theoretically elegant (topology from architecture, not just loss)

**Cons:**
- GNN on 3D volumes requires graph construction (computational overhead)
- kNN graph construction in feature space may not correspond to anatomical connectivity
- Scalability to microvascular density (1000s of branches) untested
- Multiple moving parts (graph construction + GNN + fusion)

**Feasibility on 8 GB VRAM:** Medium. TFFMBlock3D processes downsampled features
(bottleneck level), so memory overhead is modest. ViG3D dual-branch doubles encoder
cost, which may be prohibitive without modifications.

---

## 3. Cross-Cutting Themes

### 3.1 The clDice consensus

Across four independent evidence sources — TopoMortar (Valverde et al., 2025),
TopoLoRA-SAM (Khazem, 2026), the MinIVess 100-epoch experiments, and Arora et al.
(2025, "Does the Skeleton-Recall Loss Really Work?") — **clDice emerges as the most
consistently beneficial topology loss**. Arora et al. specifically challenged the
skeleton recall loss (Kirchhoff et al., 2024) by demonstrating that its reported gains
are partially attributable to training protocol confounds rather than genuine topology
improvement — a finding consistent with TopoMortar's ranking (clDice > SkelRecall).
The consensus parameters are: clDice weight ~0.5 relative to overlap loss, with
excessive weighting (lambda >= 2.0) degrading both Dice and clDice (TopoLoRA-SAM
ablation). The recently published Skea-Topo (Liu et al., 2026, Pattern Recognition)
proposes a skeleton-aware loss function targeting boundary topology that may complement
clDice, but has not yet been evaluated under controlled TopoMortar-style conditions.

However, clDice operates on a smoothed differentiable approximation of skeletonization,
not true topology — it cannot distinguish Betti-0 errors (fragmentation) from
Betti-1 errors (spurious loops), and Discrete Morse theory approaches (Gupta and Essa,
IJCV 2025; represented in MinIVess by the `toposeg` loss proxy) offer a complementary
perspective by localizing *where* topological changes occur rather than *when* they
appear on a filtration. These limitations motivate Hypothesis 1 (SDF provides richer
geometric signal) and Hypothesis 2 (centerlines provide explicit connectivity).

### 3.2 The representation hierarchy

The literature reveals a clear hierarchy of representational power for vascular
structures:

1. **Binary mask** (standard) — loses all geometric and topological information except
   occupancy. Most published segmentation methods operate here.
2. **Mask + distance field** (DDT, SDF-TopoNet) — encodes geometric information
   (vessel width, distance from boundary/centerline). Recoverable from mask via EDT,
   but predicting it directly forces feature learning.
3. **Mask + centerline map** (CentrelineHeadAdapter) — encodes skeleton proximity.
   The existing MinIVess auxiliary head.
4. **Continuous centerline points** (DeformCL, FlowAxis) — inherently connected,
   resolution-independent. Topology is a structural property.
5. **Graph with attributes** (ViG3D-UNet, SE(3)-BBSCformerGCN) — full topological
   structure with geometric attributes (radius, length, branching angles). The
   richest representation.

Each level subsumes the information of previous levels: a graph (level 5) can
reconstruct a continuous centerline (level 4), which can generate an SDF (level 2),
which can produce a binary mask (level 1). The reverse is lossy — extracting a graph
from a binary mask via skeletonization introduces discretization artifacts and is
sensitive to noise. This asymmetry explains why representation-level advances
(predicting at a higher level) outperform loss-level advances (penalizing errors at
a lower level): the network is forced to learn richer features for higher-level
prediction, which indirectly improves lower-level outputs.

The MinIVess pipeline currently operates at levels 1–3 (mask prediction with optional
centerline head). Advancing to level 4–5 requires co-evolving the metric
infrastructure: graph-level representations demand graph-level metrics (APLS, Junction
F1, BDR — already implemented in MinIVess but not wired into training loops). This
metric-method co-design requirement is often overlooked: papers that predict at level 4–5
but evaluate only at level 1 (Dice) miss the point of their own contribution.

### 3.3 Topology preservation across the pipeline

An emerging theme is that topology can be addressed at **every stage** of the pipeline,
not just the training loss:

| Stage | Method | Example | MinIVess Status |
|-------|--------|---------|----------------|
| **Preprocessing** | Topology-aware enhancement | VAOT (Dong et al., 2025) | Not implemented |
| **Data augmentation** | Disconnect-to-Connect | Valverde et al. (2025) | Not implemented |
| **Architecture** | Graph feature fusion | ViG3D, TFFMBlock3D | Implemented, not tested |
| **Representation** | Multi-task (mask + SDF + centerline) | DDT, SDF-TopoNet | Partially (centerline head) |
| **Loss function** | Topology-aware supervision | clDice, cbDice, skeleton recall, PH | 18 losses implemented |
| **Test-time adaptation** | Topology-preserving TTA | TopoTTA (Zhou et al., 2025) | Not implemented |
| **Post-processing** | Conformal refinement | ConformalRefiner, Topograph | Not implemented |
| **Evaluation** | Topology-aware metrics | clDice, BE, Junction F1, APLS | Implemented, not all wired in |

The MinIVess pipeline has invested heavily in loss functions and metrics but has gaps
at preprocessing, augmentation, TTA, and post-processing — exactly the stages that
are model-agnostic and provide "free" improvements.

### 3.4 The two-photon microscopy specificity

A critical observation: of the 60+ papers surveyed, only **three** operate on the same
imaging modality as MiniVess:
1. Haft-Javaherian et al. (2020) — PH-based CNN for 3D multiphoton brain vasculature
2. Mihelic et al. (2021) — SLAVV vectorization from 2PM images
3. Teikari et al. (2016) — early deep learning for multiphoton vasculature

All other methods were developed for CT/MR angiography (macrovascular), retinal fundus
(2D), or general tubular structures (roads, cracks). The domain gap is quantifiable:
coronary arteries have ~10–50 branches per tree with 2–5 mm caliber variation, while
MiniVess volumes contain ~1,000+ capillary branches with uniform ~5 um diameter. This
100x branching density and 1000x caliber uniformity means that methods designed for
macrovascular topology (DeformCL, ViG3D-UNet, FlowAxis) face fundamentally different
graph properties — and may fail silently on metrics that do not capture branch-level
topology (standard Dice is blind to this).

Of the three 2PM-specific papers, Haft-Javaherian et al. (2020) is the most directly
relevant precedent: they applied persistent homology as a loss term for 3D multiphoton
brain vasculature segmentation and achieved 86.5% Dice (vs. 81.6% for DeepVess, 72.7%
for 3D U-Net). Their key design decision — using a deterministic topological prior
(beta_0=1, beta_1=0 within a 21 x 21 um^2 field of view) based on known capillary
network fractal properties — exploits domain knowledge that the MinIVess project should
also leverage. However, their architecture (DeepVess extension with 3D + 2D conv + dense
layers) is outdated, and they did not use clDice (published 2021, after their work).
Their fractal-derived topological prior is directly related to recent work on fractal
feature maps for tubular structure segmentation (Huang et al., 2025).

### 3.5 Training stability and computational cost

A practical reality the field often understates: topology-aware losses are frequently
unstable during training. The MinIVess 18-loss debug sweep (6 epochs x 3 folds, peak
GPU 5767 MB, peak RAM 38.8 GB) revealed:

| Loss | DSC | clDice | Stability |
|------|-----|--------|-----------|
| centerline_ce | 0.700 | — | Stable |
| dice_ce | 0.676 | — | Stable |
| full_topo | — | 0.722 | Stable (compound) |
| cldice | — | 0.720 | Stable |
| **topo** | **0.012** | — | **Near-zero DSC** |
| **betti** | **0.016** | — | **Near-zero DSC** |
| betti_matching | FAILED | — | OOM (680K features → 323 GiB) |

Standalone topology losses collapse to degenerate solutions — a finding consistent with
TopoMortar's observation that topology losses must be combined with overlap losses. The
MinIVess `torch.isfinite(loss)` NaN guard catches gradient explosions that occur with
PH-based losses. This motivates Hypothesis 1 (SDF representation implicitly encodes
topology without unstable gradient paths) and curriculum-based approaches where topology
supervision is introduced gradually after overlap convergence.

Computational cost varies dramatically across methods. From TopoMortar (GPU hours for
full training): clDice 3.1h, SkelRecall 7.1h, cbDice 58.3h, TopoLoss 49.6h, Warping
152.1h. The 50x cost difference between clDice and Warping for marginal or no topology
improvement explains why clDice dominates in practice — **clDice may win not because it
is the best topology loss, but because it is the only one that fits the compute budget
of most researchers.** This is a more nuanced and defensible claim than "clDice is best."

---

## 4. Synthesis: Where the Field Should Go

Applying the Markov mandate for novel synthesis, we identify three previously
unconnected threads that, combined, suggest a research direction absent from any
individual paper:

### 4.1 The Width-Topology-Representation Triangle

Li et al. (2026) prove that topology alone is insufficient — width constraints are
necessary for anatomically meaningful segmentation. DDT/SDF-TopoNet show that distance
field prediction naturally encodes width. DeformCL shows that continuous centerlines
inherently preserve connectivity. **To our knowledge, no published work combines all three: continuous
centerlines with width-aware topology constraints supervised via SDF prediction.** This
synthesis would produce segmentations that are simultaneously (a) topologically correct
(connected where they should be), (b) geometrically accurate (correct vessel caliber),
and (c) resolution-independent (continuous representation).

### 4.2 The Foundation-Topology Paradox

vesselFM achieves SOTA 3D vessel segmentation without any topology-aware training,
while TopoMortar demonstrates that topology losses provide marginal gains on strong
baselines. This creates a paradox: **do foundation model features already encode
topology, making explicit topology supervision redundant?** The answer likely depends
on the structural complexity of the target domain. For macrovascular structures (few
branches, large caliber variation), pretrained features may suffice. For microvasculature
(1000s of uniform-caliber capillaries), the topology is too fine-grained to be captured
by generic pretraining. Testing vesselFM ± cbdice_cldice on MiniVess would directly
resolve this paradox.

### 4.3 The Post-Hoc Correction Gap

Despite substantial progress in topology-aware training, the field has almost entirely
neglected **post-hoc topology correction with statistical guarantees**. ConformalRefiner
(Pang et al., 2025) provides conformal risk control for threshold selection, but
operates globally. Curvi-Tracker (Heng, 2025) provides intelligent local repair, but
has no coverage guarantees. The MinIVess conformal UQ system operates at voxel level
only. The missing piece is a **conformal prediction framework operating at the graph
level** — providing calibrated confidence intervals on edge existence, branch detection
rate, and Betti numbers. This would enable principled decisions about which detected
breaks to repair, with bounded false discovery rate.

### 4.4 Specific Falsifiable Predictions

Following the Markov mandate that reviews should make specific, testable claims rather
than vague "more research is needed" statements, we commit to the following predictions:

**P1.** Multi-task SDF + centerline prediction (Hypothesis 1) will improve clDice by
>3 percentage points over cbdice_cldice alone on MiniVess, because the SDF forces
width-aware feature learning that pure mask supervision cannot provide.

**P2.** vesselFM zero-shot will achieve <0.5 DSC on MiniVess two-photon microscopy data,
because the modality gap (two-photon microscopy is absent from vesselFM's training data)
exceeds the generalization capacity of flow-matching-based domain randomization.

**P3.** Disconnect-to-Connect augmentation will improve clDice by >1 percentage point
with zero additional inference cost, representing the highest impact-per-effort ratio
of all proposed approaches.

**P4.** The TFFMBlock3D (already implemented but untested) will provide <0.5 percentage
point clDice improvement on MiniVess because kNN graphs in feature space do not
correspond to anatomical connectivity at the microvascular scale.

**P5.** Topology-aware *loss functions* (as distinct from representation changes) are
approaching a solved problem for vascular segmentation. The remaining gains from
loss-only modifications (estimated <2 percentage points clDice over cbdice_cldice on
any dataset) are dwarfed by gains from representation changes (P1's >3 points comes
from predicting SDF+centerline, not from a new loss term). We predict that no topology
loss published after 2025 will exceed clDice's rank in a TopoMortar-style controlled
evaluation when representation is held constant (binary mask).

These predictions are designed to be testable within the scope of this branch and will
serve as go/no-go gates for the implementation plan.

---

## 5. Ranked Recommendations

Based on the evidence synthesis, we rank approaches by **expected impact × feasibility
÷ implementation cost**, considering the 8 GB VRAM constraint and the goal of a
publishable contribution:

| Rank | Approach | Novel? | Expected Impact | VRAM OK? | Implementation Cost |
|------|----------|--------|----------------|----------|-------------------|
| **1** | Multi-task: mask + SDF + centerline (H1) | Yes (width-aware 3D combination) | High | Yes | Medium |
| **2** | Topology-preserving data augmentation: Disconnect-to-Connect (H4) | Test on 3D/2PM | Medium | Yes (free) | Low |
| **3** | TFFMBlock3D validation runs (H5) | Test existing code | Medium | Yes | Low |
| **4** | TopoTTA test-time adaptation (H4) | Test on 3D/2PM | Medium | Yes | Medium |
| **5** | vesselFM zero-shot + topology PEFT (H3) | Yes (vesselFM + cbdice_cldice) | Medium-High | Uncertain | Medium |
| **6** | DeformCL centerline representation (H2) | Test on microvasculature | High | Yes | High |
| **7** | Haft-Javaherian PH loss updated (H1+H5) | Update classic approach | Medium | Yes | Medium |
| **8** | ViG3D-style dual-branch (H5) | Test on microvasculature | Medium-High | Borderline | High |
| **9** | Conformal graph-level UQ (H4) | Yes (novel framework) | High (theoretical) | Yes | Very High |

**Recommended implementation order for the PR:**
1. **Immediate (low-hanging fruit):** Disconnect-to-Connect augmentation + TFFMBlock3D validation
2. **Core contribution:** Multi-task mask + SDF + centerline prediction with cbdice_cldice
3. **If resources allow:** vesselFM zero-shot benchmark + TopoTTA adaptation
4. **Future work:** DeformCL on microvasculature, conformal graph-level UQ

---

## 6. Specific Implementation Proposals

### 6.1 Multi-Task SDF-Centerline Prediction (Priority 1)

**Architecture modification:** Add two prediction heads to DynUNet:
- SDF head: 1x1x1 Conv3d → regression (predict signed distance to vessel boundary)
- Centerline distance head: 1x1x1 Conv3d → regression (predict distance to nearest
  centerline voxel)

**Loss function:** L_total = 0.5 * L_cbdice_cldice(mask_pred, mask_gt) + 0.25 * L_SDF(sdf_pred, sdf_gt) + 0.25 * L_centerline(cl_pred, cl_gt)

Where L_SDF and L_centerline use smooth L1 loss.

**GT computation:** Pre-compute SDF via `scipy.ndimage.distance_transform_edt` on both
foreground and background, with sign convention (positive outside, negative inside).
Centerline GT already exists via `compute_centreline_distance_map()`.

### 6.2 Disconnect-to-Connect Augmentation (Priority 1)

**Implementation:** During training, with probability p=0.3:
1. Compute skeleton of ground truth mask
2. Identify junction points (degree >= 3)
3. Randomly remove 1–3 short segments (< 10 voxels) connecting junctions
4. Corrupt the *input image* in the removed segment regions (zero out or add noise)
5. Keep the *ground truth mask intact* — the network must learn to predict the
   full connected vessel from corrupted input with missing evidence

**Expected effect:** Forces network to learn reconnection from partial evidence,
improving topology preservation on test data. This is analogous to masked image
modeling (MAE) but targeted at topologically critical vessel junctions.

### 6.3 vesselFM Benchmark Protocol (Priority 2)

**Zero-shot:** Apply vesselFM directly to MiniVess test volumes, measure all 8 metrics.
**Go/no-go gate:** If zero-shot DSC < 0.10, skip fine-tuning (same as SAM3 report protocol).
**If pass:** Fine-tune with LoRA (rank 16) + cbdice_cldice loss.

---

## 7. Bibliography

### Seed Papers (Read in Full)

- Bardhan, J., Hebbalaguppe, R., & Udupa, A. (2026). Curve Skeletonization in Continuous Domain for Meshes and Point Clouds. WACV 2026.
- Brito-Pacheco, D., Giannopoulos, P., & Reyes-Aldasoro, C.C. (2025). Persistent Homology in Medical Image Processing: A Literature Review. medRxiv.
- Chen, Y., Huang, G., Zhang, S., & Dai, J. (2025). Dynamic Snake Upsampling Operator and Boundary-Skeleton Weighted Loss for Tubular Structure Segmentation.
- Dong, X., et al. (2025). VAOT: Vessel-Aware Optimal Transport for Retinal Fundus Enhancement. arXiv:2511.18763.
- Guo, C., et al. (2025). SA-UNetv2: Rethinking Spatial Attention U-Net for Retinal Vessel Segmentation. arXiv:2509.11774.
- Guzzi, L., et al. (2025). Regional Hausdorff Distance Losses for Medical Image Segmentation. MLMI (MICCAI Workshop).
- Haft-Javaherian, M., et al. (2020). A Topological Encoding CNN for Segmentation of 3D Multiphoton Images of Brain Vasculature Using Persistent Homology.
- Heng, Z. (2025). Automatic Curvilinear Structure Extraction from Images. PhD Thesis, UNSW.
- Hu, X., et al. (2025). Learning to Upscale 3D Segmentations in Neuroimaging. arXiv:2505.21697.
- Khazem, S. (2026). TopoLoRA-SAM: Topology-Aware Parameter-Efficient Adaptation of Foundation Segmenters. arXiv:2601.02273.
- Li, W., Tai, X.-C., & Liu, J. (2026). Topology-Guaranteed Image Segmentation: Enforcing Connectivity, Genus, and Width Constraints. arXiv:2601.11409.
- Martin, N., Chevallet, J.-P., & Mulhem, P. (2025). From Prediction to Prompt: Leveraging nnU-Net Outputs to Guide SAM for Active Learning in 3D Dental Segmentation.
- Mihelic, S.A., et al. (2021). Segmentation-less, automated vascular vectorization robustly extracts neurovascular network statistics from in vivo two-photon images. bioRxiv.
- Podobnik, G. & Vrtovec, T. (2025). MeshMetrics: A Precise Implementation of Distance-Based Image Segmentation Metrics.
- Sekhavat, S., Jamshidian, M., Wittek, A., & Miller, K. (2025). Impact of Geometric Uncertainty on the Computation of Abdominal Aortic Aneurysm Wall Strain.
- Taheri Otaghsara, S.S. & Rahmanzadeh, R. (2025). Multi-encoder nnU-Net Outperforms Transformer Models with Self-supervised Pretraining. arXiv:2504.03474.
- Taheri Otaghsara, S.S. & Rahmanzadeh, R. (2025). F3-Net: Foundation Model for Full Abnormality Segmentation of Medical Images. arXiv:2507.08460.
- UCAD Authors (2026). Uncertainty-guided Contour-aware Displacement for Semi-supervised Medical Image Segmentation. arXiv:2601.17366.
- Usman, M., et al. (2025). Integrating Uncertainty Quantification into CFD Models of Coronary Arteries Under Steady Flow.
- Valverde, J.M., et al. (2025). TopoMortar: A Dataset to Evaluate Topology Accuracy in Image Segmentation. BMVC 2025. arXiv:2503.03365.
- Wang, J., et al. (2025). Topology-Aware Learning of Tubular Manifolds via SE(3)-Equivariant Network on Ball B-Spline Curve. NeurIPS 2025.
- Wang, Y., et al. (2020). Deep Distance Transform for Tubular Structure Segmentation in CT Scans. CVPR 2020 (arXiv 2019).
- Zhao, Z., et al. (2025). DeformCL: Learning Deformable Centerline Representation for Vessel Extraction in 3D Medical Image.
- Zhou, J., et al. (2025). TopoTTA: Topology-Enhanced Test-Time Adaptation for Tubular Structure Segmentation.
- Zhou, Y., et al. (2025). nnWNet: Rethinking the Use of Transformers in Biomedical Image Segmentation.

### Key References (Cited in Body)

- Arora, H., et al. (2025). Does the Skeleton-Recall Loss Really Work? arXiv.
- Berger, C., et al. (2026). Pitfalls of Topology-Aware Image Segmentation.
- Dhor, A., et al. (2025). TUNE++: Topology-Guided Uncertainty Estimation for Reliable 3D Medical Image Segmentation.
- Kirchhoff, S., et al. (2024). Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures. ECCV 2024.
- Liu, Z., et al. (2025). ViG3D-UNet: Volumetric Vascular Connectivity-Aware Segmentation via 3D Vision Graph Representation. IEEE JBHI.
- Liu, Z., et al. (2026). Skea-Topo: A skeleton-aware loss function for topologically accurate boundary segmentation. Pattern Recognition.
- Maier-Hein, L., et al. (2024). Metrics reloaded: recommendations for image analysis validation. Nature Methods.
- Pang, Y., et al. (2025). ConformalRefiner: Retinal Vessel Topology Reconstruction via Conformal Risk Control.
- Poon, C., et al. (2023). A dataset of rodent cerebrovasculature from in vivo multiphoton fluorescence microscopy imaging. Scientific Data.
- Shi, P., et al. (2024). Centerline Boundary Dice Loss for Vascular Segmentation. arXiv.
- Shit, S., et al. (2021). clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. CVPR 2021.
- Stucki, N., et al. (2023). Topologically Faithful Image Segmentation via Induced Matching of Persistence Barcodes. ICML 2023.
- Valverde, J.M., et al. (2025). Disconnect to Connect: A Data Augmentation Method for Improving Topology Accuracy. arXiv.
- Wittmann, B., et al. (2025). vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation. CVPR 2025.
- Wu, J., et al. (2025). SDF-TopoNet: A Two-Stage Framework for Tubular Structure Segmentation via SDF Pre-training and Topology-Aware Fine-Tuning. arXiv:2503.14523.
- Wu, J., et al. (2026). Geometric-topological deep transfer learning for precise vessel segmentation in 3D medical volumes (FlowAxis). npj Digital Medicine.

### Additional References

- Ahmed, S., et al. (2026). Topology Feature Fusion Module for 3D Volumes. WACV 2026.
- Decroocq, M., et al. (2025). Benchmarking Evaluation Metrics for Tubular Structure Segmentation in Biomedical Images.
- Gupta, S. & Essa, I. (2025). TopoSegNet: Topology-Aware Segmentation via Discrete Morse Theory. IJCV.
- Hu, X., et al. (2019). Topology-Preserving Deep Image Segmentation. NeurIPS 2019.
- Hu, X. (2022). Structure-Aware Image Segmentation with Homotopy Warping. NeurIPS 2022.
- Huang, Y., et al. (2025). Representing Topological Self-similarity Using Fractal Feature Maps for Accurate Segmentation of Tubular Structures.
- Isensee, F., et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods.
- Li, Y., et al. (2025). Plug-and-play nnSAM: A unified medical image segmentation framework. Medical Physics.
- Lin, Y., et al. (2024). VPBSD: Vessel-Pattern-Based Semi-Supervised Distillation for Efficient 3D Microscopic Cerebrovascular Segmentation. arXiv.
- Reinke, A., et al. (2024). Understanding metric-related pitfalls in image analysis validation.
- Teikari, P., et al. (2016). Deep Learning Convolutional Networks for Multiphoton Microscopy Vasculature Segmentation.
- Zhang, Z., et al. (2024). Understanding SAM's Limitations on Tree-Like Structures. (SAM geometric mismatch proof).
