---
title: "SAM3 for 3D Microvasculature: Multi-Hypothesis Research"
status: reference
created: ""
---

# SAM3 for 3D Microvasculature: A Multi-Hypothesis Research Exploration

**Status**: Living document — multi-hypothesis open-ended exploration
**Scope**: SAM family (SAM→SAM2→SAM3) for 3D vascular segmentation with topology preservation
**Context**: MiniVess MLOps platform (70 volumes, 512x512xZ, Z=5-110, 4 compound losses, 3-fold CV)
**Date**: 2026-03-02

---

## 1. The Geometry-Architecture Mismatch

Three simultaneous developments create an underexplored research frontier. First, Meta's
SAM family has evolved from spatial prompting (SAM) through video propagation (SAM2) to
concept-based segmentation (SAM3), each generation privileging different object geometries.
Second, topology-preserving losses (clDice, CAPE, skeleton recall, Betti matching) have
matured to the point where dedicated 3D networks like DynUNet achieve 0.906 clDice on
microvasculature — but only with task-specific training. Third, conformal prediction methods
for segmentation (morphological prediction sets, distance-transform calibration) now offer
coverage guarantees that no foundation model natively provides. No existing work examines
what happens when these three threads converge on the hardest segmentation geometry: thin,
branching, topologically complex 3D vessel trees.

We argue that the central question is not "Can SAM3 segment vessels?" — it demonstrably
cannot, out of the box — but rather **"Which SAM3 component (perception encoder, concept
detector, or mask tracker) best complements existing topology-aware pipelines, and what
is the minimum adaptation cost?"** This reframing shifts the conversation from replacement
to augmentation, from zero-shot evaluation to architecture-informed integration.

### 1.1 The Prompt-Topology Mismatch

SAM was designed around a deceptively simple assumption: objects are compact regions that
can be specified by points or bounding boxes. This assumption holds for everyday objects
(people, cars, animals) but fails catastrophically for tubular structures. A bounding box
around a vessel tree encloses mostly background. A single point prompt captures at most
one branch. Even multiple random points within a vessel mask exhibit spatial clustering
due to non-convexity — a failure mode that VesSAM (Kang et al. 2025) explicitly identifies
and addresses through structure-aware multi-prompting with skeleton, bifurcation, and
segment midpoint prompts.

The prompt-topology mismatch is not a fine-tuning problem. It is an architectural assumption
baked into the prompt encoder, and it explains why every SAM2 medical adaptation — MedSAM2
(Zhu & Ma et al. 2025), BioSAM-2 (Zhao et al. 2024), Ma et al. (2024) — reports
devastating vessel results. The aorta, one of the largest and simplest tubular structures
in the body, reaches only 0.64 DSC after fine-tuning (Ma et al. 2024). The inferior vena
cava manages 0.35 DSC. MedSAM2 explicitly acknowledges: "this approach inherently limits
its ability to segment highly complex anatomical structures, such as vessels with thin and
branching structures" (Zhu & Ma et al. 2025). For MiniVess microvasculature — with vessel
diameters of 5-50 um vs. the aorta's 20-30 mm (a ~1000x size difference), far more
branching, and lower contrast in two-photon microscopy — these numbers predict very poor
performance, likely below 0.20 DSC.

### 1.2 SAM3 Is Not What You Think

A widespread misconception must be corrected: SAM3 (Ravi et al. 2025) is **not** SAM2
with a bigger backbone. It is a fundamentally different architecture. The SAM2-to-SAM3 Gap
paper (Ravi et al. 2025) identifies five critical architectural divergences:

| Dimension | SAM2 | SAM3 |
|-----------|------|------|
| Paradigm | Spatial prompt segmentation | Concept-driven segmentation |
| Architecture | Vision-temporal encoder-decoder | Dual: DETR detector + SAM2 tracker sharing a Perception Encoder |
| Language | None | Large text encoder (LLaMA/Qwen) with cross-attention fusion |
| Detection | Monolithic | Decoupled: global presence head (what) + localization (where) |
| Training | Video mask tracking | Joint: segmentation + contrastive concept alignment |

The presence token innovation is particularly relevant for vessels. SAM3's detector first
determines *whether* a target concept exists before localizing instances — decoupling
recognition from localization. For vessel segmentation, this means SAM3 can reason about
"vessel-ness" as a concept (via text or image exemplar prompts) before attempting spatial
delineation. This is architecturally superior to SAM2's approach of propagating a spatial
mask through memory, which fundamentally cannot encode what a vessel *is*, only where it
*was* in the previous slice.

Crucially, SAM2 prompt engineering expertise does not transfer to SAM3. SAM3-Adapter
(2025) demonstrates this empirically, achieving new state-of-the-art on medical imaging
by designing adapter modules specifically for SAM3's dual encoder-decoder architecture.

---

## 2. The Memory Bottleneck Is Structural, Not Parametric

### 2.1 Eight Frames Is Not Enough

Every SAM2-based 3D medical segmentation approach treats volumes as video sequences,
propagating masks slice-by-slice from a prompted middle frame. SAM2's memory mechanism uses a configurable memory bank with limited temporal
context. Training uses short video clips (typically 6-8 frames), creating an effective
context window beyond which memory attention degrades significantly.

Huang et al. (2024) provide the cleanest evidence. Even with oracle ground-truth masks
as prompts, SAM2 propagation caps at ~0.56 IoU for 3D medical volumes. This is not a
fine-tuning limitation — it is a ceiling imposed by the memory mechanism itself. For
MiniVess volumes with Z-ranges of 5-110 slices, only the thinnest volumes (Z=5-8) fit
within the effective memory window.

The 2D-to-3D performance gap quantifies the damage: CT segmentation drops from 0.91 to
0.72 DSC, MRI from 0.89 to 0.68 (Ma et al. 2024). For PET imaging, SAM2 actually
*underperforms* SAM1, suggesting the video memory mechanism actively hurts when inter-slice
appearance changes are large — precisely the situation in multi-photon microscopy of
microvasculature.

### 2.2 Why Slice-by-Slice Fails for Vessels Specifically

The memory bottleneck is worse for vessels than for organs because vessels violate the
temporal locality assumption. An organ's cross-section changes gradually between adjacent
slices. A vessel's cross-section can bifurcate, disappear, or change orientation abruptly.
Memory-conditioned features that assume smooth temporal evolution will consistently lose
track of branching structures.

This explains a paradox in the MedSAM2 results: performance on organs (88% DSC) is
reasonable, while performance on vessels within the same volumes is explicitly acknowledged
as a failure mode. The same model, the same fine-tuning, the same data — but the geometry
class determines success or failure.

### 2.3 Native 3D Alternatives

SAM-Med3D (Wang et al. 2024) sidesteps the memory bottleneck entirely by replacing SAM's
2D ViT encoder with a 3D ViT that processes the entire volume at once. A single prompt
point specifies the target for the whole 3D volume, requiring 10-100x fewer prompts than
slice-by-slice approaches. The approach is trained on SA-Med3D-140K: 22K 3D images with
143K corresponding 3D masks.

However, SAM-Med3D uses SAM's original architecture (no concept understanding, no text
prompts). Extending SAM3's concept-based paradigm to native 3D processing remains unexplored,
and the barrier is substantial: the perception encoder's 2D pre-training creates a feature
space optimized for planar images. 3D adaptation would require either prohibitive volumetric
re-training or 3D adapter layers that project volumetric features into the 2D feature space
without losing inter-slice context — a non-trivial engineering challenge. This gap is worth
pursuing only if vesselFM (Section 7.2) proves insufficient for MiniVess, since vesselFM
already handles 3D natively.

### 2.4 Implication for MiniVess

The memory bottleneck eliminates slice-by-slice SAM2/SAM3 approaches for MiniVess volumes
with Z > 8 (the majority of the dataset, with Z ranging 5-110). Any viable SAM integration
must either: (a) use SAM as a per-slice feature extractor fed into a 3D aggregation backbone
(H1), (b) use SAM for prompt-guided refinement on sub-volumes where DynUNet predictions
are uncertain (H2), or (c) abandon video propagation entirely and use SAM3's concept
understanding as a standalone 2D component (H4 annotation). The slice-by-slice paradigm
is a dead end for 3D vessel segmentation.

---

## 3. VesSAM: The Only Paper That Gets Vessel Prompting Right

VesSAM (Kang et al. 2025) is, to our knowledge, the only published work that designs
SAM prompts specifically for vascular geometry. It introduces three anatomically-informed
prompt types:

1. **Skeleton prompts**: Centerline structures extracted from vessel masks
2. **Bifurcation points**: Locations where vessels branch
3. **Segment midpoints**: Central positions along individual vessel segments

These prompts are fused via hierarchical cross-attention in a multi-prompt encoder.
The results are striking: **+10% Dice and +13% IoU** over PEFT-based SAM variants, with
strong out-of-distribution generalization across 8 datasets and 5 imaging modalities.

### 3.1 Why This Matters for MiniVess

VesSAM validates a hypothesis central to the MiniVess pipeline: **vessel segmentation
requires topology-aware inputs, not just topology-aware losses.** The MiniVess pipeline
currently uses topology-preserving losses (cbdice_cldice achieves 0.906 clDice) during
training, but prompts and inputs are standard 3D patches. VesSAM's success suggests that
feeding skeleton/bifurcation information as auxiliary prompts — derived from the existing
centreline extraction pipeline (`centreline_extraction.py`) — could provide complementary
benefits.

### 3.2 The SAM3 + VesSAM Convergence

No published work combines SAM3's concept prompting with VesSAM's structure-aware
multi-prompting. This is a specific, actionable research gap:

- **SAM3 concept prompt**: "microvasculature" (text) + exemplar image crop showing vessel
- **VesSAM structure prompt**: skeleton + bifurcations + midpoints from initial DynUNet prediction
- **Two-stage inference**: DynUNet produces initial segmentation → extract topology prompts → SAM3 refines using both concept understanding and structural priors

This architecture would combine SAM3's semantic reasoning ("what is a vessel?") with
VesSAM's geometric reasoning ("where does the vessel branch?") — a pairing that neither
system achieves alone.

---

## 4. Adaptation Strategies: A Taxonomy for Vessel Segmentation

### 4.1 The LoRA vs. Full Fine-Tuning Question

The data efficiency literature reveals a surprising non-monotonic relationship between
adaptation strategy and dataset size:

| Dataset Size | Best Strategy | Evidence |
|-------------|--------------|----------|
| 0 (zero-shot) | SAM3 concept prompting | +18.3 AP over T-Rex2 on COCO (SAM3) |
| 1-10 images | LoRA Recycle (CVPR 2025) | +9.8% over fine-tuning for 1-shot |
| 5-50 images | LoRA fine-tuning (freeze encoder) | 97.6% IoU on femur with 5-20 images |
| 50-200 images | LoRA encoder+decoder | SAMora: 79-84% Dice with data fractions |
| 200-800 images | Full LoRA fine-tuning | Onco-Seg: 35 datasets, 98K cases |
| 2-5% of full dataset | Full fine-tuning (all layers) | micro-SAM: best results but longest convergence |

For MiniVess (70 fully annotated 3D volumes), this maps to approximately 1-4 volumes for
LoRA fine-tuning, or the full dataset for comprehensive adaptation. micro-SAM (Archit
et al. 2025) demonstrates that "the majority of improvement is achieved with only 2-5%
of training data" — suggesting even 2-3 MiniVess volumes could produce a competitive
specialist model.

However, a critical counterpoint from Medical Physics (2026): SAM and MedSAM "did not
outperform a standard UNet even in extremely limited data settings, contrary to the
foundation model hypothesis." This challenges the assumption that foundation models
always win with small datasets and suggests that DynUNet with topology losses may remain
superior to any SAM adaptation for MiniVess.

### 4.2 Adapter Architectures: A Comparative Analysis for Vessel Segmentation

Four adapter patterns have emerged for SAM medical adaptation. We compare them on dimensions
specifically relevant to thin-structure segmentation:

| Adapter | Local Feature Recovery | Params Added | Topology Loss Compatible | Tubular Evidence |
|---------|----------------------|-------------|-------------------------|-----------------|
| ConvLoRA (Zhang et al. 2024, ICLR) | **Strong** — conv kernels restore spatial locality | ~2-5% of backbone | Yes (any differentiable loss) | None published |
| SAM2-Adapter (Chen et al. 2024) | Moderate — multi-scale adapters | ~3-8% | Yes | None published |
| SAMed (Zhang et al. 2023) | Weak — standard LoRA, no spatial inductive bias | ~1-3% | Yes | None published |
| SAM3-Adapter (2025) | Unknown — first SAM3 adapter, architecture not yet replicated | ~5-10% | Untested | None published |

The critical observation: **no adapter has been evaluated on tubular structures.** ConvLoRA's
convolutional injection is the most architecturally promising for vessels — convolution
kernels can learn directional edge filters aligned with vessel boundaries, a capability
that pure attention-based LoRA (SAMed) cannot provide. However, this is a hypothesis, not
evidence. The relevant experiment would compare ConvLoRA vs. standard LoRA on vessel
boundary recall (not just DSC), specifically measuring thin-structure preservation via clDice.

SAMed's contribution is orthogonal: it enables SAM to perform semantic (foreground/background)
segmentation, which vessel segmentation requires. SAM natively performs instance segmentation,
which is inappropriate when the entire vessel tree is a single connected structure.

SAM3-Adapter is architecturally the most relevant for any SAM3-based integration because
it is designed for SAM3's dual encoder-decoder (DETR detector + SAM2 tracker). However,
it has no medical vessel results yet, and its parameter overhead is the highest.

**Gap**: No published work combines ConvLoRA's spatial inductive bias with SAMed's semantic
capability for vessel segmentation. A ConvLoRA-SAMed hybrid — convolutional LoRA modules
for spatial locality + semantic segmentation head — would test whether adapter design can
partially compensate for SAM's architectural tube-blindness.

### 4.3 Medical SAM3: The Direct Fine-Tuning Baseline

Medical SAM3 (2025) represents the most straightforward SAM3 medical adaptation: full
fine-tuning on 33 medical datasets with 4x H100 training. Key findings for vessels:

- **Vessel results are 2D only**: DRIVE 55.8%, CHASE_DB1 62.6%, FetoPlac 77.0% Dice
- **No topology metrics**: Only Dice reported, no clDice, no connectivity measures
- **Below specialized SOTA**: 2D retinal vessel methods exceed 80% Dice
- **Slice-by-slice 3D**: No volumetric processing, same memory bottleneck as SAM2

Medical SAM3's vessel numbers set the floor for what SAM3 can achieve with brute-force
fine-tuning. That floor (55-77% Dice, 2D) is well below MiniVess's current DynUNet
performance (0.824 DSC with dice_ce, 0.906 clDice with cbdice_cldice), confirming that
**SAM3 cannot replace the existing pipeline — the question is whether it can augment it.**

---

## 5. Uncertainty Quantification: The Missing Clinical Layer

### 5.1 UncertainSAM

UncertainSAM (Kaiser et al. 2025, ICML) introduces a Bayesian entropy formulation that
jointly decomposes uncertainty into three components:

1. **Aleatoric uncertainty**: Inherent data noise (irreducible)
2. **Epistemic uncertainty**: Model ignorance (reducible with more data)
3. **Task uncertainty**: A novel component capturing SAM's class-agnostic ambiguity

The implementation (USAM) is a lightweight post-hoc method: small MLPs trained on SAM's
internal mask and IoU tokens to predict uncertainty metrics. This is computationally cheap
and model-agnostic — it could be applied to any SAM variant without retraining the base model.

### 5.2 The Conformal Prediction Bridge

ConformalSAM (Chen et al. 2025, ICCV) directly combines conformal prediction with SAM
for semi-supervised segmentation. The framework calibrates foundation model predictions
using target domain labeled data, then filters unreliable pixel labels so only
high-confidence predictions supervise unlabeled data. The conformal prediction component
provides statistical coverage guarantees that UncertainSAM's Bayesian decomposition
cannot offer.

For MiniVess, this creates a direct integration path with the existing conformal UQ
pipeline (morphological prediction sets, distance-transform calibration, risk-controlling
prediction). The existing `MorphologicalConformalPredictor` and
`DistanceTransformConformalPredictor` could wrap any SAM3 output to provide coverage
guarantees on vessel segmentation masks — a combination no published work has explored.

### 5.3 Complementary Error Modes: SAM vs. DynUNet

A critical question for conformal prediction is whether SAM and DynUNet make *different*
errors, which would make their combined conformal sets tighter than either alone.

**DynUNet error mode**: Under-segmentation of thin structures. When DynUNet fails, it
typically misses distal vessel branches — the prediction is a subset of the ground truth.
This manifests as low recall on thin vessels and high Betti_0 error (disconnected fragments).

**SAM error mode**: Over-segmentation of compact regions. When SAM fails on vessels, it
typically expands vessel boundaries to include surrounding tissue — the prediction is a
superset of the ground truth in some regions and misses other regions entirely. This
manifests as low precision and high Betti_1 error (false loops from merged vessels).

**Implication for conformal sets**: If DynUNet under-predicts and SAM over-predicts, the
intersection of their conformal prediction sets could provide tighter bounds than either
alone. The existing `MorphologicalConformalPredictor` (dilation/erosion bands) and
`DistanceTransformConformalPredictor` (distance-based coverage) operate on single-model
outputs. Extending them to multi-model conformal sets — where the prediction region is the
calibrated intersection/union of DynUNet and SAM predictions — would require new conformal
score functions but is theoretically straightforward under split conformal prediction.

### 5.4 The Spatial Exchangeability Challenge

Standard conformal prediction assumes exchangeable calibration samples. For voxel-level
segmentation, this assumption is violated: neighboring voxels are strongly spatially
correlated. A voxel inside a vessel is likely surrounded by other vessel voxels. This
spatial correlation means naive conformal coverage guarantees are optimistic — the true
coverage is lower than the nominal level.

Two mitigations exist in the literature: (1) spatial conformal prediction with thinning
(subsample every k-th voxel to approximate independence), and (2) conformalize at the
object level rather than voxel level (treat each connected component as one sample). For
MiniVess, object-level conformal prediction on vessel segments is more natural and
preserves topology — each vessel branch becomes a calibration unit, and coverage guarantees
apply to "fraction of branches correctly segmented" rather than "fraction of voxels."

### 5.5 The Topology UQ Gap

A specific research gap: **no published work combines topology-aware uncertainty
quantification with foundation model segmentation.** UncertainSAM provides pixel-level
UQ. Conformal prediction provides set-level coverage. But neither captures topological
uncertainty — the probability that a predicted vessel tree has the correct connectivity
structure (number of connected components, branch points, loops).

The MiniVess pipeline already computes Betti number errors (BE_0, BE_1), ccDice
(connected-component Dice), and junction F1 — all topology-level metrics. Extending
conformal prediction to provide coverage guarantees on these topological properties,
rather than pixel-level properties, would be a genuinely novel contribution at the
intersection of conformal inference, topological data analysis, and foundation model
adaptation. Importantly, **this contribution is SAM-independent** — it applies to any
segmentation model and should be pursued regardless of the SAM3 integration decision.
H5 is listed alongside the SAM hypotheses for roadmap convenience, not because it
requires SAM.

---

## 6. The 3D Volumetric Frontier

### 6.1 Current State: Nobody Does Native 3D SAM3

As of March 2026, no published work adapts SAM3 for native 3D volumetric processing.
Every SAM3 medical adaptation (Medical SAM3, MedSAM3, ConceptBank) operates in 2D or
slice-by-slice pseudo-3D. SAM-Med3D (2024) provides native 3D processing but uses SAM's
original architecture without concept understanding.

This gap is not coincidental — it reflects a genuine architectural challenge. SAM3's
perception encoder uses a vision-language backbone pre-trained on large-scale image-text
data (exact corpus size undisclosed; SAM3 trains on SA-1B + SA-V + additional data).
Converting it to
process 3D volumes would require either:

1. **Volumetric re-training** of the perception encoder (prohibitively expensive)
2. **3D adapter layers** that project volumetric features into SAM3's 2D feature space
3. **Hybrid architectures** where a 3D backbone (DynUNet, SegResNet) handles volumetric
   encoding while SAM3 provides concept-level refinement (e.g., per-slice feature extraction
   aggregated by 3D convolutions)

Option 3 is the most practical for MiniVess and aligns with the MedSAM2+nnUNet fusion
paper (2025), which uses MedSAM2 features as auxiliary encodings fused with nnUNet. The
fusion achieved a modest +0.6% DSC improvement on brain vessels at 2x inference cost —
suggesting that SAM features provide marginal but statistically significant complementary
information to dedicated 3D backbones.

### 6.2 micro-SAM: A Cautionary Tale for Tubular Structures

micro-SAM (Archit et al. 2025, Nature Methods) provides the most relevant microscopy
evidence. Their generalist models trained on 8 microscopy datasets "reliably improve
segmentation only for mitochondria, nuclei and other roundish organelles." Extended Data
Fig. 8 shows that neurite and endoplasmic reticulum segmentation — tubular structures
morphologically similar to microvasculature — requires separate specialist models. The
generalist is "detrimental" for tubular morphology due to shape bias toward compact objects.

This is direct evidence that SAM's inductive biases actively harm tubular structure
segmentation when applied naively. Any MiniVess integration must account for this
morphological mismatch through:
- Specialist fine-tuning on vessel-specific data
- Topology-aware prompt generation (VesSAM-style)
- Post-processing with topology-preserving losses
- Or use as feature extractor rather than standalone segmenter

### 6.3 ProtoSAM-3D: Prototype-Based Few-Shot Volumetric Segmentation

ProtoSAM-3D (Shen et al. 2025) operates natively on 3D volumes using mask-level
prototypes for semantic classification. New classes require updating only 0.02M parameters
(a single prototype vector), making it extremely data-efficient. The approach outperforms
nnUNet and SAM-Med3D on multi-organ CT and MRI.

However, ProtoSAM-3D was validated exclusively on organ segmentation — large, convex,
well-separated structures. We predict ProtoSAM-3D will fail on vessel trees because
mask-level prototypes average over spatial structure, collapsing the topology information
that distinguishes a connected vessel tree from disconnected fragments. Specifically, we
expect acceptable DSC (the overlap metric is relatively forgiving of topology errors) but
low ccDice and high Betti number error — the prototype captures "vessel-like voxels" but
not "vessel-like connectivity." This prediction is testable: if ProtoSAM-3D achieves
\>0.7 DSC but <0.5 ccDice on MiniVess, the topology-collapsing hypothesis is confirmed.

---

## 7. The Architectural Limitation Is Fundamental, Not Fine-Tunable

### 7.1 SAM Cannot Segment Tree-Like Structures (By Design)

A critical 2024 study — "Quantifying the Limits of Segmentation Foundation Models" (Zhang
et al., arXiv:2412.04243) — provides the most rigorous evidence that SAM's limitations on
tubular structures are **architectural, not data-related**:

- SAM models struggle with dense, tree-like morphology and low textural contrast
- Performance correlates with interpretable metrics for object tree-likeness (spatial contour
  frequency, global vs. local scale variability)
- **Targeted fine-tuning FAILS to resolve this issue** — indicating a fundamental limitation
- Root cause: SAMs misinterpret local structure as global texture, causing over-segmentation
- The low-resolution mask head uses one-shot upsampling that cannot capture thin, intricate,
  or high-frequency structures accurately

This finding aligns with micro-SAM's explicit acknowledgment that generalist models are
"detrimental" for tubular structures and with MedSAM2's admission that bounding box
propagation "inherently limits its ability to segment vessels."

### 7.2 vesselFM: A Purpose-Built Alternative (CVPR 2025)

vesselFM (Wittmann, Wattenberg, Amiranashvili, Shit & Menze 2025, CVPR) is a foundation
model trained *specifically* for 3D blood vessel segmentation, using three data sources:

1. Curated annotated datasets from 17 vessel datasets (23 dataset classes)
2. Domain randomization for synthetic vessel-like structures (D_drand)
3. Flow matching-based generative model for synthetic training data (D_flow)

**Architecture**: MONAI's UNet reimplementation (Isensee et al. design — architecturally
similar to DynUNet). This means vesselFM's encoder is potentially directly compatible with
our existing DynUNet decoder for hybrid fusion.

vesselFM **outperforms all SAM-based medical segmentation foundation models** (VISTA3D,
SAM-Med3D, MedSAM-2) across 4 imaging modalities in zero-, one-, and few-shot scenarios.

**vesselFM zero-shot and few-shot results** (Dice / clDice):

| Dataset | Modality | Zero-Shot | One-Shot | Few-Shot |
|---------|----------|-----------|----------|----------|
| OCTA (mouse brain) | OCTA | 46.9 / 67.1 | 72.1 / 83.7 | 75.7 / 84.0 |
| BvEM (mouse brain) | vEM | 67.5 / 62.0 | 78.3 / 79.9 | 78.1 / 84.5 |
| SMILE-UHURA (human brain) | MRA | 74.7 / 75.3 | 76.4 / 78.4 | 78.8 / 79.4 |
| MSD8 (human liver) | CT | 29.7 / 36.1 | 36.9 / 48.7 | 45.0 / 57.3 |

Critically, vesselFM **reports clDice** — it is topology-aware by design, confirming it
takes connectivity preservation seriously (unlike any SAM-based approach).

### 7.2.1 vesselFM Gap Analysis for MiniVess

The competitive framing of "vesselFM vs. SAM3 for MiniVess" requires examining whether
vesselFM's training data covers MiniVess's specific domain:

**Modality coverage**: vesselFM's training data includes **two-photon microscopy,
light-sheet microscopy, and electron microscopy** (vEM) alongside clinical modalities
(MRA, CTA, CT, OCTA). MiniVess uses two-photon microscopy — this is **directly covered**
by vesselFM's training distribution.

**Resolution coverage**: vesselFM's training data spans 0.20-5.20 um (microscopy) to
0.26-2.50 mm (clinical). MiniVess voxel spacing ranges 0.31-4.97 um. This **falls within
vesselFM's microscopy resolution range** with excellent overlap.

**Species/organ coverage**: vesselFM includes mouse brain, rat brain, human brain, liver,
kidney, heart, and bladder vessels across human, mouse, and rat. MiniVess is mouse brain
microvasculature — **directly covered**.

**Topology metrics**: vesselFM reports both Dice and clDice for all evaluations. Few-shot
clDice reaches 84.0% (OCTA) and 84.5% (BvEM). For comparison, MiniVess DynUNet achieves
90.6% clDice — higher, but DynUNet is fully supervised on 70 volumes while vesselFM
achieves 84.5% with only a few examples.

**What vesselFM does NOT cover**: vesselFM was not compared to nnUNet. It was evaluated
against VISTA3D, SAM-Med3D, and MedSAM-2 — all foundation models. The absence of an nnUNet
baseline means we cannot directly assess vesselFM vs. our DynUNet pipeline. Additionally,
vesselFM's BvEM evaluation (closest to MiniVess) uses electron microscopy, not two-photon —
the exact modality match remains untested.

**Revised competitive assessment**: vesselFM is a **much stronger competitor** to SAM3 for
MiniVess than previously assumed. Its training data covers MiniVess's modality, resolution,
species, and organ. The most promising hybrid architecture may be **vesselFM encoder +
DynUNet decoder** (both are MONAI UNet variants, making encoder swapping feasible) rather
than SAM3 encoder + DynUNet decoder. This should be tested in Phase 2 alongside the SAM3
feature extraction experiment.

**Implication**: vesselFM should be the **primary** foundation model candidate for MiniVess.
SAM3's role is better framed as a complementary feature source, annotation accelerator, or
ensemble diversity member — not as the leading foundation model for vessel segmentation.

### 7.3 VesSAM Detailed Results: The Best SAM-Based Vessel Numbers

VesSAM (arXiv:2511.00981) provides the most comprehensive SAM-based vessel benchmarks
available. In-distribution results across 5 modalities (512x512):

| Dataset | VesSAM Dice | nnUNet Dice | MedSAM Dice |
|---------|-------------|-------------|-------------|
| LSCI (laser speckle) | 87.01 | 90.43 | 84.88 |
| Placenta | 84.61 | 81.03 | 84.22 |
| Retinal | 82.32 | 75.58 | 79.16 |
| Aorta | 95.99 | 96.78 | 96.39 |
| XCAD (coronary) | 92.14 | 83.23 | 89.72 |
| **Average** | **88.41** | **85.41** | **86.87** |

Out-of-distribution generalization (critical for clinical deployment):

| Dataset | VesSAM Dice | nnUNet Dice | MedSAM Dice |
|---------|-------------|-------------|-------------|
| **Average (5 datasets)** | **61.40** | **44.15** | **30.90** |

VesSAM's OoD advantage (+17.25% over nnUNet, +30.5% over MedSAM) demonstrates that
structure-aware prompting dramatically improves generalization — precisely because it
encodes vascular geometry as an inductive bias rather than learning it from data alone.

### 7.4 SAM for Road/Linear Structure Extraction

The SAM-based road segmentation literature provides instructive parallels for vessels:

**SAM-Road** (Hetang et al. 2024, CVPRW Best Paper): Uses SAM encoder for vectorized road
network graph extraction from aerial imagery. Comparable accuracy to RNGDet++ while being
**40x faster**. Predicts complete road network graphs spanning multiple square km in seconds.
Uses APLS and TOPO metrics — the road segmentation equivalents of clDice and ccDice.

**SAM2MS** (2025, MDPI Remote Sensing): Adapts SAM2 with Multi-Scale Subtraction Module for
road extraction. Evaluated on DeepGlobe, SpaceNet, and **Massachusetts Roads** datasets.
Cross-dataset transfer achieves competitive performance without additional training.

**TPP-SAM** (Wu et al. 2025): Uses GPS trajectory points as SAM prompts for zero-shot
road extraction — an analogy to using vessel centerlines as prompts for vessel extraction.

**Key cross-domain insight**: The road segmentation community has independently converged
on the same solutions as the vessel community: topology-aware metrics (APLS ≈ clDice),
structure-aware prompting (trajectories ≈ skeletons), and graph-based post-processing.
This convergence across domains strengthens the case for topology-informed SAM adaptation
as a general principle for thin-structure segmentation.

---

## 8. Hybrid Architectures: SAM Encoder + Specialist Decoder

The most promising integration path for MiniVess is not standalone SAM deployment but
**hybrid architectures** that combine SAM's encoder features with topology-aware decoders.
This section synthesizes 15+ papers on fusion strategies, organized by architectural
pattern rather than publication date.

### 8.1 The Encoder-Decoder Split: Why Hybrids Work When Standalone SAM Fails

A critical 2024 finding resolves an apparent contradiction: how can SAM features be useful
for vessels if SAM cannot segment vessels? The answer lies in the encoder-decoder split.

Zhang et al. (2024, arXiv:2412.04243) identify the root cause: SAM's ViT patch mechanism
creates "fragmented, locally-focused attention patterns" for tree-like objects rather than
coherent global representations. **Even when only the decoder was fine-tuned with encoder
frozen, performance degradation persisted on high tree-likeness objects.** This confirms the
encoder IS part of the problem — but it does not mean the encoder is useless.

The "How Universal Are SAM2 Features?" study (arXiv:2510.17051) provides the complementary
evidence: SAM2's Hiera encoder excels at spatially-aligned tasks (depth estimation: 3.07
RMSE) but underperforms vanilla Hiera on conceptually distant tasks. Vessel segmentation is
a spatially-aligned task, suggesting SAM2/SAM3 features contain *partial* vessel information
that lightweight adapters can unlock.

Multiple independent studies confirm this: VesSAM (+10% Dice with adapters), VesselSAM
(93.5% DSC with AtrousLoRA rank 4), and TopoLoRA-SAM (best clDice on retinal vessels with
LoRA + clDice loss) all demonstrate that adapted SAM encoder features substantially improve
vessel segmentation. The encoder features are **necessary but insufficient** — they require
vessel-aware supplementation.

### 8.2 Architectural Pattern Taxonomy

We identify five distinct fusion patterns in the literature, each with different trade-offs:

#### Pattern A: Dual-Encoder Concatenation

**nnSAM** (Li et al. 2025, Medical Physics, DOI:10.1002/mp.17481): Two parallel encoders —
frozen SAM ViT producing domain-agnostic embeddings + trainable nnUNet encoder auto-configured
per dataset. Embeddings spatially aligned and concatenated at each resolution level. Decoder
has two heads: segmentation (Dice + CE) and level-set regression (curvature loss for shape
priors). Results: brain white matter DICE 82.77% vs 79.25% for standalone nnUNet with 20
training samples. Advantage grows with fewer samples (74.55% vs 68.25% at N=5).

**MedSAM2+nnUNet** (Zhong et al. 2025, PMC:12194608): MedSAM2 multi-scale features
(256ch at 128x128 and 64x64) fused with nnUNet encoder stages E3/E7 via channel-wise
concatenation after FrequencyLoRA (FFT spectral enhancement + rank-4 LoRA) and
AttentionGate (parallel channel + spatial attention). Results on CAS2023 brain vessel
TOF-MRA: Dice 84.49% (+0.60% over nnUNet), ASD 4.97mm (−0.36mm), recall 84.37% (+0.86%).
**This is the only published hybrid tested on brain vessels.**

**Strengths**: Simple, proven on vessels, compatible with any decoder loss.
**Weaknesses**: Marginal gains (+0.6% Dice); no topology metrics reported; inference >2x slower.

#### Pattern B: Cross-Attention Fusion

**DB-SAM** (MICCAI 2024): Dual-branch (frozen SAM ViT + lightweight CNN), bilateral
cross-attention dynamically combines deep ViT features with shallow CNN features, followed
by ViT-convolution fusion block. Result: +8.8% absolute gain on 21 3D medical tasks.

**VesSAM** (Kang et al. 2025): Three-component design: (1) convolutional adapter with
depth-wise separable convolutions + channel/spatial attention in frozen ViT encoder,
(2) hierarchical two-stage cross-attention multi-prompt encoder (sparse prompts × dense
features → graph-structured features), (3) cross-modal fusion mask decoder. Average Dice
88.41%, +10% over PEFT baselines.

**Strengths**: Selective feature integration; DynUNet can learn when to attend vs. ignore SAM.
**Weaknesses**: Higher implementation complexity; cross-attention adds ~15% compute overhead.

#### Pattern C: Adapter-Based 3D Extension

**MA-SAM** (Chen et al. 2024, Medical Image Analysis): Injects 3D adapters into SAM
transformer blocks, enabling the 2D backbone to extract volumetric information. Outperforms
nnUNet by +0.9% (CT multi-organ), +2.6% (MRI prostate), +9.9% (surgical video) in Dice.

**3DSAM-adapter** (Gong et al. 2024, Medical Image Analysis): Holistic 2D-to-3D adaptation
via spatial adapters with decomposed convolution (frozen 1×14×14 + trainable 14×1×1). Only
16.96% of parameters tunable. Outperforms domain SOTA on 3/4 tumor tasks by 8-30%.

**Strengths**: Native 3D processing, parameter-efficient.
**Weaknesses**: Requires volumetric training; not yet tested on vessels.

#### Pattern D: Encoder Replacement

SAM2-UNet and SAM3-UNet replace SAM's decoder with standard U-Net decoders while retaining
SAM's encoder (Hiera or SAM3 PE) with lightweight adapters. SAM3-UNet trains with <6GB
VRAM — the most memory-efficient hybrid. **Limitation**: Discards DynUNet's 3D convolutions
and existing trained weights, making it unsuitable as a MiniVess upgrade path.

#### Pattern E: Feature Distillation

VISTA3D (MONAI, CVPR 2025) and AutoSAM (BMVC 2023) use SAM features during training only,
distilling them into smaller task-specific models. Lower inference cost but information loss
during distillation. **No vessel-specific evidence** for either approach.

### 8.3 Topology-Aware Hybrid Innovations (Deep Analysis)

Five architectures specifically address topology preservation in hybrid SAM contexts. We
analyze each in depth, with particular attention to what their ablation studies reveal about
topology-aware adaptation of foundation models for thin structures.

#### 8.3.1 TopoLoRA-SAM: The Definitive Topology-Aware SAM Adaptation

**TopoLoRA-SAM** (Khazem 2025, arXiv:2601.02273) is the most relevant single paper for
MiniVess SAM integration because it directly answers the question "Can topology-aware losses
improve SAM adaptation for thin structures?"

**Architecture** (4.9M trainable params, 5.2% of 93.7M total):
- Frozen SAM ViT-B encoder (12 transformer blocks)
- LoRA modules (rank r=16) injected into FFN layers (mlp.lin1, mlp.lin2) of all 12 blocks
  - 2.4M params; Kaiming uniform initialization for A, zeros for B
  - Scaling factor α/r modulates low-rank update magnitude
- Spatial convolutional adapter: depthwise-separable conv (3×3 depthwise → ReLU → 1×1
  pointwise) with residual connection — only 66K params
  - Operates on high-resolution feature maps for boundary preservation
- SAM's mask decoder remains trainable (2.4M params), prompt-free mode with null embeddings

**Loss function**: `L = 1.0*BCE + 1.0*Dice + 0.5*clDice` (+ optional boundary loss at λ_bd=0.0)
Training: AdamW (lr=1e-4, cosine decay to 1e-6), 50 epochs.

**Key results** (retinal vessel segmentation):

| Dataset | Dice | clDice | BFScore | vs. Mask2Former |
|---------|------|--------|---------|----------------|
| DRIVE | 0.690±0.018 | **0.678** (best) | 0.662 | +3.8% Dice |
| STARE | 0.565±0.048 | 0.524 | 0.468 | +2.1% Dice |
| CHASE_DB1 | 0.569±0.016 | **0.599** (best) | 0.603 | +8.4% Dice |
| Retina avg | **0.608** | **0.600** | **0.578** (best) | — |

Also tested on polyps (Kvasir-SEG: 0.930 Dice) and SAR imagery (SL-SSDD: 0.994 Dice),
demonstrating cross-domain applicability of the topology-aware adaptation strategy.

**Critical ablation findings for MiniVess**:

1. **LoRA is the primary driver**: +4.2 Dice, +5.1 clDice, +8.3 BFScore over decoder-only
   tuning. The spatial adapter adds only +1.2 BFScore — modest and dataset-dependent. This
   means LoRA alone captures most of the adaptation benefit; the spatial adapter is not
   essential.

2. **Rank sensitivity** (r ∈ {4, 8, 16, 32}): r=16 is optimal for thin structures. r=4
   underfits (not enough capacity for vessel topology), r=32 overfits (doubled params, no
   gain). For MiniVess with 70 training volumes (more than the retinal datasets), r=16 is
   the correct choice.

3. **clDice weight sensitivity** (λ_cl ∈ {0.0, 0.25, 0.5, 1.0, 2.0}): λ_cl=0.5 is optimal.
   Higher weights (≥2.0) **degrade accuracy** — the topology regularization overwhelms the
   region-based gradients, causing the model to hallucinate false connections. This finding
   validates our cbdice_cldice loss design (0.5 weight for clDice component).

4. **clDice improvement is most pronounced on complex branching** — images with many branch
   points show larger clDice gains than images with simple linear vessels. This is directly
   relevant to microvasculature, which has far more branching complexity than retinal vessels.

**Implication for MiniVess**: TopoLoRA-SAM validates the combination {frozen SAM ViT + LoRA
r=16 + clDice loss} as effective for thin-structure adaptation. Our proposed hybrid
architecture (Section 8.4) directly incorporates these validated hyperparameters. The key
open question is whether 3D volumetric vessels (MiniVess) require the same or different
clDice weighting as 2D retinal vessels — likely the same, given that our standalone
cbdice_cldice uses 0.5 weight and achieves 0.906 clDice.

#### 8.3.2 TopoSAM: Cross-Domain Validation from Crack Segmentation

**TopoSAM** (2025, Engineering Applications of Artificial Intelligence, DOI:
S0952197625037200) provides independent validation from a morphologically analogous domain:
structural crack segmentation. Cracks share key geometric properties with microvasculature —
thin, branching, topologically complex, with irregular geometry against noisy backgrounds.

**Architecture** (three novel components):

1. **Topology Serpentine Convolution Branch (TSCB)**: An auxiliary encoder branch parallel
   to SAM's frozen Image Encoder. Uses deformable CNN with serpentine (snake-like) sampling
   kernels that progressively follow thin-structure geometry. Unlike standard 3×3 convolutions
   whose square receptive field wastes capacity on background pixels, serpentine convolutions
   adapt their sampling trajectory to the local structure orientation. This is the crack/vessel
   analogue of Dynamic Snake Convolution (Qi et al. 2023, ICCV), which was originally designed
   for tubular structure segmentation.

2. **Cross Branch Fusor (CBF)**: Integrates high-level semantic features from SAM's Image
   Encoder (global context, textural understanding) with low-level geometric features from
   TSCB (local topology, thin-structure detail). This is a Pattern B fusion (cross-attention
   variant) where SAM provides the "what" and TSCB provides the "where/how connected."

3. **Background Adversarial Twin Learning (BATL)**: A domain-invariance strategy. A
   pretrained style transfer model creates a Dynamic Style Bank (DySBank) of background
   styles. Gram-Minimal Style Sampling generates twin training samples with identical
   crack/vessel content but different backgrounds. The model learns to ignore background
   variation and focus on intrinsic thin-structure features. Evaluated across 7 datasets
   with diverse backgrounds — demonstrates strong generalization.

**Cross-domain insights for MiniVess**:

1. **Serpentine convolution is the topology-aware counterpart to AtrousLoRA**: VesselSAM
   uses multi-scale dilation to capture different vessel scales. TopoSAM's TSCB uses
   deformable sampling to follow vessel trajectory. Both address the same problem (standard
   convolutions miss thin structures) through complementary mechanisms. A hybrid combining
   both — atrous dilation for multi-scale + serpentine sampling for orientation-tracking —
   is unexplored.

2. **CBF validates our gated cross-attention design**: TopoSAM's Cross Branch Fusor combines
   SAM global features with topology-specific local features — architecturally equivalent to
   our proposed DynUNet (Q) + SAM (K/V) cross-attention. TopoSAM's crack results confirm
   this fusion pattern works for thin structures.

3. **BATL for microscopy domain invariance**: MiniVess microscopy images have significant
   background variation (tissue staining, illumination, microscope settings). BATL's approach
   of training with background-swapped twins could improve cross-volume generalization in
   MiniVess — a known challenge when some volumes have different tissue preparation protocols.

4. **The crack-vessel isomorphism is robust**: TopoSAM's success on 7 crack datasets with
   the same architecture confirms that topology-aware SAM adaptation generalizes across
   domains sharing thin-structure geometry. This strengthens the prediction that similar
   approaches will work for microvasculature.

#### 8.3.3 VesselSAM AtrousLoRA

**VesselSAM** (2025, arXiv:2502.18185): ASPP-style multi-scale dilation (rates 1, 6, 12,
18) within LoRA forward pass at rank r=4. Only 6.8M trainable params (7.2% of SAM).
Results: 93.5% DSC on aortic vessels. Captures both fine vessel detail and broader context
simultaneously. The multi-scale approach is complementary to TopoLoRA-SAM's spatial adapter
and TopoSAM's serpentine convolution — each addresses thin-structure geometry through a
different mechanism.

#### 8.3.4 HarmonySeg

**HarmonySeg** (ICCV 2025): Deep-to-shallow decoder with varying receptive fields +
vesselness map fusion + growth-suppression balanced loss. SOTA on 2D and 3D vessels/airways.
Not SAM-based but architecturally complementary — the deep-to-shallow decoder concept could
be combined with SAM encoder features. The vesselness map fusion is the classical computer
vision analogue of TopoSAM's TSCB (both inject structural priors into the decoder).

#### 8.3.5 Synthesis: Three Mechanisms for Topology-Aware SAM Adaptation

The topology-aware SAM literature converges on three complementary mechanisms:

| Mechanism | Paper | How It Works | MiniVess Relevance |
|-----------|-------|--------------|--------------------|
| **Topology loss** | TopoLoRA-SAM | clDice forces skeleton preservation | Directly uses our cbdice_cldice |
| **Multi-scale sampling** | VesselSAM | Atrous dilation captures vessel scales | Handles MiniVess Z-range 5-110 |
| **Deformable geometry** | TopoSAM | Serpentine conv follows thin-structure trajectory | Tracks branching vessel paths |

These three mechanisms are architecturally orthogonal and could be combined in a single
adapter module: LoRA with atrous dilation for multi-scale + serpentine sampling for
orientation-tracking + clDice loss for topology supervision. No published work has attempted
this three-way combination.

### 8.4 The Recommended Hybrid for MiniVess: DynUNet + SAM3 Cross-Attention Fusion

Synthesizing the evidence across all patterns, the most promising architecture for MiniVess
combines elements from nnSAM (Pattern A), VesSAM (Pattern B), and TopoLoRA-SAM:

```
Input 3D Volume (B, 1, D, H, W)
    |
    +---> DynUNet Encoder (3D, trainable, existing trained weights)
    |         |
    |    Multi-scale 3D features F_3d^{1..4}
    |
    +---> SAM3 Perception Encoder (2D, frozen + LoRA r=16 + AtrousAdapter)
              |
         Per-slice 2D features → Axial Fusion → Pseudo-3D features F_sam^{1..4}
              |
    +---------+---------+
    |                   |
    | Gated Cross-Attention Fusion (per scale level)
    |   Q = F_3d, K/V = F_sam (SAM informs DynUNet)
    |   alpha_gate initialized to 0 (safe start, learns importance)
    |
    v
DynUNet Decoder (3D, trainable)
    |
    +---> Segmentation Head (cbdice_cldice loss)
    +---> Optional: Centreline Auxiliary Head
```

**Why this design**:

1. **DynUNet as primary (Q), SAM as auxiliary (K/V)**: If SAM features are unhelpful (as
   Zhang et al. warn for tree-like structures), the gate alpha learns to suppress them. The
   architecture cannot perform worse than standalone DynUNet.

2. **LoRA r=16 + AtrousLoRA** (dilation 1,6,12,18): TopoLoRA-SAM shows r=16 is optimal
   for vessels; VesselSAM shows ASPP-style multi-scale captures both fine and coarse features.

3. **Axial fusion** for inter-slice context: SAM processes 2D slices, but vessels span
   multiple slices. AFTer-SAM (WACV 2024) and Bio2Vol (MICCAI 2025) demonstrate effective
   2D-to-3D bridging via axial attention.

4. **cbdice_cldice loss preserved**: All topology losses are decoder-agnostic. TopoLoRA-SAM
   confirms clDice improves vessel connectivity in hybrid architectures.

5. **Memory budget**: SAM3-UNet runs at <6GB with batch size 12. DynUNet peaks at 5.7GB.
   Using SAM2-Small (~4GB) or MobileSAM with sequential processing, the hybrid fits in
   ~10-12GB — feasible on a single 16GB GPU.

### 8.5 The vesselFM Alternative

vesselFM (Wittmann et al., CVPR 2025) is a vessel-specific foundation model that eliminates
all 2D-to-3D bridging complexity by operating natively in 3D. It outperforms all SAM-based
approaches across 4 modalities. If vesselFM's encoder can be used as a frozen feature
extractor (replacing SAM3 in the architecture above), it would be the stronger foundation —
3D-native, vessel-specific, and without the patch fragmentation issues that plague SAM on
thin structures.

**vesselFM + DynUNet** is potentially the optimal hybrid: a vessel-specific foundation
encoder (vesselFM) providing domain-relevant features, fused with a topology-aware decoder
(DynUNet with cbdice_cldice). No published work has attempted this combination.

---

## 9. Research Hypotheses and Integration Architecture

Based on the evidence synthesis above, we propose five ranked hypotheses for SAM3
integration with MiniVess, ordered by expected impact-to-effort ratio:

### H1: SAM3 Perception Encoder as Auxiliary Feature Extractor (Low Risk, Moderate Reward)

**Hypothesis**: Fusing SAM3 perception encoder features with DynUNet via cross-attention
or feature concatenation will improve vessel segmentation by 0.5-2% DSC without degrading
topology metrics.

**Evidence**: MedSAM2+nnUNet fusion achieved +0.6% DSC on brain vessels (PMC 12194608).
SAM3's perception encoder, pre-trained on billions of image-text pairs (exact count undisclosed; SAM3's training
data spans SA-1B, SA-V, and additional image-text corpora), provides richer features
than SAM2's vision-only encoder.

**Architecture**: DynUNet backbone (unchanged) + SAM3 PE features via FrequencyLoRA
attention gates at each decoder level. Topology losses (cbdice_cldice) applied to the
fused output.

**Risk**: 2x inference latency. Marginal gains may not justify complexity.

**Compute**: SAM3-Large perception encoder requires ~12GB VRAM for feature extraction
(exceeds the MiniVess single-GPU 8GB budget). Either: (a) use SAM3-Base (~4GB), sacrificing
feature quality, or (b) extract features offline on a larger GPU and cache to disk (~200MB
per volume × 70 = 14GB storage). Fine-tuning FrequencyLoRA attention gates: ~2-4 GPU-hours
on an A100.

### H2: VesSAM-Style Prompting with DynUNet Initialization (Medium Risk, High Reward)

**Hypothesis**: Using DynUNet's initial segmentation to extract skeleton/bifurcation/midpoint
prompts, then refining with SAM3, will improve both DSC and clDice by leveraging SAM3's
semantic understanding of vessel-like structures.

**Evidence**: VesSAM achieved +10% Dice over PEFT-SAM baselines using structure-aware
prompts. DynUNet already achieves 0.906 clDice, providing high-quality structural priors
for prompt generation.

**Architecture**: Stage 1: DynUNet → initial mask. Stage 2: Extract centreline +
bifurcation prompts via existing `centreline_extraction.py`. Stage 3: SAM3 concept
prompt ("microvasculature") + structure prompts → refined mask. Stage 4: Topology losses
on final output.

**Risk**: Two-stage inference is slower. SAM3 may not improve on already-excellent DynUNet
topology performance.

**Compute**: Stage 1 (DynUNet) uses ~5.5GB VRAM (existing). Stage 2 (centreline extraction)
is CPU-only (~2s per volume). Stage 3 (SAM3 with prompts) requires ~8-12GB VRAM per slice.
Total inference per volume: ~30-60s (vs ~5s for DynUNet alone, a 6-12x slowdown). Fine-tuning
SAM3 with VesSAM-style prompts: ~8-16 GPU-hours on A100, requires implementing the
hierarchical cross-attention prompt encoder.

### H3: Conformal Prediction over SAM3-Augmented Ensemble (Low Risk, High Reward)

**Hypothesis**: Adding SAM3-based predictions to the existing ensemble (per-loss best,
cross-loss voting) and applying conformal calibration will provide tighter uncertainty
bounds without sacrificing coverage.

**Evidence**: ConformalSAM (ICCV 2025) demonstrates conformal prediction works with SAM
outputs. MiniVess already has `MorphologicalConformalPredictor` and
`DistanceTransformConformalPredictor`. Adding SAM3 as an ensemble member increases diversity.

**Architecture**: Existing 4-loss DynUNet ensemble + SAM3 fine-tuned prediction → conformal
calibration on held-out calibration set → coverage-guaranteed prediction sets.

**Risk**: Minimal — extends existing infrastructure without replacing anything.

**Compute**: SAM3 fine-tuning with ConvLoRA on 5-10 volumes: ~2-4 GPU-hours on A100.
Conformal calibration on held-out set: CPU-only, <1 minute. Total VRAM for ensemble
inference: same as H1 for the SAM3 member (~4-12GB), but members run sequentially not
in parallel, so 8GB GPU is sufficient with sequential processing.

### H4: SAM3 Concept Prompting for Annotation Acceleration (Zero Risk, Immediate Reward)

**Hypothesis**: Using SAM3's text/exemplar prompting for interactive annotation of new
MiniVess-like datasets will reduce annotation time by 50-85% compared to manual
segmentation.

**Evidence**: MedSAM2 reports 85-92% annotation time reduction via interactive
segmentation. micro-SAM demonstrates practical napari-based annotation workflows.
SAM3's exemplar-based prompting (providing a crop of a vessel region) enables zero-shot
initial masks for human correction. Note: text prompting ("microvasculature") is unlikely
to work because two-photon microscopy vessels are almost certainly out-of-distribution for
SAM3's text encoder. Exemplar prompts are the viable zero-shot path.

**Architecture**: SAM3 (via micro-SAM napari plugin or custom Gradio interface) →
initial vessel mask → human correction → topology-quality QC via existing pipeline.

**Risk**: Effectively zero — this is a tooling improvement, not a model change.

**Compute**: SAM3 inference via micro-SAM napari plugin: ~4-8GB VRAM. No fine-tuning
required. Human annotation time per volume with SAM3 assistance: estimated 15-30 min
(vs 60-120 min manual), based on MedSAM2's 85-92% annotation time reduction for organ
structures (likely less reduction for vessels due to prompt-topology mismatch).

### H5: Topology-Aware Conformal Prediction for SAM Outputs (High Risk, Novel Contribution)

**Hypothesis**: Extending conformal prediction to provide coverage guarantees on
topological properties (Betti numbers, connected components, junction counts) of SAM3
vessel predictions would be a novel contribution to both the conformal inference and
medical segmentation literatures.

**Evidence**: No published work combines topological metrics with conformal prediction
for segmentation. MiniVess already computes BE_0, BE_1, ccDice, and junction F1.
Risk-controlling prediction (LTT framework) supports custom risk functions.

**Architecture**: Define topological risk function R(prediction, ground_truth) = weighted
sum of BE_0 error + BE_1 error + (1 - ccDice). Apply LTT framework from existing
`RiskControllingPredictor` to find threshold lambda* that controls E[R] <= alpha.

**Risk**: Theoretical novelty is high but practical improvement over pixel-level conformal
prediction is uncertain. Topological metrics are non-differentiable and discontinuous,
which may violate conformal assumptions.

**Compute**: Negligible beyond existing pipeline. Uses existing `RiskControllingPredictor`,
existing topology metrics (BE_0, BE_1, ccDice, junction F1). The only new code is the
topological risk function definition and LTT calibration — pure CPU, <1 minute. Note:
this hypothesis does NOT require SAM3. It is SAM-independent and should be pursued
regardless of SAM3 integration decisions.

---

## 10. Devil's Advocate: The Case Against SAM3 Integration

Before committing research time to SAM3 integration, the counter-argument deserves a
serious hearing. The evidence in this report *itself* makes the strongest case against
SAM3 for MiniVess:

**The architectural argument is damning.** Zhang et al. (2024) demonstrate that SAM's
tree-like structure limitation is fundamental, not fixable by fine-tuning. micro-SAM
confirms generalist models are "detrimental" for tubular structures. MedSAM2 explicitly
acknowledges failure on vessels. If the architecture cannot segment vessels after
fine-tuning on 455K pairs (MedSAM2) or 33 medical datasets (Medical SAM3), why would
adapter modules or prompt engineering overcome this?

**vesselFM already exists.** A purpose-built vessel foundation model (CVPR 2025) that
outperforms all SAM-based approaches across 4 modalities in zero/one/few-shot settings.
If the goal is "foundation model for vessels," vesselFM is the answer. Every GPU-hour
spent on SAM3 adaptation is a GPU-hour not spent on vesselFM integration or DynUNet
topology improvements.

**The marginal gain is tiny.** The best published evidence for SAM+specialist fusion
shows +0.6% DSC at 2x inference cost (PMC:12194608). For a pipeline already achieving
0.824 DSC / 0.906 clDice, this gain is within noise of cross-validation variability.

**The opportunity cost is real.** The same 8-week research investment proposed in the
roadmap below could instead yield: (a) vesselFM integration and benchmarking on MiniVess,
(b) topology-aware conformal prediction without SAM (H5 does not actually require SAM),
or (c) improving the existing DynUNet pipeline with graph-constrained post-processing.

**Our response:** The case against SAM3 *as a segmenter* is indeed overwhelming — and we
agree. The five hypotheses below are ordered by risk precisely because the low-risk options
(H4: annotation, H3: ensemble member) do not depend on SAM3 being a good vessel segmenter.
H4 uses SAM3 as a human-in-the-loop annotation accelerator where imperfect masks are
acceptable. H3 uses SAM3 as a diversity source in an ensemble where conformal calibration
handles its errors. Only H2 and H1 bet on SAM3's segmentation quality, and both include
explicit go/no-go gates. H5 is SAM-independent. The roadmap below operationalizes this
pragmatism with hard decision criteria.

---

## 11. Practical Integration Roadmap for MiniVess

### Hardware Feasibility Matrix (RTX 2070 Super, 8GB VRAM)

The MiniVess project's primary GPU is an RTX 2070 Super (8GB VRAM). Every hypothesis must be
feasible on this hardware or specify the workaround:

| Hypothesis | Peak VRAM (naive) | Feasible on 8GB? | Workaround |
|-----------|-------------------|------------------|------------|
| **H1** (SAM3 feature extractor) | ~12GB (SAM3-Large) | **No** | Use MobileSAM (~1.5GB) or SAM3-Base (~4GB); or extract features offline on Colab/cloud GPU and cache to disk (~200MB/volume × 70 = 14GB SSD) |
| **H2** (VesSAM prompting) | ~8-12GB (SAM3 per slice) | **Marginal** | Use MobileSAM for prompt refinement (~1.5GB + DynUNet 5.5GB = 7GB total); or run stages sequentially (DynUNet → free VRAM → SAM inference) |
| **H3** (ensemble member) | ~4-8GB (ConvLoRA fine-tuning) | **Yes** with SAM-Base | Fine-tune SAM-Base with ConvLoRA (~4GB); inference runs ensemble members sequentially, not in parallel |
| **H4** (annotation tool) | ~4-8GB (SAM3 inference) | **Yes** | micro-SAM napari plugin with SAM-Base; interactive annotation is I/O-bound, not VRAM-bound |
| **H5** (topological CP) | ~0GB (CPU-only) | **Yes** | No SAM component; uses existing topology metrics and LTT framework |
| **vesselFM** (benchmark) | ~4-8GB (MONAI UNet) | **Yes** | vesselFM uses MONAI UNet architecture; comparable to DynUNet VRAM footprint |
| **Hybrid** (Section 8.4) | ~10-12GB | **No** | Run DynUNet + MobileSAM (instead of SAM3-Large): 5.5GB + 1.5GB = 7GB; or use offline feature caching |

**Key constraint**: SAM3-Large is not feasible on 8GB VRAM. All SAM3 experiments must use
SAM3-Base, MobileSAM, or offline feature extraction. This does not invalidate the research
direction — TopoLoRA-SAM's results were obtained with SAM ViT-B (not ViT-H), and
MobileSAM was specifically designed as a 1/60th-size distillation that preserves feature
quality for downstream tasks.

**Cost estimate for cloud workaround**: A100-40GB on Lambda Labs or similar: ~$1.10/hr.
Feature extraction for 70 volumes (offline): ~1-2 hours = ~$2. Fine-tuning experiments:
~4-16 hours = ~$5-18. Total cloud budget for full roadmap: **$10-40**, well within
academic budgets.

### Go/No-Go Decision Criteria

Each phase has an explicit gate. Failing the gate triggers redirection to vesselFM:

| Phase | Gate Metric | Threshold | If FAIL → Redirect To |
|-------|-----------|-----------|----------------------|
| Phase 1 (zero-shot) | SAM3 zero-shot DSC on 5 MiniVess volumes | ≥0.10 DSC | Abandon SAM3 segmentation; keep H4 (annotation) only |
| Phase 1 (zero-shot) | SAM3 zero-shot clDice | ≥0.05 clDice | Same — no topology awareness at all |
| Phase 2 (features) | Linear probe vessel AUC on SAM3 PE features | ≥0.80 AUC | SAM3 features lack vessel information; try vesselFM features |
| Phase 2 (ensemble) | 5-member ensemble DSC improvement over 4-member | ≥+0.3% DSC | SAM3 adds noise, not diversity; drop from ensemble |
| Phase 3 (VesSAM) | Two-stage DSC improvement over DynUNet alone | ≥+1.0% DSC | SAM3 refinement not worth the inference cost |
| Phase 3 (VesSAM) | clDice maintained or improved | ≥0.900 clDice | Refinement harms topology; revert to DynUNet only |

If Phase 1 gates fail, total investment is ~1 week. The roadmap is designed to fail fast.

### Phase 1: Zero-Risk Wins (Week 1)

1. **SAM3 annotation tool** (H4): Deploy SAM3 via micro-SAM napari plugin for interactive
   MiniVess re-annotation and quality control. No model changes, immediate productivity gain.

2. **Benchmark SAM3 zero-shot**: Run SAM3 with text prompt "blood vessel" and exemplar
   prompt (single MiniVess slice crop) on 5 held-out volumes. Establish the zero-shot
   baseline DSC and clDice — likely very low, but important for comparison.

### Phase 2: Feature Integration (Weeks 2-3)

3. **vesselFM zero-shot benchmark**: Run vesselFM on 5 MiniVess volumes (zero-shot and
   one-shot). vesselFM includes two-photon microscopy in its training data and covers
   MiniVess's resolution range — it may outperform SAM3 without any adaptation.
   Hardware: feasible on RTX 2070 Super (MONAI UNet architecture, ~4-6GB VRAM).

4. **SAM3 feature extraction** (H1): Extract SAM3-Base/MobileSAM perception encoder features
   for MiniVess slices (offline on cloud GPU if needed, ~$2 for 70 volumes). Evaluate via
   linear probing whether vessel-relevant information exists in the feature space.

5. **Ensemble member** (H3): Fine-tune SAM-Base with ConvLoRA on 5-10 MiniVess volumes
   (~4GB VRAM). Add as 5th ensemble member alongside existing 4-loss DynUNet models. Apply
   existing conformal calibration.

### Phase 3: Architecture Innovation (Weeks 4-8)

6. **VesSAM prompting** (H2): Implement structure-aware prompt generation from DynUNet
   predictions using MobileSAM for inference (~7GB total). Evaluate two-stage refinement.

7. **vesselFM + DynUNet hybrid**: If vesselFM zero-shot exceeds SAM3 (expected from Section
   7.2.1 analysis), test vesselFM encoder features fused with DynUNet decoder. Both are
   MONAI UNet variants, making encoder swapping architecturally natural.

8. **Topological conformal prediction** (H5): Extend `RiskControllingPredictor` with
   topology-based risk functions. Evaluate coverage guarantees on Betti numbers and
   connected components. SAM-independent — pursue regardless of other results.

---

## 12. Key Findings Summary

| # | Finding | Evidence Strength | Implication |
|---|---------|-------------------|-------------|
| 1 | SAM3 != SAM2 (fundamentally different architecture) | Strong (5 divergences) | Cannot reuse SAM2 adaptation strategies |
| 2 | Vessels are SAM2's worst category (0.14-0.64 DSC) | Strong (4 papers) | SAM2 approaches not viable for MiniVess |
| 3 | Configurable memory bank caps propagation at ~0.56 IoU | Strong (oracle experiment) | Slice-by-slice fundamentally limited for 3D |
| 4 | SAM's tree-like structure limitation is architectural, not fixable by fine-tuning | Strong (Zhang et al. 2024) | Must augment, not replace, existing pipeline |
| 5 | vesselFM outperforms all SAM-based models; covers MiniVess domain (two-photon, 0.2-5.2um) | Strong (CVPR 2025, verified) | Domain-specific FM is primary candidate; SAM3 is secondary |
| 6 | VesSAM skeleton prompting: +10% Dice, +17% OoD | Strong (8 datasets, 5 modalities) | Structure-aware prompting essential for vessels |
| 7 | LoRA fine-tuning with 5-20 images viable | Moderate (multiple papers, not on vessels) | 70 MiniVess volumes sufficient for any strategy |
| 8 | micro-SAM generalist detrimental for tubular structures | Strong (Nature Methods) | Must use specialist fine-tuning, not generalist |
| 9 | Foundation models may not beat UNet on small datasets | Moderate (Med Physics 2026) | DynUNet + topology losses may remain superior |
| 10 | No published work combines topology UQ + SAM | Strong (gap confirmed) | Novel contribution opportunity at H5 |
| 11 | Road segmentation SAM community converges on same solutions as vessel community | Strong (6+ papers) | Topology-aware prompting is domain-general |
| 12 | Hybrid SAM+specialist architectures consistently improve over standalone SAM for vessels | Strong (6+ papers, 5 patterns) | Dual-encoder fusion is the viable SAM integration path |
| 13 | nnSAM dual-encoder: +3.5% DSC, advantage grows with fewer samples | Strong (Li et al. 2025) | Foundation features most valuable in low-data regimes |
| 14 | None of 20+ hybrid SAM papers used topology-preserving losses (clDice, skeleton recall) | Strong (comprehensive survey) | Major research gap — combining topology losses with SAM hybrids |
| 15 | TopoLoRA-SAM validates {LoRA r=16 + clDice λ=0.5} as optimal for thin structures | Strong (ablation study) | Directly validates our proposed hybrid architecture hyperparameters |
| 16 | clDice improvement most pronounced on complex branching patterns | Strong (TopoLoRA-SAM) | Microvasculature (high branching) should benefit more than retinal vessels |
| 17 | Crack segmentation (TopoSAM) independently validates topology-aware SAM adaptation | Moderate (cross-domain, 7 datasets) | Thin-structure SAM adaptation generalizes across domains |
| 18 | Three complementary topology mechanisms: loss (clDice), multi-scale (atrous), deformable (serpentine) | Strong (3 independent papers) | Combining all three is an unexplored opportunity |

---

## 13. Citation Index

### SAM Family Core

- Ravi et al. (2025). "SAM 3: Segment Anything Model 3." arXiv:2511.16719.
- Ravi et al. (2024). "SAM 2: Segment Anything in Images and Videos." arXiv:2408.00611.

### Medical SAM3 Adaptations

- Medical SAM3 (2025). arXiv:2601.10880. Full fine-tuning on 33 medical datasets.
- MedSAM3 (Liu et al. 2025). arXiv:2511.19046. Text-prompted medical concept segmentation.
- ConceptBank (Pei et al. 2025). arXiv:2602.06333. Training-free SAM3 calibration.
- SAM3-Adapter (2025). arXiv:2511.19425. First adapter for SAM3 architecture.

### SAM2 Medical Evaluations

- MedSAM2 (Zhu, Ma et al. 2025). arXiv:2504.03600. 455K training pairs, vessel limitations explicit.
- BioSAM-2 (Zhao et al. 2024). arXiv:2408.03286. 10% below nnUNet on organs.
- Huang et al. (2024). arXiv:2408.00756. Zero-shot benchmark, IoU ceiling at 0.56.
- Ma, Wang et al. (2024). arXiv:2408.03322. Aorta 0.64 DSC, IVC 0.35 DSC after fine-tuning.
- MedSAM2+nnUNet Fusion (2025). PMC:12194608. +0.6% DSC on brain vessels at 2x cost.

### Vessel-Specific SAM

- VesSAM (Kang et al. 2025). arXiv:2511.00981. Skeleton/bifurcation/midpoint prompting, +10% Dice.
- VesselSAM (2025). arXiv:2502.18185. AtrousLoRA for aortic vessel segmentation, 93.5% DSC.

### Hybrid Architectures (SAM + Specialist)

- nnSAM (Li et al. 2025). Medical Physics, DOI:10.1002/mp.17481. Dual-encoder (frozen MobileSAM + nnUNet), +3.5% DSC brain WM.
- MedSAM2+nnUNet Fusion (Zhong et al. 2025). PMC:12194608. FrequencyLoRA + AttentionGate, +0.6% DSC brain vessels.
- DB-SAM (2024). MICCAI 2024. Bilateral cross-attention, +8.8% on 21 3D medical tasks.
- VISTA3D (NVIDIA MONAI, 2025). CVPR 2025. SegResNet + SAM cross-attention, 127 classes.
- SAM-UNet (2024). SA-Med2D-16M training. Frozen SAM ViT + ResNet-34/50, 88.3% overall.
- MA-SAM (Chen et al. 2024). Medical Image Analysis. FacT 3D adapters in SAM ViT-H, +0.9% CT multi-organ.
- 3DSAM-adapter (Gong et al. 2024). Medical Image Analysis. 7.8% param overhead, +8-30% on tumors.
- Med-SA (2025). Medical Image Analysis. Space-Depth Transpose + HyP-Adpt, 2% params, +34.8% over SAM.
- SAM2-UNet (Xiong & Wu 2025). Visual Intelligence. Hiera encoder + U-Net decoder, 92.8% polyp.
- H-SAM (2024). CVPR 2024. Hierarchical 2-stage decoder, +4.78% multi-organ.
- AFTer-SAM (Yan et al. 2024). WACV 2024. Axial Fusion Transformer for volumetric SAM.
- AutoSAM (2023). BMVC 2023. Auxiliary prompt encoder, no human prompts.
- De-LightSAM (2025). IEEE TCSVT. Knowledge distillation from SAM-H, 2% params.
- SAM3-UNet (2025). SAM3 encoder + U-Net decoder, <6GB VRAM.
- vesselFM (Wittmann, Wattenberg, Amiranashvili, Shit & Menze 2025). CVPR 2025. arXiv:2411.17386. Vessel-specific FM, MONAI UNet backbone, 17 datasets incl. two-photon + light-sheet microscopy, reports clDice.

### Topology-Aware SAM Adaptation

- TopoLoRA-SAM (Khazem 2025). arXiv:2601.02273. LoRA r=16 + clDice, best clDice on DRIVE (0.678).
- TopoSAM (2025). Eng. Applications of AI, DOI:S0952197625037200. TSCB + CBF + BATL for crack segmentation.
- HarmonySeg (2025). ICCV 2025. Deep-to-shallow decoder + vesselness fusion, SOTA vessels/airways.
- Dynamic Snake Convolution (Qi et al. 2023). ICCV 2023. Serpentine sampling for tubular structures.

### Adaptation Methods

- ConvLoRA (Zhang et al. 2024). arXiv:2401.17868. ICLR 2024, convolution + LoRA for SAM.
- SAMed (Zhang et al. 2023). arXiv:2304.13785. LoRA-based semantic segmentation, 81.88 DSC.
- SAM2-Adapter (Chen et al. 2024). arXiv:2408.04579. Multi-adapter for SAM2.
- Onco-Seg (2026). medRxiv. SAM3 LoRA, 35 datasets, 98K cases.
- LoRA Recycle (Hu et al. 2025). CVPR 2025. +9.8% for 1-shot learning.

### Microscopy and 3D

- micro-SAM (Archit et al. 2025). Nature Methods 22:579-591. Tubular structures need specialist.
- SAM-Med3D (Wang et al. 2024). arXiv:2310.15161. Native 3D, single point prompt for volumes.
- ProtoSAM-3D (Shen et al. 2025). Comput Med Imaging Graph 121:102501. Prototype-based 3D.

### Uncertainty Quantification

- UncertainSAM (Kaiser et al. 2025). arXiv:2505.05049. ICML 2025. Aleatoric + epistemic + task UQ.
- ConformalSAM (Chen et al. 2025). arXiv:2507.15803. ICCV 2025. Conformal prediction + SAM.
- Conformal Semantic Segmentation (Mossina & Dalmau 2024). arXiv:2405.05145. CVPR-W 2024.

### Few-Shot and ICL

- SAM3 zero-shot vs SAM2 (2025). arXiv:2511.21926. SAM3 superior on 16 medical datasets.
- SAM-MPA (2024). arXiv:2411.17363. Mask propagation + auto-prompting.
- Foundation Model vs UNet (Li et al. 2025). Medical Physics 52(3). SAM/MedSAM did not outperform UNet in limited data settings.

### SAM Architectural Analysis

- Zhang et al. (2024). arXiv:2412.04243. Quantifying SAM limits — tree-like structure failure is architectural.
- "How Universal Are SAM2 Features?" (2025). arXiv:2510.17051. SAM2 Hiera excels at spatial tasks.
- SAM for Medical Segmentation Survey (2025). PMC:12189367. Comprehensive review of SAM adaptations.

### Prompt Engineering

- SAM Prompt Engineering Survey (2025). arXiv:2507.09562. First comprehensive taxonomy.
- AutoProSAM (2025). WACV 2025. Automated prompt generation for 3D multi-organ.

---

## Appendix A: Key Numbers at a Glance

### SAM2 Vessel Performance (Fine-Tuned)

| Structure | DSC | Source |
|-----------|-----|--------|
| Aorta (CT) | 0.64 | Ma et al. 2024 |
| IVC (CT) | 0.35 | Ma et al. 2024 |
| DRIVE retinal (2D) | 0.558 | Medical SAM3 2025 |
| CHASE_DB1 retinal (2D) | 0.626 | Medical SAM3 2025 |
| FetoPlac (2D) | 0.770 | Medical SAM3 2025 |
| Brain vessels (+fusion) | +0.006 over nnUNet | PMC:12194608, 2025 |

### MiniVess DynUNet Baseline (for comparison)

| Loss | DSC | clDice | Source |
|------|-----|--------|--------|
| cbdice_cldice | ~0.78 | **0.906** | dynunet_loss_variation_v2 |
| dice_ce | **0.824** | 0.832 | dynunet_loss_variation_v2 |

### SAM3 Zero-Shot

| Metric | Value | Source |
|--------|-------|--------|
| Mask AP (LVIS) | 48.8 | SAM3 paper |
| 1-shot exemplar | +18.3 AP over T-Rex2 (COCO) | SAM3 paper |
| 10-shot | SOTA on ODinW, COCO | SAM3 paper |
| Medical zero-shot | SAM3 > SAM2 on 16 datasets | arXiv:2511.21926 |

### Adaptation Data Requirements

| Strategy | Min Images | Vessel Evidence |
|----------|-----------|-----------------|
| SAM3 concept prompt | 0-1 | None (zero-shot) |
| LoRA Recycle | 1-10 | None |
| LoRA fine-tuning | 5-50 | 97.6% IoU (femur, not vessels) |
| Full LoRA | 200-800 | Onco-Seg: 35 datasets |
| micro-SAM specialist | 2-5% of dataset | Explicit tubular limitation |
| Full fine-tuning | Full dataset | Medical SAM3: 33 datasets, 4xH100 |
