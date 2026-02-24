# Interactive Segmentation for Proofreading 3D Vascular Annotations

**Research Report — MinIVess MLOps**
**Date:** 2026-02-24
**Scope:** Interactive segmentation tools and methods for correcting/refining initial AI-generated 3D segmentation masks, with emphasis on vascular structures in medical imaging.

---

## Executive Summary

This report evaluates the current landscape of interactive segmentation for **proofreading** existing 3D medical image segmentations — a workflow where an initial AI-generated mask is corrected by a human expert rather than annotated from scratch. The investigation covers 6 seed papers, 20+ additional methods identified through systematic web research, and analysis of 5 annotation platform alternatives.

**Key findings:**

1. **nnInteractive + SlicerNNInteractive** (Isensee et al., 2025; de Vente et al., 2025) is the strongest immediately deployable solution: native 3D, all prompt types, client-server architecture, and 3D Slicer integration.
2. **3D Slicer is the pragmatic choice** over building a custom voxel frontend — the ecosystem now offers 5+ AI-assisted segmentation extensions, and client-server architectures decouple the GPU requirement from the annotation workstation.
3. **MONAI Label is no longer the only option** — it has been surpassed by nnInteractive for interactive segmentation quality, though it retains value for active learning loops.
4. **The proofreading paradigm** (AI init → interactive correction) reduces annotation burden by >30% versus from-scratch annotation (Guo et al., 2025), and online adaptation during proofreading sessions can recover 25+ Dice points on out-of-distribution data (Xu et al., 2025).
5. **Vascular structures remain challenging** for all interactive methods due to thin tubular geometry — topology-aware losses and vessel-specific models (vesselFM) matter more than the interactive segmentation tool choice.

**Recommendation:** Adopt **3D Slicer + SlicerNNInteractive** as the primary annotation/proofreading platform, with MONAI Label retained as an optional active learning orchestrator. This replaces the current equal weighting of Label Studio and MONAI Label in the PRD.

---

## 1. Problem Context

MinIVess-MLOps requires a workflow for:
- **Correcting** automated 3D vascular segmentations produced by DynUNet/SegResNet/vesselFM
- **Refining** boundary delineations on thin tubular structures (vessels 1–5 voxels diameter)
- **Scaling** annotation across datasets from multiple scanners and protocols (distribution shift)
- **Operating** with a solo researcher or small lab group (no dedicated annotation team)

The current PRD has `annotation_platform` and `annotation_workflow` decisions with no interactive segmentation-specific options. This report provides the evidence base to update those decisions.

---

## 2. Seed Paper Analysis

### 2.1 nnInteractive — Redefining 3D Promptable Segmentation
**Isensee et al. (2025).** arXiv:2503.08373. DKFZ Heidelberg. **1st place, CVPR 2025 Interactive 3D Segmentation Challenge.**

- **Architecture:** nnUNet ResEnc-L backbone with **early prompting** — prompts encoded as additional input channels (foreground clicks, background clicks, bounding box volumes, scribble maps, lasso regions, prior mask) before any feature extraction. This contrasts with SAM-style latent-space prompt injection.
- **Prompt types:** Points, bounding boxes, scribbles, lasso (2D draw → 3D mask), and **prior mask input** (critical for proofreading).
- **Training:** 120+ diverse 3D datasets spanning CT, MRI, PET, and 3D microscopy.
- **Prediction propagation:** When a large structure is detected (>1,000 border voxels change), sliding window expansion propagates the mask beyond the current patch.
- **Performance:** Substantially outperforms SAM2, SegVol, and SAM-Med3D on 3D benchmarks. Won the CVPR 2025 challenge.
- **Proofreading relevance:** The prior mask input channel directly enables feeding an existing segmentation as starting point for correction. Early prompting enables fast, reactive correction cycles.
- **Limitations:** Requires GPU with 10GB+ VRAM for inference. Windows/Linux only for server.

### 2.2 SlicerNNInteractive — 3D Slicer Extension for nnInteractive
**de Vente et al. (2025).** arXiv:2504.07991. Submitted to JOSS.

- **Architecture:** Client-server via FastAPI REST endpoints. Heavy inference runs on remote GPU server; 3D Slicer is the thin client.
- **Deployment:** Server via Docker Hub (`coendevente/nninteractive-slicer-server`) or pip. Client via GitHub (not yet in Slicer Extensions Manager).
- **Features:** All nnInteractive prompt types with keyboard shortcuts. Segment Editor integration. Server URL persisted across sessions.
- **Key advantage:** Annotation workstation does not need a GPU. macOS support for the client (server must be Linux/Windows).
- **Proofreading relevance:** This IS the production deployment path for using nnInteractive to proofread/refine segmentation masks in 3D Slicer.

### 2.3 MedSAM-Agent — Agentic Interactive Segmentation via RL
**Liu et al. (2026).** arXiv:2602.03320.

- **Architecture:** Qwen3-VL-8B multimodal LLM as policy model, orchestrating MedSAM2 or IMISNet as segmentation backend. Trained via supervised fine-tuning (449K trajectories) + GRPO reinforcement learning (9K hard samples).
- **Paradigm:** Reformulates interactive segmentation as a multi-step Markov Decision Process. Agent observes current segmentation, decides where to place corrective prompts, evaluates results, iterates up to 5 turns.
- **Results:** Average Dice=0.794 (MedSAM2 backend) across 21 datasets, 6 modalities. Below oracle single-turn box prompting (0.876).
- **Proofreading relevance:** Conceptually models the proofreading workflow (observe → decide → correct), but the 8B parameter LLM imposes significant inference latency. Research direction, not a practical tool yet.
- **Limitations:** Large model (8B params), max 5 turns, substantial GPU infrastructure (8x H20 for training), no Slicer integration.

### 2.4 Multimodal Interactive Segmentation (PhD Dissertation)
**Schon (2026).** University of Augsburg. Defended December 2025.

- **Contributions:** (1) Test-Time Adaptation of SAM using pseudo-labels from interactions; (2) SkipClick with DINOv2 backbone; (3) pseudo-depth for generalization; (4) MMFuser for multi-modal inputs.
- **Venues:** CVPR 2023/2024 workshops, WACV 2025 (best paper), CBMI 2024/2025.
- **Proofreading relevance:** The TTA contribution is directly relevant — as a user corrects segmentations, the model adapts to the specific scanner/protocol being annotated. The multi-modal fusion could benefit multi-contrast MRI annotation.
- **Limitations:** 2D only across all contributions. Primarily natural image domain.

### 2.5 K-Prism — Unified Prompting for Medical Image Segmentation
**Guo et al. (2025).** arXiv:2509.25594. Rutgers/Stanford/NYU.

- **Architecture:** Unifies semantic, in-context, and interactive segmentation. MoE decoder with dynamic routing. Mode chaining: semantic/in-context → interactive refinement.
- **Results:** NoC90=1.95, Dice@1click=89.55%, Dice@5clicks=95.50% (in-distribution). **Mode chaining reduces NoC90 by >30%** compared to interactive-only.
- **Proofreading relevance:** Directly validates the proofreading paradigm — feeding an initial mask as prior dramatically reduces correction burden. The >30% reduction in clicks when starting from an existing segmentation is the strongest quantitative evidence for the proofread-don't-reannotate approach.
- **Limitations:** 2D only. MoE adds model complexity.

### 2.6 OAIMS — Online Adaptation for Interactive Segmentation under Distribution Shift
**Xu et al. (2025).** arXiv:2503.06717. Oxford University.

- **Architecture:** U-Net with Click-Centered Gaussian (CCG) loss — Gaussian-weighted cross-entropy highest at click location. Post-Interaction (PI) and Mid-Interaction (MI) adaptation.
- **Results:** PI+MI achieves **25.5pp improvement** on out-of-distribution brain MRI after 10 corrective clicks (BRATS T1: 62.5% → 88.0% Dice at T=10). Latency: MI=0.05s GPU / 0.25s CPU; PI=0.09s GPU / 0.41s CPU. **Works on CPU.**
- **Proofreading relevance:** Directly addresses the core challenge of deploying interactive segmentation across scanners and protocols. The model adapts session-by-session using the annotator's own clicks. CPU compatibility means it could run on annotation workstations without GPU.
- **Limitations:** 2D only. Simple U-Net base. Tested only on fundus and brain pathologies.

---

## 3. Extended Landscape: Interactive Segmentation Methods (Post-Jan 2024)

### 3.1 Foundation Model–Based Interactive Segmentation

| Method | Venue/Year | Prompts | Native 3D | Slicer | Key Capability |
|--------|-----------|---------|-----------|--------|----------------|
| **nnInteractive** | CVPR 2025 Challenge 1st | Points, scribbles, boxes, lasso, mask | Yes | SlicerNNInteractive | Best 3D accuracy; prior mask for proofreading |
| **MedSAM2** | arXiv Apr 2025 | Points, boxes, masks | Via video propagation | No | 455K+ training volumes; slice-to-slice propagation |
| **ScribblePrompt** | ECCV 2024; Award at CVPR 2024 DCAMI | Scribbles, clicks, boxes | No (2D) | No | 28% annotation time reduction; 65 datasets |
| **VISTA3D** | CVPR 2025 | Class labels + clicks | Yes | Remote VISTA3D module | Dual auto+interactive; 127 CT classes |
| **SegVol** | NeurIPS 2024 Spotlight | Text + points + boxes | Yes (CT) | No | Text-guided; zoom-in-zoom-out inference |
| **SAM-Med3D** | ECCV 2024 Oral | 3D points | Yes | No | Native 3D architecture; 10–100x fewer prompts vs 2D |
| **ProtoSAM-3D** | Med Image Anal 2025 | Auto-prompt + clicks | Yes | No | Mask-level prototypes; zero/few-shot multi-organ |
| **MedCLIP-SAM** | MICCAI 2024 | Text (CLIP) | No | No | Zero-shot text-prompted segmentation |
| **μSAM** | Nature Methods 2024 | Points, boxes, scribbles | Partial (z-stacks) | napari only | Microscopy specialist |
| **ENSAM** | arXiv Sep 2025 | Points, boxes | Yes | No | Trained from scratch on <5K volumes in 6h |
| **PRISM** | MICCAI 2024 Spotlight | Points, boxes, scribbles, mask | Yes | No | Corrective refinement from prior mask; >93% Dice |
| **K-Prism** | arXiv Sep 2025 | Semantic + in-context + clicks | No (2D) | No | Mode chaining reduces NoC90 >30% |
| **OAIMS** | arXiv Mar 2025 | Clicks | No (2D) | No | Online adaptation; CPU-viable; +25pp on OOD |
| **SAM-REF** | CVPR 2025 | Points + detail refiner | No (2D) | No | Fine-grained boundary detail at click locations |
| **IKAN** | MICCAI 2025 | Clicks (KAN-based) | Partial | No | Addresses click responsiveness loss in deep nets |

### 3.2 ScribblePrompt — Detailed Analysis

**Wong et al. (2024).** ECCV 2024. MIT CSAIL. Bench-to-Bedside Paper Award at CVPR 2024 DCAMI Workshop.

ScribblePrompt trains on **simulated realistic scribbles** across 65 open-access biomedical datasets (54,000 images, 16 modality types). The key innovation is the scribble simulation algorithm — existing methods used unrealistic straight-line or random-walk scribbles that do not match how humans actually draw.

- **User study:** 28% reduction in annotation time, +15% Dice vs. next-best method.
- **MedScribble benchmark:** Curated evaluation set for scribble-based interactive segmentation.
- **Limitation for our use case:** 2D only, no 3D propagation. No Slicer or napari plugin. Would need custom integration for volumetric proofreading.
- **GitHub:** halleewong/ScribblePrompt

### 3.3 SAM2 / MedSAM2 for 3D Medical Images

The SAM2 approach to 3D volumes: treat consecutive slices as video frames, propagate annotations via the streaming memory bank.

**MedSAM2 (Bo Wang Lab, April 2025):**
- Fine-tunes SAM2's full architecture on 455K+ 3D image-mask pairs (363K CT, 77K MRI, 14K PET) + 76K video frames.
- Bidirectional propagation outperforms unidirectional.
- Best practice: prompt the slice with the **largest cross-section** of the target.
- **Key finding:** Direct transfer of SAM2 to medical imaging "is suboptimal and may even fail to converge" — medical spatial dependencies differ fundamentally from video temporal dependencies.
- No Slicer plugin — research code only (bowang-lab/MedSAM2).

### 3.4 Agentic Segmentation — Emerging Direction

Beyond MedSAM-Agent, **MedSAM-3 Agent** (November 2025) uses a medically fine-tuned SAM3 backbone + MLLM for iterative text-driven segmentation:
- MLLM plans prompts, evaluates outputs, provides semantic feedback.
- Usually converges in ≤3 rounds.
- BUSI Dice 0.8064 with Gemini 3 Pro as the MLLM.

This represents the leading edge: VLMs/LLMs automate the prompting strategy itself. Still early-stage, not practical for production annotation in 2026.

---

## 4. Annotation Platform Comparison

### 4.1 3D Slicer — The Pragmatic Choice

**Current AI-assisted segmentation extensions (as of Feb 2026):**

| Extension | Backend Model | Install Method | GPU Required (Client) | Status |
|-----------|--------------|----------------|----------------------|--------|
| **SlicerNNInteractive** | nnInteractive | GitHub (manual) | No (remote server) | Active development |
| **SlicerSegmentWithSAM** | SAM + SAM2 | Extensions Manager | Yes (local) | Stable |
| **MONAI Label** | DeepEdit, custom | Extensions Manager | No (remote server) | Stable, v0.8.5 |
| **Remote VISTA3D** | VISTA-3D | Custom module | No (remote server) | Community |
| **SlicerTotalSegmentator** | TotalSegmentator v2 | Extensions Manager | No (CPU mode) | Mature |
| **SlicerTomoSAM** | SAM | GitHub | Yes | Community |

**Advantages for vascular proofreading:**
- Industry-standard 3D medical image viewer in research (thousands of publications)
- Native DICOM/NIfTI/NRRD support
- Segment Editor with manual correction tools (paint, threshold, scissors, islands)
- Client-server architecture (SlicerNNInteractive, MONAI Label) decouples GPU from workstation
- Multi-platform (Windows, macOS, Linux) for client
- Extensible via Python — custom proofreading workflows possible

**Disadvantages:**
- Heavy desktop application (~500MB)
- Not web-accessible (no remote browser annotation)
- Extension ecosystem quality varies (some unmaintained)
- SlicerNNInteractive not yet in official Extensions Manager

### 4.2 napari

**Status:** Rapidly growing, Python-native, CZI-funded.

- **napari-nninteractive** (DKFZ-maintained): Best-in-class interactive DL segmentation.
- **SuRVoS2:** Specifically for vascular segmentation in tomographic data (supervoxel-based).
- **μSAM:** For microscopy/EM.

**Verdict:** Strong alternative if the workflow is Python-scripted. Less polished clinical UI. Good for microscopy vascular data (light-sheet, 2-photon). For clinical CT/MRI, 3D Slicer's DICOM handling is superior.

### 4.3 OHIF Viewer (Web-Based)

**OHIF-AI (CCI Bonn, 2025):** Integrates nnInteractive, SAM2, MedSAM2, SAM3, VoxTell, MedGemma in the browser. Points, scribbles, lassos, bounding boxes. Live inference + 3D propagation. Text prompts via MedGemma.

**OHIF v3.10 (2025):** Two AI segmentation tools running entirely in-browser (no server).

**Verdict:** Excellent for multi-site collaborative annotation. Zero-install. DICOM-native. However, browser GPU limitations constrain model size, and the extension ecosystem is less mature than Slicer for 3D manual editing tools.

### 4.4 ITK-SNAP

Still actively used (>8,000 citations). nnInteractive extension available. Good for lightweight annotation. Lacks modern DL integration out-of-the-box.

### 4.5 Custom Voxel Frontend — Why Not

Building a custom 3D voxel annotation frontend would require:
- 3D rendering engine (VTK, Three.js, or similar)
- DICOM/NIfTI I/O
- Multi-planar reconstruction (axial, coronal, sagittal views)
- Volume rendering
- 3D interaction primitives (point placement, lasso, scribble in 3D space)
- Undo/redo, mask I/O, label management
- Integration with interactive segmentation backends

**Estimated effort:** 6–12 months for a single developer. 3D Slicer has ~20 years of development and hundreds of contributors. The cost-benefit ratio overwhelmingly favours using Slicer.

### 4.6 Platform Recommendation Matrix

| Use Case | Recommended Platform |
|----------|---------------------|
| Clinical CT/MRI, solo researcher | 3D Slicer + SlicerNNInteractive |
| Python-native research, scripted | napari + napari-nninteractive |
| Multi-site collaboration, web | OHIF-AI |
| Microscopy vascular (light-sheet) | napari + μSAM or SuRVoS2 |
| Lightweight, minimal setup | ITK-SNAP + nnInteractive |
| Custom from scratch | **Not recommended** — use Slicer |

---

## 5. MONAI Label — Current State and Limitations

**Version:** v0.8.5 (actively maintained, Python 3.12 compatibility in progress).
**Published:** Medical Image Analysis journal, 2024.

### What MONAI Label Does Well
- **Active learning loop:** Scores unlabeled images by uncertainty, surfaces most informative for annotation, retrains on corrections.
- **Multi-viewer support:** 3D Slicer, OHIF, CVAT clients.
- **Integrated retraining:** Corrections feed directly into model updates.
- **Apps:** Radiology (DeepEdit, DeepGrow), Pathology (NuClick), Endoscopy.

### Where MONAI Label Falls Short (2026)
- **Interactive segmentation quality:** nnInteractive substantially outperforms MONAI Label's DeepEdit/DeepGrow for interactive correction quality. DeepEdit has not been updated to match nnInteractive's architecture innovations.
- **No ScribblePrompt, nnInteractive, or MedSAM2 integration:** New methods require separate extensions.
- **Complex deployment:** Server-client setup is painful for non-technical users. Docker or local Python environment required.
- **Underutilised active learning:** Most users treat it as a one-shot auto-segmenter, not leveraging the iterative learning loop.
- **Competition:** SlicerNNInteractive, SlicerSegmentWithSAM, and OHIF-AI are easier to set up for pure interactive segmentation.

### Verdict
MONAI Label's value proposition has shifted from "best interactive segmentation tool" to "best active learning orchestrator." For a workflow where the goal is iterative dataset improvement (annotate → retrain → re-annotate), MONAI Label remains relevant. For pure proofreading of existing segmentations, nnInteractive is superior.

---

## 6. The Proofreading Workflow — Design Principles

The literature converges on a standard proofreading paradigm:

### 6.1 Annotation by Iterative Deep Learning (AID)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Train model  │────▶│ Auto-segment │────▶│ Human review │
│ (initial)    │     │ (unlabelled) │     │ + correction │
└─────────────┘     └──────────────┘     └──────┬──────┘
       ▲                                         │
       │           ┌──────────────┐              │
       └───────────│ Update train │◀─────────────┘
                   │ dataset      │
                   └──────────────┘
```

### 6.2 Key Design Principles from Literature

1. **Pre-loaded overlay:** Show the initial mask as an overlay, not a blank canvas (Guo et al., 2025: >30% click reduction).
2. **Sparse click corrections:** Support point/scribble corrections, not redrawing from scratch (nnInteractive: all prompt types).
3. **3D propagation:** A single corrective interaction on one slice should update the 3D mask (nnInteractive: lasso → full 3D; MedSAM2: bidirectional propagation).
4. **Sub-second inference:** <1s per interaction is critical for usability (OAIMS: 0.05–0.25s; nnInteractive: interactive response times).
5. **Online adaptation:** Model improves during the annotation session (OAIMS: PI+MI adaptation; Schon 2026: TTA from interactions).
6. **Error visualisation:** Highlight likely false positives/negatives to guide the annotator (VessQC: uncertainty-guided curation, recall 67%→94%).
7. **Undo/redo:** Essential for iterative refinement.

### 6.3 Proposed MinIVess Proofreading Workflow

```
Phase 1: Automatic Segmentation
  ├── Run DynUNet/vesselFM on new volumes
  ├── Export NIfTI masks
  └── Quality score (uncertainty, connected component analysis)

Phase 2: Triage
  ├── Sort volumes by quality score (worst first)
  ├── Flag volumes with disconnected components, thin breaks
  └── Prioritise high-uncertainty regions

Phase 3: Interactive Proofreading (3D Slicer + SlicerNNInteractive)
  ├── Load volume + auto-segmentation overlay
  ├── Navigate to flagged regions
  ├── Place corrective prompts (points, scribbles, lasso)
  ├── nnInteractive generates updated 3D mask
  ├── Accept/reject/further-refine
  └── Export corrected mask

Phase 4: Dataset Update + Retraining (Optional: MONAI Label)
  ├── Corrected masks enter training set
  ├── Retrain model (active learning selects next batch)
  └── Repeat from Phase 1 with improved model
```

---

## 7. Vascular-Specific Considerations

### 7.1 Thin Tubular Structure Challenges

All interactive segmentation methods face specific challenges with vascular structures:

- **SAM variants struggle with thin structures:** SAM2 benchmarks (Ma et al., 2024) confirm limitations on tubular anatomy.
- **All interactive methods degrade on small structures:** AtlasSegFM evaluation (arXiv:2512.18176) shows even nnInteractive drops to ~27% Dice on organs-at-risk, and VISTA-3D's interactive branch performance on thin tubular structures is not well-characterised in its own paper. This underscores that no current interactive segmentation model is specifically optimised for vessels.
- **Topology matters more than clicks:** A single wrong voxel can break vessel connectivity. Topology-aware losses (clDice, Skeleton Recall) have 5–15x more impact than architecture choice (Phase 12 research).
- **nnInteractive is the best current option despite limitations:** Trained on 120+ diverse datasets including microscopy vessels. The nnUNet backbone is the same architecture family that dominates vascular segmentation challenges. However, vessel-specific fine-tuning or topology-aware post-processing may be needed.

### 7.2 VessQC — Uncertainty-Guided Curation

Terms et al. (2025) developed VessQC, a napari plugin for uncertainty-guided 3D segmentation curation:
- Improved error detection recall from 67% to 94%
- Uses model uncertainty to flag likely errors before human review
- Directly complementary to the proofreading workflow (Phase 2: Triage)

### 7.3 SERVAL — Retinal Vessel Assessment

SERVAL (Scientific Reports, 2025) integrates DL-based auto-initialisation + subpixel centreline refinement + interactive editing in a single GUI for retinal vessel assessment. Relevant conceptual model for the proofreading UI, though specific to 2D retinal imaging.

---

## 8. Benchmarks and Challenges

### 8.1 CVPR 2025: Foundation Models for Interactive 3D Biomedical Image Segmentation
- 5 modalities: CT, MR, PET, Ultrasound, Microscopy
- **Winner:** nnInteractive (1st place)
- Codabench competition with standardised evaluation

### 8.2 IMed-361M / IMIS-Bench (CVPR 2025)
- Largest interactive medical segmentation benchmark
- 6.4M images, 361M+ masks, 14 modalities, 204 segmentation targets
- Supports clicks, boxes, text, and combinations

### 8.3 MedScribble (ECCV 2024)
- Curated benchmark for scribble-based interactive segmentation
- Accompanies ScribblePrompt

---

## 9. Gap Analysis: What Is Missing

### 9.1 No vessel-specific interactive segmentation
No method explicitly optimises interactive prompting for tubular structures. nnInteractive trains on diverse data including some vessels, but there is no topology-aware interactive model (e.g., one that preserves vessel connectivity during iterative corrections).

### 9.2 No 3D online adaptation
OAIMS (Xu et al., 2025) demonstrates powerful online adaptation but only in 2D. Combining 3D interactive segmentation (nnInteractive) with session-level adaptation (OAIMS) is an open research direction.

### 9.3 Limited proofreading-specific evaluation
Most benchmarks evaluate from-scratch interactive segmentation. There is no standard benchmark for the proofreading scenario (given imperfect initial mask, measure clicks-to-correct). K-Prism's mode-chaining evaluation (Guo et al., 2025) is the closest.

### 9.4 No agentic proofreading in Slicer
MedSAM-Agent (Liu et al., 2026) and MedSAM-3 Agent show that LLMs can learn optimal prompting strategies, but no tool integrates this into a clinical annotation platform.

---

## 10. Recommendations for MinIVess PRD

### 10.1 annotation_platform Decision Node — Proposed Updates

**Add new option: `slicer_nninteractive`**

| Option | Prior (Current) | Prior (Updated) | Rationale |
|--------|----------------|-----------------|-----------|
| label_studio | 0.35 | 0.15 | Demoted — 2D-focused, no 3D interactive AI |
| monai_label | 0.35 | 0.20 | Retained for active learning; surpassed for interactive |
| cvat | 0.15 | 0.05 | 3D medical support still limited |
| hybrid | 0.15 | 0.10 | Slicer + MONAI Label hybrid still viable |
| **slicer_nninteractive** | — | **0.40** | Best 3D interactive segmentation; client-server; Slicer ecosystem |
| **napari_nninteractive** | — | 0.10 | Strong Python-native alternative for scripted workflows |

**New references to add:**
- `isensee2025nninteractive`: nnInteractive (CVPR 2025 challenge winner)
- `devente2025slicernninteractive`: SlicerNNInteractive (3D Slicer extension)
- `wong2024scribbleprompt`: ScribblePrompt (ECCV 2024; Bench-to-Bedside Award at CVPR 2024 DCAMI Workshop)
- `ma2025medsam2`: MedSAM2 (455K training volumes, video propagation)

### 10.2 annotation_workflow Decision Node — Proposed Updates

**Add new option: `slicer_proofread`**

| Option | Prior (Current) | Prior (Updated) | Rationale |
|--------|----------------|-----------------|-----------|
| monai_label_active | 0.35 | 0.20 | Still best for active learning loop |
| label_studio_manual | 0.30 | 0.10 | Demoted — not suited for 3D proofreading |
| hybrid | 0.20 | 0.15 | MONAI Label + Slicer proofreading |
| none | 0.15 | 0.10 | Pre-existing datasets |
| **slicer_proofread** | — | **0.45** | Auto-segment → Slicer proofread → retrain loop |

### 10.3 segmentation_models Decision Node — Reference Update

Add interactive segmentation context to `sam_variants` description:
- nnInteractive uses nnUNet backbone (same as DynUNet) — validates the architecture family for both training and interactive correction.
- AtlasSegFM evaluation (arXiv:2512.18176) shows all interactive methods including nnInteractive degrade on small structures (~27% Dice on organs-at-risk), reinforcing that vessel-specific models and topology-aware losses are needed alongside interactive tools.

### 10.4 New Decision Node: `interactive_segmentation_backend`

Consider creating a new L3-technology decision node for the interactive segmentation model specifically:

| Option | Prior | Rationale |
|--------|-------|-----------|
| nninteractive | 0.50 | CVPR 2025 winner; 3D native; all prompt types; Slicer integration |
| medsam2 | 0.20 | Large training data; video propagation for 3D |
| monai_deepedit | 0.10 | Legacy; integrated with MONAI Label active learning |
| scribbleprompt | 0.10 | Best scribble interaction; 2D only |
| vista3d_interactive | 0.10 | 127-class auto + interactive; small-structure concerns |

---

## 11. Literature Coverage Assessment

### 11.1 Top Venue Papers Covered (Post-Jan 2024)

| Paper | Venue | Year | Covered |
|-------|-------|------|---------|
| nnInteractive | CVPR 2025 Challenge 1st | 2025 | Yes (seed) |
| SlicerNNInteractive | JOSS (submitted) | 2025 | Yes (seed) |
| ScribblePrompt | ECCV 2024 | 2024 | Yes (web) |
| VISTA3D | CVPR 2025 | 2025 | Yes (web) |
| SegVol | NeurIPS 2024 Spotlight | 2024 | Yes (web) |
| SAM-Med3D | ECCV 2024 Oral | 2024 | Yes (web) |
| μSAM | Nature Methods 2024 | 2024 | Yes (web) |
| MedCLIP-SAM | MICCAI 2024 | 2024 | Yes (web) |
| ProtoSAM-3D | Medical Image Analysis 2025 | 2025 | Yes (web) |
| MedSAM2 | arXiv 2025 | 2025 | Yes (web) |
| K-Prism | arXiv 2025 | 2025 | Yes (seed) |
| OAIMS | arXiv 2025 | 2025 | Yes (seed) |
| MedSAM-Agent | arXiv 2026 | 2026 | Yes (seed) |
| Schon dissertation | Univ. Augsburg 2026 | 2026 | Yes (seed) |
| MONAI Label | Medical Image Analysis 2024 | 2024 | Yes (web) |
| IMed-361M | CVPR 2025 | 2025 | Yes (web) |
| SlicerSegmentWithSAM | arXiv 2024 | 2024 | Yes (web) |
| PRISM | MICCAI 2024 Spotlight | 2024 | Yes (reviewer) |
| SAM-REF | CVPR 2025 | 2025 | Yes (reviewer) |
| IKAN | MICCAI 2025 | 2025 | Yes (reviewer) |
| ENSAM | arXiv 2025 | 2025 | Yes (web) |

### 11.2 Known Gaps

- **MICCAI 2025 vascular-specific papers:** Graph-PAVNet (pulmonary artery/vein separation with graph-based interactive correction) and ISAC (vascular segmentation as mask completion from sparse annotations) were identified but not deeply analysed. Both are directly relevant to vascular proofreading.
- **MIDL 2025/2026:** Limited coverage — MIDL papers on interactive segmentation not found in searches.
- **Domain-specific tools:** Neuroscience proofreading tools (VAST, Neuroglancer-based) not deeply covered as they target connectomics, not clinical vascular imaging.

---

## 12. Summary of Key Evidence

| Claim | Evidence | Source |
|-------|----------|--------|
| nnInteractive is SOTA for 3D interactive segmentation | 1st place CVPR 2025 challenge | Isensee et al. (2025) |
| Proofreading from existing mask reduces clicks >30% | Mode chaining experiment | Guo et al. (2025) |
| Online adaptation recovers 25pp on OOD data | BRATS T1 experiment | Xu et al. (2025) |
| All interactive methods degrade on small structures (~27% Dice on organs-at-risk) | AtlasSegFM evaluation | arXiv:2512.18176 |
| ScribblePrompt reduces annotation time 28% | User study, 65 datasets | Wong et al. (2024) |
| Topology-aware loss > architecture choice for vessels | Phase 12 research, 5–15x impact | Multiple (Shit et al., 2021; Kirchhoff et al., 2024) |
| VessQC improves error recall 67%→94% | Uncertainty-guided curation | Terms et al. (2025) |
| MONAI Label v0.8.5 still maintained | PyPI, GitHub | Project-MONAI (2026) |
| Client-server decouples GPU from annotation | SlicerNNInteractive architecture | de Vente et al. (2025) |
| MedSAM2 treats volumes as spatial videos | 455K training volumes | Bo Wang Lab (2025) |

---

## References

- de Vente, C. et al. (2025). SlicerNNInteractive: A 3D Slicer extension for nnInteractive. arXiv:2504.07991.
- Guo, B. et al. (2025). K-Prism: Knowledge-Driven Medical Image Segmentation with Unified Prompting. arXiv:2509.25594.
- Isensee, F. et al. (2025). nnInteractive: Redefining 3D Promptable Segmentation. arXiv:2503.08373.
- Liu, J. et al. (2026). MedSAM-Agent: Towards Agentic Medical Image Segmentation via Reinforcement Learning. arXiv:2602.03320.
- Ma, J. et al. (2025). MedSAM2: Segment Anything in 3D Medical Images and Videos. arXiv:2504.03600.
- Mazurowski Lab (2024). SAM & SAM 2 in 3D Slicer: SegmentWithSAM Extension. arXiv:2408.15224.
- Schon, R. (2026). From Unimodal to Multimodal: Extending Interactive Segmentation for Multi-Sensor Scenarios. PhD dissertation, University of Augsburg.
- Terms, J. et al. (2025). VessQC: Uncertainty-guided 3D segmentation curation. napari plugin.
- Wong, H. et al. (2024). ScribblePrompt: Fast and Flexible Interactive Segmentation for Any Biomedical Image. ECCV 2024.
- Xu, J. et al. (2025). Online Adaptation for Interactive Medical Image Segmentation under Distribution Shift. arXiv:2503.06717.
- AtlasSegFM (2025). arXiv:2512.18176. Evaluation showing interactive segmentation degradation on small structures/organs-at-risk.
- Li, H. et al. (2024). PRISM: A Promptable and Robust Interactive Segmentation Model with Visual Prompts. MICCAI 2024 Spotlight. arXiv:2404.15028.
- Yu, Z. et al. (2025). SAM-REF: Introducing Image-Prompt Synergy during Interaction for Detail Enhancement in the Segment Anything Model. CVPR 2025. arXiv:2408.11535.
- He, Y. et al. (2024). VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging. arXiv:2406.05285.
- Archit, A. et al. (2024). Segment Anything for Microscopy. Nature Methods.
- Du, Y. et al. (2024). SegVol: Universal and Interactive Volumetric Medical Image Segmentation. NeurIPS 2024 Spotlight.
- Wang, H. et al. (2024). SAM-Med3D: Towards General-Purpose Segmentation Models for Volumetric Medical Images. ECCV 2024.
- Lévy, J. et al. (2025). ProtoSAM-3D: Interactive semantic segmentation in volumetric medical imaging. Medical Image Analysis.
- Ko, J. et al. (2024). MedCLIP-SAM: Bridging Text and Image Towards Universal Medical Image Segmentation. MICCAI 2024.
- Diaz-Pinto, A. et al. (2024). MONAI Label: A framework for AI-assisted Interactive Image Annotation. Medical Image Analysis.
