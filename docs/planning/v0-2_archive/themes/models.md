---
theme: models
doc_count: 31
last_synthesized: "2026-03-22"
implementation_health: partial
kg_domains: [models, training]
---

# Theme: Models -- SAM3, MambaVesselNet, VesselFM, DynUNet

This theme covers the 6-model paper lineup for the Nature Protocols manuscript:
DynUNet (baseline), MambaVesselNet++ (CNN-SSM hybrid), SAM3 Vanilla (zero-shot),
SAM3 TopoLoRA (topology-aware fine-tuning), SAM3 Hybrid (full fine-tuning), and
VesselFM (foundation model zero-shot). It includes model adapter architecture,
VRAM requirements, implementation sagas (especially SAM3), capacity matching,
and the advanced segmentation literature survey.

---

## Key Scientific Insights

### 1. The 6-Model Paper Lineup

The paper compares 6 models spanning 3 paradigms (trained-from-scratch, foundation model
fine-tuning, zero-shot) across 3 loss functions and 2 calibration settings:

| Model | Paradigm | Training | VRAM | Adapter Status |
|-------|----------|----------|------|---------------|
| DynUNet | MONAI-native from scratch | Full training | ~4 GB | COMPLETE |
| MambaVesselNet++ | CNN-SSM hybrid | Full training | ~6 GB | COMPLETE (code), GPU benchmark pending |
| SAM3 Vanilla | Foundation zero-shot | None (eval only) | ~16 GB | COMPLETE |
| SAM3 TopoLoRA | Foundation + LoRA | LoRA fine-tuning | ~18 GB | COMPLETE (code), Conv2d skip fix applied |
| SAM3 Hybrid | Foundation + full | Full fine-tuning | ~20 GB | COMPLETE (code), cloud-only |
| VesselFM | Domain-specific FMm | None (eval only) | ~16 GB | COMPLETE (code) |

Non-paper models (SegResNet, SwinUNETR, UNETR, AttentionUNet, COMMA-Mamba, ULike-Mamba)
were REMOVED 2026-03-19 to reduce maintenance burden.

**Source:** `knowledge-graph/domains/models.yaml`, `advanced-segmentation-double-check-plan.md`

### 2. SAM3 = Meta's Segment Anything Model 3 (NOT SAM2)

The most expensive mistake in the project's history: the initial SAM3 implementation was
based on SAM2 (video model) instead of SAM3 (concept-based segmentation, Nov 2025,
github.com/facebookresearch/sam3). This consumed multiple sessions before detection.
SAM3 uses a ViT-32L encoder (648M params) and ALWAYS requires real pretrained weights --
the `_StubSam3Encoder` was permanently removed on 2026-03-07.

Key constraint: SAM3 requires BF16 (not FP16). FP16 max = 65504, and SAM3's encoder
produces values that overflow FP16 during validation. This is why T4 (Turing, no BF16)
is banned -- only L4 (Ada Lovelace) or newer architectures with BF16 support work.

**Source:** `sam3-literature-research-report.md`, `sam3-installation-issues-and-synthesis.md`, metalearning `2026-03-02-sam3-implementation-fuckup.md`

### 3. SAM3 val_loss=NaN Was a Sentinel, Not Real NaN

Across 7 runs on RunPod/GCP, sam3_hybrid reported val_loss=NaN in MLflow. Eight hypotheses
were investigated. The root cause for ALL observed NaN was the validation sentinel --
validation was never actually executed:
- Run 1 (RunPod): `debug: true` forced `max_epochs=1`, sentinel set `val_interval=6 > 1`
- Runs 3-7 (GCP): SkyPilot YAML selected wrong config with `val_interval: 3 > max_epochs: 2`
- Run 2 (correct config): `val_interval=1`, AMP OFF -> val_loss=0.7264 (FINITE)

Two latent risks mitigated: FP16->BF16 dtype correction, AMP+3D NaN isolation.

**Source:** `sam3-val-loss-final-report.md`, `sam3-nan-loss-fix.md`

### 4. MambaVesselNet++ Is Iso-Param With DynUNet

The capacity matching study measured actual parameter counts (not estimates):
- DynUNet [32,64,128,256]: 5.641M (not "15-20M" as initially estimated)
- MambaVesselNet++ init_filters=32: comparable scale (within 20%)

This changes the framing from "efficiency comparison" to "architectural comparison at
comparable scale." The CNN-SSM hybrid with bidirectional Mamba layers and multi-scale
fusion provides different inductive biases than the U-Net encoder-decoder, not necessarily
better parameter efficiency.

The implementation requires `mamba-ssm` compiled with CUDA, which means a Docker base
image rebuild with `INSTALL_MAMBA=1`. This was one of the 12 glitches from the 1st
factorial pass (mambavesselnet conditions failed because mamba-ssm was not compiled).

**Source:** `mamba-model-capacity-matching.md`, `mambavesselnet-implementation-plan.xml`

### 5. DynUNet Is the Proven Workhorse

DynUNet (MONAI's nnU-Net architecture) was selected as the primary baseline based on
challenge results (SMILE-UHURA, TopCoW, KiTS, Medical Segmentation Decathlon), Isensee
et al. MICCAI 2024 benchmarks, and practical MONAI integration. Key feature: topology-
aware compound loss (Dice + CE + clDice) achieves 0.906 clDice on microvasculature.

Width ablation study: FULL [32,64,128,256] vs HALF [16,32,64,128] vs QUARTER [8,16,32,64]
was implemented to understand the capacity/performance tradeoff on the small MiniVess
dataset (70 volumes).

**Source:** `monai-segmentor-model-background-research.md`, `dynunet-ablation-plan.md`

### 6. VesselFM: Domain-Specific Foundation Model for Vessels

VesselFM (Wittmann et al. 2024) is a DynUNet pre-trained on 17 vascular datasets
(including MiniVess). It serves as a zero-shot baseline -- evaluated on DeepVess test
set without any fine-tuning. Architecture: DynUNet with [32,64,128,256,320,320] filters,
binary segmentation with sigmoid. Downloads checkpoint from HuggingFace (bwittmann/vesselFM).

**Source:** `vesselfm-plan.md`, `foundation-model-finetuning-plan.md`

### 7. TopoLoRA: Topology-Aware LoRA for SAM3

TopoLoRA (Khazem et al. 2025) applies LoRA specifically to preserve vessel topology
during SAM3 fine-tuning. Implementation note: the 1st factorial pass revealed that LoRA
was being applied to Conv2d layers (which SAM3's encoder contains), causing failures.
Fix: skip Conv2d in `_apply_lora_to_encoder()`, apply only to Linear layers.

The diff-parameterized compound loss report explores how topology-aware losses (clDice,
CAPE, Betti matching) interact with LoRA fine-tuning -- different losses may need
different LoRA targeting strategies.

**Source:** `topolora-sam3-planning-report.md`, `topolora-sam3-planning-report-diff-parameterized-compound-loss.md`

### 8. The Advanced Segmentation Literature Survey (44+ Papers)

The double-check plan surveyed 44+ papers across 8 areas through the lens of "what should
an MLOps PLATFORM implement to accelerate researcher productivity" (not model SOTA):
1. Uncertainty methods for foundation models
2. Conformal prediction + distribution shifts
3. Model merging methods (model soups, TIES)
4. 3D Mamba models with uncertainty
5. Synthetic data generation
6. Data quality and annotator pipelines
7. Latent diffusion for segmentation
8. VLMs/MLLMs for 3D medical segmentation

Key finding: Soft Dice Confidence (Borges 2025) is an O(n) model-agnostic deployment
quality gate that does not require ensembles or MC dropout.

**Source:** `advanced-segmentation-double-check-plan.md`, `advanced-segmentation-execution-plan.xml`

### 9. Planned but Not Implemented Models

Several model plans exist but were not implemented or were explicitly excluded:

| Model | Plan | Status | Reason |
|-------|------|--------|--------|
| AtlasSegFM | atlassegfm-plan.md | planned | One-shot atlas-guided customization |
| COMMA-Mamba | comma-mamba-plan.md | removed | Replaced by MambaVesselNet++ |
| MedSAM3 annotation | medsam3-annotation-plan.md | planned | Interactive annotation adapter |
| LoRA generic wrapper | foundation-model-finetuning-plan.md | implemented | PEFT LoRA for any adapter |

**Source:** Individual plan documents

---

## Architectural Decisions Made

| Decision | Outcome | Source Doc | KG Node |
|----------|---------|-----------|---------|
| Primary 3D model | DynUNet (MONAI-native, no custom code) | monai-segmentor-model-background-research.md | models.primary_3d_model |
| Foundation model | SAM3 (LoRA, Hybrid, Vanilla variants) | sam3-literature-research-report.md | models.foundation_model |
| Mamba architecture | MambaVesselNet++ (Chen et al. 2024/2025) | mambavesselnet-implementation-plan.xml | models.mamba_architecture |
| Paper model lineup | 6 models (DynUNet, MVN++, SAM3x3, VesselFM) | advanced-segmentation-double-check-plan.md | models.paper_model_comparison |
| SAM3 precision | BF16 mandatory, FP16 banned, T4 banned | sam3-val-loss-final-report.md | -- |
| Model adapter ABC | All models via ModelAdapter interface | foundation-model-finetuning-plan.md | models.primary_3d_model |
| Stub removal | _StubSam3Encoder permanently removed 2026-03-07 | sam3-stub-removal.xml | -- |
| Non-paper models | REMOVED (SegResNet, SwinUNETR, UNETR, etc.) | models.yaml | -- |
| VRAM enforcement | sam3_vram_check.py at model build time | sam3-installation-issues-and-synthesis.md | -- |

---

## Implementation Status

| Document | Type | Status | Key Deliverable |
|----------|------|--------|-----------------|
| advanced-segmentation-double-check-plan.md | plan | reference | 44-paper literature survey |
| advanced-segmentation-double-check-prompt.md | prompt | reference | User prompt for advanced segmentation |
| advanced-segmentation-execution-plan.xml | execution_plan | partial | P0-P2 improvements (SDC, conformal, ensembles) |
| atlassegfm-plan.md | plan | planned | Atlas-guided one-shot customization |
| comma-mamba-plan.md | plan | removed | Replaced by MambaVesselNet++ |
| dynunet-ablation-plan.md | plan | implemented | Width ablation (FULL/HALF/QUARTER) |
| dynunet-evaluation-plan.xml | execution_plan | implemented | DynUNet evaluation pipeline |
| foundation-model-finetuning-plan.md | plan | implemented | Generic LoRA wrapper |
| mamba-model-capacity-matching.md | document | reference | Measured param counts |
| mambavesselnet-implementation-plan.xml | execution_plan | implemented | MambaVesselNet++ adapter (PR #748) |
| mambavesselnet-overnight-optimized.xml | execution_plan | executed | Optimized overnight MambaVesselNet |
| mambavesselnet-overnight-runner.sh | script | executed | MambaVesselNet overnight script |
| mambavesselnet-test-on-dev-runpod-followup.xml | execution_plan | executed | RunPod follow-up test |
| mambavesselnet-test-on-dev-runpod.xml | execution_plan | executed | RunPod initial test |
| medsam3-annotation-plan.md | plan | planned | Interactive annotation adapter |
| monai-segmentor-model-background-research.md | document | reference | MONAI model selection survey |
| overnight-mambavesselnet.yaml | document | executed | Overnight MambaVesselNet config |
| sam3-cold-start-opus4.yaml | cold_start | reference | SAM3 session bootstrap config |
| sam3-implementation-plan.xml | execution_plan | implemented | SAM3 adapter implementation |
| sam3-installation-issues-and-synthesis-plan.xml | execution_plan | implemented | SAM3 installation fix plan |
| sam3-installation-issues-and-synthesis.md | synthesis | reference | SAM3 complete saga retrospective |
| sam3-literature-research-report.md | research_report | reference | SAM3 geometry-architecture analysis |
| sam3-nan-loss-fix.md | document | implemented | NaN sentinel root cause + fix |
| sam3-real-data-e2e-plan.xml | execution_plan | partial | SAM3 real data E2E validation |
| sam3-stub-removal.xml | execution_plan | implemented | Permanent stub removal |
| sam3-training-reference.md | document | reference | SAM3 training quick-start guide |
| sam3-val-loss-final-report.md | research_report | reference | 8-hypothesis NaN investigation |
| skypilot-compute-offloading-plan-for-vesselfm-sam3-and-synthetic-generation.xml | execution_plan | partial | Cloud compute for large models |
| topolora-sam3-planning-report-diff-parameterized-compound-loss.md | research_report | reference | TopoLoRA + compound loss interactions |
| topolora-sam3-planning-report.md | research_report | reference | TopoLoRA planning + checkpoint arch |
| vesselfm-plan.md | plan | implemented | VesselFM zero-shot adapter |

---

## Cross-References

- **Evaluation theme:** Models are factorial factors (4 models x 3 losses x 2 calibration)
- **Infrastructure theme:** Docker images package model deps, VRAM profiling
- **Cloud theme:** SAM3/VesselFM require cloud GPU (>=16 GB), T4 banned
- **Harness theme:** Model lineup tracked in KG, cold-start prompts for model sessions
- **KG domain:** `models.yaml` -- primary_3d_model, foundation_model, mamba_architecture, paper_model_comparison
- **Key metalearning:** `2026-03-02-sam3-implementation-fuckup.md`, `2026-03-15-t4-turing-fp16-nan-ban.md`, `2026-03-17-model-lineup-ignorance-massive-fuckup.md`, `2026-03-15-amp-validation-nan-3d.md`
- **Adapters directory:** `src/minivess/adapters/` -- 28 files including dynunet.py, sam3*.py, mambavesselnet.py, vesselfm.py

---

## Constituent Documents

1. `advanced-segmentation-double-check-plan.md`
2. `advanced-segmentation-double-check-prompt.md`
3. `advanced-segmentation-execution-plan.xml`
4. `atlassegfm-plan.md`
5. `comma-mamba-plan.md`
6. `dynunet-ablation-plan.md`
7. `dynunet-evaluation-plan.xml`
8. `foundation-model-finetuning-plan.md`
9. `mamba-model-capacity-matching.md`
10. `mambavesselnet-implementation-plan.xml`
11. `mambavesselnet-overnight-optimized.xml`
12. `mambavesselnet-overnight-runner.sh`
13. `mambavesselnet-test-on-dev-runpod-followup.xml`
14. `mambavesselnet-test-on-dev-runpod.xml`
15. `medsam3-annotation-plan.md`
16. `monai-segmentor-model-background-research.md`
17. `overnight-mambavesselnet.yaml`
18. `sam3-cold-start-opus4.yaml`
19. `sam3-implementation-plan.xml`
20. `sam3-installation-issues-and-synthesis-plan.xml`
21. `sam3-installation-issues-and-synthesis.md`
22. `sam3-literature-research-report.md`
23. `sam3-nan-loss-fix.md`
24. `sam3-real-data-e2e-plan.xml`
25. `sam3-stub-removal.xml`
26. `sam3-training-reference.md`
27. `sam3-val-loss-final-report.md`
28. `skypilot-compute-offloading-plan-for-vesselfm-sam3-and-synthetic-generation.xml`
29. `topolora-sam3-planning-report-diff-parameterized-compound-loss.md`
30. `topolora-sam3-planning-report.md`
31. `vesselfm-plan.md`
