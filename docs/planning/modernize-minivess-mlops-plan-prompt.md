# Modernization Plan Prompt — MinIVess MLOps

> This document preserves (1) the **verbatim user prompt** that initiated the planning, (2) the **clarification Q&A**, and (3) an **optimized prompt** rewritten for downstream LLM use.

---

## 1. Verbatim User Prompt

```text
Let's now make a "refactoring plan" to docs/modernize-minivess-mlops-plan.md as this repo was started
years ago to continue our lineage of vascular segmentation papers from initial segmentation in
https://arxiv.org/abs/1606.02382 to the dataset creation in
https://www.nature.com/articles/s41597-023-02048-8. So next step is to create a MLOps paper to produce
scientifically reproducible pipeline to address the reproducibility crisis (see e.g.
vessel_mlops.lyx which was a massive collection of different MLOps sources). However world has changed
a lot from that (SAMv2 -> SAMv3 in https://arxiv.org/abs/2512.06032) and we should now focus mostly on
using SAMv3 (https://arxiv.org/html/2511.16719v1, https://github.com/facebookresearch/sam3) in
biomedical context (http://arxiv.org/abs/2511.19046, https://arxiv.org/html/2601.10880v1) with
uncertainty (http://arxiv.org/abs/2505.05049), intelligent proofreading (e.g.
https://arxiv.org/abs/2512.24023). The focus is on MLOps and then not on the segmentation per se, so
we should use some segmentation model from MONAI (https://project-monai.github.io/) with SAMv3 (and
possible variants of it) as baseline models (or even drop the MONAI model) that the imaginary
researcher might want to use for their data. We should quantify the metrics with Metrics Reloaded
(https://doi.org/10.1038/s41592-023-02151-z, https://github.com/Project-MONAI/MetricsReloaded) and
use the best industry practices with SOTA arxiv papers for the analysis.

This project is also meant to be a portfolio project showing how I can use the recent agentic /
foundation models (in FMOps sense) for medical image analysis in production-grade fashion. The medical
image analysis might be just an excuse to further demonstrate my LLMOps/AgentOps skills for real-world
production. I have now written repo+manuscript on TSFMs (foundation-PLR) which is production-grade and
reproducible, but does not involve so much the deployment ML system design like my
music-attribution-scaffold.

Target job applications:
- Curio MLOps SME (designing 15-credit postgraduate MLOps module for UK university)
- ICEYE Forward Deployed AI Engineer (air-gapped Docker/K8s, RAG, MCP, agentic workflows, defense)
- Cohere Applied AI Engineer (production agents, LangGraph, RAG, evaluation frameworks)

So could we design the refactor so that we can add the production-grade approaches to this I did not
yet implement on dpp-agents, foundation-PLR, music-attribution-scaffold. So in a way this could be
considered as the "advanced LLMOps/MLOps/AgentOps" that is hopefully mostly an open-source academic
project. And in addition to the advanced portfolio project, a manuscript should be written about that
which could be then submitted to Nature Protocols type of journal or even lower impact journal.

Think of going beyond papers like:
- teikari-2023-minivess-mlops-vessel-segmentation
- https://doi.org/10.1016/j.jneumeth.2020.108804
- https://doi.org/10.1038/s41596-023-00881-0
- https://doi.org/10.1038/s41591-023-02540-z
- https://doi.org/10.1109/ACCESS.2023.3262138
- https://doi.org/10.1038/s41467-023-42396-y
- https://doi.org/10.1038/s41592-023-01885-0

This is non-clinical segmentation for animal models, but we should think how to ensure SaMD principles
and FDA compliance from start. Save this prompt both verbatim, and with optimization for downstream
LLM use.

We need web searches on latest SOTA techniques for:
- Experiment tracking, hyperparameter tuning, cross-validation folds
- Flexible model ensembling (same models + heterogeneous models)
- Finetuning SAMv3 or similar models, in-context learning
- Deploying from MLflow artifacts via BentoML (or alt) with ArgoCD canary deployments
- Multiple environments (prod/staging/test), Docker-based dev mandatory
- uv for Python, bun for TS/JS obligatory
- Drift detection (open-source + optional Evidently)
- Traceability/observability with OpenTelemetry/Prometheus/Grafana
- Agent-based ReAct agentic collaboration (segmentation -> classification demo)
- LLM tracing with Langfuse
- MONAI Deploy clinical tools
- AG-UI/A2UI + MONAI Label for intelligent annotation/proofreading
- SaMD principles, not overusing test set

This is quite of a task so let's start the planning with reviewer agents until convergence to an
optimal plan! Ask clarifying questions before the planning and after for next steps.
```

---

## 2. Clarification Q&A

### Q1: Rewrite Scope — Clean rewrite or incremental refactor?
**A:** Clean rewrite. Tag current state as v0.1-alpha, create CHANGELOG, then modernize everything from scratch. Not much to keep from the original codebase.

### Q2: Deployment Target — What infrastructure?
**A:** Local Docker Compose development with all services running locally (no API tokens needed in `.env`), BUT flexible to shift from local MLflow to managed cloud MLflow etc. No cloud lock-in — cloud-agnostic approach. Pulumi for IaC. All open-source tools for deployment: Nomad/Kubernetes (non-AWS). Keep it flexible for any cloud provider.

### Q3: Model Strategy — MONAI-only, SAMv3-only, or model-agnostic?
**A:** Model-agnostic pipeline. Think of it as glue code also usable for any SAMv4 or cool non-SAMv3 model. Ensembling flexibility: MONAI baseline with various seeds, different CV folds, ensemble MONAI with SAMv3, model merging, model soups, SWAG, etc. The pipeline is the product, not the model.

### Q4: Paper Target — Protocol paper, engineering paper, or both?
**A:** Both. Nature Protocols-style protocol paper AND engineering/systems paper.

---

## 3. Optimized Prompt — Claude Code / Opus 4.6

```markdown
# Task: Create Comprehensive Modernization Plan for MinIVess MLOps

## Context
MinIVess MLOps is a biomedical vascular segmentation pipeline (MONAI + PyTorch) being rewritten from
v0.1-alpha to a production-grade, model-agnostic MLOps/FMOps platform. The project serves as both:
1. An academic research platform for reproducible biomedical image segmentation
2. A portfolio project demonstrating advanced LLMOps/MLOps/AgentOps capabilities

## Research Lineage
- Vascular segmentation: arXiv:1606.02382 → Nature Scientific Data (s41597-023-02048-8)
- MLOps manuscript draft: vessel_mlops.lyx (addressing reproducibility crisis)
- Companion projects: foundation-PLR (TSFMs), music-attribution-scaffold (deployment), dpp-agents

## Architecture Requirements

### Core Pipeline (Model-Agnostic)
- **Models**: SAMv3 (Segment Anything with Concepts, Nov 2025), MONAI models (SegResNet, SwinUNETR,
  VISTA-3D), model-agnostic adapter layer for future models (SAMv4, etc.)
- **Ensembling**: Multi-seed, multi-fold CV, multi-model heterogeneous ensembles, model soups
  (greedy soup, WiSE-FT), SWAG, TIES/DARE merging
- **Uncertainty**: Conformal prediction (MAPIE), MC dropout, deep ensembles, calibration
- **Metrics**: MetricsReloaded (clDice, NSD, Hausdorff95), bootstrap CI, per-sample statistics
- **Evaluation**: SaMD-compliant test set lockout, stratified splits, fairness audits

### Infrastructure
- **Dev**: Docker Compose (all services local, zero API tokens), uv (Python), bun (TS/JS)
- **Config**: Dynaconf or Hydra-zen, Pydantic v2 validation
- **Experiment Tracking**: MLflow (local → managed), W&B optional
- **Data Versioning**: DVC with local/S3/GCS backends
- **CI/CD**: GitHub Actions → ArgoCD GitOps, canary deployments, multi-env (prod/staging/test)
- **IaC**: Pulumi (Python), cloud-agnostic (no vendor lock-in)
- **Orchestration**: Nomad or Kubernetes

### Observability & Compliance
- **Tracing**: OpenTelemetry → Jaeger/Tempo, Langfuse for LLM traces
- **Metrics**: Prometheus → Grafana dashboards
- **Drift Detection**: Evidently AI (open-source), NannyML, WhyLabs
- **SaMD/FDA**: IEC 62304 software lifecycle, ISO 13485 QMS principles, audit trail, SBOM
- **Logging**: Structured JSON, correlation IDs, immutable audit log

### Agent Layer
- **Framework**: PydanticAI or LangGraph for pipeline orchestration
- **Workflows**: ReAct agent for segmentation→classification demo, annotation review agent
- **Integration**: MONAI Label + AG-UI/A2UI for interactive annotation/proofreading
- **Clinical Deployment**: MONAI Deploy App SDK (MAP) for DICOM I/O

### Deployment
- **Serving**: BentoML or TorchServe from MLflow artifacts
- **Registry**: Docker Hub / Harbor, model registry (MLflow)
- **Environments**: Local Docker Compose → Staging K8s → Production K8s
- **Air-gapped**: Design for offline deployment (ICEYE-relevant)

## Manuscript Targets
1. **Nature Protocols**: Step-by-step reproducible MLOps protocol for biomedical segmentation
2. **Engineering paper**: Systems architecture, benchmark comparisons, lessons learned

## Output Format
Create `docs/modernize-minivess-mlops-plan.md` with:
- Executive summary
- Current state analysis (v0.1-alpha) with CHANGELOG
- Architecture overview with diagrams (Mermaid)
- Phased implementation roadmap (6 phases)
- Technology stack decisions with justification
- Manuscript outline for both papers
- Risk matrix and mitigation strategies
- Testing strategy (unit, integration, e2e, compliance)
```

---

## 4. Optimized Prompt — Gemini Deep Research (Literature Review)

```markdown
# Literature Review Request: Modern MLOps for Biomedical Image Segmentation (2024-2026)

## Scope
Conduct a comprehensive literature review of the latest research (2024-2026) on production-grade
MLOps pipelines for biomedical/medical image segmentation, with particular focus on:

### 1. Foundation Models for Segmentation
- SAM 3 / Segment Anything with Concepts (Meta, Nov 2025): architecture, promptable concept
  segmentation, in-context learning with image exemplars
- MedSAM3 / SAM3-Adapter: LoRA fine-tuning for medical imaging, multi-modal support
- MONAI VISTA-3D: unified 3D segmentation foundation model, comparison with SAM approaches
- Med-PerSAM, ProtoSAM, SAM-MPA: one-shot/few-shot medical segmentation
- Onco-Seg: SAM3 adaptation with LoRA for tumor segmentation across modalities

### 2. Model Ensembling & Uncertainty
- Model soups (greedy soup, WiSE-FT), TIES/DARE merging
- SWAG (Stochastic Weight Averaging Gaussian) for Bayesian inference
- Conformal prediction in medical imaging (MAPIE v1, crepes library)
- Deep ensemble uncertainty quantification
- Calibration methods for safety-critical predictions

### 3. MLOps Infrastructure (2025-2026 SOTA)
- Experiment tracking: MLflow 3.x, W&B, Neptune, ClearML
- Hyperparameter optimization: Optuna, Ray Tune, Ax
- Data versioning: DVC 3.x, LakeFS, Delta Lake
- Feature stores and data contracts for medical imaging
- Drift detection: Evidently AI, NannyML, WhyLabs

### 4. Deployment & Serving
- Model serving: BentoML 1.x, TorchServe, Triton Inference Server
- Clinical deployment: MONAI Deploy App SDK (MAP), DICOM interoperability
- Container orchestration: Kubernetes, HashiCorp Nomad for ML workloads
- GitOps: ArgoCD, Flux for ML model deployment
- Air-gapped deployment patterns for defense/regulated environments

### 5. Observability & Compliance
- OpenTelemetry for ML pipelines, Langfuse for LLM tracing
- SaMD (Software as a Medical Device): IEC 62304, FDA guidance
- Reproducibility frameworks and audit trails
- SBOM generation and supply chain security for ML

### 6. Agent-Based ML Pipelines
- PydanticAI, LangGraph for ML pipeline orchestration
- AG-UI / A2UI (Google, Dec 2025) for agent-driven interfaces
- MONAI Label interactive annotation with SAM integration
- ReAct patterns for automated analysis workflows

### 7. Annotation & Active Learning
- MONAI Label 0.8.5: DeepEdit, Deepgrow, SAM2 integration, 3D Slicer
- Intelligent proofreading / verification of AI segmentations
- Human-in-the-loop workflows for medical imaging

## Output Format
For each topic, provide:
- Key papers with citations (author, year, venue)
- Current SOTA approach and performance benchmarks
- Open-source implementations and tools
- Practical recommendations for a biomedical segmentation MLOps pipeline
- Gaps in the literature that could be addressed by a new protocol/systems paper

## Target Journals for Positioning
- Nature Protocols (protocol papers)
- Nature Methods (methods papers)
- Medical Image Analysis (Elsevier)
- IEEE Access (engineering-focused)
- Nature Communications (high-impact general)
```

---

*Generated: 2026-02-23 | Session: minivess-mlops modernization planning*
