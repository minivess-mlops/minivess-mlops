# MinIVess MLOps v2 — LLM Context for AI Assistants

## System Context

You are assisting with **MinIVess MLOps v2**, a comprehensive biomedical image
segmentation MLOps pipeline. The project is a **learning-first, portfolio-building**
system that demonstrates production-grade patterns for medical image analysis.

## Vision

Build the most **tool-rich, well-documented** biomedical segmentation MLOps
pipeline in open source. Every technology choice is an opportunity to learn
and demonstrate competence. No fear of overengineering — breadth over depth.

## Core Problem

3D cerebral vessel segmentation from 7T TOF-MRA images (MinIVess dataset).
The pipeline generalizes to any 3D medical segmentation task via the
ModelAdapter abstraction and domain overlay system.

## Current State (Phase 6 Complete)

- **103+ unit tests** passing across 38 modules and 30+ tools
- **6 implementation phases** complete (P0–P6)
- Stack: PyTorch 2.5+ | MONAI 1.4+ | TorchIO | TorchMetrics
- Serving: BentoML + ONNX Runtime + Gradio
- Config: Hydra-zen (training) + Dynaconf (deployment)
- Tracking: MLflow 3.10 + DuckDB analytics
- Validation: Pydantic v2 + Pandera + Great Expectations + Deepchecks
- Compliance: SaMD-principled (IEC 62304 lifecycle, model cards, audit trails)
- Observability: OpenTelemetry + Langfuse + Braintrust + LiteLLM
- Agent: LangGraph placeholder + OpenLineage
- CI/CD: GitHub Actions + CML + pre-commit

## Technology Philosophy

### DO use:
- **MONAI** for medical image transforms and model zoo
- **PyTorch 2.x** with compile mode where beneficial
- **Hydra-zen** for type-safe experiment configs
- **MLflow** as experiment tracking backbone
- **BentoML** for model serving
- **DVC** for data versioning
- **Pydantic v2** for all data contracts
- **uv** for package management (NEVER pip/conda/poetry)
- **pathlib.Path** for all file paths (NEVER string paths)

### DO NOT use:
- LangChain (use LangGraph directly or custom)
- pip/conda/poetry (use uv exclusively)
- String file paths (use pathlib.Path)
- Naive datetime (always timezone-aware UTC)

## PRD System

This project uses a **hierarchical probabilistic PRD** based on Bayesian
decision networks. The PRD lives at `docs/planning/prd/` and encodes:

1. **52 decision nodes** across 5 levels (L1 Research Goals → L5 Operations)
2. **Prior probabilities** for each option at each decision
3. **Conditional probability tables** linking parent→child decisions
4. **3 archetypes** (Solo Researcher, Lab Group, Clinical Deployment)
5. **3 scenarios** (Learning-First MVP, Research Scaffold, Clinical Production)
6. **4 domain overlays** (Vascular, Cardiac, Neuroimaging, General)

### Active Scenario: Learning-First MVP
- Archetype: Solo Researcher
- Domain: Vascular Segmentation (MinIVess)
- Priority: Breadth over depth, maximize tool exposure
- Status: Phase 6 complete, exploring next steps via PRD

## Decision-Making Framework

When making technology decisions, consider:
1. **Does it maximize learning?** (self-learning priority)
2. **Does it strengthen the portfolio?** (demonstrate production patterns)
3. **Is it MONAI-compatible?** (ecosystem alignment)
4. **Does it have a test?** (TDD is mandatory)

## Key Design Decisions (HIGH confidence, resolved)

| Decision | Choice | Confidence |
|----------|--------|-----------|
| Package manager | uv | 100% |
| ML framework | MONAI + PyTorch | 95% |
| Config (train) | Hydra-zen | 90% |
| Config (deploy) | Dynaconf | 85% |
| Experiment tracking | MLflow | 90% |
| Serving | BentoML + ONNX | 85% |
| Data versioning | DVC | 85% |
| License | MIT | 100% |

## Critical Constraints

1. **TDD mandatory** — Red→Green→Verify→Fix→Checkpoint→Converge
2. **All Python files** must have `from __future__ import annotations`
3. **UTF-8 encoding** everywhere
4. **pathlib.Path** for all file operations
5. **datetime.now(timezone.utc)** for all timestamps
6. **uv** exclusively for package management
7. **Pre-commit hooks** must pass before any commit
