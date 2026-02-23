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

## Academic Citation Standards (NON-NEGOTIABLE)

This is an **academic software project**. The PRD serves as the evidence base for a
future **peer-reviewed article**. All citation standards are mandatory.

### Rules

1. **Author-year format ONLY** — Use "Surname et al. (Year)" for 3+ authors,
   "Surname & Surname (Year)" for 2, "Surname (Year)" for 1. NEVER use numeric
   references like [1] or [Smith2024].

2. **Central bibliography** — All citations live in `bibliography.yaml`. Every
   `citation_key` in a `.decision.yaml` MUST resolve to a bibliography entry.

3. **No citation loss** — References are **append-only**. When updating a decision
   file, NEVER remove existing references. If a reference becomes outdated, add a
   note but keep the entry. The pre-commit hook will BLOCK commits that remove citations.

4. **Sub-citations mandatory** — When ingesting a paper, also extract relevant papers
   it cites and add them to `bibliography.yaml`. A single paper typically yields 3-10
   relevant sub-citations.

5. **In-text citations in rationale** — Every `rationale` field MUST contain at least
   one author-year citation. Every claim about a technology's performance must cite
   the source paper.

6. **Structured references** — Each `.decision.yaml` reference entry includes:
   - `citation_key` (links to bibliography.yaml)
   - `relevance` (why this reference matters for this decision)
   - `sections` (specific tables, figures, sections cited)
   - `supports_options` (which option_ids this reference provides evidence for)

### Guardrails

- **Pre-commit hook**: `prd-citation-check` runs `scripts/validate_prd_citations.py`
  on any PRD file change. Blocks commit if citations were removed or keys don't resolve.
- **Validate protocol**: Check 7 (Citation Integrity) in the validate protocol covers
  bibliography resolution, completeness, in-text format, and no-citation-loss.
- **SKILL.md invariants**: Invariants #6 (citation integrity), #7 (no citation loss),
  and #8 (author-year format) are enforced on every PRD operation.

### Example

In a `.decision.yaml` rationale field:
```
SegResNet (Myronenko, 2019) achieves competitive Dice scores as a lightweight
encoder-decoder with VAE regularization. VISTA-3D (He et al., 2024) represents
the next-generation foundation model approach, with LoRA fine-tuning
(Hu et al., 2022) enabling parameter-efficient adaptation.
```

## Critical Constraints

1. **TDD mandatory** — Red→Green→Verify→Fix→Checkpoint→Converge
2. **All Python files** must have `from __future__ import annotations`
3. **UTF-8 encoding** everywhere
4. **pathlib.Path** for all file operations
5. **datetime.now(timezone.utc)** for all timestamps
6. **uv** exclusively for package management
7. **Pre-commit hooks** must pass before any commit
8. **Academic citations** — Author-year format, no citation loss, sub-citations mandatory
