# Changelog

All notable changes to MinIVess MLOps are documented in this file.

---

## [v0.2-alpha] — 2026-02-25

### Summary

**Complete ground-up rewrite of MinIVess MLOps**, entirely AI-generated using Claude Code
(Anthropic Claude Opus 4/Sonnet 4) with zero human-written implementation code. This release
is an untested scaffold — the 1002 unit/integration tests pass, but no human has verified
that the ML pipeline actually trains models, produces correct segmentations, or serves
predictions in practice.

**By the numbers:**
- 91 source modules (12,421 LOC) across 11 subpackages
- 87 test modules (16,263 LOC) with 1,002 passing tests
- 88 commits since v0.1-alpha, closing 49 GitHub issues
- 448 files changed: +67,689 / -19,452 lines (net +48,237)
- ~12,000 LOC of legacy v0.1 code removed and replaced
- 52-node Bayesian PRD decision network with 3,520-entry bibliography
- 18 documented Claude Code patterns for agentic software development
- 2 full code review passes (6 parallel specialist agents each) with remediation

### Architecture (v2 rewrite)

The v0.1 monolithic codebase (~8K LOC of intertwined training/inference/logging) was
replaced with a modular architecture using 11 subpackages:

| Package | Modules | Purpose |
|---------|---------|---------|
| `adapters/` | 13 | Model-agnostic adapter ABC + 10 concrete implementations |
| `pipeline/` | 9 | Training, loss functions, HPO, metrics, QC, federated learning |
| `ensemble/` | 10 | Ensemble strategies, calibration, UQ (MC Dropout, conformal, generative) |
| `data/` | 9 | NIfTI loading, DVC integration, augmentation, drift generation |
| `serving/` | 5 | BentoML, ONNX Runtime, Gradio demo, MONAI Deploy |
| `agents/` | 6 | LangGraph orchestration, LiteLLM, Langfuse tracing |
| `observability/` | 9 | MLflow, DuckDB analytics, drift detection, lineage, telemetry |
| `compliance/` | 10 | IEC 62304, EU AI Act, fairness auditing, regulatory docs |
| `validation/` | 10 | Pandera, Great Expectations, Deepchecks, DATA-CARE, VessQC |
| `config/` | 3 | Pydantic v2 models, Hydra-zen/Dynaconf, centralized defaults |
| `utils/` | 5 | Seed management, enum registry, markdown, protocols |

### Model Adapters

All models implement `ModelAdapter` ABC with unified `forward()`, `get_config()`,
`save_checkpoint()`, `load_checkpoint()`, `export_onnx()`, and `trainable_parameters()`:

- **SegResNet** — MONAI's residual encoder-decoder (primary architecture)
- **SwinUNETR** — Shifted-window transformer for 3D segmentation
- **DynUNet** — Dynamic UNet with configurable width ablation
- **VISTA-3D** — MONAI foundation model with prompt-based segmentation
- **vesselFM** — HuggingFace-hosted vascular foundation model with weight checksums
- **AtlasSegFM** — One-shot foundation model customization
- **COMMA/Mamba** — State-space model with coordinate embedding
- **MedSAM3** — Interactive annotation adapter for medical SAM
- **LoRA** — Low-rank adaptation wrapper for any base adapter
- **Adaptation comparison** — Side-by-side adapter evaluation framework

### Training Pipeline

- **SegmentationTrainer** with dependency injection (criterion, optimizer, scheduler)
- **VesselCompoundLoss** combining Dice, cross-entropy, clDice, cbDice, and Betti losses
- **Optuna HPO** integration with configurable search spaces and study factory
- **Bootstrap confidence intervals** for segmentation metrics
- **Segmentation quality control** framework (nnQC-inspired)
- **Federated learning** support with NVIDIA FLARE patterns
- **DVC pipeline** with download, preprocess, and train stages

### Ensemble & Uncertainty Quantification

- Model soup, voting, and stacking ensemble strategies
- Temperature scaling calibration with shift-aware framework
- MC Dropout, Deep Ensembles, MAPIE conformal prediction
- Generative UQ: Probabilistic U-Net, PHiSeg, Stochastic Segmentation Networks
- WeightWatcher spectral model diagnostics

### Serving

- **BentoML** service with predict and health APIs
- **ONNX Runtime** inference engine with full export-load-predict roundtrip
- **Gradio** demo with NIfTI volume loading and slice extraction
- **MONAI Deploy** clinical deployment pathway

### Observability & MLOps

- **MLflow** experiment tracking with sorted parameter logging
- **DuckDB** in-process SQL analytics over MLflow runs
- **Two-tier drift detection** (KS tests + kernel MMD via Evidently)
- **Prediction-Powered Risk Monitoring** (PPRM) with false alarm guarantees
- **OpenLineage/Marquez** data lineage tracking
- **OpenTelemetry** instrumentation
- **Model registry** with promotion stages (dev → staging → production → archived)
- **DiLLS-style** agent diagnostics
- **whylogs** data profiling integration

### Compliance & Regulatory

- **IEC 62304** software lifecycle with audit trails and SHA-256 data hashing
- **EU AI Act** compliance checklist generation
- **ComplOps** regulatory automation with LLM-assisted document generation
- **RegOps** CI/CD pipeline extension for regulatory artifacts
- **CyclOps-inspired** fairness auditing (subgroup metrics, disparity analysis)
- **Model cards** (Mitchell et al., 2019 format)
- **CONSORT-AI** and **MI-CLEAR-LLM** reporting templates
- **Automated regulatory docs**: DHF, risk analysis, SRS, validation summaries

### Validation

- **Pydantic v2** config schemas with comprehensive negative validation tests
- **Pandera** DataFrame validation
- **Great Expectations** batch quality gates with NIfTI and metrics checks
- **Deepchecks Vision** image/model validation
- **DATA-CARE** multi-dimensional quality scoring
- **VessQC** uncertainty-guided annotation curation
- **Generic `_validate_with_schema()` wrapper** reducing validation boilerplate

### Agent Orchestration

- **LangGraph** training and comparison graphs with conditional routing
- **LiteLLM** provider-flexible LLM integration
- **Langfuse** tracing for LLM observability
- **Braintrust** offline evaluation framework

### Data Pipeline

- **EBRAINS MiniVess** dataset downloader with checksum verification
- **Synthetic NIfTI** fixtures for testing
- **Domain randomization** augmentation (SynthICL-inspired) with seed propagation
- **DVC** data versioning with `.gitkeep` structure
- **Drift generation** for synthetic distribution shift experiments

### Configuration

- **Hydra-zen** for experiment sweeps (config store with dataclass configs)
- **Dynaconf** for deployment environments (dev/staging/production/airgapped)
- **Pydantic v2** typed config models (ModelConfig, DataConfig, TrainConfig, etc.)
- **Centralized defaults** module (`config/defaults.py`)
- **Configurable adapter hyperparameters** via `architecture_params` field

### Developer Experience

- **uv** package manager (only — no pip/conda/poetry)
- **ruff** linting + formatting
- **mypy** type checking
- **pytest** with 1,002 tests (Hypothesis property-based tests included)
- **pre-commit** hooks (ruff, mypy, citation validation)
- **justfile** for common commands
- **GitHub Actions** CI (`ci-v2.yml`)
- **Docker Compose** with 12+ service profiles

### Documentation & Planning

- **CLAUDE.md** — Living contract for AI assistant behavior (7 critical rules)
- **5 Architecture Decision Records** (ADRs)
- **18 Claude Code patterns** documented from real development sessions
- **52-node hierarchical probabilistic PRD** (Bayesian decision network)
- **3,520-entry bibliography** with academic citation standards
- **R6 remediation plan** with 8 issue packages across 3 sprints
- **2 code review reports** (1st pass: 42 issues, 2nd pass: 31 issues)
- **Self-learning-iterative-coder** TDD skill with 8 protocol files

### Code Quality (review remediation)

Two full code review passes were performed by 6 parallel specialist agents:

**1st pass (R1–R4):** 42 issues found → all fixed
- Dead code removal (Sam3Adapter stub, legacy utilities)
- Typed return values (dict → dataclasses) across all adapters
- Centralized seed management
- StrEnum registry for discoverability
- Markdown utility extraction
- Exception hierarchy with `__all__` exports

**2nd pass (R5–R6):** 31 issues found → all fixed
- ModelAdapter base class consolidation (`_build_output`, `_build_config`)
- CUDA determinism (`cudnn.deterministic=True`)
- DataLoader worker seeding
- Factory naming convention (`build_*` prefix)
- Dependency version upper bounds
- Domain randomization seed fallback
- VesselFM weight checksums
- Dependency injection for Trainer and data loaders
- Validation gate refactoring
- Configurable adapter hyperparameters
- Centralized defaults
- NumPy-style docstrings

### Test Progression

| Milestone | Tests | Delta |
|-----------|-------|-------|
| Phase 0 Batch 2 | 18 | +18 |
| Phase 1 Batch 3 | 66 | +48 |
| Phase 2 | 80 | +14 |
| Phase 3 | 87 | +7 |
| Phase 4+5 | 102 | +15 |
| Phase 6 (E2E) | 102 | +0 |
| Issues #3–#51 | 562 | +460 |
| Code review R1 | 562 | +0 |
| Code review R2 | 662 | +100 |
| Code review R3 | 688 | +26 |
| Code review R4 | 813 | +125 |
| Code review R5 | 868 | +55 |
| Code review R6 Sprint 1 | 944 | +76 |
| Code review R6 Sprint 2 | 984 | +40 |
| Code review R6 Sprint 3 | 1002 | +18 |

### Breaking Changes from v0.1-alpha

- Entire source tree restructured: `src/` → `src/minivess/` package
- All legacy modules removed (`src/training/`, `src/datasets/`, `src/inference/`,
  `src/log_ML/`, `src/utils/`)
- Configuration: Hydra YAML → Hydra-zen + Dynaconf
- Package manager: Poetry → uv
- Python: 3.10 → 3.12+
- All model code now uses `ModelAdapter` ABC (no direct model instantiation)

### Known Limitations

> **This is an AI-generated scaffold. No human has verified runtime behaviour.**

- No human has run a real training loop end-to-end
- No human has verified ONNX export produces correct segmentations
- No human has tested BentoML serving with real requests
- No human has validated Gradio demo with real NIfTI volumes
- No human has verified MLflow tracking logs correct metrics
- MONAI Deploy, federated learning, and clinical deployment are structural scaffolds only
- Generative UQ models (PHiSeg, Prob-UNet, SSN) are architecture stubs awaiting training
- Foundation model adapters (VISTA-3D, vesselFM, AtlasSegFM) require weight downloads
- Fairness auditing and regulatory doc generation are template-driven, not validated against
  real regulatory submissions
- The 1,002 tests verify API contracts and data flow, not ML correctness

---

## [v0.1-alpha] — 2024-05-15

Initial prototype implementation with monolithic architecture.
See [v0.1-alpha release](https://github.com/petteriTeworworWorworwo/minivess-mlops/releases/tag/v0.1-alpha) for details.

---

## [v0.1-archive] — 2024-05-15

Archived copy of v0.1-alpha before v2 rewrite began.
