# MinIVess MLOps v2

Model-agnostic biomedical segmentation MLOps platform. Clean rewrite from v0.1-alpha.

## Design Goal #1: EXCELLENT DevEx for PhD Researchers

> **MLOps as a scaffold that frees PhD researchers from infrastructure wrangling.**
> Everything automatic by default, everything tweakable by choice.

This is the **first and most important design goal** of the entire repository. Every
feature, every configuration, every automation should be evaluated against this principle.

### Core Principles
1. **Zero-config start** — `just experiment` works out of the box on any machine
2. **Adaptive defaults** — Hardware auto-detection selects batch size, patch size, cache rate
3. **Scientific decisions stay with the researcher** — No default resampling, no implicit
   upsampling. The platform provides knobs, the researcher turns them.
4. **Model-agnostic profiles** — Same `--compute auto` works for DynUNet, SAMv3, SegResNet;
   each model maps hardware budgets differently via `configs/model_profiles/*.yaml`
5. **Dataset-agnostic patches** — Patch sizes constrained by dataset's smallest volume,
   not hardcoded. Pre-training validation ensures patches fit all volumes.
6. **Transparent automation** — Every automatic decision is logged and overridable via YAML
7. **Portfolio-grade code** — Every component demonstrates production ML engineering
8. **Division of labor via Prefect** — Prefect flows are **required** (not optional),
   separating concerns into 5 persona-based flows even for solo researchers. Each flow
   is independently testable, resumable, cacheable, and uses MLflow as the inter-flow
   contract. The first 4 flows are **core** (always run); the 5th is **best-effort**
   (runs when resources allow, failure does not block the pipeline):
   - Flow 1: Data Engineering (core)
   - Flow 2: Model Training (core)
   - Flow 3: Model Analysis (core)
   - Flow 4: Deployment (core)
   - Flow 5: Dashboard & Reporting (best-effort — paper figures, Parquet export, drift reports)

### Multi-Environment Compute
Everything must work identically on:
- Local workstation (single GPU, limited RAM)
- Intranet / on-prem servers (multi-GPU, team access)
- Ephemeral cloud instances (Docker, mounted drives)
- CI runners (CPU-only, automated)

## Quick Reference

| Aspect | Details |
|--------|---------|
| **Project Type** | application (ML pipeline + serving) |
| **Python Version** | 3.12+ |
| **Package Manager** | uv (ONLY) |
| **Linter/Formatter** | ruff |
| **Type Checker** | mypy |
| **Test Framework** | pytest |
| **Config (train)** | Hydra-zen |
| **Config (deploy)** | Dynaconf |
| **ML Framework** | PyTorch + MONAI + TorchIO + TorchMetrics |
| **Serving** | BentoML + ONNX Runtime + Gradio (demo) |
| **Experiment Tracking** | MLflow + DuckDB (analytics) |
| **Data Validation** | Pydantic v2 (schema) + Pandera (DataFrame) + Great Expectations (batch quality) |
| **Label Quality** | Cleanlab + Label Studio |
| **Model Validation** | Deepchecks Vision + WeightWatcher |
| **XAI** | Captum (3D) + SHAP (tabular only) + Quantus (meta-eval) |
| **Calibration** | MAPIE + netcal + Local Temperature Scaling |
| **Data Profiling** | whylogs |
| **LLM Observability** | Langfuse (self-hosted) + Braintrust (eval) + LiteLLM (provider flexibility) |
| **Workflow Orchestration** | Prefect 3.x (required, 5 persona-based flows: data, train, analyze, deploy, dashboard) |
| **Agent Orchestration** | LangGraph |
| **CI/CD** | GitHub Actions + CML (ML-specific PR comments) |
| **Lineage** | OpenLineage (Marquez) |

## Critical Rules

1. **uv ONLY** — Never use pip, conda, poetry, or requirements.txt. Use `uv add`, `uv sync`, `uv run`.
2. **TDD MANDATORY** — All implementation MUST follow the self-learning-iterative-coder skill (`.claude/skills/self-learning-iterative-coder/SKILL.md`). Write failing tests FIRST, then implement. No exceptions.
3. **Library-First (Non-Negotiable)** — Before implementing ANY algorithm, loss function,
   metric, or data processing step, ALWAYS search for existing implementations in established
   libraries (PyTorch, MONAI, scipy, skimage, networkx, gudhi, etc.) or peer-reviewed
   reference repos. Only write custom code when no suitable library exists. For hybrid cases
   (e.g., loss functions), use library code for non-differentiable parts (GT processing)
   and custom code only for the differentiable prediction path. Rationale: library code has
   more eyes, fewer bugs, and better performance than anything we write from scratch.
4. **Pre-commit Required** — All changes must pass pre-commit hooks before commit.
5. **Encoding** — Always specify `encoding='utf-8'` for file operations.
6. **Paths** — Always use `pathlib.Path()`, never string concatenation.
7. **Timezone** — Always use `datetime.now(timezone.utc)`, never `datetime.now()`.
8. **`from __future__ import annotations`** — At the top of every Python file.

## TDD Workflow (Non-Negotiable)

Every feature, bugfix, or refactor MUST use the self-learning-iterative-coder skill:

```
1. RED:        Write failing tests first     → .claude/skills/.../protocols/red-phase.md
2. GREEN:      Implement minimum code        → .claude/skills/.../protocols/green-phase.md
3. VERIFY:     Run tests + lint + typecheck  → .claude/skills/.../protocols/verify-phase.md
4. FIX:        If failing, targeted fix      → .claude/skills/.../protocols/fix-phase.md
5. CHECKPOINT: Git commit + state            → .claude/skills/.../protocols/checkpoint.md
6. CONVERGE:   All green? Move to next task  → .claude/skills/.../protocols/convergence.md
```

**Activation**: Before starting a multi-task implementation, run the [ACTIVATION-CHECKLIST](.claude/skills/self-learning-iterative-coder/ACTIVATION-CHECKLIST.md).

**Skill reference**: `.claude/skills/self-learning-iterative-coder/SKILL.md`

## Default Loss Function

The default single-model loss is **`cbdice_cldice`** (CbDiceClDiceLoss). This was
determined by the `dynunet_loss_variation_v2` experiment (2026-02-27) which showed:
- `cbdice_cldice` achieves **0.906 clDice** (best topology) with only −5.3% DSC penalty
- `dice_ce` has higher DSC (0.824) but significantly worse topology preservation (0.832 clDice)
- Full results: `docs/results/dynunet_loss_variation_v2_report.md`

When training a single model (not an ablation sweep), always use `cbdice_cldice` unless
the researcher explicitly requests a different loss. For multi-loss experiments, use the
experiment config YAML which specifies the full loss list.

## Quick Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -x -q

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/

# Full verify (all three gates)
uv run pytest tests/ -x -q && uv run ruff check src/ tests/ && uv run mypy src/
```

## Directory Structure (Target v2)

```
minivess-mlops/
├── src/minivess/              # Main package (renamed from src/)
│   ├── adapters/              # ModelAdapter implementations (MONAI, SAMv3, etc.)
│   ├── pipeline/              # Training, inference, evaluation pipelines
│   ├── ensemble/              # Ensembling strategies (soup, voting, conformal)
│   ├── data/                  # Data loading, profiling, DVC integration
│   ├── orchestration/         # Prefect flows + _prefect_compat.py
│   ├── serving/               # BentoML service definitions
│   ├── agents/                # LangGraph agent definitions
│   ├── observability/         # Langfuse + Braintrust integration
│   ├── compliance/            # Audit trails, SaMD lifecycle hooks
│   └── config/                # Hydra-zen + Dynaconf config schemas
├── tests/
│   ├── unit/                  # Fast, isolated tests
│   ├── integration/           # Service integration tests
│   └── e2e/                   # End-to-end pipeline tests
├── configs/                   # Hydra-zen experiment configs
├── deployment/                # Docker, docker-compose, Pulumi IaC
├── docs/                      # Architecture docs, plan, ADRs
├── .claude/                   # Claude Code configuration
│   └── skills/                # TDD skill
└── LEARNINGS.md               # Cross-session accumulated discoveries
```

## What AI Must NEVER Do

- Use pip, conda, poetry, or create requirements.txt
- Write implementation code before tests (violates TDD mandate)
- Claim tests pass without running them ("ghost completions")
- Write placeholder/stub implementations (`pass`, `TODO`, `NotImplementedError`)
- Skip pre-commit hooks
- Hardcode file paths as strings
- Use `datetime.now()` without timezone
- Commit secrets, credentials, or API keys
- Modify files marked with `# AIDEV-IMMUTABLE`
- Push untested changes

## Observability Stack

| Tool | Role | Deployment |
|------|------|-----------|
| **Prefect 3.x** | Workflow orchestration (5 flows: data, train, analyze, deploy + dashboard best-effort) | Optional — local or Docker Compose |
| **Langfuse** | Production LLM tracing, cost tracking | Self-hosted (Docker Compose) |
| **Braintrust** | Offline evaluation, CI/CD quality gates, AutoEvals | Hybrid deployment (data plane local) |
| **LangGraph** | Agent orchestration, multi-step workflows | Library (in-process) |
| **LiteLLM** | Unified LLM API, provider flexibility | Library (in-process) |
| **MLflow** | Experiment tracking, model registry | Local Docker Compose |
| **DuckDB** | In-process SQL analytics over MLflow runs | Library (in-process) |
| **Prometheus + Grafana** | Infrastructure metrics | Local Docker Compose |
| **Evidently** | Data/model drift detection | Library + Grafana export |
| **whylogs** | Lightweight data profiling | Library (in-process) |
| **OpenLineage (Marquez)** | Data lineage tracking (IEC 62304) | Local Docker Compose |
| **Deepchecks Vision** | Image data + model validation | Library (in-process) |
| **WeightWatcher** | Spectral model diagnostics | Library (in-process) |
| **CML** | ML-specific CI/CD, auto PR comments | GitHub Actions |
| **Label Studio** | Multi-annotator workflows | Local Docker Compose |

## Key Architecture Decisions

- **Model-agnostic**: All models implement `ModelAdapter` ABC (train/predict/export)
- **MONAI VISTA-3D** is primary 3D segmentation model; SAMv3 is exploratory
- **Local-first**: Docker Compose with zero cloud API tokens for development
- **SaMD-principled**: IEC 62304 lifecycle mapping, audit trails, test set lockout
- **Dual config**: Hydra-zen for experiment sweeps, Dynaconf for deployment environments

## MLflow Tracking Architecture

### Backend
- **Local filesystem** backend: `mlruns/` directory (no server required)
- Tracking URI: `mlruns` (resolved to absolute path in code)
- Each run stores params, metrics, tags, artifacts as plain files
- Run lifecycle: FINISHED (success), FAILED (exception), KILLED (abort)

### Experiments
| Experiment | Purpose | Created By |
|-----------|---------|-----------|
| `dynunet_loss_variation_v2` | Training: 4 losses x 3 folds x 100 epochs | `train_monitored.py` |
| `dynunet_half_width_v1` | Training: width ablation (filters/2) | `train_monitored.py` |
| `minivess_evaluation` | Evaluation runs (ensembles + analysis) | `analysis_flow.py` |

### Param Prefixes (standardized naming)
| Prefix | Category | Example |
|--------|----------|---------|
| (none) | Training hyperparams | `learning_rate`, `batch_size`, `training_time_seconds` |
| `arch_` | Model architecture | `arch_filters`, `arch_deep_supervision` |
| `sys_` | System/environment | `sys_python_version`, `sys_gpu_model` |
| `data_` | Dataset metadata | `data_n_volumes`, `data_train_volume_ids` |
| `loss_` | Loss function config | `loss_name`, `loss_weights` |
| `eval_` | Evaluation config | `eval_bootstrap_n`, `eval_ci_level` |
| `upstream_` | Cross-flow links | `upstream_training_run_id` |

NOTE: We use `sys_` (underscore) not `sys/` (slash) to avoid metric naming conflicts.

### Automatic Logging
`ExperimentTracker.start_run()` automatically logs:
- All `TrainingConfig` fields (17 params including weight_decay, warmup_epochs)
- Architecture params from `ModelConfig.architecture_params` (arch_ prefix)
- System info: Python, PyTorch, MONAI, CUDA, cuDNN, OS, GPU, RAM (sys_ prefix)
- Git commit hash, branch, dirty state
- MLflow system metrics (GPU/CPU/memory/disk, 12 metrics at 10s intervals)
- On failure: sets run status to FAILED with error_type tag

### Autolog Decision
`mlflow.pytorch.autolog()` is **NOT used**. It only provides full functionality with
PyTorch Lightning. This project uses vanilla PyTorch + MONAI training loops.
All logging is explicit via `ExperimentTracker`.

### Key Files
| File | Purpose |
|------|---------|
| `src/minivess/observability/tracking.py` | ExperimentTracker class |
| `src/minivess/observability/system_info.py` | System info collection |
| `src/minivess/observability/analytics.py` | DuckDB analytics over runs |
| `src/minivess/pipeline/mlruns_enhancement.py` | Post-hoc tag enhancement |
| `src/minivess/pipeline/champion_tagger.py` | Champion model tagging |
| `src/minivess/pipeline/duckdb_extraction.py` | DuckDB extraction from mlruns |
| `scripts/backfill_mlflow_metadata.py` | Retroactive run update tool |

### Retroactive Updates
Existing runs can be updated via `mlflow.start_run(run_id=existing_id)`.
**Safety rules**: never overwrite existing params (throws), preserve run status,
add `sys_backfill_note` for provenance. See `scripts/backfill_mlflow_metadata.py`.

## PRD System

The project uses a **hierarchical probabilistic PRD** (Bayesian decision network) to
manage open-ended technology decisions. This is an **academic software project** —
the PRD serves as the evidence base for a future peer-reviewed article.

- [docs/planning/prd/README.md](docs/planning/prd/README.md) — PRD navigation and overview
- [docs/planning/prd/llm-context.md](docs/planning/prd/llm-context.md) — AI assistant context
- [docs/planning/prd/bibliography.yaml](docs/planning/prd/bibliography.yaml) — Central bibliography (ALL cited works)
- [docs/planning/hierarchical-prd-planning.md](docs/planning/hierarchical-prd-planning.md) — PRD format blueprint

**PRD-Update Skill**: `.claude/skills/prd-update/SKILL.md` — Operations for maintaining
the PRD (add decisions, update priors, ingest papers, validate).

### Citation Rules (NON-NEGOTIABLE)
1. **Author-year format only** — "Surname et al. (Year)", never numeric [1]
2. **Central bibliography** — All citations in `bibliography.yaml`, decision files reference by `citation_key`
3. **No citation loss** — References are append-only. Pre-commit hook blocks citation removal.
4. **Sub-citations mandatory** — When ingesting a paper, also extract its relevant references
5. **Validation** — `uv run python scripts/validate_prd_citations.py` checks all citation invariants

## See Also

- [docs/modernize-minivess-mlops-plan.md](docs/modernize-minivess-mlops-plan.md) — Full modernization plan
- [docs/modernize-minivess-mlops-plan-prompt.md](docs/modernize-minivess-mlops-plan-prompt.md) — Original prompt and Q&A
- [.claude/skills/self-learning-iterative-coder/SKILL.md](.claude/skills/self-learning-iterative-coder/SKILL.md) — TDD skill reference
- [.claude/skills/prd-update/SKILL.md](.claude/skills/prd-update/SKILL.md) — PRD maintenance skill reference
- [wiki/](wiki/) — Legacy wiki (cloned from GitHub)
