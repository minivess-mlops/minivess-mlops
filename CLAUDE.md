# MinIVess MLOps v2

Model-agnostic biomedical segmentation MLOps platform. Clean rewrite from v0.1-alpha.

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
| **Agent Orchestration** | LangGraph |
| **CI/CD** | GitHub Actions + CML (ML-specific PR comments) |
| **Lineage** | OpenLineage (Marquez) |

## Critical Rules

1. **uv ONLY** — Never use pip, conda, poetry, or requirements.txt. Use `uv add`, `uv sync`, `uv run`.
2. **TDD MANDATORY** — All implementation MUST follow the self-learning-iterative-coder skill (`.claude/skills/self-learning-iterative-coder/SKILL.md`). Write failing tests FIRST, then implement. No exceptions.
3. **Pre-commit Required** — All changes must pass pre-commit hooks before commit.
4. **Encoding** — Always specify `encoding='utf-8'` for file operations.
5. **Paths** — Always use `pathlib.Path()`, never string concatenation.
6. **Timezone** — Always use `datetime.now(timezone.utc)`, never `datetime.now()`.
7. **`from __future__ import annotations`** — At the top of every Python file.

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
│   ├── data/                  # Data loading, DVC integration
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

## See Also

- [docs/modernize-minivess-mlops-plan.md](docs/modernize-minivess-mlops-plan.md) — Full modernization plan
- [docs/modernize-minivess-mlops-plan-prompt.md](docs/modernize-minivess-mlops-plan-prompt.md) — Original prompt and Q&A
- [.claude/skills/self-learning-iterative-coder/SKILL.md](.claude/skills/self-learning-iterative-coder/SKILL.md) — TDD skill reference
- [wiki/](wiki/) — Legacy wiki (cloned from GitHub)
