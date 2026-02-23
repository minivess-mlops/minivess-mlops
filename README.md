# MinIVess MLOps v2

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)
![pytest](https://img.shields.io/badge/tests-pytest-green)
![ruff](https://img.shields.io/badge/linter-ruff-orange)
![mypy](https://img.shields.io/badge/type%20checker-mypy-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Model-agnostic biomedical segmentation MLOps platform. Clean rewrite from v0.1-alpha.

Built on the dataset published in: Charissa Poon, Petteri Teikari *et al.* (2023),
"A dataset of rodent cerebrovasculature from in vivo multiphoton fluorescence microscopy imaging,"
*Scientific Data* 10, 141 -- doi: [10.1038/s41597-023-02048-8](https://doi.org/10.1038/s41597-023-02048-8)

---

## Architecture Overview

MinIVess v2 is organized around the **ModelAdapter** pattern: every segmentation model
(MONAI SegResNet, SwinUNETR, VISTA-3D, SAMv3, or custom) implements a common abstract
interface with six methods (`forward`, `get_config`, `load_checkpoint`, `save_checkpoint`,
`trainable_parameters`, `export_onnx`). The training engine, ensemble module, evaluation
pipeline, serving layer, and ONNX export all program against this interface.

**Pipeline stages:**

```
Data (DVC + NIfTI) --> Validation (Pydantic + Pandera + GE)
    --> Training (Hydra-zen configs, mixed precision, gradient checkpointing)
    --> Evaluation (TorchMetrics, MetricsReloaded: clDice, NSD, HD95)
    --> Ensembling (voting, greedy soup, calibration, conformal prediction)
    --> Serving (BentoML + ONNX Runtime + Gradio demo)
    --> Observability (MLflow, Langfuse, Prometheus/Grafana, OpenLineage)
```

---

## Technology Stack

| Layer | Tool | Role |
|-------|------|------|
| **Language** | Python 3.12+ | Runtime |
| **Package Manager** | uv | Dependency management (only) |
| **ML Framework** | PyTorch 2.5+ / MONAI 1.4+ / TorchIO / TorchMetrics | Training and inference |
| **Config (train)** | Hydra-zen | Experiment configs with Pydantic v2 validation |
| **Config (deploy)** | Dynaconf | Environment-layered deployment settings |
| **Data Validation** | Pydantic v2 + Pandera + Great Expectations | Schema, DataFrame, batch quality |
| **Model Validation** | Deepchecks Vision + WeightWatcher | Data integrity, spectral diagnostics |
| **Experiment Tracking** | MLflow 3.10 + DuckDB | Run tracking, model registry, SQL analytics |
| **Serving** | BentoML + ONNX Runtime + Gradio | Model serving and demo UI |
| **Calibration** | MAPIE + netcal | Conformal prediction, temperature scaling |
| **XAI** | Captum (3D) + SHAP + Quantus | Explainability and meta-evaluation |
| **Drift Detection** | Evidently AI | KS test, PSI-based drift monitoring |
| **Data Profiling** | whylogs | Lightweight statistical profiling |
| **LLM Observability** | Langfuse (self-hosted) + Braintrust | Tracing, cost tracking, offline evals |
| **Agent Orchestration** | LangGraph + LiteLLM | Multi-step workflows, provider flexibility |
| **Data Lineage** | OpenLineage (Marquez) | IEC 62304 traceability |
| **Label Quality** | Cleanlab + Label Studio | Annotation QA, multi-annotator workflows |
| **CI/CD** | GitHub Actions + CML | Lint, typecheck, test, ML-specific PR comments |
| **Linter/Formatter** | ruff | Linting and formatting |
| **Type Checker** | mypy | Static type analysis |
| **Test Framework** | pytest + Hypothesis | Unit, integration, property-based tests |
| **Infrastructure** | Docker Compose + Pulumi | Local dev stack, cloud-agnostic IaC |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/petteriTeikari/minivess_mlops.git
cd minivess_mlops

# Install dependencies (uv only -- never use pip/conda/poetry)
uv sync

# Run the test suite
uv run pytest tests/v2/ -x -q

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/

# Full three-gate verification (tests + lint + typecheck)
uv run pytest tests/v2/ -x -q && uv run ruff check src/ tests/ && uv run mypy src/
```

---

## Docker Compose Profiles

The observability and infrastructure stack is managed through Docker Compose profiles
with tiered resource usage:

```bash
# Core infrastructure (4 services, ~4 GB RAM)
# PostgreSQL, MinIO, MLflow, BentoML
docker compose -f deployment/docker-compose.yml --profile dev up

# Observability stack (7 services, ~8 GB RAM)
# Adds: Prometheus, Grafana, OpenTelemetry Collector
docker compose -f deployment/docker-compose.yml --profile monitoring up

# Full platform (12 services, ~16 GB RAM)
# Adds: Langfuse, Marquez, Label Studio, MONAI Label, Ollama
docker compose -f deployment/docker-compose.yml --profile full up
```

Copy `.env.example` to `.env` and adjust values before starting services.

---

## Directory Structure

```
minivess-mlops/
|-- src/minivess/               Main package
|   |-- adapters/               ModelAdapter ABC + implementations (SegResNet, SwinUNETR)
|   |-- agents/                 LangGraph agent definitions (training, evaluation)
|   |-- compliance/             SaMD audit trail, model cards (IEC 62304)
|   |-- config/                 Pydantic v2 config models
|   |-- data/                   Data loading, MONAI transforms, TorchIO augmentation
|   |-- ensemble/               Ensembling strategies, calibration, WeightWatcher
|   |-- observability/          MLflow tracking, DuckDB analytics, OpenTelemetry
|   |-- pipeline/               Training engine, metrics, loss functions
|   |-- serving/                BentoML service, ONNX inference, Gradio demo
|   +-- validation/             Pandera schemas, Great Expectations, Deepchecks, drift
|
|-- tests/
|   |-- v2/unit/                Fast isolated unit tests (102+ tests)
|   |-- v2/integration/         Service integration tests
|   +-- v2/e2e/                 End-to-end pipeline tests
|
|-- configs/
|   |-- experiment/             Hydra-zen experiment configs
|   +-- deployment/             Dynaconf environment settings
|
|-- deployment/
|   |-- docker-compose.yml      Profile-based Docker Compose (dev/monitoring/full)
|   +-- Dockerfile              Multi-stage build
|
|-- docs/
|   |-- adr/                    Architecture Decision Records
|   |-- modernize-minivess-mlops-plan.md   Full modernization plan
|   +-- claude-code-patterns.md            Claude Code TDD patterns
|
|-- .claude/                    Claude Code configuration
|   +-- skills/                 Self-learning iterative coder TDD skill
|
|-- pyproject.toml              uv + PEP 621 project definition
|-- CLAUDE.md                   AI development rules and quick reference
+-- LEARNINGS.md                Cross-session accumulated discoveries
```

---

## Phase Completion Status

The v2 modernization follows a phased roadmap. Phases 0 through 5 are complete:

| Phase | Name | Deliverables |
|-------|------|-------------|
| **0** | Foundation | uv + pyproject.toml, Docker Compose profiles, Pydantic v2 configs, Hydra-zen + Dynaconf, DVC 3.x, Pandera schemas, Great Expectations suites, Hypothesis tests, GitHub Actions CI, pre-commit hooks |
| **1** | Core ML Pipeline | ModelAdapter ABC, SegResNet adapter, SwinUNETR adapter, data loading (DVC + MONAI + TorchIO), training engine (mixed precision, gradient checkpointing, early stopping), MLflow integration, TorchMetrics, DuckDB analytics |
| **2** | Ensembling and Validation | Ensemble strategies (voting, mean, weighted), greedy soup, WeightWatcher spectral diagnostics, calibration (ECE/MCE + temperature scaling), Deepchecks Vision, Evidently drift detection |
| **3** | Serving | BentoML service definition, ONNX Runtime export and inference, Gradio demo frontend |
| **4** | Observability and Compliance | OpenTelemetry instrumentation, Prometheus + Grafana config, Langfuse LLM tracing, OpenLineage lineage, model cards (Mitchell et al. 2019), SaMD audit trail (IEC 62304, SHA-256 hashing) |
| **5** | Agent Layer | LangGraph agent definitions (training + evaluation state graphs), Braintrust evaluation framework, LiteLLM provider abstraction, Label Studio annotation workflows, Cleanlab label quality |
| **6** | Final Integration | End-to-end tests, ADRs, README, documentation (in progress) |

---

## Architecture Decision Records

Design rationale is documented in lightweight ADRs following the MADR format:

- [ADR-0001: Model Adapter Abstract Base Class](docs/adr/0001-model-adapter-abc.md)
- [ADR-0002: Dual Configuration System (Hydra-zen + Dynaconf)](docs/adr/0002-dual-config-system.md)
- [ADR-0003: Multi-Layer Validation ("Validation Onion")](docs/adr/0003-validation-onion.md)
- [ADR-0004: Local-First Observability Stack](docs/adr/0004-local-first-observability.md)
- [ADR-0005: Mandatory Test-Driven Development for SaMD](docs/adr/0005-tdd-mandatory.md)

---

## Development Workflow

This project enforces strict TDD (test-driven development). Every change follows the
RED-GREEN-VERIFY-FIX-CHECKPOINT-CONVERGE cycle. See [CLAUDE.md](CLAUDE.md) for the full
set of development rules.

**Three-gate verification** must pass before every commit:

1. `uv run pytest tests/v2/ -x -q` -- all tests pass
2. `uv run ruff check src/ tests/` -- no lint violations
3. `uv run mypy src/` -- no type errors

---

## Further Reading

- [Full Modernization Plan](docs/modernize-minivess-mlops-plan.md) -- architecture, tool rationale, phased roadmap
- [Claude Code Patterns](docs/claude-code-patterns.md) -- TDD patterns demonstrated during v2 development
- [Legacy Wiki](https://github.com/petteriTeikari/minivess_mlops/wiki) -- background from v0.1-alpha

---

## License

MIT
