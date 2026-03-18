# MinIVess MLOps v2

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)
![Docker](https://img.shields.io/badge/execution-Docker-2496ED?logo=docker)
![MONAI](https://img.shields.io/badge/framework-MONAI-00A86B)
![tests](https://img.shields.io/badge/tests-pytest-brightgreen)
![ruff](https://img.shields.io/badge/linter-ruff-orange)
![mypy](https://img.shields.io/badge/type%20checker-mypy-blue)

**A model-agnostic biomedical segmentation MLOps platform extending the MONAI ecosystem.**

MinIVess MLOps v2 is a research-grade software platform designed to scaffold
reproducible machine learning experimentation for preclinical biomedical imaging.
It provides Docker-per-flow isolation, SkyPilot intercloud compute, Prefect
orchestration, and a config-driven architecture where adding a new model, dataset,
or pipeline flow requires editing one YAML file -- not code. The companion
manuscript (working title: **NEUROVEX**) targets *Nature Protocols*.

The platform architecture aligns with the four pillars of the **MedMLOps framework**
([de Almeida et al., 2025](https://link.springer.com/article/10.1007/s00330-025-11654-6)):
(1) availability via containerised reproducible infrastructure,
(2) continuous monitoring and validation via drift detection and OpenLineage lineage,
(3) data protection via DVC versioning and opt-in multi-site pooling, and
(4) ease of use via zero-config defaults for PhD researchers.

Built on the dataset published in: Charissa Poon, Petteri Teikari *et al.* (2023),
"A dataset of rodent cerebrovasculature from in vivo multiphoton fluorescence
microscopy imaging," *Scientific Data* 10, 141 --
doi: [10.1038/s41597-023-02048-8](https://doi.org/10.1038/s41597-023-02048-8)

---

## Key Features

- **6 model families** behind a single `ModelAdapter` ABC: DynUNet (CNN baseline), MambaVesselNet++ (SSM hybrid), SAM3 Vanilla/TopoLoRA/Hybrid (foundation model variants), VesselFM (vessel-specific foundation model)
- **18 loss functions** -- from standard (Dice+CE) to topology-aware (clDice, CAPE, Betti matching, skeleton recall) to graph-constrained (compound graph topology)
- **12 Prefect flows** with Docker-per-flow isolation, spanning the full ML lifecycle from data engineering through biostatistics reporting
- **SkyPilot intercloud broker** ([Yang et al., NSDI'23](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng)) -- one command to launch GPU jobs on RunPod or GCP
- **OpenLineage (Marquez) data lineage** for IEC 62304 traceability -- automated audit trail for every pipeline execution
- **Evidently drift detection** + whylogs profiling + Prometheus/Grafana monitoring stack
- **BentoML + ONNX Runtime serving** with champion model discovery and Gradio demo UI
- **MetricsReloaded evaluation** -- clDice (trusted), MASD (trusted), DSC (foil) per [Maier-Hein et al. (2024)](https://doi.org/10.1038/s41592-023-02151-z)
- **3-fold cross-validation** (seed=42) with bootstrap confidence intervals and paired statistical tests
- **Conformal uncertainty quantification** -- 5 methods (split conformal, morphological, distance transform, risk-controlling, MAPIE)
- **Post-training plugins** -- 6 config-driven enhancements (SWA, Multi-SWA, model merging, calibration, CRC conformal, ConSeCo FP control)
- **Knowledge graph** -- 69+ Bayesian decision nodes across 6 layers, driving spec-driven development
- **FDA-ready audit infrastructure** -- AuditTrail, compliance module, PCCP-compatible factorial design, CycloneDX SBOM (planned)

---

## Architecture Overview

### Two-Tier Orchestration

The platform employs a **two-tier orchestration architecture** separating concerns between deterministic pipeline execution and LLM-assisted reasoning:

| Tier | Framework | Scope | Examples |
|------|-----------|-------|---------|
| **Macro-orchestration** | Prefect 3.x | Deterministic ML pipeline flows | Train, Eval, Deploy, Biostatistics |
| **Micro-orchestration** | Pydantic AI | LLM-assisted tasks within flows | Result summarisation, drift triage, figure narration |

Prefect flows execute the deterministic ML pipeline (data engineering, training,
evaluation, deployment). Within individual flows, **Pydantic AI agents** provide
LLM-assisted capabilities -- for example, summarising experiment results,
triaging drift alerts, or generating figure narratives for the manuscript. This
separation ensures that the core pipeline remains fully reproducible and
deterministic, while LLM capabilities are additive, optional, and auditable
via Langfuse tracing. See [ADR-0007](docs/adr/0007-pydantic-ai-over-langgraph.md)
for the architectural rationale.

**Planned**: CopilotKit (AG-UI protocol) + WebMCP for agentic dashboard and
annotation interfaces, enabling interactive researcher-AI collaboration.

### Three-Environment Model

| Environment | Docker | Compute | Data | Purpose |
|-------------|--------|---------|------|---------|
| **local** | Docker Compose | Local GPU (e.g., RTX 2070 Super 8 GB) | MinIO (local) | Fast iteration, `uv run pytest` |
| **env** (RunPod) | Docker image via SkyPilot | RunPod RTX 4090 (24 GB) | Network Volume (upload from local) | Quick GPU experiments |
| **staging/prod** (GCP) | Docker image via SkyPilot | GCP L4/A100 spot | GCS buckets | Production runs, paper results |

### Pipeline Architecture

```
                     Prefect Orchestration (Docker-per-flow)
                     =======================================

Flow 1: Data Eng.        Flow 2: Training         Flow 2.5: Post-Training
  DVC + NIfTI              Hydra-zen configs        6 post-hoc plugins
  TorchIO augmentation     Mixed precision           SWA + Multi-SWA
  Pandera validation       18 loss functions          Model merging (slerp)
  whylogs profiling        6 model families           Calibration (temp/Platt)
        |                        |                    CRC conformal + ConSeCo
        v                        v                           |
      MLflow  <=============== MLflow ===================> MLflow
        |                        |                           |
        v                        v                           v
Flow 3: Analysis          Flow 4: Deployment        Flow 5: Dashboard
  MetricsReloaded eval      Champion discovery        Paper figures (PNG+SVG)
  Bootstrap CIs             ONNX export + validate    DuckDB analytics
  Conformal UQ              BentoML model store       Drift reports
  Ensemble strategies       Gradio demo               Comparison tables

                 ┌──────────────────────────────────┐
                 │   OpenLineage Event Bus           │
                 │   START/COMPLETE/FAIL per flow    │
                 │   → Marquez (lineage graph)       │
                 │   → AuditTrail (IEC 62304)        │
                 └──────────────────────────────────┘
```

### Docker Three-Tier Hierarchy

```
Tier A (GPU):   nvidia/cuda:12.6.3 --> minivess-base:latest       (~8-12 GB)
Tier B (CPU):   python:3.13-slim   --> minivess-base-cpu:latest    (~1.5-2.5 GB)
Tier C (Light): python:3.13-slim   --> minivess-base-light:latest  (~1.0-1.5 GB)
```

Each tier uses a **two-stage builder-runner** pattern. Flow Dockerfiles
are thin -- only `COPY`, `ENV`, `CMD` -- they never run `apt-get` or `uv`.

---

## Regulatory Readiness and Compliance Architecture

While MinIVess is a **preclinical research platform** (rodent cerebrovasculature),
its architecture is designed to scale to clinical MLOps without retrofitting.
The compliance infrastructure supports future FDA SaMD and EU MDR/IVDR pathways.

### Current Compliance Infrastructure

| Component | Status | FDA/IEC 62304 Relevance |
|-----------|--------|------------------------|
| **OpenLineage (Marquez)** | Implemented, wiring to flows in progress | IEC 62304 §8 configuration management |
| **AuditTrail** | Implemented (127 LOC) | Test set access logging, model deployment audit |
| **IEC 62304 framework** | Partial (TraceabilityMatrix, PCCPTemplate) | Software lifecycle traceability |
| **Regulatory doc generator** | Implemented | Auto-generates DHF, Risk Analysis, SRS |
| **DVC data versioning** | Active | Data provenance and version control |
| **CycloneDX SBOM** | Planned | FDA Section 524B requirement (mandatory since 2023) |
| **Drift detection** | Implemented (Evidently) | Postmarket surveillance readiness |

### PCCP-Compatible Factorial Design

The platform's factorial experiment design (4 models x 3 losses x 2 aux_calib x
3 post-training x 2 recalibration x 5 ensemble) is architecturally equivalent to
an FDA **Predetermined Change Control Plan** (PCCP) -- it documents predetermined
model variations with pre-specified acceptance criteria and sequestered test data
validation. See [K252366 (a2z-Unified-Triage)](https://510k.innolitics.com/) for
a cleared device using the same pattern.

### Regulatory Planning Documents

| Document | Focus |
|----------|-------|
| [`regops-fda-regulation-reporting-qms-samd-iec62304-mlops-report.md`](docs/planning/regops-fda-regulation-reporting-qms-samd-iec62304-mlops-report.md) | First-pass: test set firewall, OpenLineage, PCCP, 30+ citations |
| [`fda-insights-second-pass.md`](docs/planning/fda-insights-second-pass.md) | Second-pass: SBOM, QMSR, SecOps, MLOps maturity, 60+ citations |
| [`openlineage-marquez-iec62304-report.md`](docs/planning/openlineage-marquez-iec62304-report.md) | OpenLineage/Marquez integration analysis |

---

## Model Families

Six model families for the Nature Protocols comparison:

| Model | Family | Adapter | Training Strategy | VRAM | Status |
|-------|--------|---------|-------------------|------|--------|
| **DynUNet** | CNN baseline | `adapters/dynunet.py` | Full training (100 epochs, 3 folds) | ~3.5 GB | Results available |
| **MambaVesselNet++** | SSM hybrid | `adapters/mambavesselnet.py` | Full training | TBD | Code complete |
| **SAM3 Vanilla** | Foundation (frozen) | `adapters/sam3_vanilla.py` | Zero-shot or decoder fine-tune | ~2.9 GB | GPU runs pending |
| **SAM3 TopoLoRA** | Foundation (LoRA) | `adapters/sam3_topolora.py` | LoRA fine-tune (rank=16, alpha=32) | ~16 GB | GPU runs pending |
| **SAM3 Hybrid** | Foundation (fusion) | `adapters/sam3_hybrid.py` | SAM3 features + DynUNet 3D decoder | ~6 GB | Partially validated |
| **VesselFM** | Foundation (pretrained) | `adapters/vesselfm.py` | Zero-shot + fine-tune on external data | TBD | GPU runs pending |

Every model implements the `ModelAdapter` ABC. Adding a new model = one new file
implementing this interface + one YAML config.

---

## Quick Start

### Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/) (the **only** supported package manager)
- Docker (for pipeline execution) and Docker Compose V2
- NVIDIA GPU with CUDA (optional for local development; required for training)

### Install and Verify

```bash
# Clone and install (--all-extras is REQUIRED for development)
git clone https://github.com/petteriTeikari/minivess_mlops.git
cd minivess_mlops
uv sync --all-extras

# Run the staging test suite (fast, no model loading, <3 min)
make test-staging

# Three-gate verification: tests + lint + types
make test-staging && uv run ruff check src/ tests/ && uv run mypy src/
```

### Docker Infrastructure

```bash
cp .env.example .env        # Configure environment
docker network create minivess-network
docker compose -f deployment/docker-compose.yml --profile dev up -d

# Run a training flow
docker compose --env-file .env -f deployment/docker-compose.flows.yml run --rm \
  --shm-size 8g -e EXPERIMENT=dynunet_e2e_debug train
```

---

## Cloud Execution

All cloud compute is managed through [SkyPilot](https://skypilot.readthedocs.io/) --
an intercloud broker that operates like Slurm for multi-cloud environments.
SkyPilot YAML files specify Docker images (bare VM setup is banned).

| Provider | Environment | Role | Data Storage |
|----------|------------|------|--------------|
| **RunPod** | env (dev) | Quick GPU experiments, instant provisioning | Network Volume |
| **GCP** | staging + prod | Production runs, Pulumi IaC | GCS (`gs://minivess-mlops-dvc-data`) |

Cloud configuration flows through **Hydra config groups** (`configs/cloud/`,
`configs/registry/`). Research groups with different cloud providers override
via `configs/lab/lab_name.yaml` -- zero code changes required.

---

## Knowledge Graph

The project employs a **6-layer knowledge architecture** with 69+ Bayesian
decision nodes across 11 domains for systematic architectural decision-making:

```
L0: .claude/rules/ + CLAUDE.md            -- Constitution (invariant rules)
L1: docs/planning/ + MEMORY.md            -- Hot Context (current work)
L2: knowledge-graph/navigator.yaml        -- Navigator (domain routing)
L3: knowledge-graph/decisions/*.yaml       -- Evidence (69+ decision nodes)
    knowledge-graph/domains/*.yaml         -- Materialised winners
L4: openspec/specs/                        -- Specifications (GIVEN/WHEN/THEN)
L5: src/ + tests/                          -- Implementation
```

**Information flow**: PRD decisions propagate downward through KG materialisation
to OpenSpec specifications to code. Experimental results propagate upward through
posterior updates and belief propagation.

Entry point: [`knowledge-graph/navigator.yaml`](knowledge-graph/navigator.yaml)

---

## Technology Stack

| Layer | Tool | Role |
|-------|------|------|
| Language | Python 3.12+ | Runtime |
| Package Manager | uv | Dependency management (exclusively) |
| ML Framework | PyTorch + MONAI + TorchIO | Training, augmentation, inference |
| Orchestration | Prefect 3.x | Deterministic pipeline orchestration (macro) |
| Agent Framework | Pydantic AI | LLM-assisted micro-orchestration ([ADR-0007](docs/adr/0007-pydantic-ai-over-langgraph.md)) |
| Config (train) | Hydra-zen | Experiment configs with Pydantic v2 validation |
| Config (deploy) | Dynaconf | Environment-layered deployment settings |
| Data Validation | Pydantic v2 + Pandera + Great Expectations | Schema, DataFrame, batch quality |
| Experiment Tracking | MLflow + DuckDB | Run tracking, model registry, SQL analytics |
| HPO | Optuna + ASHA | Multi-objective hyperparameter optimisation |
| Serving | BentoML + ONNX Runtime + Gradio | Model serving and demo UI |
| Data Lineage | OpenLineage (Marquez) | IEC 62304 traceability |
| Drift Detection | Evidently AI | KS test, PSI, kernel MMD |
| Data Profiling | whylogs | Lightweight statistical profiling |
| Monitoring | Prometheus + Grafana + AlertManager | Dashboards, alerting |
| Compute | SkyPilot | Intercloud broker (RunPod + GCP) |
| Infrastructure | Docker Compose + Pulumi | Local dev stack, GCP IaC |
| Linter/Formatter | ruff | Linting and formatting |
| Type Checker | mypy | Static type analysis |
| Tests | pytest + Hypothesis | Unit, integration, property-based |
| Topology | gudhi + networkx + scipy | Persistent homology, graph analysis |
| XAI | Captum + SHAP + Quantus | Explainability and meta-evaluation |
| LLM Observability | Langfuse + Braintrust + LiteLLM | Agent tracing, evals, provider flexibility |
| Compliance | AuditTrail + IEC 62304 framework + CycloneDX (planned) | FDA/MDR readiness |

---

## Testing

### Three-Tier Strategy

| Tier | Command | What Runs | Target Time |
|------|---------|-----------|-------------|
| **Staging** | `make test-staging` | No model loading, no slow, no integration | < 3 min |
| **Prod** | `make test-prod` | Everything except GPU instance tests | 5-10 min |
| **GPU** | `make test-gpu` | SAM3 + GPU-heavy tests (external GPU only) | GPU instance |

Pre-commit hooks enforce formatting, trailing whitespace, YAML validation,
knowledge graph link integrity, and bibliography citation integrity.

---

## Directory Structure

```
minivess-mlops/
|-- src/minivess/                  Main package
|   |-- adapters/                  ModelAdapter ABC + 6 model families
|   |-- pipeline/                  Training, evaluation, metrics, losses
|   |-- ensemble/                  Ensembling, UQ, calibration
|   |-- orchestration/flows/       12+ Prefect 3.x flows
|   |-- config/                    Pydantic v2 config models
|   |-- data/                      Data loading, profiling, DVC
|   |-- serving/                   BentoML, ONNX, Gradio
|   |-- observability/             MLflow tracking, OpenLineage lineage, DuckDB analytics
|   |-- agents/                    Pydantic AI micro-orchestration (ADR-0007)
|   |-- compliance/                IEC 62304 audit trail, model cards, regulatory docs
|   +-- validation/                Pandera, Great Expectations
|
|-- tests/                         Unit, integration, and E2E test suites
|-- configs/                       Hydra experiment configs, model profiles, splits
|-- deployment/                    Docker, SkyPilot, Pulumi, Grafana, Prometheus
|-- knowledge-graph/               69+ Bayesian decision nodes across 11 domains
|-- docs/                          ADRs, planning documents, research reports
+-- openspec/                      Spec-driven development (GIVEN/WHEN/THEN)
```

---

## Contributing

1. **uv only** -- never use pip, conda, or poetry. Install with `uv sync --all-extras`.
2. **TDD mandatory** -- write failing tests first, then implement.
3. **Pre-commit hooks** -- all changes must pass before commit.
4. **Three-gate verification** -- `make test-staging && uv run ruff check src/ tests/ && uv run mypy src/`
5. **Library-first** -- search for existing implementations before writing custom code.
6. **Docker is the execution model** -- all pipeline execution goes through Prefect flows in Docker containers.
7. **Config-driven** -- specific tasks, models, losses, and metrics are YAML config instantiations, not code branches.

### Architecture Decision Records

| ADR | Decision |
|-----|----------|
| [0001](docs/adr/0001-model-adapter-abc.md) | Model Adapter Abstract Base Class |
| [0002](docs/adr/0002-dual-config-system.md) | Dual Configuration System (Hydra-zen + Dynaconf) |
| [0003](docs/adr/0003-validation-onion.md) | Multi-Layer Validation ("Validation Onion") |
| [0004](docs/adr/0004-local-first-observability.md) | Local-First Observability Stack |
| [0005](docs/adr/0005-tdd-mandatory.md) | Mandatory Test-Driven Development |
| [0006](docs/adr/0006-sam3-variant-architecture.md) | SAM3 Variant Architecture |
| [0007](docs/adr/0007-pydantic-ai-over-langgraph.md) | Pydantic AI over LangGraph for Agent Orchestration |

---

## Citation

If you use this platform, please cite the underlying dataset:

> Charissa Poon, Petteri Teikari *et al.* (2023). "A dataset of rodent
> cerebrovasculature from in vivo multiphoton fluorescence microscopy imaging."
> *Scientific Data* 10, 141. doi: [10.1038/s41597-023-02048-8](https://doi.org/10.1038/s41597-023-02048-8)

---

## Roadmap

### Completed

- Foundation (uv, Docker, configs, pre-commit, knowledge graph)
- Core ML pipeline (6 model families, 18 losses, training engine)
- DynUNet baseline results (4 losses x 3 folds x 100 epochs)
- Evaluation (MetricsReloaded suite, bootstrap CIs, paired tests)
- Ensembling (7 strategies) + conformal UQ (5 methods)
- Serving (BentoML, ONNX Runtime, Gradio)
- Observability (MLflow, DuckDB, Prometheus, Grafana, Evidently, whylogs)
- Post-training plugin architecture (6 plugins, Flow 2.5)
- SAM3 integration (3 adapter variants)
- Pydantic AI agent layer (experiment summariser, drift triage, figure narration)
- FDA readiness planning (test set firewall, OpenLineage, PCCP alignment)

### In Progress

- 6-factor factorial experiment on GCP L4 spot instances
- OpenLineage flow wiring (Issue [#799](https://github.com/petteriTeikari/minivess-mlops/issues/799))
- CycloneDX SBOM generation (Issue [#821](https://github.com/petteriTeikari/minivess-mlops/issues/821))
- Nature Protocols manuscript assembly (NEUROVEX)

### Planned

- CopilotKit (AG-UI) + WebMCP for agentic dashboard/annotation
- Multi-site opt-in telemetry (PostHog, Sentry)
- Federated learning evaluation (NVIDIA FLARE vs MONAI FL)
- VesselFM integration and external-data evaluation
- QMSR production controls documentation

---

## Further Reading

- [Knowledge Graph Navigator](knowledge-graph/navigator.yaml) -- entry point for architectural decisions
- [FDA Readiness Report](docs/planning/regops-fda-regulation-reporting-qms-samd-iec62304-mlops-report.md) -- compliance gap analysis
- [FDA Insights Second Pass](docs/planning/fda-insights-second-pass.md) -- SBOM, SecOps, QMSR, PCCP
- [SAM3 Literature Report](docs/planning/sam3-literature-research-report.md) -- foundation model survey
- [Loss Variation Results](docs/results/dynunet_loss_variation_v2_report.md) -- DynUNet baseline analysis
- [GCP Setup Tutorial](docs/planning/gcp-setup-tutorial.md) -- step-by-step cloud setup

---

## License

MIT
