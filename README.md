# VASCADIA

> **VASCADIA: A MONAI-based MLOps scaffold for reproducible vasculature segmentation**

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)
![Docker](https://img.shields.io/badge/execution-Docker-2496ED?logo=docker)
![MONAI](https://img.shields.io/badge/framework-MONAI-00A86B)
![tests](https://img.shields.io/badge/tests-pytest-brightgreen)
![ruff](https://img.shields.io/badge/linter-ruff-orange)
![mypy](https://img.shields.io/badge/type%20checker-mypy-blue)
![version](https://img.shields.io/badge/version-0.2.0--beta-yellow)

**A model-agnostic biomedical segmentation MLOps platform extending the [MONAI](https://monai.io/) ecosystem.**

VASCADIA is a research-grade software platform designed to scaffold
reproducible machine learning experimentation for preclinical biomedical imaging.
It provides Docker-per-flow isolation, SkyPilot intercloud compute, Prefect
orchestration, and a config-driven architecture where adding a new model, dataset,
or pipeline flow requires editing one YAML file -- not code. The companion
manuscript targets *Nature Protocols*.

The platform architecture aligns with the four pillars of the **MedMLOps framework**
([de Almeida et al., 2025](https://link.springer.com/article/10.1007/s00330-025-11654-6)):
(1) availability via containerised reproducible infrastructure,
(2) continuous monitoring and validation via drift detection and [OpenLineage](https://openlineage.io/) lineage,
(3) data protection via DVC versioning and opt-in multi-site pooling, and
(4) ease of use via zero-config defaults for PhD researchers.

Built on the dataset published in: Charissa Poon, Petteri Teikari *et al.* (2023),
"A dataset of rodent cerebrovasculature from in vivo multiphoton fluorescence
microscopy imaging," *Scientific Data* 10, 141 --
doi: [10.1038/s41597-023-02048-8](https://doi.org/10.1038/s41597-023-02048-8)

---

## Key Features

- **6 model families** behind a single `ModelAdapter` ABC: [DynUNet](https://docs.monai.io/en/stable/networks.html) (CNN baseline), [MambaVesselNet++](https://doi.org/10.1145/3757324) (SSM hybrid), [SAM3](https://github.com/facebookresearch/sam3) Vanilla/TopoLoRA/Hybrid (foundation model variants), [VesselFM](https://arxiv.org/abs/2411.17386) (vessel-specific foundation model)
- **18 loss functions** -- from standard (Dice+CE) to topology-aware ([clDice](https://arxiv.org/abs/2003.07311), [CAPE](https://arxiv.org/abs/2504.00753), [Betti matching](https://arxiv.org/abs/2211.15272), [skeleton recall](https://arxiv.org/abs/2404.03010)) to graph-constrained (compound graph topology)
- **15 Prefect flows** with Docker-per-flow isolation, spanning the full ML lifecycle from data engineering through biostatistics reporting
- **SkyPilot intercloud broker** ([Yang et al., NSDI'23](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng)) -- one command to launch GPU jobs on RunPod or GCP
- **[OpenLineage](https://openlineage.io/) ([Marquez](https://marquezproject.ai/)) data lineage** for [IEC 62304](https://www.iso.org/standard/38421.html) traceability -- automated audit trail for every pipeline execution
- **5-layer observability** -- CUDA guard (fail-fast), GPU heartbeat (pynvml), structured epoch logging (JSONL), [Grafana LGTM](https://grafana.com/docs/opentelemetry/docker-lgtm/) backend ([OpenTelemetry](https://opentelemetry.io/) + Prometheus + Tempo + Loki), [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter) GPU hardware metrics -- Docker HEALTHCHECK on all 10 flow services
- **[Evidently](https://www.evidentlyai.com/) drift detection** + [whylogs](https://whylogs.readthedocs.io/) profiling + [Prometheus](https://prometheus.io/)/[Grafana](https://grafana.com/) monitoring stack
- **[BentoML](https://www.bentoml.com/) + [ONNX Runtime](https://onnxruntime.ai/) serving** with champion model discovery and [Gradio](https://www.gradio.app/) demo UI
- **MetricsReloaded evaluation** -- clDice (trusted), MASD (trusted), DSC (foil) per [Maier-Hein et al. (2024)](https://doi.org/10.1038/s41592-023-02151-z)
- **3-fold cross-validation** (seed=42) with bootstrap confidence intervals and paired statistical tests
- **Conformal uncertainty quantification** -- 5 methods (split conformal, morphological, distance transform, risk-controlling, MAPIE)
- **Post-training plugins** -- 7 config-driven enhancements (checkpoint averaging, subsampled ensemble, SWAG ([Maddox et al. 2019](https://arxiv.org/abs/1902.02476)), model merging, calibration, CRC conformal, ConSeCo FP control)
- **Knowledge graph** -- 75+ Bayesian decision nodes across 6 layers, driving spec-driven development
- **FDA-ready audit infrastructure** -- AuditTrail, compliance module, PCCP-compatible factorial design, [CycloneDX](https://cyclonedx.org/) SBOM (planned)

---

## Architecture Overview

### Two-Tier Orchestration: Deterministic Pipeline + Agentic Intelligence

The platform employs a **two-tier orchestration architecture** that cleanly
separates deterministic pipeline execution from LLM-assisted reasoning. This
design ensures that the core ML pipeline remains fully reproducible while
providing a natural extension point for agentic capabilities as the field matures.

| Tier | Framework | Scope | Determinism | Examples |
|------|-----------|-------|-------------|---------|
| **Macro-orchestration** | Prefect 3.x | Pipeline flows (DAG) | Fully deterministic | Train, Eval, Deploy, Biostatistics |
| **Micro-orchestration** | Pydantic AI | Tasks within flows | LLM-assisted, optional | Result summarisation, drift triage, figure narration |

**Why this separation matters.** Prefect flows execute the deterministic ML
pipeline: data engineering, training, post-training, evaluation, deployment,
biostatistics. Every run produces identical outputs given identical inputs --
the reproducibility guarantee essential for both scientific publication and
regulatory compliance. Within individual flows, **[Pydantic AI](https://ai.pydantic.dev/) agents** provide
LLM-assisted capabilities that are *additive, optional, and auditable* via
Langfuse tracing. If the LLM is unavailable, the flow runs to completion; only
the LLM-generated summaries are missing. See
[ADR-0007](docs/adr/0007-pydantic-ai-over-langgraph.md) for the rationale behind
choosing Pydantic AI over LangGraph.

**The path to "more agentic."** The two-tier architecture is explicitly designed
to grow. Current agents (experiment summariser, drift triage, figure narrator) are
read-only -- they observe flow outputs and produce text. Future agents can take
increasingly autonomous actions while remaining within the Prefect flow boundary.

**Concrete example: the Data Acquisition Flow.** The platform already includes
`acquisition_flow.py` (Flow 0) -- currently a deterministic downloader that checks
dataset availability, fetches files (VesselNN via git clone; MiniVess/DeepVess
via manual download), converts OME-TIFF to NIfTI, and logs provenance to MLflow.
This is deliberately "dumb" -- it executes a fixed acquisition plan without
intelligence. The architecture anticipates splitting this into two complementary
flows as the platform matures:

| Flow | Intelligence | What It Does |
|------|-------------|-------------|
| **Flow 0a: Batch Downloader** (current) | None (deterministic) | Downloads known datasets, converts formats, verifies checksums, logs provenance. The "PhD student onboarding" path. |
| **Flow 0b: Active Acquisition Agent** (future) | Pydantic AI agent | Real-time adaptive data acquisition *during* 2-photon microscopy experiments. Conformal bandit selects next imaging field based on segmentation uncertainty from edge inference. Decides when "enough data has been collected" for a given vascular morphology class. |

Flow 0b represents the research frontier: intelligent agents that understand when
the current dataset is insufficient, what types of vessels are under-represented,
and how to guide the microscope operator to collect the most informative next
sample. This is a natural evolution of the two-tier architecture -- Prefect
orchestrates the acquisition session, while a Pydantic AI agent reasons about
data sufficiency and acquisition strategy within the flow.

The four data-savvy agent capabilities
([Seedat et al. (2025). "What's the Next Frontier for Data-Centric AI? Data Savvy Agents!"
*ICLR DATA-FM Workshop*.](https://openreview.net/)) map directly to existing flows
and planned extensions:

| Capability | Current Flow | Current State | Future Agentic Extension |
|------------|-------------|---------------|--------------------------|
| **Proactive data acquisition** | `acquisition_flow.py` | Deterministic downloader + format conversion | Active acquisition agent guiding 2-PM microscopy, conformal bandit for field selection |
| **Sophisticated data processing** | `data_flow.py` | Pandera validation, whylogs profiling, TorchIO augmentation | Agent diagnoses data quality issues, flags annotation anomalies, suggests re-annotation |
| **Interactive evaluation** | `analysis_flow.py` | MetricsReloaded with bootstrap CIs | Agent generates natural-language summaries, proposes new eval criteria from failure modes |
| **Continual adaptation** | `drift_simulation_flow.py` | Evidently drift detection + Prometheus alerts | Agent triages drift, recommends PCCP-compliant retraining, adjusts monitoring thresholds |

Prefect ensures the deterministic backbone required for reproducibility and
regulatory compliance, while Pydantic AI agents can progressively assume
responsibility for each capability. The macro/micro boundary means every agentic
enhancement is *opt-in* -- a lab without LLM access runs the same pipeline with
the same results; they simply lack the AI-generated summaries and recommendations.

**Planned agentic UI**: CopilotKit (AG-UI protocol) + WebMCP for agentic dashboard
and annotation interfaces, enabling interactive researcher-AI collaboration where
the interface itself adapts to the researcher's workflow.

### Three-Environment Model

| Environment | Docker | Compute | Data | Purpose |
|-------------|--------|---------|------|---------|
| **local** | Docker Compose | Local GPU (e.g., RTX 2070 Super 8 GB) | [MinIO](https://min.io/) (local) | Fast iteration, `uv run pytest` |
| **env** (RunPod) | Docker image via SkyPilot | RunPod RTX 4090 (24 GB) | Network Volume (upload from local) | Quick GPU experiments |
| **staging/prod** (GCP) | Docker image via SkyPilot | GCP L4/A100 spot | GCS buckets | Production runs, paper results |

### Pipeline Architecture

```
                     Prefect Orchestration (Docker-per-flow)
                     =======================================

Flow 1: Data Eng.        Flow 2: Training (parent + 2 sub-flows)
  DVC + NIfTI              Hydra-zen configs
  TorchIO augmentation     Mixed precision           Sub-flow 1: Training
  Pandera validation       18 loss functions            → "none" cell (free)
  whylogs profiling        6 model families          Sub-flow 2: Post-Training
        |                        |                     → SWAG (Maddox 2019)
        |                        |                     → 7 plugins total
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

### Observability Architecture

Every flow and every Docker container has production-grade observability,
enforced by AST tests that verify context managers are **called**, not just imported.

| Layer | Component | What It Monitors | Implementation |
|-------|-----------|-----------------|----------------|
| **1. Fail-Fast Guard** | `require_cuda_context()` | CUDA driver/toolkit mismatch | Raises `RuntimeError` before any GPU allocation |
| **2. GPU Heartbeat** | `GpuHeartbeatMonitor` | GPU utilisation, memory, temperature | Background thread writes `heartbeat.json` every 30s |
| **3. Structured Epoch Logging** | `StructuredEventLogger` | Per-epoch train/val loss, dice, LR, ETA | JSONL events to `events.jsonl` + `sys.stdout.flush()` |
| **4. Telemetry Backend** | [Grafana LGTM](https://grafana.com/docs/opentelemetry/docker-lgtm/) | Traces, metrics, logs (unified) | Single container: [OpenTelemetry](https://opentelemetry.io/) Collector + Prometheus + Tempo + Loki + Grafana |
| **5. GPU Hardware Metrics** | [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter) | GPU util%, memory, temp, ECC errors, PCIe | Prometheus scrape at `:9400`, pre-built Grafana dashboard |

**Docker HEALTHCHECK** on all 10 flow services: GPU flows check `heartbeat.json`
staleness, CPU flows check `events.jsonl` staleness. `docker ps` shows
`(healthy)` / `(unhealthy)` for every container.

**Prefect task hooks** on all 77 `@task` decorators: automatic timing and failure
logging for every pipeline task, visible in Prefect UI at `localhost:4200`.

Activate the observability stack:
```bash
docker compose --env-file .env -f deployment/docker-compose.yml --profile observability up -d
```

---

## Regulatory Readiness and Compliance Architecture

While VASCADIA is a **preclinical research platform** (rodent cerebrovasculature),
its architecture is designed to scale to clinical MLOps without retrofitting.
The compliance infrastructure supports future [FDA SaMD](https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd) and EU MDR/IVDR pathways.

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

The platform's 4-layer factorial experiment design is architecturally equivalent to
an FDA **Predetermined Change Control Plan** (PCCP):

| Layer | Factors | Execution |
|-------|---------|-----------|
| **A: Training** | 4 models x 3 losses x 2 aux_calib = 24 cells | Cloud GPU (SkyPilot) |
| **B: Post-Training** | {none, SWAG} = 2 methods | Same GPU job (parent flow) |
| **C: Analysis** | 2 recalibration x 5 ensemble | Local CPU |
| **D: Biostatistics** | Analytical choices | Local CPU |

Each layer documents predetermined model variations with pre-specified acceptance
criteria and sequestered test data validation. See
[K252366 (a2z-Unified-Triage)](https://510k.innolitics.com/) for a cleared device
using the same pattern.

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
git clone https://github.com/petteriTeikari/vascadia.git
cd vascadia
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

| Provider | Environment | Role | Data Storage | MLflow |
|----------|------------|------|--------------|--------|
| **RunPod** | env (dev) | Quick GPU experiments, instant provisioning | Network Volume | [DagsHub](https://dagshub.com/) (remote) |
| **GCP** | staging + prod | Production runs, Pulumi IaC | GCS (`gs://minivess-mlops-dvc-data`) | [DagsHub](https://dagshub.com/) (remote) |

Cloud configuration flows through **Hydra config groups** (`configs/cloud/`,
`configs/registry/`). Research groups with different cloud providers override
via `configs/lab/lab_name.yaml` -- zero code changes required.

---

## Knowledge Graph

The project employs a **6-layer knowledge architecture** with 75+ Bayesian
decision nodes across 11 domains for systematic architectural decision-making:

```
L0: .claude/rules/ + CLAUDE.md            -- Constitution (invariant rules)
L1: docs/planning/ + MEMORY.md            -- Hot Context (current work)
L2: knowledge-graph/navigator.yaml        -- Navigator (domain routing)
L3: knowledge-graph/decisions/*.yaml       -- Evidence (75+ decision nodes)
    knowledge-graph/domains/*.yaml         -- Materialised winners
L4: openspec/specs/                        -- Specifications (GIVEN/WHEN/THEN)
L5: src/ + tests/                          -- Implementation
```

**Information flow**: PRD decisions propagate downward through KG materialisation
to OpenSpec specifications to code. Experimental results propagate upward through
posterior updates and belief propagation.

Entry point: [`knowledge-graph/navigator.yaml`](knowledge-graph/navigator.yaml)

### Context Management Tooling

The knowledge graph is supplemented by automated context management infrastructure
that prevents knowledge loss across Claude Code sessions:

| Tool | Purpose | Scale |
|------|---------|-------|
| **[code-review-graph](https://github.com/tirth8205/code-review-graph)** MCP | Tree-sitter structural code graph with blast radius analysis | 12,729 nodes, 85,399 edges |
| **Metalearning search** | DuckDB full-text search over failure pattern docs | 90 docs indexed |
| **Config-to-code graph** | Maps Hydra YAML configs to Python consumers | 97 YAML files, 624 edges |
| **Decision registry** | DO_NOT_RE_ASK lookup table for decided questions | 10 entries (100% coverage) |
| **Planning SOP** | Mandatory 6-step pre-planning context load | `.claude/rules/planning-sop.md` |
| **Analytics dashboards** | Violation frequency, memory churn, registry coverage | `scripts/context_analytics.py` |

Skills: `/search-metalearning` (search failure patterns), `/plan-context-load` (pre-planning SOP).

---

## Technology Stack

| Layer | Tool | Role |
|-------|------|------|
| Language | Python 3.12+ | Runtime |
| Package Manager | uv | Dependency management (exclusively) |
| ML Framework | [PyTorch](https://pytorch.org/) + [MONAI](https://monai.io/) + [TorchIO](https://torchio.readthedocs.io/) | Training, augmentation, inference |
| Orchestration | Prefect 3.x | Deterministic pipeline orchestration (macro) |
| Agent Framework | [Pydantic AI](https://ai.pydantic.dev/) | LLM-assisted micro-orchestration ([ADR-0007](docs/adr/0007-pydantic-ai-over-langgraph.md)) |
| Config (train) | [Hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) | Experiment configs with Pydantic v2 validation |
| Config (deploy) | [Dynaconf](https://www.dynaconf.com/) | Environment-layered deployment settings |
| Data Validation | [Pydantic](https://docs.pydantic.dev/) v2 + [Pandera](https://pandera.readthedocs.io/) + [Great Expectations](https://greatexpectations.io/) | Schema, DataFrame, batch quality |
| Experiment Tracking | [MLflow](https://mlflow.org/) + [DuckDB](https://duckdb.org/) | Run tracking, model registry, SQL analytics |
| HPO | [Optuna](https://optuna.org/) + ASHA | Multi-objective hyperparameter optimisation |
| Serving | [BentoML](https://www.bentoml.com/) + [ONNX Runtime](https://onnxruntime.ai/) + [Gradio](https://www.gradio.app/) | Model serving and demo UI |
| Data Lineage | [OpenLineage](https://openlineage.io/) ([Marquez](https://marquezproject.ai/)) | [IEC 62304](https://www.iso.org/standard/38421.html) traceability |
| Drift Detection | [Evidently](https://www.evidentlyai.com/) AI | KS test, PSI, kernel MMD |
| Data Profiling | [whylogs](https://whylogs.readthedocs.io/) | Lightweight statistical profiling |
| Monitoring | [Prometheus](https://prometheus.io/) + [Grafana](https://grafana.com/) + AlertManager | Dashboards, alerting |
| Observability Backend | [Grafana LGTM](https://grafana.com/docs/opentelemetry/docker-lgtm/) | Unified OTel Collector + Prometheus + Tempo + Loki |
| GPU Metrics | [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter) | Hardware GPU metrics (Prometheus format) |
| Telemetry | [OpenTelemetry](https://opentelemetry.io/) | Traces, metrics, logs standard |
| Compute | SkyPilot | Intercloud broker (RunPod + GCP) |
| Infrastructure | [Docker Compose](https://docs.docker.com/compose/) + [Pulumi](https://www.pulumi.com/) | Local dev stack, GCP IaC |
| Linter/Formatter | ruff | Linting and formatting |
| Type Checker | mypy | Static type analysis |
| Tests | pytest + [Hypothesis](https://hypothesis.readthedocs.io/) | Unit, integration, property-based |
| Topology | [gudhi](https://gudhi.inria.fr/) + [networkx](https://networkx.org/) + [scipy](https://scipy.org/) | Persistent homology, graph analysis |
| XAI | [Captum](https://captum.ai/) + [SHAP](https://shap.readthedocs.io/) + [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus) | Explainability and meta-evaluation |
| LLM Observability | [Langfuse](https://langfuse.com/) + [Braintrust](https://www.braintrust.dev/) + [LiteLLM](https://docs.litellm.ai/) | Agent tracing, evals, provider flexibility |
| Compliance | AuditTrail + [IEC 62304](https://www.iso.org/standard/38421.html) framework + [CycloneDX](https://cyclonedx.org/) (planned) | [FDA](https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd)/MDR readiness |

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
vascadia/
|-- src/minivess/                  Main package
|   |-- adapters/                  ModelAdapter ABC + 6 model families
|   |-- pipeline/                  Training, evaluation, metrics, losses
|   |-- ensemble/                  Ensembling, UQ, calibration
|   |-- orchestration/flows/       15 Prefect 3.x flows (all with observability context managers)
|   |-- config/                    Pydantic v2 config models
|   |-- data/                      Data loading, profiling, DVC
|   |-- serving/                   BentoML, ONNX, Gradio
|   |-- observability/             MLflow tracking, GPU heartbeat, structured logging, OTel, DuckDB analytics
|   |-- agents/                    Pydantic AI micro-orchestration (ADR-0007)
|   |-- compliance/                IEC 62304 audit trail, model cards, regulatory docs
|   +-- validation/                Pandera, Great Expectations
|
|-- tests/                         Unit, integration, and E2E test suites
|-- configs/                       Hydra experiment configs, model profiles, splits
|-- deployment/                    Docker, SkyPilot, Pulumi, Grafana, Prometheus
|-- knowledge-graph/               75+ Bayesian decision nodes across 11 domains
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

### Recommended Developer Tools (not required to run the pipeline)

- **[code-review-graph](https://github.com/tirth8205/code-review-graph)** -- MCP server for structural code analysis. Blast radius queries, test coverage mapping, complexity hotspots:
  ```
  pip install code-review-graph && code-review-graph install && code-review-graph build
  ```
- **[duckdb-skills](https://github.com/duckdb/duckdb-skills)** -- Claude Code plugin for interactive DuckDB queries on biostatistics output:
  ```
  /plugin marketplace add duckdb/duckdb-skills
  /plugin install duckdb-skills@duckdb-skills
  ```

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
- Observability (MLflow, DuckDB, Prometheus, Grafana, Evidently, whylogs, Grafana LGTM, DCGM Exporter, GPU heartbeat, structured epoch logging, Docker HEALTHCHECK on all 10 services, Prefect task hooks on all 77 tasks)
- Post-training plugin architecture (7 plugins including SWAG, Flow 2.5)
- SAM3 integration (3 adapter variants)
- Pydantic AI agent layer (experiment summariser, drift triage, figure narration)
- FDA readiness planning (test set firewall, OpenLineage, PCCP alignment)

### In Progress

- Biostatistics flow polishing — statistical engine verified on fixture DuckDB (stratified permutation, BCa/percentile CI, hierarchical gatekeeping, specification curve). Training: dice_ce 3 folds complete on DagsHub MLflow, cbdice_cldice pending
- OTel trace propagation from Python to Grafana Tempo ([#974](https://github.com/petteriTeikari/vascadia/issues/974))
- Dashboard flow as observability consumer ([#975](https://github.com/petteriTeikari/vascadia/issues/975))
- `prefect-opentelemetry` package integration ([#976](https://github.com/petteriTeikari/vascadia/issues/976))
- Behavioral end-to-end observability verification test ([#977](https://github.com/petteriTeikari/vascadia/issues/977))
- 4-layer factorial experiment on RunPod + DagsHub MLflow (24 training cells x 2 post-training x analysis layers)
- OpenLineage flow wiring (Issue [#799](https://github.com/petteriTeikari/vascadia/issues/799))
- [CycloneDX](https://cyclonedx.org/) SBOM generation (Issue [#821](https://github.com/petteriTeikari/vascadia/issues/821))

### Planned

- [CopilotKit](https://www.copilotkit.ai/) (AG-UI) + WebMCP for agentic dashboard/annotation
- Multi-site opt-in telemetry ([PostHog](https://posthog.com/), [Sentry](https://sentry.io/))
- Federated learning evaluation ([NVIDIA FLARE](https://nvidia.github.io/NVFlare/) vs [MONAI FL](https://flower.ai/docs/examples/quickstart-monai.html))
- QMSR production controls documentation
- **Science backlog**: calibration-aware ensembles ([#896](https://github.com/petteriTeikari/vascadia/issues/896)), greedy ensemble selection ([#894](https://github.com/petteriTeikari/vascadia/issues/894)), snapshot ensembles ([#895](https://github.com/petteriTeikari/vascadia/issues/895)), spec curve analysis ([#898](https://github.com/petteriTeikari/vascadia/issues/898)), uncertainty-guided eval ([#897](https://github.com/petteriTeikari/vascadia/issues/897)), topology-critical calibration ([#899](https://github.com/petteriTeikari/vascadia/issues/899)), VLM calibration ([#798](https://github.com/petteriTeikari/vascadia/issues/798)), federated learning ([#842](https://github.com/petteriTeikari/vascadia/issues/842)), Syne Tune HPO ([#861](https://github.com/petteriTeikari/vascadia/issues/861)), AI card stack ([#864](https://github.com/petteriTeikari/vascadia/issues/864)), KG provenance ([#938](https://github.com/petteriTeikari/vascadia/issues/938))

---

## Further Reading

- [Knowledge Graph Navigator](knowledge-graph/navigator.yaml) -- entry point for architectural decisions
- [FDA Readiness Report](docs/planning/regops-fda-regulation-reporting-qms-samd-iec62304-mlops-report.md) -- compliance gap analysis
- [FDA Insights Second Pass](docs/planning/fda-insights-second-pass.md) -- SBOM, SecOps, QMSR, PCCP
- [SAM3 Literature Report](docs/planning/sam3-literature-research-report.md) -- foundation model survey
- [Loss Variation Results](docs/results/dynunet_loss_variation_v2_report.md) -- DynUNet baseline analysis
- [GCP Setup Tutorial](docs/planning/gcp-setup-tutorial.md) -- step-by-step cloud setup
- [Train + Post-Training Flow Merger](docs/planning/training-and-post-training-into-two-subflows-under-one-flow.md) -- parent flow with 2 sub-flows
- [Context Management Upgrade Plan](.claude/context-management-upgrade-plan.md) -- Issue #906, 5-phase knowledge compounding fix

---

## License

Apache-2.0 (license review pending for non-commercial academic use)
