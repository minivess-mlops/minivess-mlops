# MinIVess MLOps v2

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)
![Docker](https://img.shields.io/badge/execution-Docker-2496ED?logo=docker)
![MONAI](https://img.shields.io/badge/framework-MONAI-00A86B)
![tests](https://img.shields.io/badge/tests-pytest-brightgreen)
![ruff](https://img.shields.io/badge/linter-ruff-orange)
![mypy](https://img.shields.io/badge/type%20checker-mypy-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Model-agnostic biomedical segmentation MLOps platform extending the MONAI ecosystem.**
MinIVess MLOps v2 is a production-grade scaffold that frees PhD researchers from
infrastructure wrangling -- everything automatic by default, everything tweakable by
choice. It provides Docker-per-flow isolation, SkyPilot intercloud compute, Prefect
orchestration, and a config-driven architecture where adding a new model, dataset, or
flow requires editing one YAML file. The companion paper (working title: **NEUROVEX**)
targets *Nature Protocols*.

Built on the dataset published in: Charissa Poon, Petteri Teikari *et al.* (2023),
"A dataset of rodent cerebrovasculature from in vivo multiphoton fluorescence
microscopy imaging," *Scientific Data* 10, 141 --
doi: [10.1038/s41597-023-02048-8](https://doi.org/10.1038/s41597-023-02048-8)

---

## Key Features

- **6 model families** behind a single `ModelAdapter` ABC: DynUNet (CNN baseline), MambaVesselNet++ (SSM hybrid), SAM3 Vanilla/TopoLoRA/Hybrid (foundation model variants), VesselFM (vessel-specific foundation model)
- **18 loss functions** -- from standard (Dice+CE) to topology-aware (clDice, CAPE, Betti matching, skeleton recall) to graph-constrained (compound graph topology)
- **5 core Prefect flows** with Docker-per-flow isolation, plus additional auxiliary flows (biostatistics, drift simulation, synthetic generation, HPO)
- **SkyPilot intercloud broker** ([Yang et al., NSDI'23](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng)) -- one command to launch GPU jobs on RunPod or GCP
- **Evidently drift detection** + **whylogs profiling** + Prometheus/Grafana monitoring stack
- **BentoML + ONNX Runtime serving** with champion model discovery and Gradio demo UI
- **MetricsReloaded evaluation** -- clDice (trusted), MASD (trusted), DSC (foil) per [Maier-Hein et al. (2024)](https://doi.org/10.1038/s41592-023-02151-z)
- **3-fold cross-validation** (seed=42) with bootstrap confidence intervals and paired statistical tests
- **Conformal uncertainty quantification** -- 5 methods (split conformal, morphological, distance transform, risk-controlling, MAPIE)
- **Post-training plugins** -- 6 config-driven enhancements (SWA, Multi-SWA, model merging, calibration, CRC conformal, ConSeCo FP control)
- **Knowledge graph** -- 69 Bayesian decision nodes across 6 layers, driving spec-driven development

---

## Architecture Overview

### Three-Environment Model

| Environment | Docker | Compute | Data | Purpose |
|-------------|--------|---------|------|---------|
| **local** | Docker Compose | Local GPU (e.g., RTX 2070 Super 8 GB) | MinIO (local) | Fast iteration, `uv run pytest` |
| **env** (RunPod) | Docker image via SkyPilot | RunPod RTX 4090 (24 GB) | Network Volume (upload from local) | Quick GPU experiments |
| **staging/prod** (GCP) | Docker image via SkyPilot | GCP L4/A100 spot | GCS buckets | Production runs, paper results |

The laptop is always the control plane. After any remote training job, MLflow runs are
synced back to the local `mlruns/` directory. Subsequent flows (analysis, deploy,
dashboard) consume local MLflow data seamlessly.

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
```

### Inter-Flow Contract

All flows communicate through **MLflow** as the shared state layer. The
`find_upstream_run()` utility discovers upstream artifacts (checkpoints, metrics,
configs) by experiment name and run tags. Checkpoint formats use a standardized
envelope (`checkpoint_format.yaml`) so any downstream consumer can load any
upstream model.

### Docker Three-Tier Hierarchy

```
Tier A (GPU):   nvidia/cuda:12.6.3 --> minivess-base:latest       (~8-12 GB)
Tier B (CPU):   python:3.13-slim   --> minivess-base-cpu:latest    (~1.5-2.5 GB)
Tier C (Light): python:3.13-slim   --> minivess-base-light:latest  (~1.0-1.5 GB)
```

Each tier uses a **two-stage builder-runner** pattern (BuildKit). Flow Dockerfiles
are thin -- only `COPY`, `ENV`, `CMD` -- they never run `apt-get` or `uv`.

| Tier | Base Image | Flows |
|------|-----------|-------|
| A (GPU) | `minivess-base:latest` | train, hpo, post_training, analyze, deploy, data, annotation, monailabel, acquisition, benchmark |
| B (CPU) | `minivess-base-cpu:latest` | biostatistics |
| C (Light) | `minivess-base-light:latest` | dashboard, dashboard-api, pipeline |

There are 20 Dockerfiles in `deployment/docker/` serving infrastructure services and
per-flow containers.

---

## Model Families

Six model families for the Nature Protocols paper comparison (see
`knowledge-graph/decisions/L3-technology/paper_model_comparison.yaml`):

| Model | Family | Adapter | Training Strategy | VRAM | Status |
|-------|--------|---------|-------------------|------|--------|
| **DynUNet** | CNN baseline | `adapters/dynunet.py` | Full training (100 epochs, 3 folds) | ~3.5 GB | Results available |
| **MambaVesselNet++** | SSM hybrid | `adapters/mambavesselnet.py` | Full training | TBD | Code complete, GPU runs pending |
| **SAM3 Vanilla** | Foundation (frozen) | `adapters/sam3_vanilla.py` | Zero-shot or decoder fine-tune | ~2.9 GB | GPU runs pending |
| **SAM3 TopoLoRA** | Foundation (LoRA) | `adapters/sam3_topolora.py` | LoRA fine-tune (rank=16, alpha=32) | ~16 GB | GPU runs pending |
| **SAM3 Hybrid** | Foundation (fusion) | `adapters/sam3_hybrid.py` | SAM3 features + DynUNet 3D decoder | ~6 GB | Partially validated |
| **VesselFM** | Foundation (pretrained) | `adapters/vesselfm.py` | Zero-shot + fine-tune on external data | TBD | GPU runs pending |

Every model implements the `ModelAdapter` ABC:

```python
class ModelAdapter(ABC):
    def forward(volume) -> SegmentationOutput     # (B, C, D, H, W) logits
    def get_config() -> ModelInfo                  # family, params, extras
    def load_checkpoint(path)                      # restore weights
    def save_checkpoint(path)                      # persist weights
    def trainable_parameters() -> int              # count for logging
    def export_onnx(path, example_input)           # ONNX export
```

Adding a new model = one new file implementing this interface + one YAML config.

### Key Results: DynUNet Baseline (4 losses x 3 folds x 100 epochs)

| Loss | DSC | clDice | HD95 | NSD |
|------|-----|--------|------|-----|
| `dice_ce` | **0.824** | 0.832 | 3.46 | 0.891 |
| `cbdice_cldice` | 0.779 | **0.906** | 4.12 | 0.847 |
| `dice_ce_cldice` | 0.802 | 0.868 | 3.67 | 0.872 |
| `focal` | 0.788 | 0.841 | 3.89 | 0.859 |

**Default loss: `cbdice_cldice`** -- best topology preservation (0.906 clDice) with
only -5.3% DSC penalty.

---

## Prefect Flows

All pipeline execution goes through Prefect flows running in Docker containers.
Each flow has an independently configurable compute target via deployment config.

| Flow | File | Docker Tier | Purpose |
|------|------|-------------|---------|
| **Data Engineering** | `data_flow.py` | GPU (A) | DVC, NIfTI loading, validation, profiling |
| **Training** | `train_flow.py` | GPU (A) | Model training with Hydra-zen config |
| **Post-Training** | `post_training_flow.py` | GPU (A) | SWA, merging, calibration, conformal |
| **Analysis** | `analysis_flow.py` | GPU (A) | MetricsReloaded evaluation, bootstrap CIs |
| **Deployment** | `deploy_flow.py` | GPU (A) | Champion discovery, ONNX export, BentoML |
| **Dashboard** | `dashboard_flow.py` | Light (C) | Paper figures, tables, drift reports |
| **HPO** | `hpo_flow.py` | GPU (A) | Optuna + ASHA hyperparameter optimization |
| **Biostatistics** | `biostatistics_flow.py` | CPU (B) | Statistical analysis, DuckDB exports |
| **Drift Simulation** | `drift_simulation_flow.py` | GPU (A) | Synthetic drift experiments |
| **Synthetic Generation** | `synthetic_generation_flow.py` | GPU (A) | Synthetic data generation |
| **Acquisition** | `acquisition_flow.py` | GPU (A) | Dataset download and DVC versioning |
| **Annotation** | `annotation_flow.py` | GPU (A) | Label Studio integration |

---

## Quick Start

### Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/) (the **only** supported package manager)
- Docker (for pipeline execution) and Docker Compose V2
- NVIDIA GPU with CUDA (optional for local development; required for training)

### Install and Verify

```bash
# Clone the repository
git clone https://github.com/petteriTeikari/minivess_mlops.git
cd minivess_mlops

# Install dependencies (--all-extras is REQUIRED for development)
uv sync --all-extras

# Run the staging test suite (fast, no model loading, <3 min)
make test-staging

# Lint + format + typecheck (three-gate verification)
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/
```

### Docker Compose Infrastructure

```bash
# Copy environment config
cp .env.example .env
# Edit .env to set MODEL_CACHE_HOST_PATH and other values

# Create the shared network
docker network create minivess-network

# Core infrastructure (PostgreSQL, MinIO, MLflow, Prefect)
docker compose -f deployment/docker-compose.yml --profile dev up -d

# Initialize volume ownership (one-time)
make init-volumes

# Run a training flow
docker compose --env-file .env -f deployment/docker-compose.flows.yml run --rm \
  --shm-size 8g -e EXPERIMENT=dynunet_e2e_debug train
```

### Docker Compose Profiles

| Profile | Services | Approx. RAM |
|---------|----------|-------------|
| `dev` | PostgreSQL, MinIO, MLflow, BentoML | ~4 GB |
| `monitoring` | + Prometheus, Grafana, OpenTelemetry | ~8 GB |
| `full` | + Langfuse, Marquez, Label Studio, Ollama | ~16 GB |

---

## Cloud Execution

### SkyPilot Intercloud Broker

All cloud compute is managed through [SkyPilot](https://skypilot.readthedocs.io/) --
an intercloud broker that works like Slurm for multi-cloud. SkyPilot YAML files specify
Docker images (bare VM setup is banned for reproducibility).

```bash
# RunPod: quick GPU experiments
sky jobs launch deployment/skypilot/dev_runpod.yaml

# GCP: production runs with spot instances
sky jobs launch deployment/skypilot/train_production.yaml

# Sync MLflow results back to local after RunPod training
make dev-gpu-sync
```

### Two-Provider Architecture

| Provider | Environment | Role | Data Storage |
|----------|------------|------|--------------|
| **RunPod** | env (dev) | Quick GPU experiments, instant provisioning | Network Volume (data from local disk) |
| **GCP** | staging + prod | Production runs, paper results, Pulumi IaC | GCS (`gs://minivess-mlops-dvc-data`) |

- **RunPod** is fully standalone -- no GCP account required. Data is uploaded from the
  researcher's local disk. Ideal for paper readers getting started.
- **GCP** (`europe-north1`) provides the full managed stack: GCS buckets, Artifact
  Registry, Cloud SQL (PostgreSQL), spot instance recovery via SkyPilot.
- Cloud config flows through **Hydra config groups** (`configs/cloud/`, `configs/registry/`).
  Labs with different providers override via `configs/lab/lab_name.yaml` -- zero code changes.

### Per-Flow Compute Routing

Each Prefect flow has an independently configurable compute target. The platform provides
flows; the researcher decides where each runs:

```
Flow 1: Data Eng.    --> local Docker, or cloud instance for large datasets
Flow 2: Training     --> RunPod RTX 4090, or GCP L4 spot, or local GPU
Flow 3: Analysis     --> local, or cloud GPU for large-scale sliding window eval
Flow 4: Deployment   --> local, or CI runner, or cloud
Flow 5: Dashboard    --> local, or Vercel/Render for static hosting
```

---

## Dataset

### Primary: MiniVess

70 volumes of rodent cerebrovasculature from *in vivo* multiphoton fluorescence
microscopy. 3D NIfTI format.

- **Splits**: 3-fold cross-validation, seed=42 (`configs/splits/3fold_seed42.json`) -- 47 train / 23 val
- **Origin**: [EBRAINS](https://ebrains.eu/) repository
- **Versioning**: DVC (MinIO local, GCS cloud)

### External Test Sets

| Dataset | Volumes | Use | Reference |
|---------|---------|-----|-----------|
| DeepVess | -- | External validation | Haft-Javaherian et al. (2019) |
| TubeNet | -- | External validation | -- |
| VesselNN | -- | External validation | -- |

**VesselFM data leakage warning**: VesselFM was pre-trained on 17 datasets *including*
MiniVess. Evaluation of VesselFM uses DeepVess/TubeNet only -- never MiniVess.

---

## Loss Functions

18 loss functions organized by provenance:

| Loss | Type | Source/Reference |
|------|------|-----------------|
| `dice_ce` | LIBRARY | MONAI `DiceCELoss` |
| `dice` | LIBRARY | MONAI `DiceLoss` |
| `focal` | LIBRARY | MONAI `FocalLoss` |
| `cb_dice` | LIBRARY | Class-balanced Dice |
| `cbdice` | LIBRARY | Class-balanced Dice (standalone) |
| `cldice` | HYBRID | clDice -- soft skeleton ([Shit et al., CVPR 2021](https://arxiv.org/abs/2003.07311)) |
| `dice_ce_cldice` | COMPOUND | 0.5 DiceCE + 0.5 clDice |
| `cbdice_cldice` | COMPOUND | 0.5 cbDice + 0.5 dice_ce_cldice (**default/champion**) |
| `centerline_ce` | HYBRID | Centerline-weighted cross-entropy |
| `skeleton_recall` | EXPERIMENTAL | [Kirchhoff et al. (ECCV 2024)](https://arxiv.org/abs/2404.09404) |
| `cape` | EXPERIMENTAL | [CAPE (MICCAI 2025)] |
| `betti_matching` | EXPERIMENTAL | Persistent homology Betti matching |
| `toposeg` | EXPERIMENTAL | TopoSegNet critical points |
| `topo` | EXPERIMENTAL | Topological loss |
| `warp` | EXPERIMENTAL | Warping-based loss |
| `betti` | EXPERIMENTAL | Betti number error |
| `full_topo` | EXPERIMENTAL | Compound topology |
| `graph_topology` | EXPERIMENTAL | Graph-constrained topology |

**Top 3 for vessel segmentation**:
1. **`cbdice_cldice`** (champion) -- best topology preservation (0.906 clDice)
2. **`dice_ce_cldice`** -- balanced overlap + topology (0.868 clDice)
3. **`dice_ce`** -- best volumetric overlap (0.824 DSC)

---

## Metrics

Metric selection follows the [MetricsReloaded](https://doi.org/10.1038/s41592-023-02151-z)
framework (Maier-Hein et al., *Nature Methods* 2024) for 3D vascular segmentation.

### Trusted Metrics (Primary)

| Metric | What It Measures | Why Trusted |
|--------|-----------------|-------------|
| **clDice** | Vessel topology preservation (centerline connectivity) | MetricsReloaded recommends topology-aware metrics for tubular structures |
| **MASD** | Mean average surface distance | More robust than HD95 to single-voxel outliers |

### Foil Metric (Included Deliberately)

| Metric | What It Measures | Why Foil |
|--------|-----------------|----------|
| **DSC** | Volumetric overlap (Dice coefficient) | Volume-biased: high Dice possible while missing all thin branches. Included to demonstrate misleading rankings |

### Optional Metrics (Supplementary)

HD95, ASSD, NSD, Betti-0 error, Betti-1 error, Junction F1

---

## Knowledge Graph

The project uses a **6-layer knowledge architecture** with 69 Bayesian decision nodes
across 11 domains:

```
L0: .claude/rules/ + CLAUDE.md            -- Constitution (invariant rules)
L1: docs/planning/ + MEMORY.md            -- Hot Context (current work)
L2: knowledge-graph/navigator.yaml        -- Navigator (domain routing)
L3: knowledge-graph/decisions/*.yaml       -- Evidence (69 decision nodes)
    knowledge-graph/domains/*.yaml         -- Materialized winners
L4: openspec/specs/                        -- Specifications (GIVEN/WHEN/THEN)
L5: src/ + tests/                          -- Implementation (actual code)
```

**Information flow**: PRD decisions propagate downward through KG materialization to
OpenSpec specifications to code. Experiment results propagate upward through KG
posterior updates to PRD belief propagation.

Decision nodes span 5 levels:

| Level | Scope | Examples |
|-------|-------|---------|
| L1 | Research goals | Publication target, reproducibility level, project purpose |
| L2 | Architecture | Model adapter pattern, flow topology, Docker architecture, ensemble strategy |
| L3 | Technology | Loss functions, metrics, HPO engine, dataset strategy, model comparison |
| L4 | Infrastructure | Cloud providers, container strategy, CI/CD, Docker registry, IaC |
| L5 | Operations | Drift monitoring, dashboarding, cost optimization, audit trail |

Entry point: [`knowledge-graph/navigator.yaml`](knowledge-graph/navigator.yaml)

---

## Observability

### MLflow Experiment Tracking

- Local filesystem backend (`mlruns/`) for development
- Cloud Run MLflow server with GCS artifact storage for production
- DuckDB analytics for SQL queries over experiment data
- PostgreSQL for MLflow metadata + Optuna HPO storage (SQLite is banned)

### Prometheus + Grafana Monitoring

```bash
# Start monitoring stack
docker compose -f deployment/docker-compose.yml --profile monitoring up -d
```

- Prometheus scrapes MLflow, BentoML, and custom flow metrics
- Grafana dashboards for training progress, drift alerts, and cost tracking
- AlertManager for drift and performance alerts

### Evidently Drift Detection

- `DataDriftPreset` with KS test and PSI for feature drift
- Kernel MMD for embedding drift monitoring
- Drift alerting via JSONL log, webhook, and Prometheus export

### whylogs Data Profiling

Lightweight statistical profiling of input data distributions, integrated into the
data engineering flow.

### Additional Observability

- **Langfuse** (self-hosted) for LLM tracing and cost tracking
- **Braintrust** for LLM evaluation and offline evals
- **OpenLineage** (Marquez) for IEC 62304 data lineage traceability
- **Captum** (3D) + **SHAP** + **Quantus** for explainability (XAI)

---

## Docker Architecture

### Three-Tier Multi-Stage Builds

All execution (training, evaluation, serving, deployment) runs inside Docker containers.
Docker is the execution model -- the reproducibility guarantee.

```
                          Builder Stage              Runner Stage
                     +-------------------+     +-------------------+
Tier A (GPU):        | nvidia/cuda:12.6  |     | nvidia/cuda:12.6  |
                     | devel + uv + deps | --> | runtime + .venv   |
                     +-------------------+     +-------------------+

Tier B (CPU):        | python:3.13-slim  |     | python:3.13-slim  |
                     | uv + scipy/pandas | --> | .venv only        |
                     +-------------------+     +-------------------+

Tier C (Light):      | python:3.13-slim  |     | python:3.13-slim  |
                     | uv + prefect/API  | --> | .venv only        |
                     +-------------------+     +-------------------+
```

### Volume Mount Rules

Every artifact that must survive the container is explicitly volume-mounted:

| Mount | Purpose |
|-------|---------|
| `/data` | Input datasets |
| `/mlruns` | MLflow tracking |
| `/checkpoints` | Model weights |
| `/logs` | Training JSONL/CSV logs |
| `/configs` | YAML configs + split definitions |

### Docker Registry Strategy

| Environment | Registry | Rationale |
|-------------|----------|-----------|
| RunPod (dev) | Docker Hub | Public, zero-auth pull |
| GCP (staging/prod) | Google Artifact Registry | Same-region as GCS, ADC auth |
| GitHub Actions CI | GHCR | Internal build artifacts only |

Registry choice is config-driven (`configs/registry/`), not hardcoded.

---

## Testing

### Three-Tier Strategy

| Tier | Command | What Runs | Target Time |
|------|---------|-----------|-------------|
| **Staging** | `make test-staging` | No model loading, no slow, no integration | < 3 min |
| **Prod** | `make test-prod` | Everything except GPU instance tests | 5-10 min |
| **GPU** | `make test-gpu` | SAM3 + GPU-heavy tests (external GPU only) | GPU instance |

```bash
make test-staging    # PR readiness -- fast, no models
make test-prod       # Pre-merge -- includes model loading + slow
make test-gpu        # RunPod / intranet GPU -- SAM3 forward passes
```

### Test Infrastructure

- **535 test files** across unit, integration, and E2E suites
- **pytest** + **Hypothesis** (property-based testing)
- Markers: `@pytest.mark.model_loading`, `@pytest.mark.slow`, `@pytest.mark.gpu_heavy`
- `tests/gpu_instance/` excluded from default pytest collection -- zero noise in CI
- Three-gate quality check: tests + lint (ruff) + types (mypy)

### Quality Gates

Every change must pass:
1. **Tests** -- `make test-staging` (fast) or `make test-prod` (thorough)
2. **Lint** -- `uv run ruff check src/ tests/` (no violations)
3. **Types** -- `uv run mypy src/` (no type errors)

Pre-commit hooks enforce formatting, trailing whitespace, YAML validation, and
bibliography integrity.

---

## Technology Stack

| Layer | Tool | Role |
|-------|------|------|
| Language | Python 3.12+ | Runtime |
| Package Manager | uv | Dependency management (exclusively) |
| ML Framework | PyTorch + MONAI + TorchIO | Training, augmentation, inference |
| Orchestration | Prefect 3.x | Persona-based flow orchestration |
| Config (train) | Hydra-zen | Experiment configs with Pydantic v2 validation |
| Config (deploy) | Dynaconf | Environment-layered deployment settings |
| Data Validation | Pydantic v2 + Pandera + Great Expectations | Schema, DataFrame, batch quality |
| Experiment Tracking | MLflow + DuckDB | Run tracking, model registry, SQL analytics |
| HPO | Optuna + ASHA | Multi-objective hyperparameter optimization |
| Serving | BentoML + ONNX Runtime + Gradio | Model serving and demo UI |
| Drift Detection | Evidently AI | KS test, PSI, kernel MMD |
| Data Profiling | whylogs | Lightweight statistical profiling |
| Monitoring | Prometheus + Grafana + AlertManager | Dashboards, alerting |
| Compute | SkyPilot | Intercloud broker (RunPod + GCP) |
| Infrastructure | Docker Compose + Pulumi | Local dev stack, GCP IaC |
| CI/CD | GitHub Actions + CML | Manual-trigger workflows (credits-aware) |
| Linter/Formatter | ruff | Linting and formatting |
| Type Checker | mypy | Static type analysis |
| Tests | pytest + Hypothesis | Unit, integration, property-based, E2E |
| Topology | gudhi + networkx + scipy | Persistent homology, graph analysis |
| XAI | Captum + SHAP + Quantus | Explainability and meta-evaluation |
| LLM Observability | Langfuse + Braintrust + LiteLLM | Tracing, evals, provider flexibility |
| Lineage | OpenLineage (Marquez) | IEC 62304 traceability |

---

## Ensemble Strategies and Uncertainty Quantification

### Ensemble Strategies (7)

Mean, majority vote, weighted, greedy soup, SWAG, TIES-DARE, learned stacking --
configured via `EnsembleStrategy` enum in `ensemble/ensemble_builder.py`.

### Conformal Uncertainty Quantification (5 methods)

| Method | Class | Reference |
|--------|-------|-----------|
| Split conformal | `ConformalPredictor` | Vovk et al. (2005) |
| Morphological bands | `MorphologicalConformalPredictor` | ConSeMa-inspired |
| Distance transform | `DistanceTransformConformalPredictor` | CLS/FNR control via EDT |
| Risk-controlling | `RiskControllingPredictor` | LTT framework |
| MAPIE integration | `MapieConformalSegmentation` | MAPIE library wrapper |

Additional UQ: MC Dropout, Deep Ensembles, Calibration Shift Analysis.

### Post-Training Plugins (6)

Config-driven post-hoc enhancements (Flow 2.5, between train and analysis):

| Plugin | Method | Calibration Data? |
|--------|--------|-------------------|
| SWA | Uniform checkpoint averaging | No |
| Multi-SWA | M independent SWA models | No |
| Model Merging | Linear, SLERP, layer-wise merge | No |
| Calibration | Temperature scaling, isotonic, Platt | Yes |
| CRC Conformal | Conformal prediction + heatmaps | Yes |
| ConSeCo FP Control | Threshold/erosion shrinking | Yes |

Enable/disable via `configs/post_training/default.yaml`.

---

## Directory Structure

```
minivess-mlops/
|-- src/minivess/                  Main package (340 Python files)
|   |-- adapters/                  ModelAdapter ABC + 6 families
|   |-- pipeline/                  Training, evaluation, metrics, losses
|   |-- ensemble/                  Ensembling, UQ, calibration
|   |-- orchestration/flows/       12+ Prefect 3.x flows
|   |-- config/                    Pydantic v2 config models
|   |-- data/                      Data loading, profiling, DVC
|   |-- serving/                   BentoML, ONNX, Gradio
|   |-- observability/             MLflow tracking, DuckDB analytics
|   |-- agents/                    LangGraph agent definitions
|   |-- compliance/                IEC 62304 audit trail, model cards
|   +-- validation/                Pandera, Great Expectations
|
|-- tests/                         535 test files
|   |-- v2/unit/                   Fast isolated unit tests
|   |-- v2/integration/            Service integration tests
|   |-- v2/e2e/                    End-to-end pipeline tests
|   +-- gpu_instance/              SAM3 + GPU-heavy (excluded from default collection)
|
|-- configs/
|   |-- experiments/               Experiment YAML configs
|   |-- model_profiles/            Per-model VRAM/compute profiles
|   |-- splits/                    Cross-validation fold definitions
|   |-- cloud/                     Cloud provider configs (Hydra groups)
|   +-- registry/                  Docker registry configs
|
|-- deployment/
|   |-- docker/                    20 Dockerfiles (3 base + flow + infra)
|   |-- docker-compose.yml         Infrastructure services (profiles: dev/monitoring/full)
|   |-- docker-compose.flows.yml   Per-flow Docker services
|   |-- skypilot/                  SkyPilot YAML configs (RunPod + GCP)
|   |-- pulumi/gcp/                GCP infrastructure as code
|   |-- grafana/                   Dashboard configs
|   |-- prometheus/                Scrape configs
|   +-- seccomp/                   Per-flow seccomp profiles
|
|-- knowledge-graph/
|   |-- navigator.yaml             Entry point -- domain routing
|   |-- domains/                   11 materialized domain files
|   |-- decisions/                 69 Bayesian decision nodes (L1-L5)
|   +-- manuscript/                Scientific claims + section stubs
|
|-- docs/
|   |-- adr/                       Architecture Decision Records
|   |-- planning/                  Implementation plans, research reports
|   +-- results/                   Experiment reports and figures
|
+-- openspec/                      Spec-driven development (GIVEN/WHEN/THEN)
```

---

## Contributing

1. **uv only** -- never use pip, conda, or poetry. Install with `uv sync --all-extras`.
2. **TDD mandatory** -- write failing tests first, then implement. RED-GREEN-VERIFY cycle.
3. **Pre-commit hooks** -- all changes must pass before commit (ruff, mypy, YAML validation, bibliography integrity).
4. **Three-gate verification** -- `make test-staging && uv run ruff check src/ tests/ && uv run mypy src/`
5. **Library-first** -- search for existing implementations before writing custom code. Use MONAI, PyTorch, scipy, etc.
6. **Docker is the execution model** -- all pipeline execution goes through Prefect flows in Docker containers.
7. **Config-driven** -- specific tasks, models, losses, and metrics are YAML config instantiations, not code branches.

### Architecture Decision Records

| ADR | Decision |
|-----|----------|
| [0001](docs/adr/0001-model-adapter-abc.md) | Model Adapter Abstract Base Class |
| [0002](docs/adr/0002-dual-config-system.md) | Dual Configuration System (Hydra-zen + Dynaconf) |
| [0003](docs/adr/0003-validation-onion.md) | Multi-Layer Validation ("Validation Onion") |
| [0004](docs/adr/0004-local-first-observability.md) | Local-First Observability Stack |
| [0005](docs/adr/0005-tdd-mandatory.md) | Mandatory Test-Driven Development for SaMD |
| [0006](docs/adr/0006-sam3-variant-architecture.md) | SAM3 Variant Architecture |

---

## Citation

If you use this platform, please cite the underlying dataset:

> Charissa Poon, Petteri Teikari *et al.* (2023). "A dataset of rodent
> cerebrovasculature from in vivo multiphoton fluorescence microscopy imaging."
> *Scientific Data* 10, 141. doi: [10.1038/s41597-023-02048-8](https://doi.org/10.1038/s41597-023-02048-8)

All internal citations follow the **author-year format** ("Surname et al. (Year)")
with hyperlinks. The central bibliography is maintained in `bibliography.yaml` with
an append-only policy enforced by pre-commit hooks.

---

## Roadmap

### Completed

- Foundation (uv, Docker, configs, pre-commit)
- Core ML pipeline (6 model families, 18 losses, training engine)
- DynUNet baseline results (4 losses x 3 folds x 100 epochs)
- Evaluation (MetricsReloaded suite, bootstrap CIs, paired tests)
- Ensembling (7 strategies, greedy soup, calibration)
- Conformal UQ (5 methods + MC Dropout + deep ensembles)
- Serving (BentoML, ONNX Runtime, Gradio)
- Deployment flow (champion discovery, ONNX export, promotion)
- Observability (MLflow, DuckDB, Prometheus, Grafana, Evidently, whylogs)
- Post-training plugin architecture (6 plugins, Flow 2.5)
- SAM3 integration (3 adapter variants, VRAM checks, SDPA enforcement)
- MambaVesselNet++ adapter (hybrid CNN-SSM)
- Knowledge graph (69 decision nodes, 11 domains, 6 layers)
- Docker three-tier hierarchy (20 Dockerfiles)

### In Progress

- 6-model GPU benchmark runs on GCP L4 spot
- SAM3 variant training (Vanilla, TopoLoRA, Hybrid)
- Nature Protocols manuscript assembly (NEUROVEX)

### Planned

- VesselFM integration and external-data evaluation
- Full HPO sweep (Optuna + ASHA)
- Advanced conformal methods
- Federated learning (MONAI FL)
- SBOM generation and EU AI Act compliance

---

## Further Reading

- [Knowledge Graph Navigator](knowledge-graph/navigator.yaml) -- entry point for all architecture decisions
- [MetricsReloaded Report](docs/MetricsReloaded.html) -- metric selection rationale
- [SAM3 Literature Report](docs/planning/sam3-literature-research-report.md) -- foundation model survey
- [Loss Variation Results](docs/results/dynunet_loss_variation_v2_report.md) -- DynUNet baseline analysis
- [Docker Base Improvement Plan](docs/planning/docker-base-improvement-plan.md) -- three-tier Docker architecture
- [GCP Setup Tutorial](docs/planning/gcp-setup-tutorial.md) -- step-by-step cloud setup
- [Legacy Wiki](https://github.com/petteriTeikari/minivess_mlops/wiki) -- background from v0.1-alpha

---

## License

MIT
