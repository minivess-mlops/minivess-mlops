# MinIVess MLOps v2

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)
![tests](https://img.shields.io/badge/tests-pytest-brightgreen)
![ruff](https://img.shields.io/badge/linter-ruff-orange)
![mypy](https://img.shields.io/badge/type%20checker-mypy-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Model-agnostic biomedical segmentation MLOps platform.** A production-grade scaffold
that frees PhD researchers from infrastructure wrangling — everything automatic by
default, everything tweakable by choice.

Built on the dataset published in: Charissa Poon, Petteri Teikari *et al.* (2023),
"A dataset of rodent cerebrovasculature from in vivo multiphoton fluorescence
microscopy imaging," *Scientific Data* 10, 141 --
doi: [10.1038/s41597-023-02048-8](https://doi.org/10.1038/s41597-023-02048-8)

---

## What Is This?

MinIVess MLOps v2 is a **complete rewrite** of [v0.1-alpha](https://github.com/petteriTeikari/minivess_mlops/wiki),
designed as a **portfolio-grade reference implementation** of an end-to-end ML pipeline for
3D biomedical image segmentation. It demonstrates:

1. **Model-agnostic architecture** -- 9 model families (DynUNet, SAM3, VesselFM,
   COMMA/Mamba, TFFM, ...) behind a single `ModelAdapter` ABC
2. **18 loss functions** -- from standard (Dice+CE) to topology-aware (clDice, CAPE,
   Betti matching, skeleton recall) to graph-constrained (compound graph topology)
3. **Conformal uncertainty quantification** -- 5 methods (split conformal, morphological,
   distance transform, risk-controlling, MAPIE)
4. **Post-training plugin architecture** -- 6 configurable post-hoc plugins (SWA,
   Multi-SWA, model merging, calibration, CRC conformal, ConSeCo FP control)
5. **7 Prefect flows** -- persona-based orchestration (data, train, post-training,
   analyze, deploy, dashboard, qa)
6. **Reproducible experiments** -- all flows verified end-to-end with real data
   (70 volumes, 4 losses, 3 folds, 100 epochs)
7. **SaMD-principled** -- IEC 62304 lifecycle mapping, audit trails, model cards

The platform is an **academic software project** -- the PRD system (78 Bayesian decision
nodes across 5 levels) serves as the evidence base for a peer-reviewed article.

---

## Key Results

### DynUNet Baseline (4 losses x 3 folds x 100 epochs)

| Loss | DSC | clDice | HD95 | NSD |
|------|-----|--------|------|-----|
| `dice_ce` | **0.824** | 0.832 | 3.46 | 0.891 |
| `cbdice_cldice` | 0.779 | **0.906** | 4.12 | 0.847 |
| `dice_ce_cldice` | 0.802 | 0.868 | 3.67 | 0.872 |
| `focal` | 0.788 | 0.841 | 3.89 | 0.859 |

**Default loss: `cbdice_cldice`** -- best topology preservation (0.906 clDice) with
only -5.3% DSC penalty. Full results in
[`docs/results/dynunet_loss_variation_v2_report.md`](docs/results/dynunet_loss_variation_v2_report.md).

### SAM3 Variants (In Progress)

Three SAM3 (Segment Anything Model 3) variants to quantify foundation model limitations
on 3D microvessel segmentation:

| Variant | Architecture | Loss | Expected DSC | Status |
|---------|-------------|------|-------------|--------|
| **Vanilla** | Frozen SAM3 ViT-32L + trainable decoder | `dice_ce` | 0.35-0.55 | Training |
| **TopoLoRA** | + LoRA on FFN (mlp.lin1/lin2) | `cbdice_cldice` | +10-20% clDice | Pending |
| **Hybrid** | SAM3 features + DynUNet 3D decoder | `cbdice_cldice` | Best SAM variant | Planned |

See [ADR-0006](docs/adr/0006-sam3-variant-architecture.md) for architecture decisions.

---

## Architecture Overview

### Pipeline

```
                        Prefect Orchestration (7 flows)
                        ============================

Flow 1: Data Engineering     Flow 2: Training       Flow 2.5: Post-Training
  DVC + NIfTI                  Hydra-zen configs      6 post-hoc plugins
  TorchIO augmentation         Mixed precision         SWA + Multi-SWA
  Pydantic validation          Grad checkpointing     Model merging (slerp)
  Pandera schemas              18 loss functions       Calibration (temp/Platt)
  Dataset profiling            12 model families       CRC conformal + ConSeCo
          |                          |                        |
          v                          v                        v
        MLflow <=================  MLflow  ===============> MLflow
          |                          |                        |
          v                          v                        v
Flow 3: Analysis             Flow 4: Deployment     Flow 5: Dashboard
  8-metric evaluation          Champion discovery      Paper figures (PNG+SVG)
  Bootstrap CIs                ONNX export + validate  Parquet + DuckDB export
  Ensemble strategies          BentoML model store     Comparison tables (MD+TEX)
  Conformal UQ                 Gradio demo             Drift reports
  Graph topology               MONAI Deploy MAP
  Post-training discovery      DEV -> STAGING -> PROD
                                                     Flow 6: QA (best-effort)
                                                       MLflow data integrity
                                                       Ghost run cleanup
                                                       Param validation
```

### ModelAdapter Pattern

Every segmentation model implements 6 methods:

```python
class ModelAdapter(ABC):
    def forward(volume) -> SegmentationOutput     # (B, C, D, H, W) logits
    def get_config() -> ModelInfo                  # family, params, extras
    def load_checkpoint(path)                      # restore weights
    def save_checkpoint(path)                      # persist weights
    def trainable_parameters() -> int              # count for logging
    def export_onnx(path, example_input)           # ONNX export
```

The training engine, ensemble module, evaluation pipeline, serving layer, and ONNX
export all program against this interface. Adding a new model = one new file.

### Model Families (9)

| Family | Module | Description |
|--------|--------|-------------|
| `dynunet` | `adapters/dynunet.py` | MONAI DynUNet -- primary 3D baseline |
| `vesselfm` | `adapters/vesselfm.py` | VesselFM foundation model |
| `comma_mamba` | `adapters/comma.py` | COMMA Mamba state-space model |
| `sam3_vanilla` | `adapters/sam3_vanilla.py` | SAM3 frozen encoder + decoder |
| `sam3_topolora` | `adapters/sam3_topolora.py` | SAM3 + LoRA + topology loss |
| `sam3_hybrid` | `adapters/sam3_hybrid.py` | SAM3 features + DynUNet 3D decoder |
| `sam3_lora` | `adapters/lora.py` | Generic LoRA adapter |
| `multitask_dynunet` | `adapters/multitask_adapter.py` | Multi-task with auxiliary heads |
| `custom` | user-defined | Bring your own adapter |

Model dispatch is handled by `build_adapter(config)` in `adapters/model_builder.py`.

### Loss Functions (18)

| Loss | Type | Source |
|------|------|--------|
| `dice_ce` | LIBRARY | MONAI `DiceCELoss` |
| `dice` | LIBRARY | MONAI `DiceLoss` |
| `focal` | LIBRARY | MONAI `FocalLoss` |
| `cb_dice` | LIBRARY | Class-balanced Dice |
| `cldice` | HYBRID | clDice (soft skeleton) |
| `dice_ce_cldice` | LIBRARY-COMPOUND | 0.5 DiceCE + 0.5 clDice |
| `cbdice_cldice` | LIBRARY-COMPOUND | 0.5 cbDice + 0.5 dice_ce_cldice |
| `centerline_ce` | HYBRID | Centerline-weighted CE |
| `skeleton_recall` | EXPERIMENTAL | Kirchhoff (ECCV 2024) |
| `cape` | EXPERIMENTAL | CAPE (MICCAI 2025) |
| `betti_matching` | EXPERIMENTAL | Betti matching (persistent homology) |
| `toposeg` | EXPERIMENTAL | TopoSegNet critical points |
| `topo` | EXPERIMENTAL | Topological loss |
| `warp` | EXPERIMENTAL | Warping-based loss |
| `betti` | EXPERIMENTAL | Betti number error |
| `full_topo` | EXPERIMENTAL | Compound topology |
| `graph_topology` | EXPERIMENTAL | Graph-constrained topology |
| `cbdice` | LIBRARY | Class-balanced Dice (standalone) |

### Evaluation Metrics (13+)

**Main paper metrics (8):** DSC, HD95, ASSD, NSD, clDice, Betti-0 error, Betti-1 error, Junction F1

**Graph topology metrics:** ccDice, APLS, skeleton recall, BDR, Murray's law compliance,
persistence distance

All topology metrics in `pipeline/topology_metrics.py`. MONAI built-ins used where available.

### Conformal Uncertainty Quantification (5 methods)

| Method | Class | Reference |
|--------|-------|-----------|
| Split conformal | `ConformalPredictor` | Vovk et al. (2005) |
| Morphological bands | `MorphologicalConformalPredictor` | ConSeMa-inspired |
| Distance transform | `DistanceTransformConformalPredictor` | CLS / FNR control via EDT |
| Risk-controlling | `RiskControllingPredictor` | LTT framework |
| MAPIE integration | `MapieConformalSegmentation` | MAPIE library wrapper |

Additional UQ: `MCDropoutPredictor`, `DeepEnsemblePredictor`, `CalibrationShiftAnalyzer`.

### Ensemble Strategies (7)

Mean, majority vote, weighted, greedy soup, SWAG, TIES-DARE, learned stacking --
configured via `EnsembleStrategy` enum.

### Post-Training Plugins (6)

Config-driven post-hoc enhancement plugins (Flow 2.5, best-effort between train and analyze):

| Plugin | Wraps | Cal Data? |
|--------|-------|-----------|
| **SWA** | `model_soup.uniform_swa()` | No |
| **Multi-SWA** | M independent SWA models (subsampled) | No |
| **Model Merging** | `linear_merge()`, `slerp_merge()`, `layer_wise_merge()` | No |
| **Calibration** | Temperature scaling, isotonic regression, spatial Platt | Yes |
| **CRC Conformal** | `CRCPredictor` + `varisco_heatmap()` | Yes |
| **ConSeCo FP Control** | Threshold/erosion shrinking with conformal guarantees | Yes |

Each plugin implements `PostTrainingPlugin` protocol. Enable/disable via
`configs/post_training/default.yaml`. See
[`docs/planning/post-training-plugins-and-swa-planning.md`](docs/planning/post-training-plugins-and-swa-planning.md).

---

## Technology Stack

| Layer | Tool | Role |
|-------|------|------|
| **Language** | Python 3.12+ | Runtime |
| **Package Manager** | uv | Dependency management (only -- never pip/conda) |
| **ML Framework** | PyTorch 2.5+ / MONAI 1.4+ / TorchIO / TorchMetrics | Training and inference |
| **Orchestration** | Prefect 3.x | 7 persona-based flows (required, not optional) |
| **Config (train)** | Hydra-zen | Experiment configs with Pydantic v2 validation |
| **Config (deploy)** | Dynaconf | Environment-layered deployment settings |
| **Data Validation** | Pydantic v2 + Pandera + Great Expectations | Schema, DataFrame, batch quality |
| **Model Validation** | Deepchecks Vision + WeightWatcher | Data integrity, spectral diagnostics |
| **Experiment Tracking** | MLflow 3.10 + DuckDB | Run tracking, model registry, SQL analytics |
| **Serving** | BentoML + ONNX Runtime + Gradio | Model serving and demo UI |
| **Calibration** | MAPIE + netcal + conformal methods | Conformal prediction, temperature scaling |
| **XAI** | Captum (3D) + SHAP + Quantus | Explainability and meta-evaluation |
| **Drift Detection** | Evidently AI | KS test, PSI-based drift monitoring |
| **Data Profiling** | whylogs | Lightweight statistical profiling |
| **LLM Observability** | Langfuse (self-hosted) + Braintrust | Tracing, cost tracking, offline evals |
| **Agent Orchestration** | LangGraph + LiteLLM | Multi-step workflows, provider flexibility |
| **Data Lineage** | OpenLineage (Marquez) | IEC 62304 traceability |
| **Label Quality** | Cleanlab + Label Studio | Annotation QA, multi-annotator workflows |
| **CI/CD** | GitHub Actions (6 workflows) + CML | Lint, typecheck, test, ML-specific PR comments |
| **Linter/Formatter** | ruff | Linting and formatting |
| **Type Checker** | mypy | Static type analysis |
| **Test Framework** | pytest + Hypothesis | Unit, integration, property-based, E2E |
| **Infrastructure** | Docker Compose + Pulumi | Local dev stack, cloud-agnostic IaC |
| **Topology** | gudhi + networkx + scipy | Persistent homology, graph analysis |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/petteriTeikari/minivess_mlops.git
cd minivess_mlops

# Install dependencies (uv only -- never use pip/conda/poetry)
uv sync

# Run the full test suite
uv run pytest tests/v2/ -x -q

# Lint + format + typecheck (three-gate verification)
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/

# Run a debug training experiment (DynUNet, 1 epoch, 2 folds)
uv run python scripts/run_experiment.py --experiment dynunet_e2e_debug

# Run the full reproducibility pipeline (all 7 flows)
uv run python scripts/run_full_pipeline.py
```

### Docker Compose Profiles

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

## For Reviewers: Pipeline-to-Code Mapping

| Pipeline Stage | Entry Point | Key Source Files |
|---------------|------------|-----------------|
| **Data ingestion** | `scripts/download_minivess.py` | `data/profiler.py`, `data/validation.py` |
| **Training** | `scripts/train_monitored.py` | `adapters/model_builder.py`, `pipeline/loss_functions.py` |
| **Experiment runner** | `scripts/run_experiment.py` | `config/adaptive_profiles.py`, `config/model_profiles.py` |
| **Evaluation** | `scripts/compare_experiments.py` | `pipeline/comparison.py`, `pipeline/topology_metrics.py` |
| **Champion selection** | `scripts/tag_champions.py` | `pipeline/champion_tagger.py`, `pipeline/deploy_champion_discovery.py` |
| **ONNX export** | `scripts/register_models.py` | `pipeline/deploy_onnx_export.py` |
| **Serving** | `serving/onnx_service.py` | `serving/bento_model_import.py`, `serving/deploy_artifacts.py` |
| **Dashboard** | `scripts/run_dashboard_real.py` | `orchestration/flows/dashboard_flow.py` |
| **Full pipeline** | `scripts/run_full_pipeline.py` | `orchestration/trigger.py` (PipelineTriggerChain) |
| **Artifact verification** | `scripts/verify_all_artifacts.py` | 73 validation checks |
| **Paper figures** | `scripts/assemble_paper_artifacts.py` | 25 paper-ready artifacts |

### Quality Gates

Every commit must pass three gates:

1. **Tests** -- `uv run pytest tests/v2/ -x -q` (unit, integration, E2E)
2. **Lint** -- `uv run ruff check src/ tests/` (no violations)
3. **Types** -- `uv run mypy src/` (no type errors)

Pre-commit hooks enforce ruff formatting, trailing whitespace, YAML validation, and
PRD citation integrity (append-only bibliography).

---

## Experiment Configurations (19)

| Config | Model | Purpose |
|--------|-------|---------|
| `dynunet_losses.yaml` | DynUNet | 4 losses x 3 folds x 100 epochs (main baseline) |
| `dynunet_half_width.yaml` | DynUNet | Width ablation (filters/2) |
| `dynunet_topology.yaml` | DynUNet | Topology-aware losses |
| `dynunet_topology_all_approaches.yaml` | DynUNet | All topology approaches |
| `dynunet_graph_topology.yaml` | DynUNet | Graph-constrained topology |
| `dynunet_multitask_ablation.yaml` | DynUNet | Multi-task auxiliary heads |
| `dynunet_tffm_ablation.yaml` | DynUNet+TFFM | Dense GAT ablation |
| `dynunet_d2c_ablation.yaml` | DynUNet | Domain-to-class ablation |
| `dynunet_all_losses_debug.yaml` | DynUNet | All 18 losses smoke test |
| `dynunet_e2e_debug.yaml` | DynUNet | Quick E2E debug |
| `sam3_vanilla_baseline.yaml` | SAM3 Vanilla | Frozen encoder + decoder |
| `sam3_topolora_topology.yaml` | SAM3 TopoLoRA | + LoRA + topology loss |
| `sam3_hybrid_fusion.yaml` | SAM3 Hybrid | SAM3 + DynUNet fusion |
| `sam3_vanilla_debug.yaml` | SAM3 Vanilla | Debug (6 epochs, 3 folds) |
| `sam3_topolora_debug.yaml` | SAM3 TopoLoRA | Debug (6 epochs, 3 folds) |
| `sam3_hybrid_debug.yaml` | SAM3 Hybrid | Debug (6 epochs, 3 folds) |
| `sam3_all_debug.yaml` | All SAM3 | Smoke test all variants |
| `dynunet_graph_topology_debug.yaml` | DynUNet | Graph topology debug |
| `dynunet_topology_all_approaches_debug.yaml` | DynUNet | All topology debug |

---

## Directory Structure

```
minivess-mlops/
|-- src/minivess/                  Main package (207 Python files, 39K LOC)
|   |-- adapters/                  ModelAdapter ABC + 12 families (27 files)
|   |   |-- base.py                ModelAdapter ABC, SegmentationOutput
|   |   |-- model_builder.py       build_adapter() factory dispatch
|   |   |-- dynunet.py             MONAI DynUNet (primary 3D baseline)
|   |   |-- sam3_backbone.py       SAM3 ViT-32L encoder wrapper
|   |   |-- sam3_vanilla.py        SAM3 frozen encoder + decoder
|   |   |-- sam3_topolora.py       SAM3 + LoRA on FFN layers
|   |   |-- sam3_hybrid.py         SAM3 features + DynUNet 3D fusion
|   |   |-- multitask_adapter.py   Generic multi-task (config-driven heads)
|   |   |-- tffm_wrapper.py        TFFM dense GAT wrapper
|   |   +-- ...                    vesselfm, comma, mamba
|   |-- pipeline/                  Training, evaluation, metrics, losses
|   |   |-- loss_functions.py      18 loss functions with build_loss_function()
|   |   |-- comparison.py          Cross-loss comparison + paired bootstrap
|   |   |-- topology_metrics.py    13+ topology/graph metrics
|   |   |-- deploy_onnx_export.py  ONNX export + validation
|   |   |-- deploy_champion_discovery.py  Champion model selection
|   |   +-- ...                    train engine, inference, checkpoints
|   |-- ensemble/                  Ensembling, UQ, calibration
|   |   |-- strategies.py          7 ensemble strategies
|   |   |-- conformal.py           Split conformal prediction
|   |   |-- morphological_conformal.py   Morphological bands (ConSeMa)
|   |   |-- distance_conformal.py  Distance transform conformal (CLS)
|   |   |-- risk_control.py        Risk-controlling prediction (LTT)
|   |   |-- mapie_conformal.py     MAPIE library integration
|   |   |-- mc_dropout.py          MC Dropout UQ
|   |   |-- deep_ensembles.py      Deep ensemble UQ
|   |   +-- calibration.py         Temperature scaling, ECE/MCE
|   |-- orchestration/             Prefect 3.x flows
|   |   |-- flows/data_flow.py     Flow 1: Data Engineering
|   |   |-- flows/post_training_flow.py  Flow 2.5: Post-Training Plugins
|   |   |-- flows/analysis_flow.py Flow 3: Model Analysis
|   |   |-- flows/annotation_flow.py  Label Studio integration
|   |   |-- flows/dashboard_flow.py   Flow 5: Dashboard (best-effort)
|   |   |-- deploy_flow.py         Flow 4: Deployment
|   |   |-- trigger.py             PipelineTriggerChain (7 flows)
|   |   +-- _prefect_compat.py     Graceful degradation for CI
|   |-- pipeline/post_training_plugins/  6 post-hoc plugin implementations
|   |   |-- swa.py                 SWA plugin (checkpoint averaging)
|   |   |-- multi_swa.py           Multi-SWA plugin (M independent models)
|   |   |-- model_merging.py       Linear/SLERP/layer-wise merging
|   |   |-- calibration.py         Temperature scaling, isotonic, Platt
|   |   |-- crc_conformal.py       CRC conformal prediction
|   |   +-- conseco_fp_control.py  ConSeCo FP rate control
|   |-- config/                    Pydantic v2 config models
|   |   |-- models.py              ModelFamily, ModelConfig, DataConfig
|   |   |-- adaptive_profiles.py   HardwareBudget, auto compute detection
|   |   |-- model_profiles.py      Per-model YAML profiles
|   |   |-- deploy_config.py       DeployConfig, ChampionCategory
|   |   +-- post_training_config.py  PostTrainingConfig + plugin sub-configs
|   |-- data/                      Data loading, profiling, DVC integration
|   |-- serving/                   BentoML, ONNX inference, Gradio demo
|   |-- observability/             MLflow tracking, DuckDB analytics
|   |-- agents/                    LangGraph agent definitions
|   |-- compliance/                SaMD audit trail, model cards
|   +-- validation/                Pandera, Great Expectations, Deepchecks
|
|-- tests/                         238 test files, 53K LOC, 3214 tests
|   |-- v2/unit/                   Fast isolated unit tests
|   |-- v2/integration/            Service integration tests
|   +-- v2/e2e/                    End-to-end pipeline tests
|
|-- configs/
|   |-- experiments/               19 experiment YAML configs
|   |-- post_training/             Post-training plugin config (default.yaml)
|   |-- model_profiles/            Per-model VRAM/compute profiles
|   +-- splits/                    Cross-validation fold definitions
|
|-- scripts/                       26 operational scripts
|   |-- train_monitored.py         Crash-resistant training with checkpoints
|   |-- run_experiment.py          YAML-driven experiment runner
|   |-- run_full_pipeline.py       Full 7-flow pipeline trigger
|   |-- verify_all_artifacts.py    73 artifact validation checks
|   |-- assemble_paper_artifacts.py  25 paper-ready figures/tables
|   +-- ...                        analysis, export, monitoring scripts
|
|-- deployment/
|   |-- docker-compose.yml         Profile-based (dev/monitoring/full)
|   |-- docker-compose.flows.yml   Per-flow Docker services (7 flows)
|   |-- docker/Dockerfile.post_training  Post-training flow container
|   +-- Dockerfile                 Multi-stage build
|
|-- docs/
|   |-- adr/                       6 Architecture Decision Records
|   |-- planning/prd/              78 Bayesian decision nodes (5 levels)
|   |-- results/                   Experiment reports and figures
|   +-- planning/                  Implementation plans and research
|
|-- outputs/                       Verified pipeline artifacts
|   |-- analysis/                  Figures (PNG+SVG), tables (MD+TEX)
|   |-- paper_artifacts/           Paper-ready assembled artifacts
|   +-- duckdb/parquet/            Parquet exports (DuckDB gitignored)
|
|-- .claude/                       Claude Code configuration
|   |-- skills/                    TDD skill, PRD update skill
|   +-- metalearning/              Cross-session corrective insights
|
|-- pyproject.toml                 uv + PEP 621 project definition
|-- CLAUDE.md                      AI development rules and quick reference
+-- LEARNINGS.md                   Cross-session accumulated discoveries
```

---

## Architecture Decision Records

| ADR | Decision |
|-----|----------|
| [0001](docs/adr/0001-model-adapter-abc.md) | Model Adapter Abstract Base Class |
| [0002](docs/adr/0002-dual-config-system.md) | Dual Configuration System (Hydra-zen + Dynaconf) |
| [0003](docs/adr/0003-validation-onion.md) | Multi-Layer Validation ("Validation Onion") |
| [0004](docs/adr/0004-local-first-observability.md) | Local-First Observability Stack |
| [0005](docs/adr/0005-tdd-mandatory.md) | Mandatory Test-Driven Development for SaMD |
| [0006](docs/adr/0006-sam3-variant-architecture.md) | SAM3 Variant Architecture |

---

## PRD System

The project uses a **hierarchical probabilistic PRD** (Bayesian decision network) with
78 decision nodes across 5 levels:

- **L1** -- Strategic direction (overall objectives)
- **L2** -- Architecture choices (pipeline structure, data flow)
- **L3** -- Technology selection (specific tools and libraries)
- **L4** -- Deployment decisions (serving, promotion, monitoring)
- **L5** -- Implementation details (loss functions, metrics, augmentation)

Each node carries prior/posterior probabilities and evidence. The PRD serves as the
structured evidence base for a future peer-reviewed article. See
[`docs/planning/prd/README.md`](docs/planning/prd/README.md).

---

## Codebase Statistics

| Metric | Count |
|--------|-------|
| Model families | 12 |
| Loss functions | 18 |
| Post-training plugins | 6 |
| Experiment configs | 19 |
| Conformal/UQ methods | 7 |
| Ensemble strategies | 7 |
| Topology metrics | 13+ |
| PRD decision nodes | 78 |
| ADRs | 6 |
| CI workflows | 6 |
| Prefect flows | 7 |

---

## Development Workflow

This project enforces **strict TDD** (test-driven development). Every change follows the
RED-GREEN-VERIFY-FIX-CHECKPOINT-CONVERGE cycle defined in the
[self-learning-iterative-coder skill](.claude/skills/self-learning-iterative-coder/SKILL.md).

### Multi-Environment Compute

Everything works identically on:
- **Local workstation** -- single GPU, limited RAM (tested on RTX 2070 Super 8GB)
- **Intranet servers** -- multi-GPU, team access
- **Cloud instances** -- Docker, mounted drives
- **CI runners** -- CPU-only, automated (GitHub Actions)

Hardware auto-detection selects batch size, patch size, and cache rate via
`AdaptiveComputeProfile` in `config/adaptive_profiles.py`.

### Task-Agnostic Architecture

This is an **MLOps platform**, not a single-use-case repo. Multi-task adapters, losses,
metrics, and data pipelines are GENERIC and config-driven. Specific tasks (SDF edge
detection, centerline extraction, artery/vein classification, etc.) are YAML config
instantiations, not code. See CLAUDE.md Core Principle #8.

---

## Roadmap

### Completed

- [x] Foundation (uv, Docker, configs, CI, pre-commit)
- [x] Core ML pipeline (12 model families, 18 losses, training engine)
- [x] Evaluation (8-metric suite, bootstrap CIs, paired tests)
- [x] Ensembling (7 strategies, greedy soup, calibration)
- [x] Conformal UQ (5 methods + MC Dropout + deep ensembles)
- [x] Graph topology (13+ metrics, TFFM, centreline head)
- [x] Serving (BentoML, ONNX Runtime, Gradio)
- [x] Deployment flow (champion discovery, ONNX export, promotion)
- [x] Observability (MLflow, DuckDB, OpenLineage, Langfuse)
- [x] Real-data E2E pipeline verification (all flows)
- [x] SAM3 integration: backbone, decoder, 3 adapter variants, 153 tests
- [x] Post-training plugin architecture (6 plugins, 96 tests, Flow 2.5)

### In Progress

- [ ] SAM3 debug training (vanilla/topolora/hybrid, 6 epochs x 3 folds)
- [ ] SAM3 vs DynUNet comparison report

### Planned

- [ ] SAM3 full training (50 epochs, production configs)
- [ ] VesselFM integration (`feat/vesselfm` branch)
- [ ] Advanced conformal methods (SACP, RW-CP, TUNE++, Generative CP)
- [ ] Topograph post-processing, learned reconnection
- [ ] Paper manuscript assembly

---

## Further Reading

- [Full Modernization Plan](docs/modernize-minivess-mlops-plan.md) -- architecture, tool rationale, phased roadmap
- [Claude Code Patterns](docs/claude-code-patterns.md) -- TDD patterns demonstrated during v2 development
- [SAM3 Literature Report](docs/planning/sam3-literature-research-report.md) -- 80+ paper survey on SAM for segmentation
- [Post-Training Plugins Plan](docs/planning/post-training-plugins-and-swa-planning.md) -- Plugin architecture, SWA vs SWAG distinction, 6 plugins
- [Graph Topology Analysis](docs/planning/graph-connectivity-analysis.md) -- 80+ paper survey on topology metrics
- [Loss Variation Results](docs/results/dynunet_loss_variation_v2_report.md) -- DynUNet baseline analysis
- [PRD Overview](docs/planning/prd/README.md) -- Bayesian decision network navigation
- [Legacy Wiki](https://github.com/petteriTeikari/minivess_mlops/wiki) -- background from v0.1-alpha

---

## License

MIT
