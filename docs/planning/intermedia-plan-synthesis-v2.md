# MinIVess MLOps v2 — Intermediate Plan Synthesis v2

**Date:** 2026-03-08
**Builds on:** `intermedia-plan-synthesis.md` (v1, 2026-03-07)
**Branch:** main (pre-execution baseline)
**Synthesizes:** v1 synthesis + 25 planning docs + CLAUDE.md + MEMORY.md + 5 metalearning docs + script-consolidation.xml
**Author:** Synthesis via deep read of 50+ source files across 5 parallel research agents

---

## What Changed Since v1

v1 (2026-03-07) identified two P0 blockers (#367 standalone scripts, #369 Docker volume
mounts). Since then:

1. **Docker volume mounts FIXED** — `docker-compose.flows.yml` now has explicit volumes
   for ALL 9 services (acquisition, data, train, post_training, analyze, deploy, dashboard,
   qa, biostatistics, annotation, hpo, pipeline). This closes most of #369.

2. **Hydra-zen composition gap DISCOVERED** — The root cause of all config reproducibility
   issues: `train_flow.py` bypasses Hydra entirely. Uses argparse + 9-key minimal dict
   instead of `compose_experiment_config()`. `log_hydra_config()` exists but is NEVER
   called. This is now documented in `script-consolidation.xml` v3 as Phase 0.

3. **Overnight autonomous execution model CREATED** — `overnight-master-plan.xml` with
   4 child plans (01-04) targeting issues #304, #343, #365, #401, #424, #469, #474.
   Execution script at `overnight-prefect-docker-monai.sh` (2 children, already defined).

4. **Pydantic AI micro-orchestration MERGED** — PR #500 (6 phases, 19 tasks). Agent
   infrastructure is now in place.

5. **Total Prefect flow decoupling MERGED** — PR #478 (20-task plan). All flows
   independently testable.

6. **Biostatistics flow MERGED** — PR #459 (Phases 0-9). Full statistical analysis
   pipeline in place.

---

## Part 1: Architecture Status Matrix

### 1.1 The Config Pipeline (Critical Gap)

The intended pipeline (documented in CLAUDE.md Rule #23 and `configs/README.md`):

```
configs/experiment/*.yaml          ← Hydra experiment YAML
    ↓ compose_experiment_config()
Resolved config dict               ← Full merged config (data, model, training, checkpoint)
    ↓ training_flow(config_dict=...)
train_one_fold_task()              ← Extracts params, executes training
    ↓ tracker.log_hydra_config()
MLflow artifact:                   ← config/resolved_config.yaml (THE record)
    ↓ downstream flows read via MLflow API
post_training/analysis flows       ← Discover upstream config from artifacts
```

**Current reality:**

```
train_flow.py __main__             ← argparse + env vars
    ↓ builds 9-key minimal dict    ← Missing: data_dir, splits_file, seed,
    ↓                                 checkpoint config, architecture_params
training_flow(individual params)   ← NOT config_dict
    ↓ tracker.start_run()          ← Logs system info, training config
    ↓ tracker.log_hydra_config()   ← NEVER CALLED
MLflow run                         ← No resolved_config.yaml artifact
    ↓ downstream flows search by tag
analysis_flow                      ← Uses "loss_function" tag, but train logs "loss_name"
```

**Consequence:** Every training run lacks the resolved config artifact. Downstream flows
can't discover upstream config. Experiments are NOT reproducible from MLflow alone.

### 1.2 Flow Implementation Levels (Updated)

| # | Flow | File | Real Training? | Hydra Integration | Docker Ready | Volume Mounts |
|---|------|------|---------------|-------------------|-------------|---------------|
| 0 | Acquisition | `acquisition_flow.py` | N/A (data download) | NO | YES (Dockerfile exists) | YES (raw_data, logs) |
| 1 | Data | `data_flow.py` | N/A (splits/validation) | NO | YES | YES (raw_data:ro, data_cache, configs_splits, mlruns) |
| 2 | **Train** | `train_flow.py` | **YES but bypasses Hydra** | **NO — B1** | YES | YES (data:ro, splits:ro, checkpoints, mlruns, logs) |
| 2.5 | Post-Training | `post_training_flow.py` | YES (SWA, calibration) | NO | YES | YES (checkpoints:ro, data:ro, post_training_out, mlruns) |
| 3 | Analysis | `analysis_flow.py` | YES (eval, ensemble) | NO | YES | YES (data:ro, checkpoints:ro, splits:ro, mlruns, outputs) |
| 4 | Deploy | `deploy_flow.py` | YES (ONNX, BentoML) | NO | YES | YES (checkpoints:ro, mlruns:ro, outputs, bentoml) |
| 5 | Dashboard | `dashboard_flow.py` | YES (reports) | NO | YES | YES (mlruns:ro, analysis:ro, outputs) |
| 6 | QA | `qa_flow.py` | YES (validation) | NO | YES | YES (mlruns:ro, dashboard_out) |
| - | Bio | `biostatistics_flow.py` | YES (stats) | NO | YES | YES (mlruns:ro, outputs, configs:ro) |
| - | Annotation | `annotation_flow.py` | YES (labeling) | NO | YES | YES (bentoml:ro, data:ro, mlruns) |
| - | HPO | `hpo_flow.py` | YES (Optuna) | NO | YES | YES (data:ro, splits:ro, checkpoints, mlruns, logs) + GPU |
| - | Pipeline | `pipeline_flow.py` | YES (orchestrator) | NO | YES | YES (mlruns:ro) |

**Key finding:** Docker volume mounts are now complete (v1 said 5 flows had zero mounts —
this has been fixed). The remaining critical gap is **Hydra integration** — no flow calls
`compose_experiment_config()`.

### 1.3 Blocker Inventory

| ID | Severity | Description | Fix Location | Status |
|----|----------|-------------|-------------|--------|
| B1 | CRITICAL | train_flow.py bypasses Hydra composition | train_flow.py __main__ | OPEN — script-consolidation Phase 0 |
| B2 | CRITICAL | log_hydra_config() never called | train_flow.py + tracking.py | OPEN — Phase 0.3 |
| B3 | CRITICAL | MLflow tag mismatch: loss_name vs loss_function | train_flow.py + builder.py | OPEN — Phase 0.5 |
| B4 | CRITICAL | eval_fold2_dsc gate rejects debug runs | builder.py:221 | OPEN — Phase 1.1 |
| B5 | CRITICAL | Checkpoints not logged as MLflow artifacts | trainer.py | OPEN — Phase 1.2 |
| B6 | CRITICAL | Analysis flow can't find debug experiment | analysis_flow.py | OPEN — Phase 1.3 |
| B7 | HIGH | post_training_out volume missing from analyze | docker-compose.flows.yml | OPEN — Phase 1.4 |
| B8 | HIGH | load_checkpoint silently returns random weights | builder.py | OPEN — Phase 1.5 |

---

## Part 2: Planning Document Inventory

### 2.1 Active Plans (Execute Now)

| Plan | File | Issues | Priority | Est. Duration |
|------|------|--------|----------|---------------|
| **Script Consolidation** | `script-consolidation.xml` | — | P0 | 10-14 hours |
| **Overnight Master** | `overnight-master-plan.xml` | #304,#343,#365,#401,#424,#469,#474 | P0 | 3-5 hours |
| **Overnight Child 01** | `overnight-child-01.xml` | #424,#469 | P0 | 30-45 min |
| **Overnight Child 02** | `overnight-child-02.xml` | #474,#343 | P0 | 90-120 min |
| **Overnight Child 03** | `overnight-child-03.xml` | #304 | P2 | 30-45 min |
| **Overnight Child 04** | `overnight-child-04.xml` | #365,#401 | P1 | 45-60 min |
| **Prefect/Docker Optimization** | `overnight-child-prefect-docker.xml` | #434,#503,#504 | P1 | overnight |
| **MONAI Multi-Strategy Eval** | `overnight-child-monai-eval.xml` | #505 | P1 | overnight |

### 2.2 Execution Order (Correct Sequencing)

The script-consolidation plan (Phases 0-5) **MUST execute BEFORE** the overnight master
plan because:

1. Phase 0 (Hydra bridge) fixes the config pipeline that all flows depend on
2. Phase 1 (blocker fixes) unblocks debug training scenarios
3. Phase 2 (debug configs) creates the experiment YAMLs that overnight plans will use
4. The overnight master plan's Branch 4 (training resume + HPO grid) depends on
   `compose_experiment_config()` working correctly

**Correct order:**
```
script-consolidation Phase 0-1  (Hydra bridge + blockers)    ← foundational
script-consolidation Phase 2-5  (debug configs + shell)      ← usability
overnight-master-plan child 01  (infrastructure hardening)   ← quick wins
overnight-master-plan child 02  (MONAI ecosystem audit)      ← core value
overnight-master-plan child 03  (test tiers)                 ← DevEx
overnight-master-plan child 04  (training resume + HPO)      ← features
overnight-prefect-docker        (Prefect/Docker optimization)← infrastructure
overnight-monai-eval            (multi-strategy eval)        ← analysis
```

### 2.3 Research Reports (Reference Only — Not Executable)

These informed architectural decisions but don't have implementation tasks:

| Report | File | Key Insight |
|--------|------|-------------|
| Topology-Aware Segmentation | `topology-aware-segmentation-literature-research-report.md` | CbDice+clDice = default loss |
| SAM3 Literature | `sam3-literature-research-report.md` | ViT-32L, 1008x1008, SDPA mandatory |
| Conformal UQ | `conformal-uq-segmentation-report.md` | MAPIE + netcal for calibration |
| Data Engineering Quality | `data-engineering-quality-etl-report.md` | Pandera + Great Expectations |
| Interactive Segmentation | `interactive-segmentation-report.md` | SAM3 for annotation flow |
| Advanced Ensembling | `advanced-ensembling-bootstrapping-report.md` | 4 ensemble strategies |
| Monitoring Research | `monitoring-research-report.md` | Evidently + whylogs |
| MLOps Practices | `mlops-practices-report.md` | Prefect + MLflow + Docker stack |
| Regulatory Ops | `regulatory-ops-report.md` | IEC 62304, EU AI Act |

### 2.4 Metalearning Documents (Must-Read Before Sessions)

| Document | Key Lesson |
|----------|-----------|
| `2026-03-02-sam3-implementation-fuckup.md` | SAM3 != SAM2. Web-search first. |
| `2026-03-02-session-failure-self-reflection.md` | Confabulation costs hours |
| `2026-03-04-skip-mypy-hook-failure.md` | Never skip pre-commit without consent |
| `2026-03-05-silent-fallback-failure.md` | Silent fallback = cosmetic success = lie |
| `2026-03-06-standalone-script-antipattern.md` | scripts/*.py NOT a run path |
| `2026-03-06-regex-ban.md` | import re banned for structured data |
| `2026-03-07-silent-existing-failures.md` | Every failure needs immediate action |
| `2026-03-07-ci-reenabled-without-permission.md` | Never re-enable disabled infra |
| `2026-03-07-docker-volume-mount-violation.md` | /tmp forbidden for artifacts |

---

## Part 3: Model Portfolio

### 3.1 Models Fitting 8GB GPU (Verified)

| Model | Family | Peak VRAM | Status | Adapter File | Notes |
|-------|--------|-----------|--------|-------------|-------|
| DynUNet | MONAI native | 3.5 GB | Production | `dynunet_adapter.py` | Default, 100-epoch verified |
| SAM3 Vanilla | Foundation | 2.9 GB | Training verified | `sam3_adapter.py` | SDPA mandatory, val_roi_size differs |
| SAM3 Hybrid | Foundation | 7.5 GB | Marginal | `sam3_adapter.py` | HF encoder + MONAI decoder |
| CoMMA Mamba | SSM | ~2.5 GB | Planned | — | Vision Mamba, capacity matching needed |
| U-Like Mamba | SSM | ~2.5 GB | Planned | — | U-Net + Mamba hybrid |

### 3.2 MONAI Native Models (Overnight Child 02)

| Model | MONAI Class | Est. Adapter Lines | Priority |
|-------|------------|-------------------|----------|
| SegResNet | `monai.networks.nets.SegResNet` | <100 | P0 |
| SwinUNETR | `monai.networks.nets.SwinUNETR` | <100 | P0 |
| UNETR | `monai.networks.nets.UNETR` | <100 | P1 |
| AttentionUnet | `monai.networks.nets.AttentionUnet` | <100 | P1 |

### 3.3 Ensemble Strategies

| Strategy | Requires | Verified |
|----------|---------|----------|
| `per_loss_single_best` | 1 loss, 3 folds | YES (quasi-E2E) |
| `all_loss_single_best` | 4 losses, 3 folds | YES (quasi-E2E) |
| `per_loss_all_best` | 1 loss, all folds above threshold | YES |
| `all_loss_all_best` | All losses, all qualifying folds | YES |

### 3.4 Post-Training Plugins

| Plugin | Description | Status |
|--------|------------|--------|
| `swa` | Stochastic Weight Averaging | Implemented |
| `multi_swa` | Multi-SWA (ensemble of SWA models) | Implemented |
| `model_merging` | Weight interpolation between models | Implemented |
| `calibration` | Temperature scaling + reliability diagrams | Implemented |
| `crc_conformal` | Conformal Risk Control | Implemented |
| `conseco_fp_control` | Conservative-selective FP control | Implemented |

---

## Part 4: Progressive Disclosure Institutional Knowledge System

### 4.1 Problem

Context amnesia is the primary failure mode across Claude Code sessions. With the
codebase at ~50K lines of Python and ~100 planning documents, each session starts with
partial context and risks:

1. **Re-implementing from scratch** — ignoring existing Hydra composition, Prefect flows,
   Docker infrastructure that already works
2. **Creating parallel systems** — new config parsers, separate debug frameworks, custom
   merge logic that duplicates Hydra-zen
3. **Goal substitution** — optimizing for "make loss go down" instead of demonstrating
   reproducible MLOps infrastructure

### 4.2 Design: Three-Layer Progressive Disclosure

```
Layer 0: Always loaded (< 200 lines each)
├── CLAUDE.md                    ← 23 rules, design goals, architecture overview
├── MEMORY.md                    ← Critical patterns, verified configs, known issues
└── configs/README.md            ← Config system architecture, composition pipeline

Layer 1: Loaded on file access (folder-level CLAUDE.md)
├── src/minivess/adapters/CLAUDE.md    ← Model adapter rules, SAM3 VRAM table
├── src/minivess/orchestration/CLAUDE.md  ← Flow rules, STOP protocol  [TO CREATE]
├── src/minivess/config/CLAUDE.md      ← Hydra-zen rules, compose API  [TO CREATE]
├── src/minivess/ensemble/CLAUDE.md    ← Ensemble strategy rules       [TO CREATE]
├── deployment/CLAUDE.md               ← Docker rules, volume mounts   [TO CREATE]
└── tests/CLAUDE.md                    ← Test rules, markers, fixtures [TO CREATE]

Layer 2: Deep reference (planning docs, metalearning)
├── docs/planning/intermedia-plan-synthesis-v2.md  ← This document
├── docs/planning/script-consolidation.xml         ← Hydra bridge plan
├── docs/planning/overnight-master-plan.xml        ← Issue closure plan
├── .claude/metalearning/*.md                      ← Failure post-mortems
└── docs/planning/*-report.md                      ← Research reports
```

### 4.3 Layer 1 CLAUDE.md Contents (To Create)

#### `src/minivess/orchestration/CLAUDE.md`
- STOP Protocol: S(Docker), T(Prefect), O(volumes), P(provenance)
- Flow naming: `{flow_name}_flow()` is the `@flow` decorated function
- Task naming: `{task_name}_task()` is the `@task` decorated function
- Inter-flow contract: MLflow artifacts ONLY (no shared filesystem)
- Docker gate: `_require_docker_context()` — escape hatch `MINIVESS_ALLOW_HOST=1` for pytest only
- Config pipeline: EXPERIMENT env var → compose_experiment_config() → training_flow(config_dict=...)

#### `src/minivess/config/CLAUDE.md`
- Hydra-zen = single source of truth (Rule #23)
- compose_experiment_config(): the ONLY way to get a resolved config
- Config groups: data/, model/, training/, checkpoint/
- Experiment configs: configs/experiment/*.yaml (22+ configs)
- Debug configs: configs/experiment/debug_*.yaml (same format, same directory)
- BANNED: parallel config systems, custom merge scripts, argparse bypass
- Pipeline: YAML → compose → resolved dict → log_hydra_config() → MLflow artifact

#### `src/minivess/ensemble/CLAUDE.md`
- 4 ensemble strategies: per_loss_single_best, all_loss_single_best, per_loss_all_best, all_loss_all_best
- discover_training_runs_raw(): requires "loss_function" tag (NOT "loss_name")
- eval_fold2_dsc gate: make configurable for debug scenarios
- load_checkpoint(): MUST raise on state_dict mismatch (never return random weights)
- All ensemble building goes through MLflow run discovery — no filesystem traversal

#### `deployment/CLAUDE.md`
- Three-layer Docker hierarchy: nvidia/cuda → minivess-base → Dockerfile.{flow}
- Flow Dockerfiles NEVER run apt-get or uv — all deps in Dockerfile.base
- docker-compose.flows.yml: 12 services, all with explicit volume mounts
- Volume mount rules: ALL artifacts must be volume-mapped, /tmp FORBIDDEN
- Network: minivess-network (external, create with `docker network create`)
- GPU: only train and hpo services reserve GPU devices

#### `tests/CLAUDE.md`
- All tests use `MINIVESS_ALLOW_HOST=1` to bypass Docker gate
- `PREFECT_DISABLED=1` for unit tests (no server required)
- Markers: `@pytest.mark.slow` for integration tests (>5s)
- Fixtures: `tmp_path` for ephemeral test data (NOT tempfile.mkdtemp)
- Never import from scripts/ — those are migration utilities

### 4.4 MEMORY.md Topic Files (To Create)

Move detailed content from MEMORY.md into topic files:

| Topic File | Content |
|-----------|---------|
| `config-system.md` | Hydra-zen pipeline, compose API, known gaps |
| `docker-infrastructure.md` | Volume mounts, Dockerfile layers, network |
| `model-portfolio.md` | All models, VRAM budgets, adapter patterns |
| `overnight-execution.md` | How to structure child plans, runner scripts |

---

## Part 5: What Must Happen Next (Prioritized)

### Priority 0: Hydra-zen Bridge (script-consolidation Phase 0-1)

This is the **single most important change** in the entire repo. Without it:
- No training run has a resolved config artifact in MLflow
- No experiment is reproducible from MLflow alone
- Debug configs can't work (they depend on compose_experiment_config())
- Downstream flows can't discover upstream config

**Tasks:** 10 tasks across Phase 0 (5 tasks) and Phase 1 (5 tasks). See
`script-consolidation.xml` for full details.

### Priority 1: Debug Experiment Configs (script-consolidation Phase 2-5)

Create 5 debug experiment YAMLs (standard Hydra format), shell wrapper, justfile
integration, multi-flow chaining. **Blocked by Priority 0.**

### Priority 2: Overnight Master Plan (4 child plans)

Infrastructure hardening (#424, #469), MONAI ecosystem audit (#474, #343), test tiers
(#304), training resume + HPO grid (#365, #401). **Partially blocked by Priority 0**
(child 04 depends on config pipeline).

### Priority 3: Layer 1 CLAUDE.md Files

Create 5 folder-level CLAUDE.md files. No code changes — documentation only. Can be done
in parallel with any other work.

### Priority 4: MEMORY.md Consolidation

Trim MEMORY.md to <200 lines by moving detailed content to topic files. No code changes.

---

## Part 6: Known Risks

### 6.1 Hydra-zen Compose May Fail in Docker

`compose_experiment_config()` uses Hydra Compose API which requires `configs/` on the
Python path. Inside Docker, `configs/` is at `/app/configs/`. The Hydra config path
resolution may need adjustment. `compose.py` has a `_compose_with_manual_merge()` fallback
for when Hydra Compose fails — this fallback is the safety net.

### 6.2 SAM3 Models Need HF_TOKEN

SAM3 vanilla and hybrid require `HF_TOKEN` for downloading pretrained weights from
HuggingFace. Docker containers must have `HF_TOKEN` in their environment.
`docker-compose.flows.yml` already passes `HF_TOKEN: ${HF_TOKEN:-}`.

### 6.3 Mamba Models Not Yet Implemented

CoMMA Mamba and U-Like Mamba have research plans (`comma-mamba-plan.md`,
`mamba-model-capacity-matching.md`) but no adapter code. These are placeholder entries
in model portfolio and debug_all_models.yaml.

### 6.4 Analysis Flow Uses Regex (BANNED)

`analysis_flow.py:1033` uses `re.compile(r"^(.+)_fold(\d+)$")`. Must be replaced with
`name.rsplit("_fold", 1)` per the regex ban rule. Currently a dormant violation.

### 6.5 Overnight Execution Requires Stable main

Child plans create branches from main. If main is broken (test failures, import errors),
ALL child plans fail. Run `uv run pytest tests/ -x -q` before starting overnight execution.

---

## Appendix A: File Cross-Reference

### Config System Files
| File | Purpose |
|------|---------|
| `configs/base.yaml` | Hydra entry point: defaults list + top-level values |
| `configs/data/minivess.yaml` | Data config group |
| `configs/model/dynunet.yaml` | Model selection group |
| `configs/training/default.yaml` | Training hyperparams group |
| `configs/checkpoint/standard.yaml` | Checkpoint strategy group |
| `configs/experiment/*.yaml` | 22+ experiment configs (including debug_*) |
| `src/minivess/config/compose.py` | Hydra Compose API bridge (compose_experiment_config) |
| `src/minivess/observability/tracking.py` | ExperimentTracker + log_hydra_config() |
| `configs/README.md` | Config system documentation |

### Orchestration Files
| File | Purpose |
|------|---------|
| `src/minivess/orchestration/flows/train_flow.py` | Training Prefect flow |
| `src/minivess/orchestration/flows/analysis_flow.py` | Analysis Prefect flow |
| `src/minivess/orchestration/flows/post_training_flow.py` | Post-training plugins |
| `src/minivess/orchestration/flows/deploy_flow.py` | ONNX + BentoML + promotion |
| `src/minivess/orchestration/flows/dashboard_flow.py` | Reporting |
| `src/minivess/orchestration/flows/qa_flow.py` | Data integrity |
| `src/minivess/orchestration/_prefect_compat.py` | Prefect compatibility layer |
| `src/minivess/orchestration/deployments.py` | Flow deployment configs |

### Docker Files
| File | Purpose |
|------|---------|
| `deployment/docker/Dockerfile.base` | Layer 2: all shared deps |
| `deployment/docker/Dockerfile.train` | Layer 3: train CMD |
| `deployment/docker-compose.flows.yml` | 12 per-flow services |
| `deployment/docker-compose.yml` | Infrastructure (PostgreSQL, MinIO, MLflow, Prefect) |

### Planning Files (Active)
| File | Purpose |
|------|---------|
| `docs/planning/script-consolidation.xml` | Hydra bridge + debug mode plan |
| `docs/planning/overnight-master-plan.xml` | 7-issue closure plan |
| `docs/planning/overnight-child-{01..04}.xml` | Individual child plans |
| `docs/planning/overnight-prefect-docker-monai.sh` | Overnight runner script |
| `docs/planning/minivess-vision-enforcement-plan.md` | STOP Protocol design |
| `docs/planning/minivess-vision-enforcement-plan-execution.xml` | 14-task execution |

### Metalearning Files
| File | Lesson |
|------|--------|
| `.claude/metalearning/2026-03-02-sam3-implementation-fuckup.md` | Verify before implementing |
| `.claude/metalearning/2026-03-05-silent-fallback-failure.md` | Never silent-fail |
| `.claude/metalearning/2026-03-06-standalone-script-antipattern.md` | Prefect flows only |
| `.claude/metalearning/2026-03-06-regex-ban.md` | No regex for structured data |
| `.claude/metalearning/2026-03-07-silent-existing-failures.md` | Fix all failures |
| `.claude/metalearning/2026-03-07-ci-reenabled-without-permission.md` | Never re-enable |
