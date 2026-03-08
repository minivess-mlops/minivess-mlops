# Orchestration — Prefect Flows

## STOP Protocol (Non-Negotiable)

Before ANY training or pipeline execution, verify ALL four checks:

- **S**(ource): Running inside Docker container? `_require_docker_context()` enforces this.
- **T**(racking): Prefect orchestration active? `PREFECT_DISABLED=1` → only for pytest.
- **O**(utputs): All artifact paths volume-mounted? `/tmp` → REJECT.
- **P**(rovenance): Reproducible on another machine? Host-env dependent → REJECT.

Escape hatch: `MINIVESS_ALLOW_HOST=1` for **pytest ONLY** — never in scripts or production.

## Flow Architecture

Each Prefect flow runs in its own Docker container. Flows communicate through
**MLflow artifacts ONLY** — no shared filesystem, no direct function calls.

### Naming Conventions
- Flow function: `{name}_flow()` or `run_{name}_flow()` — decorated with `@flow`
- Task function: `{name}_task()` — decorated with `@task`
- File: `flows/{name}_flow.py` — one flow per file

### Config Pipeline (CRITICAL — script-consolidation.xml Phase 0)

The INTENDED pipeline (not yet implemented in train_flow.py):
```
EXPERIMENT env var → compose_experiment_config() → resolved dict
→ training_flow(config_dict=resolved) → tracker.log_hydra_config()
→ MLflow artifact: config/resolved_config.yaml
```

**Current gap:** `train_flow.py` bypasses Hydra-zen. Uses argparse + 9-key dict.
Never calls `compose_experiment_config()`. Never calls `log_hydra_config()`.
Fix: `script-consolidation.xml` Phase 0.

### Inter-Flow Contract

Downstream flows discover upstream runs via MLflow:
1. Search by `experiment_name` tag
2. Read `config/resolved_config.yaml` artifact for full config
3. Read model checkpoints from MLflow artifact URIs
4. Tag: `loss_function` (NOT `loss_name` — there's a mismatch to fix)

## Files

| File | Purpose |
|------|---------|
| `flows/train_flow.py` | Training flow — GPU, fold iteration, MLflow logging |
| `flows/analysis_flow.py` | Analysis — ensemble building, evaluation, comparison |
| `flows/post_training_flow.py` | Post-training — SWA, calibration, conformal |
| `flows/deploy_flow.py` | Deployment — ONNX export, BentoML, promotion |
| `flows/dashboard_flow.py` | Dashboard — paper figures, reports |
| `flows/qa_flow.py` | QA — MLflow integrity, ghost runs, param validation |
| `_prefect_compat.py` | Prefect compatibility layer (CI/test fallback) |
| `deployments.py` | Flow deployment configurations |
| `trigger.py` | PipelineTriggerChain for cascading flows |

## What AI Must NEVER Do

- Run training outside Docker (no `uv run python scripts/train_monitored.py`)
- Set `PREFECT_DISABLED=1` outside of pytest
- Create flows that bypass MLflow for inter-flow communication
- Use `/tmp` or `tempfile.mkdtemp()` for any artifact that must survive the container
- Add `argparse` params that bypass Hydra-zen composition
