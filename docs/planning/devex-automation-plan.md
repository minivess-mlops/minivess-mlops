# DevEx Automation Plan: Single-Command Multi-Environment Compute

**Status**: DRAFT
**Created**: 2026-02-25
**Context**: [dynunet-evaluation-plan.xml](dynunet-evaluation-plan.xml) | [monai-performance-optimization-plan.md](monai-performance-optimization-plan.md)

## Problem Statement

Running ML experiments currently requires too many manual steps:

1. Clear stale swap (`sudo swapoff -a && sudo swapon -a`)
2. Start Docker services (`just dev`)
3. Wait for services to be healthy (manual `curl` checks)
4. Select correct compute profile manually
5. Launch training with correct flags
6. Monitor for crashes, manually resume
7. Inspect results across multiple UIs (MLflow, terminal, CSV)

**Goal**: Everything behind a **single function call** that works identically on:
- Local workstation (Ubuntu, RTX 2070 Super, 64 GB RAM)
- Ephemeral cloud GPU instances (Docker, mounted drives)
- On-prem multi-GPU servers (DGX-style)

## Reference Patterns

### From dpp-agents
- **Makefile-driven workflows**: `make up`, `make down`, `make logs` with Docker Compose
- **docker-up.sh pre-flight**: Validates Docker daemon, env vars, port availability before starting
- **direnv (.envrc)**: Auto-activates virtualenv, sets environment variables on `cd`
- **Overnight pipeline runner**: Long-running scripts with structured logging
- **Environment validation scripts**: Check dependencies before execution

### From foundation-PLR
- **Multi-Dockerfile architecture**: Separate `Dockerfile.dev`, `Dockerfile.test`, `Dockerfile.R`
- **Extensive Makefile** (40+ targets): `make reproduce` as the single entry point
- **Two-block reproduction**: Block 1 = data processing, Block 2 = analysis
- **Test tiering**: unit → integration → e2e, each runnable independently
- **Pinned base images**: SHA256 digests for reproducibility
- **`make reproduce`**: The gold standard — one command, full pipeline

### From minivess-mlops (current state)
- **justfile** (14 recipes): `just train`, `just dev`, `just verify`
- **Docker Compose profiles**: dev / monitoring / full (3 tiers)
- **Compute profiles**: cpu, gpu_low, gpu_high, dgx_spark, cloud_single, cloud_multi
- **CheckpointManager**: JSON-based crash recovery in train_monitored.py
- **SystemMonitor**: Background resource tracking with abort callbacks
- **Gaps**: No preflight checks, no hardware auto-detection, no unified experiment runner, no `.env.example`

## Architecture

### Layer Model

```
┌──────────────────────────────────────────────────────────┐
│  Layer 4: User Interface                                 │
│  just experiment --config experiments/dynunet_losses.yaml │
│  (single command, zero manual steps)                     │
├──────────────────────────────────────────────────────────┤
│  Layer 3: Experiment Runner (Python)                     │
│  scripts/run_experiment.py                               │
│  - Loads experiment YAML config                          │
│  - Orchestrates: preflight → train → evaluate → report   │
│  - Delegates to existing train.py / train_monitored.py   │
├──────────────────────────────────────────────────────────┤
│  Layer 2: Preflight & Environment (Python + Shell)       │
│  scripts/preflight.py                                    │
│  - Hardware detection → auto compute profile             │
│  - Service health checks (MLflow, MinIO, Postgres)       │
│  - Swap/memory hygiene (non-sudo fallback)               │
│  - Disk space validation                                 │
│  - GPU driver / CUDA version check                       │
│  - DVC data availability check                           │
├──────────────────────────────────────────────────────────┤
│  Layer 1: Infrastructure (Docker Compose + justfile)     │
│  - Docker service lifecycle (start/wait/health)          │
│  - Profile selection (dev/monitoring/full)                │
│  - Volume mounts (data, artifacts, configs)              │
│  - Resource limits on containers                         │
└──────────────────────────────────────────────────────────┘
```

### Experiment Config Format (YAML)

Instead of long CLI flag lists, experiments are defined as YAML files:

```yaml
# configs/experiments/dynunet_losses.yaml
experiment:
  name: dynunet_loss_variation
  description: "Compare loss functions for DynUNet on MiniVess"

training:
  losses: [dice_ce, cbdice, dice_ce_cldice, warp]
  num_folds: 3
  max_epochs: 50
  splits_file: configs/splits/3fold_seed42.json

compute:
  profile: auto          # auto-detect from hardware, or: gpu_low, gpu_high, etc.
  docker_profile: dev    # which Docker Compose profile to start
  require_gpu: true      # fail-fast if no GPU available

evaluation:
  metrics: [dsc, centreline_dsc, measured_masd]
  bootstrap_iterations: 1000

monitoring:
  system_monitor: true
  monitor_interval_sec: 10
  memory_abort_process_rss_gb: 55  # per-process abort threshold

resume:
  enabled: true          # auto-resume from checkpoint if available
  checkpoint_dir: null   # auto-generated if null

debug:
  enabled: false         # override: 1 epoch, 2 folds, 10 volumes
```

## Implementation Plan

### Phase 1: Preflight System (`scripts/preflight.py`)

**Goal**: Automated environment validation that works on any machine.

```python
# Checks to implement (each returns pass/warn/fail):
class PreflightChecks:
    def check_gpu_available() -> CheckResult       # nvidia-smi, CUDA version
    def check_ram_available() -> CheckResult        # Minimum 16 GB free
    def check_disk_space() -> CheckResult           # Minimum 20 GB free in work dir
    def check_swap_health() -> CheckResult          # Warn if >5 GB swap used
    def check_dvc_data() -> CheckResult             # Required data files exist
    def check_docker_services() -> CheckResult      # MLflow/MinIO/Postgres responding
    def check_env_vars() -> CheckResult             # Required vars set
    def check_python_deps() -> CheckResult          # Key imports succeed
    def auto_detect_compute_profile() -> str        # Return best profile for hardware
```

**Swap handling strategy** (no sudo required):
- **Local workstation**: Warn + suggest `sudo swapoff -a && sudo swapon -a`
- **Docker/cloud**: Swap is clean on fresh instances; skip check
- **Detect environment**: Check if running inside Docker via `/.dockerenv` or cgroup

**Hardware auto-detection logic**:
```
No GPU detected                    → cpu
GPU VRAM < 6 GB                    → cpu (warn: GPU too small)
GPU VRAM 6-10 GB, RAM < 32 GB     → gpu_low
GPU VRAM 6-10 GB, RAM >= 32 GB    → gpu_low
GPU VRAM 10-24 GB                  → gpu_high
GPU VRAM 24-48 GB                  → dgx_spark
GPU VRAM >= 48 GB, single GPU     → cloud_single
Multiple GPUs                      → cloud_multi
```

### Phase 2: Experiment Runner (`scripts/run_experiment.py`)

**Goal**: Single entry point that orchestrates the full pipeline.

```
run_experiment.py --config experiments/dynunet_losses.yaml [--debug] [--resume] [--dry-run]
```

**Execution flow**:
1. Parse experiment YAML config
2. Run preflight checks (fail-fast on critical failures)
3. Auto-detect compute profile (if `profile: auto`)
4. Start Docker services if needed (via subprocess → `just dev`)
5. Wait for service health (polling with timeout)
6. Delegate to `train_monitored.py` with correct flags
7. Aggregate results: cross-loss comparison table, MLflow links
8. Write summary report to `logs/{experiment_name}/report.md`

**Dry-run mode**: Validates config + preflight without training. Useful for CI and pre-launch verification on cloud instances.

### Phase 3: Docker Training Image (`deployment/Dockerfile.train`)

**Goal**: Self-contained training image for cloud/ephemeral instances.

```dockerfile
# Key changes from current Dockerfile:
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Use uv (not pip/poetry)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

# Entrypoint: preflight → experiment runner
ENTRYPOINT ["uv", "run", "python", "scripts/run_experiment.py"]
CMD ["--config", "configs/experiments/default.yaml"]
```

**Volume mounts** (expected by image):
```bash
docker run --gpus all \
  -v /data/minivess:/data:ro \           # Input data (read-only)
  -v /output/artifacts:/artifacts \       # Output: checkpoints, logs
  -v /output/mlruns:/mlruns \             # MLflow tracking directory
  minivess-train:latest \
  --config configs/experiments/dynunet_losses.yaml
```

### Phase 4: justfile Integration

**New recipes** to add to justfile:

```just
# Single-command experiment execution
experiment config="configs/experiments/default.yaml" *flags="":
    uv run python scripts/run_experiment.py --config {{config}} {{flags}}

# Experiment with debug overrides
experiment-debug config="configs/experiments/default.yaml":
    uv run python scripts/run_experiment.py --config {{config}} --debug

# Preflight check only (no training)
preflight:
    uv run python scripts/preflight.py

# Dry-run: validate config + preflight, no training
experiment-dry-run config="configs/experiments/default.yaml":
    uv run python scripts/run_experiment.py --config {{config}} --dry-run

# Build training Docker image
build-train:
    docker build -f deployment/Dockerfile.train -t minivess-train:latest .

# Run experiment in Docker (cloud/ephemeral pattern)
experiment-docker config="configs/experiments/default.yaml" data_dir="/data/minivess":
    docker run --gpus all \
      -v {{data_dir}}:/data:ro \
      -v $(pwd)/artifacts:/artifacts \
      -v $(pwd)/mlruns:/mlruns \
      minivess-train:latest \
      --config {{config}}
```

### Phase 5: `.env.example` and Secrets Template

Create `.env.example` with all required variables documented:

```bash
# PostgreSQL (MLflow + Langfuse backend)
POSTGRES_USER=minivess
POSTGRES_PASSWORD=changeme
POSTGRES_DB=mlflow
POSTGRES_MULTIPLE_DATABASES=langfuse

# MinIO (S3-compatible artifact store)
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=changeme

# Grafana (optional, monitoring profile)
GRAFANA_USER=admin
GRAFANA_PASSWORD=changeme

# Langfuse (optional, full profile)
LANGFUSE_SECRET=changeme
LANGFUSE_SALT=changeme

# Compute (auto-detected if unset)
# COMPUTE_PROFILE=gpu_low
# DATA_DIR=/data/minivess
# ARTIFACT_DIR=/artifacts
```

### Phase 6: CI/CD Integration (GitHub Actions)

Experiment configs become CI/CD artifacts:

```yaml
# .github/workflows/experiment.yml
on:
  workflow_dispatch:
    inputs:
      config:
        description: 'Experiment config file'
        required: true
        default: 'configs/experiments/dynunet_losses.yaml'

jobs:
  train:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - run: uv sync
      - run: uv run python scripts/run_experiment.py --config ${{ inputs.config }}
      - uses: actions/upload-artifact@v4
        with:
          name: experiment-results
          path: logs/
```

## Environment Detection Matrix

| Signal | Local Workstation | Docker Container | Cloud Instance | CI Runner |
|--------|-------------------|------------------|----------------|-----------|
| `/.dockerenv` exists | No | Yes | Maybe | Yes |
| `KUBERNETES_SERVICE_HOST` set | No | No | Maybe | Maybe |
| `CI` env var set | No | No | No | Yes |
| `nvidia-smi` available | Yes (if GPU) | Yes (if --gpus) | Yes (if GPU) | Yes (if GPU) |
| Swap used > 5 GB | Possible (stale) | No (fresh) | No (fresh) | No |
| Docker socket available | Yes | Maybe (DinD) | Yes | Yes |
| MLflow at localhost:5000 | After `just dev` | Via docker network | Via docker network | Via service |

**Detection function**:
```python
def detect_environment() -> Literal["local", "docker", "cloud", "ci"]:
    if os.environ.get("CI"):
        return "ci"
    if Path("/.dockerenv").exists():
        return "docker"
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return "cloud"
    return "local"
```

## Migration Path

### Step 1 (Immediate — blocks Phase 2-6 of dynunet-evaluation-plan)
- [ ] Create `scripts/preflight.py` with hardware auto-detection
- [ ] Create `configs/experiments/dynunet_losses.yaml` experiment config
- [ ] Create `.env.example`

### Step 2 (Short-term — enables single-command experiments)
- [ ] Create `scripts/run_experiment.py` experiment runner
- [ ] Add `experiment` recipe to justfile
- [ ] Integrate preflight into train_monitored.py startup

### Step 3 (Medium-term — enables cloud execution)
- [ ] Create `deployment/Dockerfile.train` (uv-based, not pip)
- [ ] Add Docker resource limits to docker-compose.yml
- [ ] Add service healthchecks for MLflow, BentoML, Langfuse
- [ ] Add `experiment-docker` recipe to justfile

### Step 4 (Longer-term — enables CI/CD experiments)
- [ ] Create `.github/workflows/experiment.yml`
- [ ] Add CML integration for PR-level experiment comparison
- [ ] Add experiment result caching (skip re-runs for identical configs + data)

## Design Decisions

### D1: justfile over Makefile
**Decision**: Keep justfile (already established in project)
**Rationale**: justfile has cleaner syntax, better argument handling, and is already the project standard. Makefile is more portable but minivess already chose justfile. No migration needed.

### D2: YAML experiment configs over CLI flags
**Decision**: YAML files in `configs/experiments/`
**Rationale**: CLI flags don't compose well (long commands, easy to typo), aren't version-controlled by default, and can't capture the full experiment intent. YAML is human-readable, diffable, and can be reviewed in PRs. The experiment runner translates YAML → CLI flags for train_monitored.py.

### D3: Python preflight over shell scripts
**Decision**: `scripts/preflight.py` (Python, not bash)
**Rationale**: Python has better cross-platform support, can import project modules for config validation, integrates naturally with the existing Python toolchain, and can use the same Pydantic models for config validation. Shell scripts are fragile for complex hardware detection.

### D4: Auto-detect with manual override
**Decision**: `compute.profile: auto` as default, explicit profiles as override
**Rationale**: Auto-detection removes the most common manual step (choosing the right profile) while still allowing power users to force a specific configuration. The `--manual` flag pattern the user mentioned maps to explicit profile selection in the YAML config.

### D5: Process RSS for memory management (not system-wide RAM)
**Decision**: Monitor and abort based on process RSS, not system-wide memory
**Rationale**: System-wide RAM metrics are misleading when swap is stale from previous OOM crashes (the bug we hit). Process RSS accurately reflects the training process's actual memory footprint. This is especially important on shared machines and cloud instances with other services running.

### D6: Non-sudo fallback for swap
**Decision**: Warn about stale swap but never require sudo
**Rationale**: Requiring sudo breaks: (a) Docker containers (no sudo), (b) cloud instances (may not have root), (c) CI runners (definitely no root), (d) shared servers (shouldn't have root). The preflight system warns the user and suggests the fix, but the training pipeline works correctly regardless because it uses process RSS-based memory management.

## Success Criteria

1. **Zero manual steps**: `just experiment --config X.yaml` runs end-to-end without user intervention
2. **Environment agnostic**: Same command works on local workstation, Docker, and cloud
3. **Crash resilient**: Automatic resume from last checkpoint on re-run
4. **Self-diagnosing**: Preflight catches 95% of issues before training starts
5. **Reproducible**: Experiment YAML + git SHA + DVC hash = identical results
6. **Observable**: System metrics, training metrics, and experiment results all captured automatically
