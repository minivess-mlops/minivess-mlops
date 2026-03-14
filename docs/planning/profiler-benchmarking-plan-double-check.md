# Profiler & Benchmarking Plan Double-Check: Infrastructure Timing + Cost Analysis

**Status**: DRAFT
**Date**: 2026-03-14
**Parent Plan**: [profiler-benchmarking-plan.md](./profiler-benchmarking-plan.md)
**Execution Plan**: [profiler-benchmarking-plan-execution.xml](./profiler-benchmarking-plan-execution.xml)
**Branch**: `feat/skypilot-runpod-gpu-offloading`

---

## 1. Motivation: What the Existing Plan Misses

The existing profiler-benchmarking plan (#643, 8 sub-issues) focuses on **training loop
profiling** — per-epoch CUDA kernel timing, memory allocation tracking, WeightWatcher
spectral diagnostics, and GPU benchmark caching. These are essential for understanding
*where time goes inside training*.

However, the plan has **zero coverage** of the infrastructure stages that dominate
wall-clock time and actual cost for cloud GPU runs:

```
TIMELINE OF A CLOUD GPU RUN (actual, measured 2026-03-14):

 T=0          T=2m           T=10m        T=12m       T=13m      T=13m15s
  |            |              |            |            |           |
  ├── LAUNCH ──├── SETUP ─────├── DVC ─────├── VERIFY ──├── TRAIN ──┤
  │  SkyPilot  │  apt-get     │  dvc pull  │  nvidia    │  2 epoch  │
  │  provision │  deadsnakes  │  from      │  -smi,     │  DynUNet  │
  │  + rsync   │  + uv +      │  UpCloud   │  torch     │  smoke    │
  │  workdir   │  uv sync     │  S3        │  imports   │  test     │
  │            │              │            │            │           │
  └── ~2 min ──┘── ~8 min ────┘── ~2 min ──┘── ~1 min ──┘── 1m15s ──┘

  BILLING STARTS HERE ─────────────────────────────────────────────►
  (pod active from T=0, but GPU is IDLE until T=12m)
```

**The training was 1m15s. The infrastructure overhead was ~12 minutes.**
At $0.40/hr (A40 on-demand), the 2-epoch smoke test cost ~$0.09 total, but only
~$0.008 was effective GPU work. The **effective GPU cost was 11x the hourly rate.**

For a 50-epoch DynUNet run (~25 minutes), the setup amortizes to ~48% overhead.
For 100 epochs (~50 minutes), it amortizes to ~24%. But we don't KNOW these numbers
because **nothing is timed**.

### 1.1 The Gap

| What | Existing Plan | This Double-Check |
|------|--------------|-------------------|
| Per-epoch CUDA profiling | YES (torch.profiler) | - |
| Memory allocation tracking | YES (profile_memory) | - |
| GPU benchmark cache | YES (TFLOPS, bandwidth) | - |
| WeightWatcher diagnostics | YES (alpha, stable_rank) | - |
| Pre-training sanity checks | YES (shape, gradient, NaN) | - |
| **Instance provisioning time** | NO | **YES** |
| **Docker image pull time** | NO | **YES** |
| **Dependency installation time** | NO | **YES** |
| **DVC data pull time** | NO | **YES** |
| **Model weight download time** | NO | **YES** |
| **First-epoch CUDA JIT time** | NO | **YES** |
| **Setup phase total time** | NO | **YES** |
| **Hourly instance cost** | NO | **YES** |
| **Actual incurred cost** | NO | **YES** |
| **Effective GPU cost analysis** | NO | **YES** |
| **Amortization break-even** | NO | **YES** |
| **HPO single vs multi-instance** | NO | **YES** |

---

## 2. Infrastructure Timing Requirements

Every major operation that constitutes a bottleneck MUST be timed by default. Timing
data is logged as **MLflow params** (one-time values) or **MLflow metrics** (per-epoch
values), and summarized in a structured **JSONL artifact** for programmatic analysis.

### 2.1 Timing Taxonomy

All infrastructure timing params use the `setup_` prefix to distinguish from training
profiling (`prof_`) and diagnostics (`diag_`):

```
Metric Prefix Taxonomy (extended):
  setup_       — infrastructure timing (new — this document)
  cost_        — cost tracking and analysis (new — this document)
  prof_        — torch.profiler metrics (existing plan)
  diag_ww_     — WeightWatcher diagnostics (existing plan)
  diag_        — pre-training checks (existing plan)
  sys_bench_   — GPU benchmark cache (existing plan)
  sys_gpu_     — per-epoch GPU utilization (existing system_monitor.py)
```

### 2.2 Setup Phase Timing (MLflow Params)

These are logged ONCE per MLflow run as params (not metrics, since they occur once):

| Param Name | Unit | Description | Where Measured |
|-----------|------|-------------|----------------|
| `setup_total_seconds` | float | Total setup phase wall time | `setup:` block wrapper |
| `setup_python_install_seconds` | float | Python 3.13 install (deadsnakes PPA) | `setup:` Step 1 |
| `setup_uv_install_seconds` | float | uv package manager install | `setup:` Step 2 |
| `setup_uv_sync_seconds` | float | `uv sync --all-extras` or `uv sync` | `setup:` Step 3 |
| `setup_dvc_config_seconds` | float | DVC init + remote config | `setup:` Step 4a |
| `setup_dvc_pull_seconds` | float | `dvc pull -r upcloud` data download | `setup:` Step 4b |
| `setup_dvc_pull_bytes` | int | Bytes downloaded by DVC | `setup:` Step 4b |
| `setup_model_weights_seconds` | float | HuggingFace model weight download | `setup:` Step 7 |
| `setup_model_weights_bytes` | int | Bytes downloaded for model weights | `setup:` Step 7 |
| `setup_workdir_sync_seconds` | float | SkyPilot rsync workdir upload | Launcher script (if measurable) |
| `setup_workdir_sync_bytes` | int | Bytes transferred by rsync | Launcher script |
| `setup_verification_seconds` | float | Pre-flight checks (nvidia-smi, torch import, MONAI import) | `setup:` Step 8 |
| `setup_docker_pull_seconds` | float | Docker image pull time (staging/prod only, N/A for dev) | SkyPilot internal |
| `setup_cache_status` | str | "cold" / "warm_venv" / "warm_all" — cache state | Heuristic |

Note: First/steady epoch timing uses `prof_` prefix (training profiling, not setup):

| Param Name | Unit | Description | Where Measured |
|-----------|------|-------------|----------------|
| `prof_first_epoch_seconds` | float | First training epoch (includes CUDA JIT) | `trainer.py` epoch 0 |
| `prof_steady_epoch_seconds` | float | Second training epoch (steady-state) | `trainer.py` epoch 1 |

### 2.3 Instance Metadata (MLflow Params)

| Param Name | Unit | Description | Source |
|-----------|------|-------------|--------|
| `cost_instance_hourly_usd` | float | Hourly cost of instance (on-demand or spot) | SkyPilot / env var |
| `cost_instance_sku` | str | Cloud instance SKU (e.g., gpu_1x_a4000) | SkyPilot |
| `cost_instance_cloud` | str | Cloud provider (runpod, lambda, aws, gcp) | SkyPilot |
| `cost_instance_region` | str | Cloud region | SkyPilot |
| `cost_instance_spot` | bool | Whether spot/preemptible instance | SkyPilot |
| `cost_instance_disk_gb` | int | Disk size provisioned | SkyPilot YAML |

### 2.4 Computed Cost Analysis (MLflow Params, logged post-training)

| Param Name | Unit | Description | Computation |
|-----------|------|-------------|-------------|
| `cost_total_wall_seconds` | float | Total run time (setup + training) | End - Start |
| `cost_total_usd` | float | Total incurred cost | `wall_seconds / 3600 * hourly_rate` |
| `cost_setup_usd` | float | Cost of setup phase alone | `setup_seconds / 3600 * hourly_rate` |
| `cost_training_usd` | float | Cost of training phase alone | `training_seconds / 3600 * hourly_rate` |
| `cost_effective_gpu_rate` | float | Effective $/hr for GPU work only | `total_cost / (training_seconds / 3600)` |
| `cost_setup_fraction` | float | Fraction of cost spent on setup | `setup_cost / total_cost` |
| `cost_gpu_utilization_fraction` | float | Fraction of wall time spent on GPU | `training_seconds / wall_seconds` |
| `cost_epochs_to_amortize_setup` | int | Epochs needed for setup < 10% of total cost | Computed |
| `cost_break_even_epochs` | int | Epochs where effective rate < 2x hourly rate | Computed |

### 2.5 Infrastructure Timing Artifact (JSONL)

In addition to individual MLflow params, a structured JSONL artifact is logged at
`timing/infrastructure_timing.jsonl` containing every timed operation as a record:

```json
{"phase": "setup", "operation": "python_install", "start_utc": "2026-03-14T10:00:00Z", "end_utc": "2026-03-14T10:00:35Z", "duration_seconds": 35.2, "bytes_transferred": null, "notes": "deadsnakes PPA, Python 3.13.12"}
{"phase": "setup", "operation": "uv_install", "start_utc": "2026-03-14T10:00:35Z", "end_utc": "2026-03-14T10:00:42Z", "duration_seconds": 7.1, "bytes_transferred": null, "notes": "uv 0.10.10"}
{"phase": "setup", "operation": "uv_sync", "start_utc": "2026-03-14T10:00:42Z", "end_utc": "2026-03-14T10:05:30Z", "duration_seconds": 288.0, "bytes_transferred": null, "notes": "--all-extras, 450 packages"}
{"phase": "setup", "operation": "dvc_pull", "start_utc": "2026-03-14T10:05:35Z", "end_utc": "2026-03-14T10:07:20Z", "duration_seconds": 105.0, "bytes_transferred": 2749883382, "notes": "upcloud S3, 350 files"}
{"phase": "setup", "operation": "model_weights", "start_utc": "2026-03-14T10:07:20Z", "end_utc": "2026-03-14T10:07:25Z", "duration_seconds": 5.0, "bytes_transferred": 0, "notes": "dynunet, no HF download needed"}
{"phase": "training", "operation": "epoch_0", "start_utc": "2026-03-14T10:08:00Z", "end_utc": "2026-03-14T10:08:45Z", "duration_seconds": 45.0, "bytes_transferred": null, "notes": "includes CUDA JIT compilation"}
{"phase": "training", "operation": "epoch_1", "start_utc": "2026-03-14T10:08:45Z", "end_utc": "2026-03-14T10:09:20Z", "duration_seconds": 35.0, "bytes_transferred": null, "notes": "steady-state"}
{"phase": "cost", "operation": "summary", "start_utc": "2026-03-14T10:00:00Z", "end_utc": "2026-03-14T10:09:20Z", "duration_seconds": 560.0, "hourly_rate_usd": 0.40, "total_cost_usd": 0.062, "setup_cost_usd": 0.053, "training_cost_usd": 0.009, "effective_gpu_rate_usd": 2.79, "setup_fraction": 0.856}
```

This JSONL format enables:
- DuckDB `read_json_auto('timing/*.jsonl')` for SQL analytics
- Grafana JSON data source for dashboard panels
- Cross-run comparison via MLflow artifact search

---

## 3. Implementation: How to Time Each Phase

### 3.1 SkyPilot YAML `setup:` Block (Shell Timing)

The `setup:` block in SkyPilot YAML runs as a bash script. Timing is captured via
shell builtins and written to a JSON file that the Python `run:` block reads:

```bash
setup: |
  set -ex
  # Key=value timestamps — parsed by Python in run: block (no bc, no JSON escaping)
  TIMING_FILE="${HOME}/sky_workdir/timing_setup.txt"
  echo "setup_start=$(date +%s.%N)" > "$TIMING_FILE"

  # Step 1: Python install
  echo "python_install_start=$(date +%s.%N)" >> "$TIMING_FILE"
  if ! python3.13 --version 2>/dev/null; then
    apt-get update -qq && apt-get install -y -qq software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update -qq
    apt-get install -y -qq python3.13 python3.13-venv python3.13-dev
  fi
  echo "python_install_end=$(date +%s.%N)" >> "$TIMING_FILE"

  # Step 2: uv install
  echo "uv_install_start=$(date +%s.%N)" >> "$TIMING_FILE"
  if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  echo "uv_install_end=$(date +%s.%N)" >> "$TIMING_FILE"

  # Step 3: uv sync (includes venv creation)
  echo "uv_sync_start=$(date +%s.%N)" >> "$TIMING_FILE"
  uv venv --python python3.13 .venv && uv sync --all-extras
  echo "uv_sync_end=$(date +%s.%N)" >> "$TIMING_FILE"

  # Step 4: DVC config + pull
  echo "dvc_config_start=$(date +%s.%N)" >> "$TIMING_FILE"
  dvc init --no-scm -f && dvc remote add ... && dvc remote modify ...
  echo "dvc_config_end=$(date +%s.%N)" >> "$TIMING_FILE"

  echo "dvc_pull_start=$(date +%s.%N)" >> "$TIMING_FILE"
  dvc pull -r upcloud
  echo "dvc_pull_end=$(date +%s.%N)" >> "$TIMING_FILE"

  # Step 7: Model weights
  echo "model_weights_start=$(date +%s.%N)" >> "$TIMING_FILE"
  # ... model-specific pre-cache
  echo "model_weights_end=$(date +%s.%N)" >> "$TIMING_FILE"

  # Step 8: Verification
  echo "verification_start=$(date +%s.%N)" >> "$TIMING_FILE"
  nvidia-smi && python -c "import torch; ..." && python -c "import monai; ..."
  echo "verification_end=$(date +%s.%N)" >> "$TIMING_FILE"

  echo "setup_end=$(date +%s.%N)" >> "$TIMING_FILE"
```

The Python parser in `run:` reads this with `str.split("=")` and computes durations:

```python
def parse_setup_timing(timing_file: Path) -> dict[str, float]:
    timestamps: dict[str, float] = {}
    for line in timing_file.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        timestamps[key.strip()] = float(value.strip())
    # Compute durations
    durations = {}
    for op in ["python_install", "uv_install", "uv_sync", "dvc_config", "dvc_pull",
               "model_weights", "verification"]:
        start_key, end_key = f"{op}_start", f"{op}_end"
        if start_key in timestamps and end_key in timestamps:
            durations[op] = timestamps[end_key] - timestamps[start_key]
    durations["setup_total"] = timestamps.get("setup_end", 0) - timestamps.get("setup_start", 0)
    return durations
```

### 3.2 Python `run:` Block (MLflow Integration)

The `run:` block reads `timing_setup.json` and logs all values to MLflow:

```python
# In train_flow.py or a new timing utility
import json
from pathlib import Path

def log_infrastructure_timing(tracker: ExperimentTracker) -> None:
    timing_file = Path.cwd() / "timing_setup.json"
    if timing_file.exists():
        data = json.loads(timing_file.read_text(encoding="utf-8"))
        for entry in data["timings"]:
            tracker.log_param(f"setup_{entry['op']}_seconds", entry["seconds"])
        # Log artifact
        tracker.log_artifact(str(timing_file), "timing")
```

### 3.3 Cost Computation (Post-Training)

```python
def compute_and_log_cost_analysis(
    tracker: ExperimentTracker,
    setup_seconds: float,
    training_seconds: float,
    hourly_rate_usd: float,
    epoch_count: int,
) -> None:
    total_seconds = setup_seconds + training_seconds
    total_cost = total_seconds / 3600 * hourly_rate_usd
    setup_cost = setup_seconds / 3600 * hourly_rate_usd
    training_cost = training_seconds / 3600 * hourly_rate_usd

    effective_rate = total_cost / (training_seconds / 3600) if training_seconds > 0 else float("inf")
    setup_fraction = setup_cost / total_cost if total_cost > 0 else 0.0

    # Epochs needed to amortize setup to < 10% of total
    if epoch_count > 0:
        seconds_per_epoch = training_seconds / epoch_count
        # setup_cost / (setup_cost + N * epoch_cost) < 0.10
        # => N > 9 * setup_seconds / seconds_per_epoch
        epochs_to_amortize = int(9 * setup_seconds / seconds_per_epoch) + 1 if seconds_per_epoch > 0 else 0
    else:
        epochs_to_amortize = 0

    # Epochs where effective rate < 2x hourly rate
    # total_cost / (training_hours) < 2 * hourly_rate
    # => (setup + training) / training < 2
    # => setup < training
    # => N > setup_seconds / seconds_per_epoch
    break_even = int(setup_seconds / seconds_per_epoch) + 1 if seconds_per_epoch > 0 else 0

    tracker.log_param("cost_total_wall_seconds", round(total_seconds, 1))
    tracker.log_param("cost_total_usd", round(total_cost, 4))
    tracker.log_param("cost_setup_usd", round(setup_cost, 4))
    tracker.log_param("cost_training_usd", round(training_cost, 4))
    tracker.log_param("cost_effective_gpu_rate", round(effective_rate, 4))
    tracker.log_param("cost_setup_fraction", round(setup_fraction, 4))
    tracker.log_param("cost_gpu_utilization_fraction", round(training_seconds / total_seconds, 4))
    tracker.log_param("cost_epochs_to_amortize_setup", epochs_to_amortize)
    tracker.log_param("cost_break_even_epochs", break_even)
```

### 3.4 Hourly Rate Source

The hourly rate must be injected, not guessed:

| Environment | Source | Mechanism |
|-------------|--------|-----------|
| Dev (RunPod) | SkyPilot `sky status --cloud runpod` | Launcher script writes `INSTANCE_HOURLY_USD` env var |
| Staging/Prod (Lambda) | SkyPilot provision response | Same mechanism |
| Local | N/A (free) | `INSTANCE_HOURLY_USD=0.0` |
| Docker (local) | N/A (electricity only) | `INSTANCE_HOURLY_USD=0.0` |

The launcher script (`scripts/launch_dev_runpod.py`, `scripts/launch_smoke_test.py`)
already provisions via SkyPilot Python API, which returns the instance price. This
should be passed as an env var to the run:

```python
# In launcher script, after sky.launch():
# Get instance cost from SkyPilot status
status = sky.status(cluster_names=["minivess-dev"])
hourly_cost = status[0].get("hourly_cost", 0.0)
# Pass via env var
envs["INSTANCE_HOURLY_USD"] = str(hourly_cost)
```

---

## 4. Cost Analysis: Key Questions to Answer

### 4.1 Setup Amortization Analysis

**Question**: How many epochs must we run to make setup time insignificant?

| Model | Setup (est.) | Epoch Time | Epochs for <10% setup | Epochs for <5% setup |
|-------|-------------|------------|----------------------|---------------------|
| DynUNet (A4000) | ~10 min (600s) | ~30 sec | 180 | 380 |
| DynUNet (RTX 4090) | ~10 min (600s) | ~15 sec | 360 | 760 |
| SAM3 Vanilla (A40) | ~12 min (720s) | ~90 sec | 72 | 152 |
| SAM3 Hybrid (A40) | ~12 min (720s) | ~120 sec | 54 | 114 |
| VesselFM (A100) | ~15 min (900s) | ~180 sec | 45 | 95 |

**Derivation**: `N > 9 * setup_seconds / seconds_per_epoch` for <10% setup fraction,
`N > 19 * setup_seconds / seconds_per_epoch` for <5% setup fraction. Derivation:
`setup / (setup + N * epoch) < 0.10` ⟹ `9 * setup < N * epoch` ⟹ `N > 9 * setup / epoch`.

**Insight**: Faster GPUs have HIGHER amortization breakpoints because per-epoch time
drops but setup remains constant. An A100 doing 5-second DynUNet epochs needs 1080
epochs to amortize 10 minutes of setup to <10%! This means short smoke tests will
always have high effective GPU costs, and that is acceptable — smoke tests are for
correctness, not cost efficiency. Real training runs (50-200 epochs) amortize well.

### 4.2 Single-Instance vs Multi-Instance HPO

**Question**: Should we run HPO trials on one persistent instance or spin up one per trial?

| Strategy | Pros | Cons |
|----------|------|------|
| **Single instance, sequential trials** | 1x setup cost; warm caches; deterministic | GPU idle during model init; no parallelism |
| **Multi-instance, parallel trials** | Nx speedup; fault isolation | Nx setup cost; DVC pull per instance |
| **Single instance, multi-GPU** | 1x setup; GPU parallelism | Requires multi-GPU instance (expensive); complex scheduling |
| **Hybrid: single instance, Optuna worker pool** | 1x setup; parallel trials within instance | GPU memory contention; complex worker management |

**Analysis** (assuming 20-trial HPO, 50 epochs each, DynUNet on A4000 at $0.16/hr):

- Training per trial: 50 epochs * 30s = 25 min
- Setup per instance: 10 min (one-time)
- 20 trials sequential: 20 * 25 min = 500 min training + 10 min setup = 510 min = 8h 30m

| Strategy | Instances | Setup Cost | Instance-Hours | Total Cost | Wall Time |
|----------|-----------|-----------|----------------|-----------|-----------|
| Sequential (1 inst) | 1 | $0.027 | 8.5h * 1 = 8.5h | $1.36 | 8h 30m |
| 5 parallel | 5 | $0.133 | 1.77h * 5 = 8.83h | $1.41 | 1h 46m |
| 10 parallel | 10 | $0.267 | 0.93h * 10 = 9.33h | $1.49 | 56m |
| 20 parallel | 20 | $0.533 | 0.58h * 20 = 11.67h | $1.87 | 35m |

Note: Total cost = sum of (per-instance wall time * hourly_rate) across all instances.
Each instance pays its own setup cost. More instances = more total instance-hours
(because each pays setup) but less wall time.

**Conclusion**: For A4000 at $0.16/hr, the cost difference between sequential and
20-parallel is only $0.51 (~37%) while wall time drops from 8.5 hours to 35 minutes.
**Parallelism is almost free for cheap GPUs.** Each additional instance adds only
$0.027 in setup overhead ($0.16/hr * 10min/60). For expensive GPUs (A100 at $1.10/hr),
setup cost per instance is $0.18, still modest but adds up: 20 parallel A100s spend
$3.67 on setup alone vs $0.18 for sequential.

### 4.3 Docker Image Pull vs Bare-Metal Setup

**Question**: Is Docker image pull or `uv sync` faster?

| Approach | Time (cold) | Time (warm) | Notes |
|----------|-------------|-------------|-------|
| Docker pull (21.4 GB GHCR) | 20-60 min | ~0 (cached) | Depends on network; RunPod = no docker-in-docker |
| `uv sync --all-extras` | ~5-8 min | ~0 (cached venv) | Depends on pip resolver; consistent |
| SkyPilot default image | ~0 (pre-cached) | ~0 | Always warm on RunPod/Lambda |

**Conclusion for Dev**: `uv sync` on SkyPilot default image is 4-10x faster than
Docker pull on first run. On subsequent runs (warm venv via SkyPilot storage mount),
both are near-instant.

### 4.4 Effective GPU Cost

**Question**: What is our effective GPU cost for different run lengths?

Given:
- A40 on-demand: $0.40/hr
- Setup time: 10 minutes (fixed)
- Epoch time: 35 seconds (DynUNet, 4-volume smoke test)

| Epochs | Training Time | Total Time | Effective $/hr | Overhead |
|--------|--------------|-----------|----------------|----------|
| 2 | 1m 10s | 11m 10s | $3.83 | 9.6x |
| 5 | 2m 55s | 12m 55s | $1.78 | 4.4x |
| 10 | 5m 50s | 15m 50s | $1.09 | 2.7x |
| 25 | 14m 35s | 24m 35s | $0.67 | 1.7x |
| 50 | 29m 10s | 39m 10s | $0.54 | 1.3x |
| 100 | 58m 20s | 68m 20s | $0.47 | 1.2x |
| 200 | 116m 40s | 126m 40s | $0.43 | 1.1x |

**Insight**: The break-even point (effective rate < 2x hourly) is ~17 epochs.
For smoke tests (2 epochs), the effective cost is **9.6x** the hourly rate.
This is expected and acceptable — smoke tests are for correctness, not cost efficiency.

---

## 5. Implementation Phases

### Phase A: Shell Timing in SkyPilot YAML (This Issue)

**Scope**: Add `date +%s.%N` timing to every major step in `dev_runpod.yaml` and
`smoke_test_gpu.yaml` setup blocks. Write timing data to JSON file.

**Files to modify**:
- `deployment/skypilot/dev_runpod.yaml` — setup block timing
- `deployment/skypilot/smoke_test_gpu.yaml` — setup block timing

**Effort**: 1 hour

### Phase B: Python Timing Integration

**Scope**: Read timing JSON in `train_flow.py`, log to MLflow. Add cost computation
utility. Add `INSTANCE_HOURLY_USD` env var injection in launcher scripts.

**Files to modify/create**:
- `src/minivess/observability/infrastructure_timing.py` — new module
- `src/minivess/orchestration/flows/train_flow.py` — call timing logger
- `scripts/launch_dev_runpod.py` — inject hourly cost env var
- `scripts/launch_smoke_test.py` — inject hourly cost env var

**Effort**: 3 hours

### Phase C: DuckDB + Grafana Analysis Pipeline

**Scope**: Extend DuckDB extraction to include `setup_*` and `cost_*` params.
Create Grafana dashboard panels for cost analysis. Add cost analysis to Dashboard Flow.

**Files to modify/create**:
- `src/minivess/pipeline/duckdb_extraction.py` — extract timing/cost params
- Dashboard Flow adapter for cost panel generation
- Grafana JSON dashboard model for timing/cost panels

**Effort**: 4 hours

### Phase D: Cross-Run Analysis Reports

**Scope**: Generate automated reports comparing infrastructure timing across:
- Cloud providers (RunPod vs Lambda vs AWS)
- GPU types (A4000 vs RTX 4090 vs A40 vs A100)
- Models (DynUNet vs SAM3 vs VesselFM)
- Run configurations (smoke test vs full training)

Output: Markdown report + SVG charts logged as MLflow artifacts.

**Effort**: 4 hours

---

## 6. Checklist: All Major Operations That MUST Be Timed

### 6.1 Pre-Training (Infrastructure)

- [ ] **Instance provisioning** — SkyPilot `sky.launch()` → SSH available (measured in launcher)
- [ ] **Docker image pull** — For staging/prod with `image_id: docker:...` (SkyPilot internal)
- [ ] **Workdir rsync** — SkyPilot rsync of project files to remote (measured in launcher)
- [ ] **apt-get + PPA** — deadsnakes PPA add + apt-get update (included in Python install)
- [ ] **Python install** — Python 3.13 via deadsnakes (first run only)
- [ ] **uv venv + install** — `uv venv` + uv package manager installation (first run only)
- [ ] **uv sync** — `uv sync --all-extras` dependency installation
- [ ] **DVC config** — `dvc init --no-scm -f` + `dvc remote add` + 3x `dvc remote modify`
- [ ] **DVC pipeline files** — `cat > dvc.yaml` + `cat > dvc.lock` (instant, but timed)
- [ ] **DVC pull** — `dvc pull -r upcloud` data download (with bytes transferred + throughput)
- [ ] **Splits copy** — `cp configs/splits/*.json` (instant, but confirms data path)
- [ ] **Directory creation** — `mkdir -p checkpoints logs` (instant, but timed)
- [ ] **Model weight download** — HuggingFace `hf_hub_download` or `from_pretrained` (log model name)
- [ ] **Verification/pre-flight** — `nvidia-smi`, `python -c "import torch"`, MONAI import, file checks
- [ ] **CUDA warmup** — First `torch.cuda.is_available()` + first tensor allocation
- [ ] **Model instantiation** — `build_model()` call (weight loading + GPU transfer)
- [ ] **DataLoader initialization** — First `CacheDataset` construction + caching
- [ ] **Hydra config composition** — `compose_experiment_config()` resolution time
- [ ] **MLflow first connection** — Initial HTTP connection to remote MLflow server

### 6.2 Training Phase

- [ ] **First epoch** — Includes CUDA JIT compilation, cuDNN benchmarking
- [ ] **Steady-state epoch** — Second epoch onward (representative timing)
- [ ] **Validation epoch** — Full validation pass (may include sliding_window_inference)
- [ ] **Per-epoch data loading time** — DataLoader I/O wait time (existing `prof_` plan)
- [ ] **Per-epoch forward pass** — Forward pass GPU time (existing `prof_` plan)
- [ ] **Per-epoch backward pass** — Backward + optimizer step (existing `prof_` plan)
- [ ] **Per-epoch memory stats** — Peak allocated/reserved CUDA memory (existing `prof_` plan)
- [ ] **Checkpoint save** — `torch.save()` for best/last/periodic checkpoints
- [ ] **MLflow logging overhead** — Time spent in MLflow API calls per epoch

### 6.3 Post-Training

- [ ] **WeightWatcher analysis** — Spectral diagnostics (existing plan)
- [ ] **Cleanlab label quality** — Label issue detection (existing plan)
- [ ] **ONNX export** — Model export time
- [ ] **Checkpoint upload to MLflow** — Network transfer of checkpoints to remote MLflow
- [ ] **Total run time** — Wall clock start to finish
- [ ] **Cost computation** — All cost_ metrics (hourly rate, total, effective)
- [ ] **Run status** — `cost_run_status` tag: completed / failed / preempted

### 6.4 Infrastructure Metadata (Not Timed, But Logged)

- [ ] **Cloud provider** — runpod, lambda, aws, gcp, local
- [ ] **Instance SKU** — Cloud instance type (e.g., gpu_1x_a4000)
- [ ] **GPU model** — Already logged as `sys_gpu_model`; no duplication
- [ ] **Region** — Cloud region
- [ ] **Spot/on-demand** — Pricing tier
- [ ] **Billing granularity** — per_second (RunPod) / per_hour (AWS) / per_minute (Lambda)
- [ ] **Disk size** — Provisioned disk GB
- [ ] **Network bandwidth** — DVC pull throughput (bytes/second)
- [ ] **DVC endpoint region** — UpCloud S3 endpoint for throughput analysis
- [ ] **Cache status** — cold / warm_venv / warm_all
- [ ] **Launch mode** — sky_launch / sky_exec / sky_jobs_launch
- [ ] **Idle auto-stop minutes** — From SkyPilot YAML config

---

## 7. Grafana Dashboard Design

### 7.1 Cost Analysis Panels

| Panel | Type | Data Source | Query |
|-------|------|-----------|-------|
| Setup vs Training Cost | Stacked bar | DuckDB/Parquet | `SELECT run_id, cost_setup_usd, cost_training_usd FROM runs` |
| Effective GPU Rate Over Time | Time series | DuckDB/Parquet | `SELECT start_time, cost_effective_gpu_rate FROM runs ORDER BY start_time` |
| Setup Amortization Curve | Line chart | DuckDB/Parquet | `SELECT epoch_count, cost_setup_fraction FROM runs` |
| Cloud Provider Cost Comparison | Grouped bar | DuckDB/Parquet | `SELECT cost_instance_cloud, AVG(cost_effective_gpu_rate) FROM runs GROUP BY 1` |
| Hourly Rate vs Effective Rate | Scatter | DuckDB/Parquet | `SELECT cost_instance_hourly_usd, cost_effective_gpu_rate, cost_instance_sku FROM runs` |

### 7.2 Infrastructure Timing Panels

| Panel | Type | Data Source | Query |
|-------|------|-----------|-------|
| Setup Phase Breakdown | Stacked bar | DuckDB/Parquet | `SELECT run_id, setup_python_install_seconds, setup_uv_sync_seconds, setup_dvc_pull_seconds FROM runs` |
| Setup Time Trend | Time series | DuckDB/Parquet | `SELECT start_time, setup_total_seconds FROM runs ORDER BY start_time` |
| DVC Pull Speed | Time series | DuckDB/Parquet | `SELECT start_time, setup_dvc_pull_bytes / setup_dvc_pull_seconds AS bytes_per_sec FROM runs` |
| First vs Steady Epoch | Grouped bar | DuckDB/Parquet | `SELECT model, prof_first_epoch_seconds, prof_steady_epoch_seconds FROM runs` |

---

## 8. Relationship to Existing Plan

This document is **additive** to the existing profiler-benchmarking plan. It does NOT
replace any of the 8 sub-issues (#644-#651). Instead, it adds a new Phase (A/B/C/D)
focused on infrastructure timing and cost analysis.

```
EXISTING PLAN:
  Phase 0: ProfilingConfig + Hydra YAML (#644)
  Phase 1: PyTorch Profiler Core (#645, #646, #647)
  Phase 2: Pre/Post Training Diagnostics (#648, #649)
  Phase 3: GPU Benchmark Container (#650, #651)

THIS DOUBLE-CHECK (NEW):
  Phase A: Shell timing in SkyPilot YAML
  Phase B: Python timing integration + MLflow logging
  Phase C: DuckDB + Grafana cost analysis pipeline
  Phase D: Cross-run analysis reports

DEPENDENCY: Phase B depends on Phase 0 (ProfilingConfig exists).
            Phase C depends on Phase B (timing data available).
            Phase D depends on Phase C (DuckDB pipeline exists).
```

---

## 9. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Shell timing `date +%s.%N` not available on all images | Timing fails silently | Fallback to `date +%s` (integer seconds); check in setup preamble |
| SkyPilot doesn't expose hourly cost in Python API | Cost analysis incomplete | Manual lookup table in `configs/cloud_pricing.yaml`; SkyPilot `sky show-gpus` for prices |
| JSONL artifact too large for many-epoch runs | MLflow storage bloat | Cap at 1000 entries; summarize in DuckDB |
| Timing overhead interferes with training | Biased measurements | Shell `date` calls are <1ms each; JSONL writes are <1ms; total overhead < 0.01% |
| Cross-cloud clock skew | UTC timestamps differ | Always use `datetime.now(timezone.utc)`; never rely on wall-clock comparison across hosts |
| Billing granularity varies | AWS rounds to 1 hour; RunPod bills per-second | Log `cost_billing_granularity` param; compute `cost_billing_rounded_usd` for actual charges |
| Spot preemption wastes setup cost | 12 min setup at $1.10/hr = $0.22 wasted per preemption | Track `cost_run_status` tag (completed/failed/preempted); compute `cost_wasted_usd` |
| Idle time between runs | 15 min auto-stop = $0.10 idle cost at A40 rates | Log `cost_idle_minutes` from SkyPilot config; include in total cost |
| Egress fees (AWS/GCP) | $0.09-0.12/GB outbound data | Not tracked initially; add when multi-cloud comparison is implemented |

---

## 10. Success Criteria

- [ ] Every `make dev-gpu` run produces a `timing/infrastructure_timing.jsonl` MLflow artifact
- [ ] Every run has `cost_total_usd` and `cost_effective_gpu_rate` params in MLflow
- [ ] DuckDB pipeline extracts all `setup_*` and `cost_*` params into queryable Parquet
- [ ] Grafana dashboard shows cost analysis panels (at least: effective GPU rate, setup breakdown)
- [ ] Dashboard Flow generates a cost summary table in its output report
- [ ] Cross-run comparison report can answer: "Is RunPod A4000 or Lambda A100 cheaper for 50-epoch DynUNet?"
- [ ] `INSTANCE_HOURLY_USD` added to `.env.example` (Rule #22 compliance)
- [ ] Failed/preempted runs tracked with `cost_wasted_usd` or `cost_run_status` tag

---

## 11. Reviewer Changelog (Round 1)

Three reviewer agents evaluated this plan. Below are the critical findings and resolutions.

### Reviewer 1: Infrastructure Timing

| ID | Severity | Finding | Resolution |
|----|----------|---------|-----------|
| M2 | HIGH | Docker image pull time for Lambda Labs not in timing taxonomy | ADDED: `setup_docker_pull_seconds` param |
| M3 | HIGH | SkyPilot provisioning API call time not in params | ADDED to checklist 6.1; measurement via launcher timestamps |
| M10 | MEDIUM | `uv venv --python python3.13` time not separate from `uv sync` | Covered by `setup_uv_sync_seconds` (includes venv creation) |
| M11 | MEDIUM | Verification/pre-flight check time not timed | ADDED: `setup_verification_seconds` param |
| M15 | MEDIUM | SAM3 vs VesselFM weight download size not distinguished | Log `setup_model_weights_model` string alongside seconds |
| M16 | MEDIUM | Checkpoint upload to remote MLflow not timed | ADDED to checklist 6.3 |
| M17 | MEDIUM | `sky exec` warm re-run not tagged | ADDED: `setup_cache_status` param (cold/warm_venv/warm_all) |
| M19 | MEDIUM | Workdir rsync measurement mechanism unspecified | Measure via launcher: timestamp before/after `sky.launch()` |
| M20 | LOW | DVC S3 endpoint region not logged | Log `setup_dvc_pull_endpoint` param |
| U1 | MEDIUM | Setup/training boundary unclear | `setup_` = everything before `train_flow.py` runs; `run:` preamble is setup |
| U3 | MEDIUM | First epoch per-fold ambiguity | Log `prof_first_epoch_seconds` for fold 0 only |
| U6 | MEDIUM | `sky launch` vs `sky jobs launch` billing difference | Note: `idle_minutes_to_autostop` adds to cost; track via `cost_idle_minutes` |
| U7 | LOW | Only 2/5 SkyPilot YAMLs get timing | Phase A covers dev + smoke; prod YAMLs added in Phase B |

### Reviewer 2: Cost Analysis

| ID | Severity | Finding | Resolution |
|----|----------|---------|-----------|
| E2 | **CRITICAL** | Section 4.1 amortization table values 10x too low | **FIXED**: DynUNet A4000 was 18, corrected to 180 |
| E5 | **HIGH** | Section 4.2 HPO cost table internally inconsistent | **FIXED**: Recomputed with correct total = hourly * wall_time * instances |
| I8 | MEDIUM | JSONL example effective_gpu_rate_usd wrong ($0.24 → $2.79) | **FIXED** |
| M1 | LOW | Egress fees not captured | Acknowledged; add note for AWS/GCP multi-cloud |
| M3 | MEDIUM | Spot preemption wasted-cost not modeled | Track via `cost_run_status` tag (completed/failed/preempted) |
| M5 | MEDIUM | Failed runs not tracked separately | ADDED to success criteria |
| I1 | LOW | Billing granularity (per-second vs per-hour) not documented | Add `cost_billing_granularity` param (per_second/per_hour) |
| I2 | MEDIUM | Warm vs cold runs not tagged | RESOLVED by `setup_cache_status` param |
| I7 | LOW | Amortization over RUNS (not epochs) for dev workflow | Deferred — useful but requires session-level tracking |

### Reviewer 3: MLflow Integration

| ID | Severity | Finding | Resolution |
|----|----------|---------|-----------|
| 1a | HIGH | `setup_first_epoch_seconds` straddles two domains | **FIXED**: Moved to `prof_first_epoch_seconds` |
| 1b | MEDIUM | `cost_instance_sku` overlaps `sys_gpu_model` | **FIXED**: Renamed to `cost_instance_sku` |
| 2a | MEDIUM | Measured values should be MLflow metrics, not params | ACCEPTED: Log `cost_total_usd` etc. as metrics at step=0 |
| 2c | HIGH | `INSTANCE_HOURLY_USD` not in `.env.example` (Rule #22) | ADDED to success criteria checklist |
| 2d | MEDIUM | JSONL schema inconsistent across record types | Cost summary → separate `timing/cost_summary.json` artifact |
| 2e | MEDIUM | Shell-generated JSON fragile (bc, escaping) | Use key=value timestamps in shell, parse in Python |
| 2f | HIGH | Grafana has no DuckDB datasource plugin | Use dashboard-api FastAPI endpoint (existing port 8090) |
| 4b | MEDIUM | Need `setup_cache_status` param | Already added (Reviewer 1 M17) |
| 4e | LOW | DuckDB excluded prefixes need `setup_`, `cost_` | Add to `_EXCLUDED_METRIC_PREFIXES` tuple |
| 4g | HIGH | `INSTANCE_HOURLY_USD` needs `.env.example` entry | Added |

### Design Decision: Shell Timing Format

Per Reviewer 3 (finding 2e), **do NOT generate JSON in bash**. Instead, write simple
key=value timestamps:

```bash
# In setup: block
echo "python_install_start=$(date +%s.%N)" >> "$TIMING_FILE"
# ... do work ...
echo "python_install_end=$(date +%s.%N)" >> "$TIMING_FILE"
```

Then parse in Python with `str.split("=")` — no `bc`, no JSON escaping, no comma handling.

### Design Decision: Grafana Integration Path

Grafana has no native DuckDB plugin. The integration path is:

1. Add `/api/v1/cost-analysis` endpoint to existing `dashboard-api` service (port 8090)
2. Query DuckDB from FastAPI, return JSON
3. Configure Grafana JSON API datasource pointing to `dashboard-api:8090`
4. Dashboard panels query the JSON API endpoint

This is consistent with the existing dashboard architecture.
