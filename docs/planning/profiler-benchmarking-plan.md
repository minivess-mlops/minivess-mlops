# Profiler and Benchmarking Plan for MinIVess MLOps

**Status**: DRAFT
**Issue**: [#564 — Dockerized GPU benchmark suite for external instances](https://github.com/petteriTeikari/minivess-mlops/issues/564)
**Date**: 2026-03-13
**Branch**: `feat/profiler-benchmarking`

---

## Abstract

This document presents an integrated profiling and benchmarking architecture for the
MinIVess MLOps platform. The system addresses three complementary problems: (1) per-epoch
computational profiling of the training loop to identify GPU utilization bottlenecks,
CPU-GPU data transfer inefficiencies, and memory fragmentation; (2) run-once GPU capability
benchmarking for ephemeral cloud instances (RunPod, AWS, GCP) to eliminate redundant
hardware probing; and (3) pre-training and post-training model diagnostics that extend
beyond loss curves to spectral analysis, label quality estimation, and structured sanity
checks. All profiling artifacts are logged to MLflow as first-class artifacts, enabling
retrospective comparison of identical models across heterogeneous compute environments
(local RTX 2070 Super 8 GB vs. cloud A100 40 GB vs. intranet multi-GPU). The design is
YAML-configurable via Hydra-zen, default-ON for all models, and compatible with the
existing Docker-per-flow isolation architecture. Implementation follows strict TDD
methodology across four phases totaling an estimated 14 GitHub issues.

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Background and Related Work](#2-background-and-related-work)
3. [Architectural Decision Matrix](#3-architectural-decision-matrix)
4. [System Architecture](#4-system-architecture)
5. [Implementation Plan](#5-implementation-plan)
6. [Hydra Configuration Schema](#6-hydra-configuration-schema)
7. [MLflow Artifact Taxonomy](#7-mlflow-artifact-taxonomy)
8. [Test Plan](#8-test-plan)
9. [GitHub Issue Breakdown](#9-github-issue-breakdown)
10. [Risk Analysis](#10-risk-analysis)
11. [References](#11-references)

---

## 1. Introduction and Motivation

### 1.1 The Problem

The MinIVess MLOps platform currently supports 13 model families across three compute
tiers (local 8 GB GPU, intranet multi-GPU, cloud spot instances). While the existing
`system_monitor.py` provides coarse-grained GPU utilization snapshots at 10-second
intervals via MLflow system metrics, this approach has three critical blind spots:

1. **No per-operation granularity.** We know that an epoch took 45 seconds but not whether
   the bottleneck was in the forward pass, the loss computation, gradient scaling, or data
   loading. The PyTorch Profiler provides operator-level CUDA kernel timing that exposes
   exactly which operation dominates wall time.

2. **No cloud vs. local validation.** Cloud A100 instances cost $1.10--$3.00/hour
   (spot pricing). Without measured throughput (images/sec, TFLOPS utilization), we cannot
   verify that a cloud GPU actually outperforms the local RTX 2070 Super for a given model
   and batch size. Anecdotal reports on community forums confirm that naive cloud migration
   frequently underperforms local hardware due to storage I/O, PCIe topology, and driver
   mismatches ([Reddit, 2023](https://www.reddit.com/r/CUDA/comments/180996l/what_is_your_experience_developing_on_a_cloud/)).

3. **No structured model diagnostics.** Loss curves alone cannot distinguish an undertrained
   model from an overfit model from a model with corrupted weights. Spectral analysis
   (WeightWatcher), label quality estimation (Cleanlab), and structured forward-pass
   sanity checks (torch-test patterns) provide orthogonal diagnostic signals that are
   currently absent from the pipeline.

### 1.2 Design Constraints

All solutions must satisfy the MinIVess MLOps constraints:

- **Docker-per-flow isolation**: Profiling runs inside the training container; no host
  process access is assumed.
- **Hydra-zen single source of truth**: All profiling configuration flows through
  `configs/base.yaml` and experiment overrides.
- **MLflow as inter-flow contract**: All profiling artifacts are logged to MLflow, not
  saved to local filesystem outside Docker volumes.
- **Default ON**: Profiling is enabled by default for every model, on every compute
  environment. The overhead budget is <=5% wall-time increase.
- **MONAI-first**: Use `monai.utils.profiling.PerfContext` where it fits; extend with
  PyTorch Profiler where MONAI lacks coverage.
- **No standalone scripts**: Profiling is integrated into the Prefect training flow,
  not a separate `python scripts/profile.py` command.

### 1.3 Scope

This plan covers:

- **In scope**: PyTorch Profiler integration into `trainer.py`, GPU benchmark container,
  WeightWatcher post-training diagnostics, Hydra config schema, MLflow artifact logging,
  TDD test specifications.
- **Out of scope**: NVIDIA Nsight Systems GUI-based interactive profiling (requires X11
  forwarding incompatible with headless Docker), JAX profiling (not applicable), real-time
  Grafana dashboards for profiling data (deferred to Dashboard Flow improvements).

---

## 2. Background and Related Work

### 2.1 PyTorch Profiler Architecture

The PyTorch Profiler (`torch.profiler`) is a context-manager-based tracing system that
wraps CUPTI (CUDA Profiling Tools Interface) for GPU kernel timing and the PyTorch
autograd dispatcher for CPU operator tracing. Key API components:

- **`torch.profiler.profile()`**: Context manager accepting `activities` (CPU, CUDA),
  `schedule` (wait/warmup/active/repeat cycle), `on_trace_ready` (callback for trace
  export), `profile_memory` (allocation tracking), `with_stack` (Python stack capture),
  `with_flops` (FLOP estimation), and `record_shapes` (tensor shape logging).

- **`torch.profiler.schedule()`**: Defines a cyclic profiling window:
  `skip_first` initial steps are ignored, then repeating cycles of `wait` (inactive) +
  `warmup` (tracing on, results discarded) + `active` (tracing on, results kept). The
  `repeat` parameter caps the number of cycles. This design amortizes profiler overhead
  across long training runs.

- **`torch.profiler.record_function()`**: Named context manager for labeling code regions
  (e.g., `"forward"`, `"loss"`, `"backward"`, `"data_loading"`).

- **`prof.key_averages()`**: Aggregates trace events by operator name. Supports sorting
  by `self_cuda_time_total`, `cpu_time_total`, `self_cpu_memory_usage`, and grouping by
  Python stack depth (`group_by_stack_n`).

- **`prof.export_chrome_trace()`**: Exports trace to Chrome `chrome://tracing` JSON format
  for visual timeline analysis.

The profiler's overhead when using `schedule()` with targeted active windows is documented
at "a few percent" for timeline tracing, increasing to ~10-15% with `profile_memory=True`
and `with_stack=True` ([Eunomia, 2025](https://eunomia.dev/blog/2025/04/21/gpu-profiling-under-the-hood-an-implementation-focused-survey-of-modern-accelerator-tracing-tools/)).

### 2.2 NVIDIA Profiling Ecosystem

NVIDIA provides two complementary profilers:

- **Nsight Systems (nsys)**: Low-overhead (~2-5%) system-wide timeline profiler. Captures
  CPU-GPU synchronization, CUDA API calls, kernel launches, memory transfers, and PCIe
  bandwidth. Ideal for identifying data loading stalls and kernel launch latency. Produces
  `.nsys-rep` reports viewable in the Nsight Systems GUI.

- **Nsight Compute (ncu)**: Kernel-level profiler that replays individual CUDA kernels
  multiple times (~46 passes) to collect hardware counter metrics (SM occupancy, memory
  bandwidth utilization, warp stall reasons). Significant overhead; suitable for targeted
  kernel optimization only.

For MinIVess, Nsight Systems is the relevant comparison to PyTorch Profiler. However,
`nsys` requires host-level privileges and produces binary `.nsys-rep` files that cannot
be directly logged to MLflow as human-readable artifacts. PyTorch Profiler wraps CUPTI
internally and produces JSON/Chrome trace output natively.

### 2.3 MONAI Profiling Utilities

MONAI provides `monai.utils.profiling.PerfContext`, a lightweight timing context manager:

```python
from monai.utils.profiling import PerfContext

with PerfContext() as pc:
    output = model(images)
print(f"Forward pass: {pc.total_time:.3f}s")
```

`PerfContext` measures wall-clock time only (no CUDA kernel breakdown, no memory tracking).
It is useful for coarse timing of pipeline stages but insufficient for operator-level
profiling.

### 2.4 Pre-Training and Post-Training Diagnostics

The talk "Honey, I broke the PyTorch model" (Hoffman, 2024) proposes a structured
debugging methodology for PyTorch models organized into pre-training checks (verifying
model structure and data integrity before GPU time is spent) and post-training checks
(diagnosing model quality beyond loss curves). Key tools:

- **torch-test / torchtest**: Synthetic-input sanity checks — verify that loss decreases
  on a single batch, gradients flow to all parameters, outputs have expected shape, no
  NaN/Inf in outputs. Catches architecture bugs before full training.

- **WeightWatcher**: Spectral analysis of weight matrices via Random Matrix Theory.
  Computes power-law exponent `alpha` per layer; `alpha_weighted > 5.0` indicates poor
  generalization. Already implemented in `src/minivess/ensemble/weightwatcher.py`.

- **Cleanlab**: Label quality estimation via confident learning. Identifies likely
  mislabeled training examples. Already a dependency in `pyproject.toml` (quality group).

### 2.5 Existing Infrastructure in MinIVess

| Component | File | Current State |
|-----------|------|---------------|
| Training loop | `src/minivess/pipeline/trainer.py` | No profiling integration |
| System monitor | `scripts/system_monitor.py` | 10s GPU snapshots via `pynvml` |
| MLflow tracking | `src/minivess/observability/tracking.py` | `log_epoch_metrics()`, `log_artifact()` |
| WeightWatcher | `src/minivess/ensemble/weightwatcher.py` | Implemented, not integrated into flow |
| GPU VRAM check | `src/minivess/adapters/sam3_vram_check.py` | SAM3-specific, not generic |
| Model profiles | `configs/model_profiles/*.yaml` | 13 profiles with VRAM estimates |
| Config compose | `src/minivess/config/compose.py` | Hydra Compose API bridge |
| Benchmark issue | GitHub #564 | Open, no implementation |

---

## 3. Architectural Decision Matrix

### D1: Primary Profiling Tool

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **PyTorch Profiler only** | Native Python API; JSON/Chrome trace output; `schedule()` for training loops; `profile_memory` + `with_flops`; no extra binary dependencies; logs directly to MLflow | No PCIe bandwidth, no NVLink metrics, no CPU instruction sampling | **SELECTED** |
| **NVIDIA Nsight Systems only** | Lowest overhead (~2%); PCIe + NVLink + CPU sampling; most complete timeline | Binary `.nsys-rep` format; requires `nsys` CLI in Docker image; cannot produce JSON for MLflow; needs GUI for visualization | Rejected |
| **Both (PyTorch Profiler default + optional Nsight)** | Best of both worlds | Complexity; Nsight requires privileged container; doubles artifact size | Deferred (Phase 4) |

**Rationale**: PyTorch Profiler provides sufficient granularity for training loop
optimization (operator-level CUDA timing, memory allocation, FLOP estimation) and
produces MLflow-friendly artifacts (JSON Chrome traces, tabular summaries). Nsight Systems
is deferred to Phase 4 as an optional deep-dive tool for kernel-level optimization, gated
behind a `profiling.nsight_enabled: false` config flag.

### D2: Profiling Schedule Strategy

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Profile all epochs** | Complete data | 5-15% overhead across entire training; terabyte-scale trace files for 100-epoch runs | Rejected |
| **Profile first N epochs only** | Simple; captures warmup + steady-state; predictable overhead | Misses late-training behavior (LR schedule changes, early stopping boundary) | Rejected |
| **Configurable schedule (first N + last epoch)** | Captures warmup, steady-state, AND final epoch; configurable N; bounded overhead | Slightly more complex schedule logic | **SELECTED** |

**Implementation**: Use `torch.profiler.schedule()` with `skip_first=0, wait=0, warmup=1, active=N-1, repeat=1` for the first N epochs, then trigger a single active window on the final epoch via a custom `on_trace_ready` callback. Default: `profiling.epochs: 5` for real training, `profiling.epochs: 2` for debug configs.

### D3: Trace Export Format

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Chrome trace JSON only** | Universal viewer (`chrome://tracing`); MLflow artifact; parseable | Large files (10-100 MB per trace) | **SELECTED (primary)** |
| **TensorBoard plugin** | Rich UI with timeline, memory, kernel views | Requires TensorBoard server; not in MinIVess stack; duplicates MLflow role | Rejected |
| **Both Chrome + TensorBoard** | Maximum flexibility | Doubled storage; maintenance burden | Deferred |
| **Chrome trace + summary CSV** | Chrome for visual, CSV for programmatic analysis | Extra file per run | **SELECTED (secondary)** |

**Implementation**: Export Chrome trace as MLflow artifact at `profiling/chrome_trace_epoch_{N}.json`. Additionally, extract `key_averages()` table as CSV artifact at `profiling/op_summary.csv` and log aggregate metrics (total CUDA time, total CPU time, peak memory, data loading fraction) as MLflow params with `prof_` prefix.

### D4: Memory Profiling Approach

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **`torch.profiler.profile(profile_memory=True)`** | Integrated with operator timeline; per-op allocation tracking; Chrome trace shows memory | ~5% additional overhead; allocation-level, not fragmentation-level | **SELECTED** |
| **`torch.cuda.memory_stats()` snapshots** | Fragmentation metrics; no profiler overhead; can run every epoch | No per-operator attribution; snapshot-only | **SELECTED (complementary)** |
| **`torch.cuda.memory_snapshot()` + `torch.cuda._record_memory_history()`** | Full allocation history with Python stack traces | Extremely verbose; GB-scale output; experimental API | Deferred |

**Implementation**: Use `profile_memory=True` in the profiler schedule (first N epochs).
Additionally, log `torch.cuda.memory_stats()` summary every epoch as MLflow metrics:
`prof_cuda_allocated_peak_mb`, `prof_cuda_reserved_peak_mb`, `prof_cuda_active_peak_mb`,
`prof_cuda_num_alloc_retries`, `prof_cuda_num_ooms`. This captures fragmentation
indicators (reserved vs. allocated gap) without profiler overhead.

### D5: Data Loading Bottleneck Detection

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **`record_function("data_loading")` wrapper** | Visible in Chrome trace timeline; measures total data wait time per batch | Manual instrumentation of DataLoader iteration | **SELECTED** |
| **Automatic `num_workers` tuning** | Finds optimal parallelism | Non-deterministic; varies per machine; adds pre-training overhead | Deferred (Phase 3) |
| **MONAI `ThreadDataLoader` auto-detection** | MONAI-native | Only relevant if using MONAI's DataLoader variant | Informational only |

**Implementation**: Wrap the `for batch in loader` iteration in `train_epoch()` and
`validate_epoch()` with `torch.profiler.record_function("data_loading")`. This labels
the data fetch time in the Chrome trace, making it trivially visible as a separate bar.
Additionally, compute `data_loading_fraction = data_time / (data_time + compute_time)`
per epoch and log as `prof_data_loading_fraction`.

### D6: Pre-Training Check Library

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Custom sanity checks (torch-test patterns)** | Tailored to 3D segmentation; no extra dependency; exact checks we need | Must implement and maintain | **SELECTED** |
| **`torchtest` package** | Ready-made; popular | Last PyPI release 2023; limited to classification patterns; no 3D volume support | Rejected |
| **Both** | Coverage | Dependency on unmaintained package | Rejected |

**Checks to implement** (inspired by Hoffman, 2024):

1. **Output shape check**: `model(random_input).logits.shape == (B, C, H, W, D)`
2. **Loss decreases on single batch**: 5 steps of `train_epoch()` on 1 batch; loss must
   decrease by >1%.
3. **Gradients flow**: After one backward pass, verify `param.grad is not None` and
   `torch.isfinite(param.grad).all()` for all trainable parameters.
4. **No NaN/Inf in output**: `torch.isfinite(output.logits).all()` on random input.
5. **Parameter count matches profile**: Compare `model.trainable_parameters()` against
   `configs/model_profiles/{name}.yaml` expected range.
6. **VRAM usage matches profile**: Compare measured peak VRAM against model profile
   `model_overhead_mb` estimate (within 50% tolerance).

### D7: Post-Training Check Library

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **WeightWatcher only** | Already implemented; spectral quality gate; lightweight | Single diagnostic signal | Partial |
| **WeightWatcher + Cleanlab** | Spectral + label quality; both already in dependencies | Cleanlab requires predicted probabilities on training set | **SELECTED** |
| **WeightWatcher + Cleanlab + Deepchecks Vision** | Most comprehensive | Deepchecks is heavy; overlaps with existing evaluation suite | Deferred |

**Implementation**: After training completes (in `train_flow.py`), run:
1. `weightwatcher.analyze_model()` -- log `prof_ww_alpha_weighted`, `prof_ww_log_norm`,
   `prof_ww_passed_gate` to MLflow.
2. Cleanlab `find_label_issues()` on training set predictions -- log
   `prof_cleanlab_n_issues`, `prof_cleanlab_issue_fraction` to MLflow, and save the
   issue index list as artifact `profiling/cleanlab_label_issues.json`.

---

## 4. System Architecture

### 4.1 Component Diagram

```
                    Hydra Config (configs/profiling/default.yaml)
                              |
                              v
    +---------------------------------------------------+
    |              SegmentationTrainer.fit()              |
    |                                                    |
    |  +-----------+    +------------------+             |
    |  | Pre-Train |    | torch.profiler   |             |
    |  | Checks    |    | .profile()       |             |
    |  | (Phase 2) |    |                  |             |
    |  +-----+-----+    | schedule(        |             |
    |        |           |   wait=0,       |             |
    |        v           |   warmup=1,     |             |
    |  MLflow params     |   active=N-1,   |             |
    |  prof_precheck_*   |   repeat=1)     |             |
    |                    |                  |             |
    |                    | record_function: |             |
    |                    |   "data_loading" |             |
    |                    |   "forward"      |             |
    |                    |   "loss"         |             |
    |                    |   "backward"     |             |
    |                    |   "optimizer"    |             |
    |                    +--------+---------+             |
    |                             |                      |
    |                    on_trace_ready callback          |
    |                             |                      |
    |                    +--------v---------+             |
    |                    | TraceHandler     |             |
    |                    |                  |             |
    |                    | export_chrome()  |             |
    |                    | key_averages()   |             |
    |                    | memory_stats()   |             |
    |                    +--------+---------+             |
    |                             |                      |
    +---------------------------------------------------+
                              |
                              v
    +---------------------------------------------------+
    |                  MLflow Artifacts                   |
    |                                                    |
    |  profiling/                                        |
    |    chrome_trace_epoch_1.json                       |
    |    chrome_trace_epoch_5.json                       |
    |    chrome_trace_final.json                         |
    |    op_summary.csv                                  |
    |    memory_timeline.csv                             |
    |    cleanlab_label_issues.json                      |
    |                                                    |
    |  MLflow Params (prof_ prefix):                     |
    |    prof_cuda_time_total_ms                         |
    |    prof_cpu_time_total_ms                          |
    |    prof_gpu_utilization_pct                        |
    |    prof_data_loading_fraction                      |
    |    prof_peak_memory_mb                             |
    |    prof_memory_fragmentation_pct                   |
    |    prof_top_op_name / prof_top_op_cuda_pct         |
    |    prof_ww_alpha_weighted                          |
    |    prof_ww_passed_gate                             |
    |    prof_cleanlab_n_issues                          |
    |    prof_precheck_output_shape_ok                   |
    |    prof_precheck_loss_decreases                    |
    |    prof_precheck_grads_finite                      |
    +---------------------------------------------------+
                              |
                              v
    +---------------------------------------------------+
    |           GPU Benchmark Container (Phase 3)        |
    |                                                    |
    |  docker compose run --rm gpu-benchmark             |
    |                                                    |
    |  Probes: VRAM, compute capability, driver,         |
    |          peak memory under model forward pass,     |
    |          throughput (images/sec per model)          |
    |                                                    |
    |  Output: /benchmarks/gpu_benchmark.yaml            |
    |          (mounted volume, cached across runs)      |
    +---------------------------------------------------+
```

### 4.2 Integration Points

#### 4.2.1 `trainer.py` Integration

The `SegmentationTrainer.fit()` method gains an optional `torch.profiler.profile`
context manager, activated when `profiling.enabled: true` in the Hydra config. The
profiler wraps the epoch loop, with `prof.step()` called at the end of each epoch.

The `train_epoch()` and `validate_epoch()` methods gain `record_function()` annotations
around four regions: `"data_loading"`, `"forward"`, `"loss_compute"`, and
`"backward_optimizer"`. These annotations are always present (zero overhead when profiler
is inactive) and provide labeled regions in the Chrome trace.

#### 4.2.2 `tracking.py` Integration

`ExperimentTracker` gains a new method `log_profiling_summary()` that accepts the
profiler's `key_averages()` output and logs:

- Aggregate metrics as MLflow params (single values per run, `prof_` prefix)
- Chrome trace files as MLflow artifacts under `profiling/`
- Summary CSV as MLflow artifact under `profiling/`

#### 4.2.3 `train_flow.py` Integration

The Prefect training flow reads `profiling.*` config keys from the composed experiment
config and passes them to `SegmentationTrainer`. Post-training checks (WeightWatcher,
Cleanlab) are called after `trainer.fit()` returns, within the same MLflow run context.

#### 4.2.4 Config Integration

A new Hydra config group `profiling` is added at `configs/profiling/default.yaml`,
included in `configs/base.yaml` defaults list.

---

## 5. Implementation Plan

### Phase 1: Core PyTorch Profiler Integration (4 issues)

**Goal**: Instrument `trainer.py` with `torch.profiler`, export Chrome traces to MLflow,
log summary metrics. Default ON.

#### Issue P1.1: Hydra Profiling Config Group

**Files**:
- `configs/profiling/default.yaml` (new)
- `configs/profiling/disabled.yaml` (new)
- `configs/profiling/deep.yaml` (new)
- `configs/base.yaml` (add `profiling` to defaults)
- `src/minivess/config/models.py` (add `ProfilingConfig`)

**TDD Spec**:
- RED: `test_profiling_config_default()` -- compose default config, assert
  `profiling.enabled == True`, `profiling.epochs == 5`.
- RED: `test_profiling_config_debug_override()` -- compose `debug_single_model`, assert
  `profiling.epochs == 2`.
- RED: `test_profiling_config_disabled()` -- compose with `profiling=disabled`, assert
  `profiling.enabled == False`.
- GREEN: Implement config files and Pydantic model.

#### Issue P1.2: `record_function` Annotations in Training Loop

**Files**:
- `src/minivess/pipeline/trainer.py` (modify `train_epoch`, `validate_epoch`)

**TDD Spec**:
- RED: `test_train_epoch_has_record_function_labels()` -- Run 1 epoch with profiler
  active, assert `"data_loading"`, `"forward"`, `"loss_compute"`, `"backward_optimizer"`
  appear in `prof.key_averages()` event names.
- RED: `test_record_function_zero_overhead_without_profiler()` -- Run 1 epoch without
  profiler, measure wall time. Run again with `record_function` annotations but no
  profiler context. Assert wall time delta < 1%.
- GREEN: Add `record_function` context managers around the four regions in `train_epoch()`
  and `validate_epoch()`.

#### Issue P1.3: Profiler Context Manager in `fit()`

**Files**:
- `src/minivess/pipeline/trainer.py` (modify `fit()`)
- `src/minivess/pipeline/profiler_handler.py` (new -- trace handler callback)

**TDD Spec**:
- RED: `test_fit_with_profiling_produces_chrome_trace()` -- Run `fit()` with
  `profiling.enabled=True, profiling.epochs=2`, assert Chrome trace JSON file exists
  in checkpoint_dir.
- RED: `test_fit_profiling_schedule_correct_epochs()` -- With `profiling.epochs=3,
  max_epochs=10`, verify profiler is active for epochs 1-3 and the final epoch (10)
  only. Assert exactly 4 Chrome trace files.
- RED: `test_fit_profiling_disabled_no_overhead()` -- With `profiling.enabled=False`,
  assert no trace files produced, wall time within 1% of un-instrumented baseline.
- RED: `test_profiler_handler_extracts_summary()` -- Assert handler produces dict with
  keys `cuda_time_total_ms`, `cpu_time_total_ms`, `peak_memory_mb`,
  `data_loading_fraction`, `top_op_name`.
- GREEN: Implement profiler context manager wrapping the epoch loop in `fit()`, with
  custom `on_trace_ready` callback.

#### Issue P1.4: MLflow Profiling Artifact Logging

**Files**:
- `src/minivess/observability/tracking.py` (add `log_profiling_summary()`)
- `src/minivess/pipeline/trainer.py` (call tracker from profiler handler)

**TDD Spec**:
- RED: `test_log_profiling_summary_params()` -- Mock MLflow, call
  `log_profiling_summary()`, assert params logged with `prof_` prefix.
- RED: `test_log_profiling_chrome_trace_artifact()` -- Assert Chrome trace logged
  under `profiling/` artifact path.
- RED: `test_log_profiling_op_summary_csv()` -- Assert CSV with columns
  `name, self_cpu_time, self_cuda_time, calls, cpu_memory, cuda_memory` logged.
- GREEN: Implement `log_profiling_summary()` and integrate with trainer.

### Phase 2: Pre-Training and Post-Training Checks (4 issues)

**Goal**: Structured model diagnostics before and after training, logged to MLflow.

#### Issue P2.1: Pre-Training Sanity Checks Module

**Files**:
- `src/minivess/pipeline/pretrain_checks.py` (new)
- `tests/v2/unit/test_pretrain_checks.py` (new)

**TDD Spec**:
- RED: `test_output_shape_check_passes()` -- DynUNet with known input shape, assert pass.
- RED: `test_output_shape_check_fails_wrong_channels()` -- Model with wrong out_channels,
  assert fail with descriptive message.
- RED: `test_loss_decreases_check()` -- 5 optimization steps on random data, loss must
  decrease by >1%.
- RED: `test_gradients_finite_check()` -- After backward pass, all `param.grad` finite.
- RED: `test_no_nan_inf_output()` -- Forward pass on random input, no NaN/Inf.
- RED: `test_param_count_matches_profile()` -- Compare `model.trainable_parameters()`
  against model profile YAML expected range.
- GREEN: Implement `run_pretrain_checks()` returning `PreTrainCheckResult` dataclass.

#### Issue P2.2: Pre-Training Checks Integration into Training Flow

**Files**:
- `src/minivess/orchestration/flows/train_flow.py` (add pre-train check call)
- `src/minivess/observability/tracking.py` (add `log_pretrain_checks()`)

**TDD Spec**:
- RED: `test_train_flow_runs_pretrain_checks()` -- Mock train flow, assert
  `run_pretrain_checks()` called before `trainer.fit()`.
- RED: `test_pretrain_check_failure_aborts_training()` -- When `pretrain_checks.abort_on_failure: true` and checks fail, training does not proceed.
- RED: `test_pretrain_results_logged_to_mlflow()` -- Assert `prof_precheck_*` params
  logged.
- GREEN: Integrate into flow, respecting `profiling.pretrain_checks.enabled` config.

#### Issue P2.3: WeightWatcher Post-Training Integration

**Files**:
- `src/minivess/orchestration/flows/train_flow.py` (add post-train WeightWatcher call)
- `src/minivess/ensemble/weightwatcher.py` (already exists -- minor API refinement)

**TDD Spec**:
- RED: `test_weightwatcher_logged_after_training()` -- After `fit()`, assert
  `prof_ww_alpha_weighted` logged as MLflow param.
- RED: `test_weightwatcher_gate_failure_tagged()` -- When alpha > threshold, assert
  MLflow tag `prof_ww_gate = "FAILED"`.
- GREEN: Call `analyze_model()` after `fit()`, log results via tracker.

#### Issue P2.4: Cleanlab Label Quality Post-Training Check

**Files**:
- `src/minivess/pipeline/label_quality.py` (new)
- `tests/v2/unit/test_label_quality.py` (new)

**TDD Spec**:
- RED: `test_cleanlab_finds_issues_on_noisy_labels()` -- Synthetic dataset with 10%
  flipped labels, assert `n_issues > 0`.
- RED: `test_cleanlab_results_logged_to_mlflow()` -- Assert `prof_cleanlab_n_issues`
  and `prof_cleanlab_issue_fraction` logged.
- RED: `test_cleanlab_issue_json_artifact()` -- Assert JSON artifact with issue indices
  logged under `profiling/`.
- GREEN: Implement `find_label_issues()` wrapper, integrate into train flow.

### Phase 3: GPU Benchmark Container (3 issues)

**Goal**: Close GitHub issue #564. Run-once GPU capability probing with cached results.

#### Issue P3.1: GPU Benchmark Script

**Files**:
- `src/minivess/compute/gpu_benchmark.py` (new)
- `tests/v2/unit/test_gpu_benchmark.py` (new -- YAML parsing, no GPU needed)

**TDD Spec**:
- RED: `test_benchmark_yaml_schema()` -- Load a sample `gpu_benchmark.yaml`, validate
  schema with Pydantic model (`BenchmarkResult`).
- RED: `test_benchmark_cache_skip()` -- When YAML exists and is < 24h old, assert
  benchmark is skipped.
- RED: `test_benchmark_capability_check()` -- Given YAML with `sam3_hybrid: true`,
  assert `is_model_feasible("sam3_hybrid")` returns True.
- RED: `test_benchmark_missing_gpu_graceful()` -- On CPU-only, assert benchmark produces
  valid YAML with `gpu_count: 0` and all models marked infeasible.
- GREEN: Implement `run_gpu_benchmark()` and `load_benchmark_cache()`.

#### Issue P3.2: Benchmark Docker Service

**Files**:
- `deployment/docker/Dockerfile.benchmark` (new)
- `deployment/docker-compose.flows.yml` (add `gpu-benchmark` service)
- `.env.example` (add `BENCHMARK_CACHE_DIR` variable)

**TDD Spec**:
- RED: `test_dockerfile_benchmark_builds()` -- `docker build` succeeds.
- RED: `test_benchmark_service_volume_mount()` -- Assert `/benchmarks` volume mount
  in compose file.
- GREEN: Create Dockerfile and compose service definition.

#### Issue P3.3: Benchmark Integration with Training Flow

**Files**:
- `src/minivess/orchestration/flows/train_flow.py` (read benchmark cache)
- `src/minivess/compute/gpu_profile.py` (new -- cache reader API)

**TDD Spec**:
- RED: `test_train_flow_reads_benchmark_cache()` -- Mock cached YAML, assert training
  flow logs `sys_benchmark_gpu_model`, `sys_benchmark_throughput` as MLflow params.
- RED: `test_train_flow_warns_on_infeasible_model()` -- Benchmark says SAM3 hybrid
  infeasible, training flow logs warning and sets `prof_benchmark_feasible: false`.
- GREEN: Integrate cache reader into training flow.

### Phase 4: Advanced Profiling and Cloud Comparison (3 issues)

**Goal**: Cross-environment comparison tooling and optional deep profiling.

#### Issue P4.1: CUDA Memory Stats Per-Epoch Logging

**Files**:
- `src/minivess/pipeline/trainer.py` (add memory stats collection in `fit()`)

**TDD Spec**:
- RED: `test_memory_stats_logged_every_epoch()` -- After training, assert
  `prof_cuda_allocated_peak_mb` present in MLflow metrics at each step.
- RED: `test_memory_fragmentation_computed()` -- Assert
  `prof_cuda_fragmentation_pct = (reserved - allocated) / reserved * 100` computed
  correctly.
- GREEN: Add `torch.cuda.memory_stats()` collection after each epoch, log via tracker.

#### Issue P4.2: Cloud vs. Local Comparison Dashboard Adapter

**Files**:
- `src/minivess/orchestration/flows/dashboard_sections.py` (add profiling comparison)
- `configs/dashboard/profiling_comparison.yaml` (new)

**TDD Spec**:
- RED: `test_comparison_table_generated()` -- Given two MLflow runs (one with
  `sys_gpu_model=RTX 2070 Super`, one with `sys_gpu_model=A100`), assert comparison
  table includes `throughput_ratio`, `cost_efficiency_ratio`, `memory_headroom_pct`.
- GREEN: Implement comparison adapter reading `prof_*` params from MLflow runs.

#### Issue P4.3: Optional Nsight Systems Integration

**Files**:
- `src/minivess/pipeline/nsight_profiler.py` (new)
- `configs/profiling/nsight.yaml` (new)

**TDD Spec**:
- RED: `test_nsight_disabled_by_default()` -- Default config has
  `profiling.nsight_enabled: false`.
- RED: `test_nsight_produces_report()` -- When enabled and `nsys` is available, assert
  `.nsys-rep` file produced and logged as artifact.
- GREEN: Implement optional Nsight wrapper with graceful degradation.

---

## 6. Hydra Configuration Schema

### 6.1 Config Group: `configs/profiling/default.yaml`

```yaml
# Default profiling config — enabled for all models and environments.
# Overhead budget: <5% wall-time increase via schedule() windowing.

enabled: true

# Number of epochs to actively profile (from start of training).
# Additionally, the FINAL epoch is always profiled regardless of this setting.
# Real training: 5 epochs captures warmup + steady-state transition.
# Debug configs override to 2.
epochs: 5

# PyTorch Profiler activities to trace
activities:
  - cpu
  - cuda

# Memory profiling (adds ~5% overhead; disable for minimal-overhead runs)
profile_memory: true

# Record operator input shapes (useful for diagnosing shape mismatches)
record_shapes: true

# Estimate FLOPs per operator (requires record_shapes: true)
with_flops: true

# Capture Python stack traces (adds ~3% overhead; useful for debugging)
with_stack: false

# Pre-training sanity checks
pretrain_checks:
  enabled: true
  abort_on_failure: false  # Log warning but continue training
  loss_decrease_steps: 5
  loss_decrease_min_pct: 1.0

# Post-training diagnostics
posttrain_checks:
  weightwatcher:
    enabled: true
    alpha_threshold: 5.0
  cleanlab:
    enabled: false  # Requires full training set prediction pass; opt-in

# CUDA memory stats (per-epoch, independent of profiler schedule)
memory_stats:
  enabled: true
  log_every_n_epochs: 1

# Nsight Systems (optional deep profiling)
nsight_enabled: false
```

### 6.2 Config Group: `configs/profiling/disabled.yaml`

```yaml
# Explicitly disable all profiling (e.g., for production HPO sweeps
# where every second counts).

enabled: false
pretrain_checks:
  enabled: false
posttrain_checks:
  weightwatcher:
    enabled: false
  cleanlab:
    enabled: false
memory_stats:
  enabled: false
nsight_enabled: false
```

### 6.3 Config Group: `configs/profiling/deep.yaml`

```yaml
# Deep profiling — all features enabled including stack traces.
# ~15% overhead. Use for targeted debugging sessions.

enabled: true
epochs: 10
activities:
  - cpu
  - cuda
profile_memory: true
record_shapes: true
with_flops: true
with_stack: true
pretrain_checks:
  enabled: true
  abort_on_failure: true
posttrain_checks:
  weightwatcher:
    enabled: true
    alpha_threshold: 5.0
  cleanlab:
    enabled: true
memory_stats:
  enabled: true
  log_every_n_epochs: 1
nsight_enabled: false
```

### 6.4 Pydantic Model: `ProfilingConfig`

```python
class PreTrainCheckConfig(BaseModel):
    enabled: bool = True
    abort_on_failure: bool = False
    loss_decrease_steps: int = Field(default=5, ge=1)
    loss_decrease_min_pct: float = Field(default=1.0, ge=0.0)

class WeightWatcherCheckConfig(BaseModel):
    enabled: bool = True
    alpha_threshold: float = Field(default=5.0, gt=0.0)

class CleanlabCheckConfig(BaseModel):
    enabled: bool = False

class PostTrainCheckConfig(BaseModel):
    weightwatcher: WeightWatcherCheckConfig = Field(
        default_factory=WeightWatcherCheckConfig
    )
    cleanlab: CleanlabCheckConfig = Field(
        default_factory=CleanlabCheckConfig
    )

class MemoryStatsConfig(BaseModel):
    enabled: bool = True
    log_every_n_epochs: int = Field(default=1, ge=1)

class ProfilingConfig(BaseModel):
    enabled: bool = True
    epochs: int = Field(default=5, ge=1)
    activities: list[str] = Field(default=["cpu", "cuda"])
    profile_memory: bool = True
    record_shapes: bool = True
    with_flops: bool = True
    with_stack: bool = False
    pretrain_checks: PreTrainCheckConfig = Field(
        default_factory=PreTrainCheckConfig
    )
    posttrain_checks: PostTrainCheckConfig = Field(
        default_factory=PostTrainCheckConfig
    )
    memory_stats: MemoryStatsConfig = Field(
        default_factory=MemoryStatsConfig
    )
    nsight_enabled: bool = False
```

### 6.5 Debug Config Override

Existing debug experiment configs (e.g., `debug_single_model.yaml`) gain:

```yaml
# Profiling override for debug runs — fewer epochs
profiling:
  epochs: 2
  with_stack: true  # Always useful in debug
  posttrain_checks:
    cleanlab:
      enabled: false  # No full training set in debug
```

### 6.6 `configs/base.yaml` Update

```yaml
defaults:
  - data: minivess
  - model: dynunet
  - training: default
  - checkpoint: standard
  - profiling: default      # NEW
  - _self_
```

---

## 7. MLflow Artifact Taxonomy

### 7.1 Artifacts (logged under `profiling/` path)

| Artifact | Format | Size (est.) | When Produced |
|----------|--------|-------------|---------------|
| `chrome_trace_epoch_{N}.json` | Chrome trace JSON | 10-50 MB | Each profiled epoch |
| `chrome_trace_final.json` | Chrome trace JSON | 10-50 MB | Final epoch |
| `op_summary.csv` | CSV | <100 KB | End of profiling window |
| `memory_timeline.csv` | CSV | <50 KB | End of training |
| `cleanlab_label_issues.json` | JSON | <1 MB | Post-training (if enabled) |
| `pretrain_check_report.json` | JSON | <10 KB | Pre-training |

### 7.2 MLflow Params (`prof_` prefix, logged once per run)

| Param | Type | Description |
|-------|------|-------------|
| `prof_enabled` | bool | Whether profiling was active |
| `prof_epochs_profiled` | int | Number of epochs actually profiled |
| `prof_cuda_time_total_ms` | float | Total CUDA kernel time (profiled epochs) |
| `prof_cpu_time_total_ms` | float | Total CPU time (profiled epochs) |
| `prof_data_loading_fraction` | float | Fraction of epoch spent in data loading |
| `prof_peak_memory_mb` | float | Peak CUDA memory during profiled epochs |
| `prof_memory_fragmentation_pct` | float | `(reserved - allocated) / reserved * 100` |
| `prof_top_op_name` | str | Most time-consuming CUDA operator |
| `prof_top_op_cuda_pct` | float | Percentage of total CUDA time from top op |
| `prof_total_flops` | float | Estimated total FLOPs (profiled epochs) |
| `prof_throughput_img_per_sec` | float | Images processed per second |
| `prof_precheck_all_passed` | bool | All pre-training checks passed |
| `prof_precheck_output_shape_ok` | bool | Output shape matches expected |
| `prof_precheck_loss_decreases` | bool | Loss decreased on sanity batch |
| `prof_precheck_grads_finite` | bool | All gradients finite after backward |
| `prof_ww_alpha_weighted` | float | WeightWatcher alpha (post-training) |
| `prof_ww_log_norm` | float | WeightWatcher log-norm |
| `prof_ww_passed_gate` | bool | Alpha below threshold |
| `prof_cleanlab_n_issues` | int | Number of likely label issues found |
| `prof_cleanlab_issue_fraction` | float | Fraction of training set flagged |

### 7.3 MLflow Metrics (`prof_` prefix, logged per-epoch step)

| Metric | Step | Description |
|--------|------|-------------|
| `prof_cuda_allocated_peak_mb` | epoch | Peak allocated CUDA memory |
| `prof_cuda_reserved_peak_mb` | epoch | Peak reserved CUDA memory |
| `prof_cuda_num_alloc_retries` | epoch | CUDA allocator retries (fragmentation) |
| `prof_epoch_cuda_time_ms` | epoch | CUDA time for this epoch (profiled only) |
| `prof_epoch_data_fraction` | epoch | Data loading fraction this epoch |

---

## 8. Test Plan

### 8.1 Test Tier Mapping

| Test | Tier | Rationale |
|------|------|-----------|
| `test_profiling_config_*` | Staging | Config parsing, no GPU |
| `test_pretrain_checks_*` (unit) | Staging | Synthetic data, CPU |
| `test_record_function_annotations` | Prod | Requires model forward pass |
| `test_fit_with_profiling_*` | Prod | Requires GPU for CUDA profiling |
| `test_memory_stats_*` | Prod | Requires CUDA device |
| `test_gpu_benchmark_yaml_*` | Staging | YAML parsing only |
| `test_gpu_benchmark_run` | GPU | Actual GPU probing |
| `test_weightwatcher_*` | Prod | Model loading required |
| `test_cleanlab_*` | Prod | Training set prediction required |
| `test_cloud_comparison_*` | Staging | Mock MLflow data |

### 8.2 Test Count Estimate

| Phase | Unit Tests | Integration Tests | Total |
|-------|-----------|-------------------|-------|
| Phase 1 | 12 | 4 | 16 |
| Phase 2 | 14 | 4 | 18 |
| Phase 3 | 8 | 2 | 10 |
| Phase 4 | 6 | 2 | 8 |
| **Total** | **40** | **12** | **52** |

---

## 9. GitHub Issue Breakdown

### Summary Table

| # | Issue | Title | Phase | Priority | Est. |
|---|-------|-------|-------|----------|------|
| P1.1 | [#644](https://github.com/petteriTeikari/minivess-mlops/issues/644) | Add Hydra profiling config group + ProfilingConfig model | 1 | P0 | 2h |
| P1.2 | [#645](https://github.com/petteriTeikari/minivess-mlops/issues/645) | Add `record_function` annotations to trainer.py | 1 | P0 | 1h |
| P1.3 | [#646](https://github.com/petteriTeikari/minivess-mlops/issues/646) | Integrate `torch.profiler.profile` context in `fit()` | 1 | P0 | 4h |
| P1.4 | [#647](https://github.com/petteriTeikari/minivess-mlops/issues/647) | MLflow profiling artifact logging (`log_profiling_summary`) | 1 | P0 | 3h |
| P2.1 | [#648](https://github.com/petteriTeikari/minivess-mlops/issues/648) | Pre-training sanity checks module | 2 | P1 | 3h |
| P2.2 | — | Integrate pre-training checks into train_flow.py | 2 | P1 | 2h |
| P2.3 | [#649](https://github.com/petteriTeikari/minivess-mlops/issues/649) | WeightWatcher post-training integration in train_flow | 2 | P1 | 1h |
| P2.4 | — | Cleanlab label quality post-training check | 2 | P2 | 3h |
| P3.1 | [#650](https://github.com/petteriTeikari/minivess-mlops/issues/650) | GPU benchmark script + cache YAML | 3 | P1 | 4h |
| P3.2 | [#651](https://github.com/petteriTeikari/minivess-mlops/issues/651) | Dockerfile.benchmark + compose service | 3 | P1 | 2h |
| P3.3 | — | Benchmark cache integration with training flow | 3 | P1 | 2h |
| P4.1 | — | CUDA memory stats per-epoch logging | 4 | P2 | 2h |
| P4.2 | — | Cloud vs. local comparison dashboard adapter | 4 | P2 | 3h |
| P4.3 | — | Optional Nsight Systems integration | 4 | P3 | 3h |

**Total estimated effort**: ~35 hours across 14 issues.

### Dependency Graph

```
P1.1 ──> P1.2 ──> P1.3 ──> P1.4
                     |
                     v
              P2.1 ──> P2.2
              P2.3 (parallel)
              P2.4 (parallel)
                     |
                     v
              P3.1 ──> P3.2 ──> P3.3
                     |
                     v
              P4.1 (parallel)
              P4.2 (parallel)
              P4.3 (parallel)
```

Phase 1 is the critical path. Phase 2 can start after P1.3. Phase 3 can start after
Phase 1 completes. Phase 4 issues are independent.

### Closing Issue #564

Issue #564 is fully addressed by Phase 3 (P3.1 + P3.2 + P3.3), which implements:

- `src/minivess/compute/gpu_benchmark.py` -- probe GPU + run micro-benchmarks + write YAML
- `src/minivess/compute/gpu_profile.py` -- read cached YAML, expose capability API
- `deployment/docker/Dockerfile.benchmark` -- thin benchmark image
- `deployment/docker-compose.flows.yml` -- `gpu-benchmark` service
- `tests/v2/unit/test_gpu_benchmark.py` -- YAML schema tests (no GPU needed)

The original #564 scope is extended by Phases 1-2 (profiling + diagnostics) and Phase 4
(cloud comparison), which address the broader user requirement of "knowing if GPU use
is optimal."

---

## 10. Risk Analysis

### R1: Profiler Overhead Exceeds Budget

**Risk**: `profile_memory=True` + `with_flops=True` adds >5% overhead.
**Mitigation**: `schedule()` limits active profiling to N epochs. Default `with_stack=False`
reduces overhead. Emergency escape: `profiling=disabled` override.
**Monitoring**: Log `prof_overhead_pct = (profiled_epoch_time - unprofiled_epoch_time) / unprofiled_epoch_time` for the first profiled and first unprofiled epochs.

### R2: Chrome Trace File Size

**Risk**: Chrome traces for SAM3 (648M params) may exceed 100 MB per epoch.
**Mitigation**: Limit `record_shapes` to profiled epochs only. Consider
`prof.export_stacks()` for compressed output if traces exceed 200 MB.
**Monitoring**: Log `prof_trace_size_mb` as MLflow metric.

### R3: WeightWatcher Compatibility with Frozen Backbones

**Risk**: WeightWatcher may produce misleading alpha values for SAM3's frozen ViT backbone
(large layers with unchanged pretrained weights).
**Mitigation**: Filter WeightWatcher analysis to trainable layers only via
`watcher.analyze(layers=trainable_layer_indices)`.
**Monitoring**: Log `prof_ww_num_layers_analyzed` vs. total layers.

### R4: Cleanlab Memory on Full Training Set

**Risk**: Running Cleanlab on all 47 training volumes (full-resolution 3D) may exceed
available RAM.
**Mitigation**: Default `cleanlab.enabled: false`. When enabled, subsample to
`max_cleanlab_volumes: 10` and operate on 2D slices, not full 3D volumes.

### R5: GPU Benchmark Cache Invalidation

**Risk**: Cached `gpu_benchmark.yaml` becomes stale after driver update or GPU swap.
**Mitigation**: Include `driver_version` and `cuda_version` in cache key. If current
system info does not match cached values, re-run benchmark automatically.

---

## 11. References

### PyTorch Profiler

1. [PyTorch Profiler -- Official API Documentation. `torch.profiler` module reference.](https://docs.pytorch.org/docs/stable/profiler.html)

2. [PyTorch Profiler Tutorial -- Beginner Guide. "Profiling your PyTorch Module," covering `profile()`, `key_averages()`, `record_function()`, and memory profiling.](https://docs.pytorch.org/tutorials/beginner/profiler.html)

3. [PyTorch Profiler Recipe. Training loop integration with `schedule()`, `on_trace_ready`, Chrome trace export.](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

### MONAI

4. [MONAI Profiling Utilities. `monai.utils.profiling` module with `PerfContext` timer.](https://github.com/Project-MONAI/MONAI/blob/dev/monai/utils/profiling.py)

### GPU Profiling Surveys

5. [Eunomia (2025). "GPU Profiling Under the Hood: An Implementation-Focused Survey of Modern Accelerator Tracing Tools." Comprehensive survey of CUPTI, Nsight Systems, Nsight Compute, HPCToolkit, and DCGM.](https://eunomia.dev/blog/2025/04/21/gpu-profiling-under-the-hood-an-implementation-focused-survey-of-modern-accelerator-tracing-tools/)

### Cloud GPU Profiling and Monitoring

6. [RunPod. "Monitoring and Debugging AI Model Deployments." Guide for RunPod-specific GPU monitoring.](https://www.runpod.io/articles/guides/monitoring-and-debugging-ai-model-deployments)

7. [Atlantic.net. "How to Profile and Debug GPU Performance for Machine Learning Models on Ubuntu 24.04 GPU Server."](https://www.atlantic.net/gpu-server-hosting/how-to-profile-and-debug-gpu-performance-for-machine-learning-models-on-ubuntu-24-04-gpu-server/)

8. [Medium / ServerwalaInfra. "Debugging GPU Issues with AI: How Grok Simplifies Troubleshooting."](https://medium.com/@serverwalainfra/debugging-gpu-issues-with-ai-how-grok-simplifies-troubleshooting-0e7ccaadb109)

9. [Reddit r/CUDA. "What is your experience developing on a cloud?" Community discussion on cloud vs. local GPU development pain points (2023).](https://www.reddit.com/r/CUDA/comments/180996l/what_is_your_experience_developing_on_a_cloud/)

10. [Google Cloud. "TensorBoard Profiler for Vertex AI." Cloud-native profiling integration.](https://docs.cloud.google.com/vertex-ai/docs/training/tensorboard-profiler)

11. [Spheron. "GPU Cost Optimization Playbook." Strategies for reducing cloud GPU spend.](https://www.spheron.network/blog/gpu-cost-optimization-playbook/)

12. [Massed Compute. "How to use NVIDIA tools to debug and profile GPU memory usage in cloud-based deep learning environments."](https://massedcompute.com/faq-answers/?question=How%20to%20use%20NVIDIA%20tools%20to%20debug%20and%20profile%20GPU%20memory%20usage%20in%20cloud-based%20deep%20learning%20environments?)

13. [NeevCloud. "How to Monitor Cloud GPU Use for Model Training and Inference."](https://blog.neevcloud.com/how-to-monitor-cloud-gpu-use-for-model-training-and-inference)

14. [Towards Data Science. "Remote Development and Debugging on the Cloud (AWS, Azure, GCP) for Deep Learning."](https://towardsdatascience.com/remote-development-and-debugging-on-the-cloud-aws-azure-gcp-for-deep-learning-computer-vision-5333fc698769/)

15. [AWS SageMaker. "Debugger Profiling Report Walkthrough." SageMaker-specific profiling artifacts.](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-profiling-report-walkthrough.html)

### Other Profiling Frameworks

16. [JAX Profiling Documentation. JAX-specific GPU profiling (not applicable to PyTorch but included for completeness).](https://docs.jax.dev/en/latest/profiling.html)

17. [LobeHub. "GPU CLI Tools." Collection of GPU monitoring CLI utilities.](https://lobehub.com/skills/gpu-cli-gpu-gpu)

### Model Diagnostics

18. Hoffman, C. "Honey, I broke the PyTorch model -- Debugging custom PyTorch models in a structured manner." Talk at PyData / Research Center for Trustworthy Data Science and Security, University Alliance Ruhr. Key packages: torch-test, WeightWatcher, Cleanlab, PlaitPy, Zumo Lab ZPy. [Slides available on speaker's GitHub -- preprint pending]

19. [Martin, C. H. and Mahoney, M. W. (2021). "Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Training." *Journal of Machine Learning Research*, 22(165), 1--73.](https://jmlr.org/papers/v22/20-410.html) -- Theoretical foundation for WeightWatcher's power-law alpha metric.

20. [Northcutt, C. G., Jiang, L., and Chuang, I. L. (2021). "Confident Learning: Estimating Uncertainty in Dataset Labels." *Journal of Artificial Intelligence Research*, 70, 1373--1411.](https://jair.org/index.php/jair/article/view/12125) -- Theoretical foundation for Cleanlab's label issue detection.

---

## Appendix A: Example Chrome Trace Analysis

After training with profiling enabled, retrieve the trace from MLflow:

```python
import mlflow

client = mlflow.MlflowClient()
artifacts = client.list_artifacts(run_id, path="profiling")
# Download chrome_trace_epoch_1.json
client.download_artifacts(run_id, "profiling/chrome_trace_epoch_1.json", "/tmp/traces/")
# Open chrome://tracing in Chrome browser, load the JSON file
```

The Chrome trace shows a timeline with labeled regions:

```
|-- data_loading --|-- forward --|-- loss_compute --|-- backward_optimizer --|
|      120ms       |    45ms     |      8ms         |        35ms           |
```

If `data_loading` dominates, investigate:
- `num_workers` too low (increase in `DataConfig`)
- Disk I/O bottleneck (check `iostat` in system_monitor)
- Cache rate too low for dataset size (increase MONAI CacheDataset rate)

## Appendix B: GPU Benchmark YAML Schema

```yaml
# gpu_benchmark.yaml -- generated by minivess-gpu-benchmark container
# Location: /benchmarks/gpu_benchmark.yaml (Docker volume mount)
# Cache validity: re-run if driver_version or cuda_version changes

schema_version: "1.0"
timestamp: "2026-03-13T14:30:00Z"

instance:
  provider: "runpod"         # runpod | aws | gcp | local | intranet
  hostname: "runpod-abc123"
  gpu_model: "NVIDIA A100-SXM4-40GB"
  gpu_count: 1
  vram_gb: 40.0
  cuda_version: "12.6"
  driver_version: "560.35.03"
  compute_capability: "8.0"
  pcie_bandwidth_gbps: 32.0  # measured via bandwidthTest

benchmarks:
  # Per-model micro-benchmarks (10 forward passes, batch=1)
  dynunet:
    peak_vram_mb: 3500
    throughput_img_per_sec: 12.4
    mean_forward_ms: 80.6
    std_forward_ms: 2.1
  sam3_vanilla:
    peak_vram_mb: 2900
    throughput_img_per_sec: 4.2
    mean_forward_ms: 238.0
    std_forward_ms: 5.3
  sam3_hybrid:
    peak_vram_mb: 7500
    throughput_img_per_sec: 2.1
    mean_forward_ms: 476.0
    std_forward_ms: 8.7

capabilities:
  # Binary feasibility (peak_vram < 85% of total vram)
  dynunet: true
  sam3_vanilla: true
  sam3_hybrid: true
  vesselfm: true
  max_batch_size:
    dynunet: 8
    sam3_vanilla: 4
    sam3_hybrid: 2
    vesselfm: 2
```

## Appendix C: Profiler Integration Code Sketch

This sketch illustrates the target state of `trainer.py:fit()` after Phase 1
implementation. It is not production code -- the actual implementation will follow
TDD methodology.

```python
# In SegmentationTrainer.fit() — conceptual sketch

from torch.profiler import profile, ProfilerActivity, schedule, record_function

def fit(self, train_loader, val_loader, *, fold_id=0, checkpoint_dir=None):
    profiling_cfg = self.config.profiling  # ProfilingConfig from Hydra

    # Build profiler schedule: active for first N epochs + final epoch
    prof_schedule = schedule(
        skip_first=0,
        wait=0,
        warmup=1,
        active=profiling_cfg.epochs - 1,
        repeat=1,
    ) if profiling_cfg.enabled else None

    activities = []
    if "cpu" in profiling_cfg.activities:
        activities.append(ProfilerActivity.CPU)
    if "cuda" in profiling_cfg.activities:
        activities.append(ProfilerActivity.CUDA)

    trace_handler = TraceHandler(
        tracker=self.tracker,
        checkpoint_dir=checkpoint_dir,
    )

    ctx = profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=trace_handler,
        profile_memory=profiling_cfg.profile_memory,
        record_shapes=profiling_cfg.record_shapes,
        with_flops=profiling_cfg.with_flops,
        with_stack=profiling_cfg.with_stack,
    ) if profiling_cfg.enabled else contextlib.nullcontext()

    with ctx as prof:
        for epoch in range(self.config.max_epochs):
            train_result = self.train_epoch(train_loader)
            # ... validation, checkpointing, etc. (existing code) ...

            if prof is not None:
                prof.step()

    # Log final profiling summary
    if profiling_cfg.enabled and self.tracker is not None:
        self.tracker.log_profiling_summary(trace_handler.summary)
```

```python
# In SegmentationTrainer.train_epoch() — record_function annotations

def train_epoch(self, loader):
    self.model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        with record_function("data_loading"):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            images = batch["image"]
            labels = batch["label"]

        self.optimizer.zero_grad()
        with autocast(...):
            with record_function("forward"):
                output = self.model(images)
            with record_function("loss_compute"):
                loss = self._compute_loss(output, batch, labels)

        with record_function("backward_optimizer"):
            self.scaler.scale(loss).backward()
            # ... gradient clipping, optimizer step ...

        running_loss += loss.item()
        num_batches += 1

    return EpochResult(loss=running_loss / max(num_batches, 1))
```

---

*This plan was produced for the MinIVess MLOps project. Implementation follows
TDD methodology as specified in `.claude/skills/self-learning-iterative-coder/SKILL.md`.*
