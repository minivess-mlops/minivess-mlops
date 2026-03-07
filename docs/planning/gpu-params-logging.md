# GPU Efficiency Logging Plan

**Date:** 2026-03-06
**Branch:** feat/flow-biostatistics
**Status:** Planning

---

## Motivation

The current monitor line:

```
[MONITOR] RAM: 35.5/62.7 GB (57%) | Swap: 6.3/16.0 GB | GPU: 5360/8192 MB | Process RSS: 8.8 GB
```

shows **only VRAM allocation** — a static number once the model is loaded. It tells us
nothing about whether the GPU is actually doing useful work. The critical signal for
training efficiency is **volatile GPU-Util (compute utilization %)**, which reveals
data-pipeline starvation, thermal throttling, and algorithmic dead time.

The investigation found:
- `scripts/system_monitor.py` **already collects** `gpu_utilization_percent` and
  `gpu_temperature_c` via `nvidia-smi` subprocess — but neither appears in the console log.
- `gpu_memory_bandwidth_utilization` (Mem-BW%), `power_w`, and `sm_clock_mhz` (throttle
  detection) are **not collected at all**.
- Console status is printed every 60 s and mixes static (VRAM) with dynamic (GPU util)
  at the same rate — the wrong cadence for each.
- The CSV captures everything but is never uploaded to MLflow as an artifact.
- `mlflow.start_run(log_system_metrics=True)` is already called, which logs 12 MLflow
  system metrics at 10 s intervals — but at epoch granularity we have no GPU efficiency
  summary in the run.

---

## Key Concepts: What Each Metric Means

| Metric | What it measures | Healthy | Pathological |
|--------|-----------------|---------|--------------|
| **GPU-Util %** (`utilization.gpu`) | % of time ≥1 kernel was running | > 90 % sustained | < 60 %, oscillating → data starvation |
| **Mem-BW-Util %** (`utilization.memory`) | % of time memory interface was busy | Varies by model | Low + high GPU-Util → compute-bound (fine) |
| **VRAM used MB** | Allocated device memory | 80–95 % of total | > 95 % → OOM risk |
| **Temperature °C** | Die temperature | < 83 °C | > 87 °C → throttle imminent |
| **SM clock MHz** | Current streaming-multiprocessor clock | At rated base clock | Below base → thermal throttle (silent FLOPS drop) |
| **Power W** | Board power draw | Near TDP under load | Well below TDP with low util → GPU starved |
| **CPU % (all cores avg)** | Processor load | < 80 % | > 90 % with low GPU-Util → DataLoader bottleneck |

**The single most important diagnostic combination:**
- `gpu_util_pct < 60 AND cpu_percent > 85` → DataLoader / augmentation bottleneck
- `gpu_util_pct ≈ 100 AND sm_clock < base_clock * 0.92` → thermal throttle

---

## Current State (What Already Works)

| Component | Status |
|-----------|--------|
| `ResourceSnapshot` dataclass — all fields exist | ✅ collected |
| `gpu_utilization_percent` collected via nvidia-smi | ✅ collected, ❌ not in console log |
| `gpu_temperature_c` collected via nvidia-smi | ✅ collected, ❌ not in console log |
| `cpu_percent` collected | ✅ collected, ❌ not prominently shown |
| `gpu_memory_bandwidth_util` (Mem-BW%) | ❌ not collected |
| `gpu_power_w` | ❌ not collected |
| `gpu_sm_clock_mhz` | ❌ not collected |
| CSV written to `logs/monitor/system_metrics.csv` | ✅ written |
| CSV uploaded to MLflow as artifact | ❌ not uploaded |
| Per-epoch GPU efficiency summary in MLflow | ❌ not logged |
| Data pipeline uses `nvidia-smi` subprocess (~100 ms/call) | ⚠️ slow; ok for 5 s polling |

---

## Planned Changes

### 1. Add Missing GPU Efficiency Fields to `ResourceSnapshot`

**File:** `scripts/system_monitor.py`

Add to `ResourceSnapshot` dataclass:

```python
gpu_mem_bw_util_pct: float | None = None   # memory interface busy time %
gpu_power_w: float | None = None           # board power draw (Watts)
gpu_sm_clock_mhz: int | None = None        # SM clock — drop = thermal throttle
```

Collect via extended `nvidia-smi --query-gpu` call:

```
--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm,...
```

These four fields are available on all NVIDIA GPUs that support NVML (all Maxwell+).

> **pynvml vs nvidia-smi subprocess:** pynvml is faster (no fork, direct NVML binding)
> and is the correct choice for sub-second polling. At our current 5–10 s interval the
> subprocess overhead is negligible (~100 ms vs 5000 ms). Migrate to pynvml in a
> follow-up if we need < 1 s resolution. For now extend the nvidia-smi query string.

---

### 2. Two-Tier Console Log (Different Cadences)

Replace the single 60 s console line with two distinct lines at different rates.

**Tier A — GPU Efficiency line (every epoch, printed after the loss line)**

Printed by the trainer at the end of each epoch when the monitor provides a snapshot.
Two design constraints must be satisfied simultaneously:

1. **Human-readable at a glance** — a developer scanning a terminal should immediately
   see whether the GPU is working efficiently.
2. **LLM/machine-parseable** — log analysis tools, LLM summarisers, and regex parsers
   must be able to extract each field without ambiguity.

**Format specification:**

```
f #1/3: Epoch 65/100 — train_loss=0.2061 val_loss=0.2327 lr=2.99e-05
f #1/3: [GPU]  util=87%  bw=45%  temp=71C  pwr=185W | cpu=34% | vram=5360/8192MB  clk=1980MHz  [OK]
```

**Design decisions:**

| Choice | Reason |
|--------|--------|
| `[GPU]` type tag at start of second line | Log parsers can `grep '\[GPU\]'` to extract GPU lines without ambiguity |
| `key=value` pairs for every field | Parseable as `dict(item.split('=') for item in ...)` — no positional parsing |
| `%` kept in values (`util=87%`) | Human-readable; machine strips `%` trivially with `int(v.rstrip('%'))` |
| `C` not `°C` | ASCII only — `°` breaks some log forwarders and unicode-unaware parsers |
| `vram=5360/8192MB` — fraction not split | Human compact form; for MLflow the two values are logged separately (`vram_used_mb`, `vram_total_mb`) |
| `clk=1980MHz` | SM clock for throttle detection — if it drops > 8% from baseline, status changes |
| `[OK]` / `[STARVE]` / `[THROTTLE]` suffix | Single token status — greppable, scannable; LLM can extract status field without parsing all metrics |
| `\|` as group separator | Consistent with existing `[MONITOR]` line; breaks the line into 3 visual zones: GPU / CPU / memory |
| No unicode arrows `↑↓→` | Replaced by `[OK]`/`[STARVE]`/`[THROTTLE]` — deterministic, ASCII, no rendering dependency |

**Status token logic:**

```python
if gpu_util < 60 and cpu_pct > 85:
    status = "[STARVE]"   # DataLoader is the bottleneck
elif temp_c > 87 or sm_clock < base_clock * 0.92:
    status = "[THROTTLE]" # Thermal throttle, FLOPS degraded
else:
    status = "[OK]"
```

**Tier B — Slow resource line (every 10 epochs)**

Replaces the current 60 s `[MONITOR]` line. Focus on the static/slow-changing values:

```
f #1/3: [MEM]  ram=35.5/62.7GB(57%)  swap=6.3/16.0GB  vram=5360/8192MB  rss=8.8GB
```

Same `[TYPE] key=value` format as the GPU line — parseable with the same regex.

**Full epoch output (two lines):**

```
2026-03-06 19:45:27 trainer INFO: f #1/3: Epoch 65/100 — train_loss=0.2061 val_loss=0.2327 lr=2.99e-05
2026-03-06 19:45:27 trainer INFO: f #1/3: [GPU]  util=87%  bw=45%  temp=71C  pwr=185W | cpu=34% | vram=5360/8192MB  clk=1980MHz  [OK]
```

Every 10th epoch, a third line:

```
2026-03-06 19:45:27 monitor INFO: f #1/3: [MEM]  ram=35.5/62.7GB(57%)  swap=6.3/16.0GB  vram=5360/8192MB  rss=8.8GB
```

**Starvation/throttle warnings** (printed once when threshold first crossed):

```
2026-03-06 19:45:27 monitor WARNING: [STARVE] GPU-util=38% cpu=92% — DataLoader bottleneck. Try: num_workers up, pin_memory=True, persistent_workers=True, cache_rate up
2026-03-06 19:45:27 monitor WARNING: [THROTTLE] temp=88C sm_clock=1695MHz (base=1980MHz, drop=14%) — GPU FLOPS silently degraded. Check cooling.
```

**Machine-parseable output — JSONL, not regex:**

Regex is **banned** in this repo (CLAUDE.md rule #16,
`.claude/metalearning/2026-03-06-regex-ban.md`). The human-readable `[GPU]` line is for
terminals only. For log analysis, the `SystemMonitor` emits a **JSONL** (JSON Lines)
file alongside the CSV, one JSON object per snapshot:

```python
# system_monitor.py — emit alongside CSV
import json, dataclasses
snapshot_dict = dataclasses.asdict(snapshot)
jsonl_file.write(json.dumps(snapshot_dict) + "\n")
```

```python
# Consumer — no regex, just json.loads()
for line in Path("system_metrics.jsonl").read_text().splitlines():
    snap = json.loads(line)
    print(snap["gpu_utilization_percent"], snap["gpu_temperature_c"])
```

The JSONL file is uploaded to MLflow as `system_monitor/system_metrics.jsonl`.
The `ResourceSnapshot` dataclass is the schema — adding a field automatically
appears in the JSONL without any format string changes.

This requires passing the latest `ResourceSnapshot` into the trainer's epoch callback.
Add an optional `SystemMonitor` reference to `SegmentationTrainer.__init__`.

**Automatic bottleneck warning (printed once when threshold crossed):**

```
[STARVE] GPU-util=38% cpu=92% — DataLoader bottleneck. Try: num_workers up, pin_memory=True, persistent_workers=True, cache_rate up
[THROTTLE] temp=88C sm_clock=1695MHz (base=1980MHz, drop=14%) — GPU FLOPS silently degraded. Check cooling.
```

---

### 3. Per-Epoch GPU Summary → MLflow Metrics

At each epoch, compute a window average of GPU snapshots taken during that epoch and
log to MLflow. This gives the GPU efficiency timeline alongside the loss curves.

**New MLflow metric keys per epoch:**

| Key | Value | Description |
|-----|-------|-------------|
| `gpu_util_pct_mean` | float | Mean volatile GPU-Util over epoch |
| `gpu_util_pct_min` | float | Min (catches starvation dips) |
| `gpu_temp_c_max` | float | Max temperature (catches throttle) |
| `gpu_sm_clock_mhz_min` | int | Min SM clock (throttle detection) |
| `gpu_power_w_mean` | float | Mean power draw |
| `cpu_pct_mean` | float | Mean CPU utilization |

These are **metrics** (stepped by epoch), not params. They go into the MLflow metrics
store (PostgreSQL in production, `mlruns/` in dev) alongside `train_loss`, `val_loss`.

**Implementation approach:**

The `SystemMonitor` background thread continuously collects snapshots into a ring buffer.
The trainer calls `monitor.epoch_summary()` at the end of each epoch which drains and
aggregates the buffer, returning `dict[str, float]`. The trainer logs this dict via
`tracker.log_epoch_metrics()`.

No new MLflow API calls needed — reuses the existing `log_epoch_metrics` path.

---

### 4. Upload CSV as MLflow Artifact at Run End

**File:** `scripts/train_monitored.py`

After `monitor.stop()`, log the CSV as a MLflow artifact:

```python
summary = monitor.stop()
if monitor.csv_path.exists():
    tracker.log_artifact(monitor.csv_path, artifact_path="system_monitor")
```

This stores the full time-series (every 5–10 s over the full run) in the MLflow artifact
store (MinIO S3 in production, `mlruns/artifacts/` in dev). Useful for forensic analysis
of starvation patterns.

**PostgreSQL vs artifact store:**
- **Epoch-level aggregates** (6 keys listed above) → MLflow metrics store (PostgreSQL)
  — indexed, queryable via MLflow API, appear in the UI as charts
- **Full time-series CSV** → MLflow artifact store (MinIO S3)
  — not queryable but downloadable; the raw evidence

---

### 5. Console Log Format Finalization

**Proposed epoch output (in training terminal):**

```
2026-03-06 19:45:27 trainer INFO: f #1/3: Epoch 65/100 — train_loss=0.2061 val_loss=0.2327 lr=2.99e-05
2026-03-06 19:45:27 trainer INFO: f #1/3: [GPU]  util=87%  bw=45%  temp=71C  pwr=185W | cpu=34% | vram=5360/8192MB  clk=1980MHz  [OK]
```

Status token (`[OK]` / `[STARVE]` / `[THROTTLE]`) replaces unicode arrows.
Every 10th epoch also prints the slow resource line:

```
2026-03-06 19:45:27 monitor INFO: f #1/3: [MEM]  ram=35.5/62.7GB(57%)  swap=6.3/16.0GB  vram=5360/8192MB  rss=8.8GB
```

---

## Known Debt in train_monitored.py (found during planning, 2026-03-06)

| Issue | Severity | Notes |
|-------|----------|-------|
| `/tmp` checkpoint dirs never `shutil.rmtree()`'d | Medium | Resource leak; checkpoints ARE uploaded to MLflow first, so not lost |
| `train_flow.py` Prefect training flow is a stub | High | Returns placeholder string; CLAUDE.md mandates Prefect as required |
| `tracker.log_pyfunc_model()` never called | Low | Checkpoints logged as raw `.pth` artifacts; fine for now, needed for Model Registry |

The checkpoints in `/tmp` are **not the canonical copy** — they are uploaded to
`mlruns/artifacts/checkpoints/` during training. Losing the `/tmp` dir is fine.
The Prefect stub is the real gap to close in a follow-up.

---

## Implementation Tasks (TDD order)

| # | Task | File(s) | Size |
|---|------|---------|------|
| T1 | Add `gpu_mem_bw_util_pct`, `gpu_power_w`, `gpu_sm_clock_mhz` to `ResourceSnapshot` | `system_monitor.py` | S |
| T2 | Extend `nvidia-smi` query to collect new fields | `system_monitor.py` | S |
| T3 | Add `epoch_summary()` method to `SystemMonitor` (drain ring buffer, return agg dict) | `system_monitor.py` | M |
| T4 | Add GPU efficiency `[GPU]` line to trainer epoch log (`system_monitor` ref optional) | `trainer.py` | S |
| T5 | Log epoch GPU summary to MLflow via `log_epoch_metrics` | `trainer.py` | S |
| T6 | Print `[STARVE]` warning when `gpu_util < 60 AND cpu > 85` | `system_monitor.py` | S |
| T7 | Print `[THROTTLE]` warning when `temp > 87 OR sm_clock < base * 0.92` | `system_monitor.py` | S |
| T8 | Upload CSV artifact to MLflow at run end | `train_monitored.py` | S |
| T9 | `[MEM]` slow resource line every 10 epochs (remove wall-clock timer) | `system_monitor.py` + `trainer.py` | S |
| T10 | `shutil.rmtree(checkpoint_dir.parent)` after fold completes | `train_monitored.py` | XS |
| T11 | Unit tests for `epoch_summary()`, new fields, warning logic, log format | `tests/v2/unit/test_system_monitor.py` | M |

---

## What NOT to Do

- **No pynvml migration** in this iteration — the subprocess overhead is fine at 5–10 s
  polling. Migrate only if we need sub-second resolution (e.g., per-batch logging).
- **No `torch.profiler`** integration here — that's a separate deep-profiling story for
  when a specific bottleneck needs kernel-level diagnosis.
- **No MONAI WorkflowProfiler** — adds complexity, marginal gain over the pynvml path
  for the DevEx goal of "GPU is idle → tell me why".
- **Do not log the CSV to PostgreSQL** — it's a blob (100 KB per fold), belongs in the
  artifact store, not the relational backend.

---

## Out-of-Scope (Future Work)

- Per-batch GPU utilization (requires `torch.profiler` or CUPTI)
- Multi-GPU monitoring (single-GPU assumed for MiniVess workloads)
- DALI / streaming dataset integration for DataLoader bottleneck resolution
- NVTX annotations for Nsight Systems profiling
- Automatic `num_workers` tuning when starvation is detected

---

## References

- nvidia-smi query fields: `nvidia-smi --help-query-gpu`
- pynvml API: `pip show nvidia-ml-py` → same NVML C library bindings as nvidia-smi
- MONAI profiling utils: `monai.utils.profiling.{PerfContext,WorkflowProfiler}`
- HuggingFace GPU training guide: https://huggingface.co/docs/transformers/v4.23.0/en/perf_train_gpu_one
- GPU-Util vs Mem-BW-Util explained: https://discuss.pytorch.org/t/volatile-gpu-util-0-with-high-memory-usage/52047
- DataLoader bottleneck diagnosis: https://stackoverflow.com/questions/47298447/how-to-fix-low-volatile-gpu-util-with-tensorflow-gpu-and-keras
