# RAM Crash Investigation Report: Biostatistics Test Memory Exhaustion

**Date**: 2026-03-21
**Branch**: `test/debug-factorial-run`
**Severity**: P0 — System unresponsive, all 67.3 GB RAM + 17.2 GB swap exhausted
**Blocker**: Cannot execute XML plans until resolved

## 1. Incident Summary

Running biostatistics flow tests consumed 66.8 GB (99.3%) of system RAM plus
exhausted all 17.2 GB swap. The system became completely unresponsive. Screenshot
evidence shows all 16 CPU cores at 50-90% utilization, memory at 99.3%, and
swap at 100%.

**Sequence of events (from screenshot)**:
1. `rm -rf ~/Dropbox/github-personal/minivess-mlops/file:*` — MLflow URI pollution cleanup
2. `make test-staging 2>&1 | tail -10` — ran with timeout 5m
3. `MINIVESS_ALLOW_HOST=1 uv run pytest tests/v2/unit/test_biostatistics_flow.py::TestFlowRunsWithMockData::test_flow_runs_with_mock_data` — running 1m 48s when memory maxed
4. System became unresponsive — terminal crashed

## 2. Multi-Hypothesis Root Cause Analysis

### H1: Missing Task Mocks in `test_flow_runs_with_mock_data`

**Probability: HIGH (70%)**

The test decorates 13 `@patch` calls but the flow has 16 tasks. Three tasks
run with **real implementations** against mock data:

| Task | Mocked? | Impact |
|------|---------|--------|
| `task_discover_source_runs` | YES | — |
| `task_validate_source_completeness` | YES | — |
| `task_build_duckdb` | YES | — |
| `task_compute_pairwise` | YES | — |
| `task_compute_bayesian` | YES | — |
| `task_compute_variance` | YES | — |
| `task_compute_rankings` | YES | — |
| `task_generate_figures` | YES | — |
| `task_generate_tables` | YES | — |
| `task_build_lineage` | YES | — |
| `task_log_mlflow` | YES | — |
| `_load_config` | YES | — |
| `_build_per_volume_data` | YES | — |
| **`task_compute_specification_curve`** | **NO** | Permutation loop |
| **`task_compute_rank_concordance`** | **NO** | Cross-metric computation |
| **`_resolve_factor_names`** | **NO** | Parses MagicMock as YAML path |

**Critical detail**: `mock_config = MagicMock()` means `config.factorial_yaml`
is a truthy MagicMock (not None), causing `_resolve_factor_names` to attempt
`parse_factorial_yaml(MagicMock())`. This is caught by try/except, but the
import of `factorial_config` may trigger cascading imports.

**More critically**: With `per_volume_data = {}`, the unmocked tasks should
return quickly (empty data → no iterations). BUT `MagicMock()` arithmetic
behavior is unpredictable — `MagicMock() // 20` returns a MagicMock, and
`range(MagicMock)` can trigger `__index__` which may return a huge number
or loop indefinitely depending on Python version.

**Counter-evidence**: `config.n_bootstrap` IS explicitly set to `100` in the
test (line 70), so `100 // 20 = 5` permutations. However, `config.metrics`
is set to `["val_dice"]` and `per_volume_data = {}`, so the permutation loop
skips entirely (`specs` is empty).

### H2: Zombie Processes from `make test-staging` Timeout

**Probability: MEDIUM (50%)**

`make test-staging` was run with a 5-minute timeout BEFORE the specific test.
If pytest spawns worker processes (via `pytest-xdist` or subprocess-based
fixtures) and the timeout kills only the parent, child processes continue
consuming memory. The subsequent test then runs on a system with 50%+
memory already consumed.

**Evidence**: The screenshot shows `make test-staging` output truncated,
suggesting it was interrupted. The specific test was then started immediately.

### H3: Specification Curve Permutation Memory Leak (Structural)

**Probability: MEDIUM (40%) — for production runs, not this specific test**

The `_permutation_test()` function in `biostatistics_specification_curve.py`
(lines 284-358) has structural memory issues:

```
Per permutation:
  - Creates new permuted_data dict (old one NOT explicitly deleted)
  - Creates all_scores list per metric
  - Creates shuffled numpy array per metric
  - Creates Specification objects per pair × metric × aggregation
  - No gc.collect() between iterations
```

**Production-scale memory projection** (24 conditions, 8 metrics, 500 permutations):
- C(24,2) = 276 pairs × 8 metrics × 2 aggregations = 4,416 specifications/permutation
- 500 permutations × 4,416 = 2,208,000 Specification objects
- Each ~120 bytes → 265 MB minimum
- With Python GC delays + fragmentation: 1-5 GB
- In Prefect context: potentially 10-30 GB

### H4: DuckDB Memory Pool from Module Import

**Probability: LOW (15%)**

DuckDB allocates an internal memory pool on import. The biostatistics flow
imports `biostatistics_duckdb` at module level. If multiple test processes
import DuckDB, each gets its own default memory pool (typically 75% of
system RAM). Multiple concurrent DuckDB pools could consume 40+ GB.

### H5: Large conftest Fixtures or PyTorch Model Loading

**Probability: LOW (10%)**

The `tests/v2/unit/conftest.py` only defines a `MockMamba` PyTorch module
(lightweight). No large data fixtures. PyTorch itself allocates CUDA context
on first use, but this test doesn't use GPU operations.

### H6: Prefect Worker Memory Overhead

**Probability: LOW (10%)**

Running `run_biostatistics_flow.fn()` with `.fn()` bypasses Prefect orchestration
but still imports the full Prefect runtime. If Prefect pre-allocates connection
pools, task result caches, or logging buffers, this could add 1-2 GB overhead
but not 60+ GB.

## 3. Foundation-PLR Precedent Analysis

The foundation-PLR project experienced the **exact same failure pattern**:
62 GB RAM crash during MLflow-to-DuckDB extraction. Their root cause and
solutions are directly applicable.

### Root Cause (foundation-PLR)
- **Batch accumulation**: 410 MLflow runs accumulated in memory before writing to DuckDB
- **Location**: `extraction_flow.py` lines 210-243
- **Impact**: 6-24 hour stalled extractions with zero progress visibility

### Solutions Implemented (foundation-PLR)

| Solution | Description | File |
|----------|-------------|------|
| **MemoryMonitor** | `psutil.Process().memory_info().rss` — warn at 12 GB, critical at 14 GB | `streaming_duckdb_export.py:217-284` |
| **Per-run streaming** | Load → Extract → Write → Delete → GC per run | `streaming_duckdb_export.py:534-677` |
| **CheckpointManager** | DuckDB `extraction_checkpoints` table for crash resume | `streaming_duckdb_export.py:291-400` |
| **Explicit cleanup** | `del metrics_data; gc.collect()` after each run | `streaming_duckdb_export.py:670-677` |
| **Minimal test fixtures** | 5 runs × 5 samples (not production-scale) | `test_streaming_duckdb_export.py` |
| **Normalized schema** | Child tables with FK (not denormalized) — 9 tables | `streaming_duckdb_export.py` |
| **Progress logging** | Heartbeat every 60s + progress tracker | `streaming_duckdb_export.py` |

### Key Insight
DuckDB itself was NOT the memory culprit — it was Python-level accumulation
of intermediate data structures BEFORE writing to DuckDB. The fix is always:
**write immediately per item, never batch**.

## 4. Decision Matrix: Fix Strategies

### Immediate Fixes (unblock XML plan execution)

| Fix | Effort | Impact | Risk | Priority |
|-----|--------|--------|------|----------|
| **F1: Add missing mocks** to `test_flow_runs_with_mock_data` | 15 min | Eliminates unmocked real code paths | None | **P0** |
| **F2: Replace `MagicMock()` config** with real `BiostatisticsConfig()` | 30 min | Eliminates MagicMock arithmetic surprises | Low | **P0** |
| **F3: Add memory limit to pytest** via `resource.setrlimit(RLIMIT_AS, 8GB)` in conftest | 20 min | Hard cap prevents system crash | May cause test failures | **P1** |

### Structural Fixes (prevent recurrence)

| Fix | Effort | Impact | Risk | Priority |
|-----|--------|--------|------|----------|
| **F4: Add `gc.collect()` to permutation loop** every 50 iterations | 10 min | Prevents GC delay accumulation | Slight slowdown | **P1** |
| **F5: Implement MemoryMonitor** (port from foundation-PLR) | 2 hr | Process-level memory guardrails | New dependency (psutil) | **P2** |
| **F6: Stream Specification objects** to DuckDB/Parquet instead of list | 4 hr | Eliminates in-memory accumulation | Architecture change | **P2** |
| **F7: Cap `n_bootstrap` default** to 2,000 (from 10,000) | 5 min | Reduces permutation count 5x | May affect statistical power | **P2** |
| **F8: Generator-based permutation test** (yield instead of list) | 2 hr | O(1) memory per permutation | Requires refactoring | **P2** |

### Testing Infrastructure Fixes

| Fix | Effort | Impact | Risk | Priority |
|-----|--------|--------|------|----------|
| **F9: Add mock count assertion** to conftest | 30 min | Catches missing mocks automatically | Maintenance overhead | **P1** |
| **F10: Add `pytest-timeout` per-test** (300s max) | 10 min | Prevents infinite loops | May kill slow tests | **P1** |
| **F11: Add `pytest-memray` profiling** marker for staging tests | 1 hr | Catch memory issues before they crash | New dev dependency | **P2** |
| **F12: Process isolation** for heavy tests via `pytest-forked` | 30 min | Memory freed per test | Slower test execution | **P2** |

### Design-Level Fixes

| Fix | Effort | Impact | Risk | Priority |
|-----|--------|--------|------|----------|
| **F13: 32 GB laptop benchmark** — add CI-local memory budget | 2 hr | All tests run on modest hardware | Need to profile current usage | **P1** |
| **F14: Tiered test data sizes** — `tiny`/`small`/`full` fixtures | 3 hr | Right-size data per test tier | More fixture maintenance | **P2** |
| **F15: DuckDB `PRAGMA memory_limit`** in all DuckDB connections | 15 min | Cap DuckDB's internal allocations | May cause DuckDB OOM errors | **P2** |

## 5. Recommended Fix Order

### Phase 1: Unblock (today, ~45 min)
1. **F1**: Add missing mocks for `task_compute_specification_curve`,
   `task_compute_rank_concordance`, and `_resolve_factor_names`
2. **F2**: Replace `MagicMock()` config with real `BiostatisticsConfig()`
   instance using test-appropriate defaults
3. **F10**: Add `pytest-timeout=300` to staging tier

### Phase 2: Guardrails (this week, ~3 hr)
4. **F4**: Add periodic `gc.collect()` to `_permutation_test()`
5. **F3**: Add `resource.setrlimit(RLIMIT_AS, 8*1024*1024*1024)` to
   test conftest (soft limit 8 GB)
6. **F9**: Add mock count validation to biostatistics flow tests
7. **F13**: Establish 32 GB memory budget for `make test-staging`

### Phase 3: Architecture (next sprint)
8. **F5**: Port MemoryMonitor from foundation-PLR
9. **F8**: Refactor permutation test to use generators
10. **F6**: Stream Specification objects to DuckDB instead of in-memory list

## 6. Specification Curve Memory Model

### Current Architecture (In-Memory)

```
compute_specification_curve()
├── Generate ALL specifications → list[Specification]  ← ALL IN MEMORY
├── Apply BH-FDR correction → new list[Specification]  ← DOUBLES MEMORY
├── Sort by effect size                                 ← IN PLACE (OK)
└── _permutation_test()
    └── for _ in range(n_permutations):                 ← LOOP
        ├── permuted_data = {}                          ← NEW DICT EACH ITER
        ├── pool + shuffle per metric                   ← ARRAYS
        ├── compute ALL specs for permuted data         ← SPECS IN MEMORY
        └── compute median                              ← SINGLE FLOAT
        # NO del, NO gc.collect()                       ← LEAK
```

### Target Architecture (Streaming)

```
compute_specification_curve()
├── Generate specs → yield Specification                ← GENERATOR
├── Write to DuckDB/Parquet on disk                     ← STREAMING WRITE
├── BH-FDR correction from disk                         ← SQL QUERY
└── _permutation_test()
    └── for i in range(n_permutations):
        ├── permuted_data = _shuffle_in_place()         ← REUSE ARRAYS
        ├── perm_median = _compute_median_effect()      ← SINGLE FLOAT
        ├── if i % 50 == 0: gc.collect()                ← PERIODIC GC
        └── del permuted_data                           ← EXPLICIT CLEANUP
```

### Memory Projections

| Scenario | Conditions | Metrics | Permutations | Specs | Current RAM | Target RAM |
|----------|-----------|---------|-------------|-------|-------------|------------|
| **Unit test (empty)** | 0 | 1 | 5 | 0 | ~50 MB | ~50 MB |
| **Unit test (small)** | 4 | 1 | 100 | 600 | ~100 MB | ~60 MB |
| **Debug run** | 24 | 8 | 100 | 441,600 | ~2 GB | ~200 MB |
| **Production** | 24 | 8 | 500 | 2,208,000 | **~5-30 GB** | ~500 MB |
| **Full factorial** | 720 | 8 | 500 | ~2B | **OOM** | ~2 GB |

The full factorial (720 conditions) is **impossible with current architecture**:
C(720,2) = 259,080 pairs × 8 metrics × 2 agg = 4,145,280 specs/permutation.
500 permutations = 2,072,640,000 total specs. This MUST use streaming.

## 7. Foundation-PLR Code to Port

### MemoryMonitor (minimal port)

```python
# Port from: foundation_PLR/src/data_io/streaming_duckdb_export.py:217-284
import gc
import os
from dataclasses import dataclass

@dataclass
class MemoryMonitor:
    warning_threshold_gb: float = 4.0   # Conservative for 32 GB target
    critical_threshold_gb: float = 6.0

    def check(self) -> tuple[float, str]:
        try:
            import psutil
            mem_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        except ImportError:
            return 0.0, "unknown"

        if mem_gb >= self.critical_threshold_gb:
            return mem_gb, "critical"
        elif mem_gb >= self.warning_threshold_gb:
            return mem_gb, "warning"
        return mem_gb, "ok"

    def enforce(self) -> None:
        mem_gb, status = self.check()
        if status == "critical":
            gc.collect()
            mem_after, _ = self.check()
            logger.warning(
                "Memory critical: %.1f GB → %.1f GB after GC", mem_gb, mem_after
            )
```

### Periodic GC in Permutation Loop

```python
def _permutation_test(...) -> float:
    monitor = MemoryMonitor()
    for i in range(n_permutations):
        # ... existing code ...

        # Periodic cleanup
        if i % 50 == 0:
            del permuted_data  # Explicit cleanup
            monitor.enforce()
            if i % 100 == 0:
                logger.debug("Permutation %d/%d, RSS=%.1f GB", i, n_permutations, monitor.check()[0])

    return (n_extreme + 1) / (n_permutations + 1)
```

## 8. Test Fix: Complete Mock Coverage

```python
# CURRENT (missing 3 mocks):
@patch("...task_log_mlflow")
@patch("...task_build_lineage")
# ... 11 more ...
@patch("..._build_per_volume_data")
def test_flow_runs_with_mock_data(self, ...):

# FIXED (all tasks mocked):
@patch("...task_log_mlflow")
@patch("...task_build_lineage")
@patch("...task_generate_tables")
@patch("...task_generate_figures")
@patch("...task_compute_rank_concordance")   # <-- ADDED
@patch("...task_compute_specification_curve") # <-- ADDED
@patch("...task_compute_rankings")
@patch("...task_compute_variance")
@patch("...task_compute_pairwise")
@patch("...task_compute_bayesian")
@patch("...task_build_duckdb")
@patch("...task_validate_source_completeness")
@patch("...task_discover_source_runs")
@patch("..._resolve_factor_names")           # <-- ADDED
@patch("..._load_config")
@patch("..._build_per_volume_data")
def test_flow_runs_with_mock_data(self, ...):
    # Use real config instead of MagicMock
    from minivess.config.biostatistics_config import BiostatisticsConfig
    config = BiostatisticsConfig(
        mlruns_dir=tmp_path / "mlruns",
        output_dir=tmp_path / "output",
        experiment_names=["test"],
        metrics=["val_dice"],
        primary_metric="val_dice",
        n_bootstrap=100,  # Small for tests
    )
    mock_load_config.return_value = config
```

## 9. Verification Plan

After applying fixes:

```bash
# 1. Verify the specific test passes quickly
time MINIVESS_ALLOW_HOST=1 uv run pytest \
  tests/v2/unit/test_biostatistics_flow.py::TestFlowRunsWithMockData \
  -v --timeout=60

# 2. Monitor memory during staging tests
/usr/bin/time -v make test-staging 2>&1 | grep "Maximum resident"

# 3. Profile biostatistics specification curve tests
MINIVESS_ALLOW_HOST=1 uv run pytest \
  tests/v2/unit/biostatistics/test_specification_curve.py \
  -v --timeout=120

# 4. Full staging suite with memory cap (target: <8 GB peak)
ulimit -v 8388608 && make test-staging
```

## 10. Implementation Record (TDD Execution Log)

**Date**: 2026-03-21
**Methodology**: Self-learning iterative coder TDD skill (RED→GREEN→VERIFY→FIX→CHECKPOINT)
**Result**: All 3 phases implemented. Full staging suite: **5498 passed, 3 skipped, 0 failed** (4m 23s)

### What Was Implemented

| Fix | Status | What Changed | Verification |
|-----|--------|-------------|--------------|
| **F1**: Missing mocks | DONE | Added `@patch` for `task_compute_specification_curve`, `task_compute_rank_concordance`, `_resolve_factor_names` | Test runs in 13s (was crashing at 62 GB) |
| **F2**: MagicMock config | DONE | Replaced `MagicMock()` with real `BiostatisticsConfig(metrics=["dsc"], n_bootstrap=100)`. Added real `SpecificationCurveResult` and `RankConcordanceResult` return values. | Eliminates MagicMock arithmetic surprises |
| **F3**: Memory cap | DONE | `resource.setrlimit(RLIMIT_AS, 32 GB)` in `tests/v2/unit/conftest.py` (Linux only) | Prevents system crash; 32 GB allows full suite to pass |
| **F4**: Periodic GC | DONE | Pre-allocated shuffle buffers (reused in-place), `del permuted_data` + `monitor.enforce()` every 50 iterations | All 11 spec curve tests pass |
| **F5**: MemoryMonitor | DONE | New `src/minivess/observability/memory_monitor.py` — 3-tier (ok/warning/critical), psutil RSS, fallback if psutil missing | 9 dedicated tests pass |
| **F8**: In-place shuffling | DONE | Pre-compute `metric_structures` dict once, `rng.shuffle(buffer)` in-place each iteration — zero new array allocations per permutation | Statistical correctness preserved |
| **F9**: Mock count validation | DONE | AST-based test introspects flow module for `@task` decorators and `@patch` targets, asserts full coverage | Catches regressions automatically |

### Key Decisions Made During Implementation

1. **32 GB vs 8 GB memory cap**: Started with 8 GB but Prefect + PyTorch + numpy imports
   alone consume ~6 GB. After 2800+ tests, matplotlib figure creation pushes past 16 GB
   due to RLIMIT_AS counting virtual (not resident) memory. Settled on 32 GB — still
   prevents the 62 GB crash while allowing the full suite to pass.

2. **Pre-allocated shuffle buffers**: Instead of the report's "generator-based" approach,
   the actual fix pre-computes `metric_structures = {metric: (np.array, structure)}` once
   and shuffles the buffer in-place each iteration. This is simpler and achieves O(1)
   extra allocation per permutation — the buffer is reused, not recreated.

3. **MemoryMonitor integrated into permutation loop**: Rather than creating a separate
   streaming architecture (F6 from the original plan), the MemoryMonitor is invoked
   every 50 iterations inside `_permutation_test()`. This is simpler than full streaming
   and sufficient for the 24-condition debug/production scenarios. Full streaming to
   DuckDB is only needed if we ever run the 720-condition full factorial spec curve.

4. **AST-based mock validation (not regex)**: The mock count test uses `ast.parse()` to
   find `@task` decorators and `@patch` targets — following Rule #16 (regex banned for
   structured data). It checks only callables actually invoked in `run_biostatistics_flow`,
   not all task functions defined in the module (e.g., `narrate_figures` is defined but
   not called in the flow).

5. **Real config instead of MagicMock**: The test now uses `BiostatisticsConfig(metrics=["dsc"])`
   instead of `MagicMock()`. This eliminates the entire class of bugs where MagicMock
   attributes return other MagicMocks that propagate through arithmetic operations.

### Reproducing This Approach in a New Repo

This investigation + fix pattern is generalizable to any repo with Prefect flows or
other orchestration frameworks. Here's the step-by-step recipe:

**Step 1: Detect the Problem**
- System monitor shows RSS approaching total RAM during test execution
- Tests that pass in isolation fail in full suite (memory accumulation)
- pytest `-x` stops at different tests each run (non-deterministic OOM)

**Step 2: Investigate Root Cause**
```bash
# Profile memory of the suspect test
/usr/bin/time -v uv run pytest tests/path/to/suspect_test.py 2>&1 | grep "Maximum resident"

# Check which functions are NOT mocked
python -c "
import ast
tree = ast.parse(open('src/flow.py').read())
tasks = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
         for d in n.decorator_list if isinstance(d, ast.Name) and d.id == 'task'}
print(f'Tasks in flow: {tasks}')
"
```

**Step 3: Fix (in order)**
1. **Mock all tasks** in flow tests — count `@task` decorators vs `@patch` decorators
2. **Replace `MagicMock()` config** with real Pydantic model instances
3. **Add `resource.setrlimit`** to conftest (Linux: RLIMIT_AS; macOS: RLIMIT_RSS)
4. **Add MemoryMonitor** to any loop that creates >100 intermediate objects
5. **Pre-allocate buffers** for shuffle/permutation operations
6. **Add AST-based mock count test** to prevent regressions

**Step 4: Validate**
```bash
# Single test: should complete in <30s
time uv run pytest tests/path/to/flow_test.py -v --timeout=60

# Full suite: should complete without OOM
make test-staging  # or equivalent

# Memory profile: peak RSS should be <8 GB for unit tests
/usr/bin/time -v make test-staging 2>&1 | grep "Maximum resident"
```

**Step 5: Guard Against Recurrence**
- AST-based test validates mock coverage (catches new unmocked tasks)
- `resource.setrlimit` prevents system crash even if a new leak is introduced
- MemoryMonitor logs RSS at runtime so leaks are visible in CI logs

### Foundation-PLR Patterns Ported

| Pattern | Foundation-PLR Location | Minivess Location | Adaptation |
|---------|------------------------|-------------------|------------|
| `MemoryMonitor` dataclass | `streaming_duckdb_export.py:217-284` | `src/minivess/observability/memory_monitor.py` | Lowered thresholds (4/6 GB vs 12/14 GB), added `_check_without_psutil` fallback |
| Per-iteration cleanup | `streaming_duckdb_export.py:670-677` | `biostatistics_specification_curve.py:_permutation_test` | `del permuted_data` + `monitor.enforce()` every 50 iter |
| Pre-allocated buffers | N/A (foundation-PLR loads from disk) | `biostatistics_specification_curve.py:310-324` | `metric_structures` dict with numpy buffers shuffled in-place |
| Minimal test fixtures | `test_streaming_duckdb_export.py` (5 runs × 5 samples) | `test_biostatistics_flow.py` (empty `per_volume_data={}`) | Flow test uses real config + all-mocked tasks |

### Files Changed

| File | Change |
|------|--------|
| `tests/v2/unit/test_biostatistics_flow.py` | +3 `@patch` decorators, real `BiostatisticsConfig`, `TestMockCoverage` class |
| `tests/v2/unit/conftest.py` | `resource.setrlimit(RLIMIT_AS, 32 GB)` on Linux |
| `src/minivess/pipeline/biostatistics_specification_curve.py` | Pre-allocated buffers, in-place shuffle, `MemoryMonitor.enforce()` every 50 iter |
| `src/minivess/observability/memory_monitor.py` | **NEW** — MemoryMonitor dataclass (ported from foundation-PLR) |
| `tests/v2/unit/test_memory_monitor.py` | **NEW** — 9 tests for MemoryMonitor |

## 11. Cross-References

- Metalearning: `.claude/metalearning/2026-03-21-ram-crash-biostatistics-test.md`
- Foundation-PLR streaming: `foundation_PLR/src/data_io/streaming_duckdb_export.py`
- Foundation-PLR plan: `foundation_PLR/docs/planning/prefect-mlflow-to-duckdb-test-suite-improvement.md`
- Foundation-PLR restrictions: `foundation_PLR/docs/planning/computation-doublecheck-plan-and-restriction-to-mlflow-duckdb-conversion.md`
- Biostatistics flow: `src/minivess/orchestration/flows/biostatistics_flow.py`
- Specification curve: `src/minivess/pipeline/biostatistics_specification_curve.py`
- MemoryMonitor: `src/minivess/observability/memory_monitor.py`
- Flow test: `tests/v2/unit/test_biostatistics_flow.py`
- Config: `src/minivess/config/biostatistics_config.py`
