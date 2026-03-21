# RAM Crash: Biostatistics Test Consumed 62+ GB RAM

**Date**: 2026-03-21
**Severity**: P0 — System became unresponsive, required hard reset
**Category**: Resource exhaustion, test design failure
**Branch**: `test/debug-factorial-run`

## Incident

Running `make test-staging` followed by
`MINIVESS_ALLOW_HOST=1 uv run pytest tests/v2/unit/test_biostatistics_flow.py::TestFlowRunsWithMockData::test_flow_runs_with_mock_data`
consumed 66.8 GB of 67.3 GB RAM (99.3%) plus exhausted all 17.2 GB swap,
making the desktop completely unresponsive.

## Root Cause Analysis (Multi-Hypothesis)

### H1: Missing Mocks in `test_flow_runs_with_mock_data` (PROBABLE)

The test mocks 13 tasks but **misses 3**:
- `task_compute_specification_curve` (line 454 of flow)
- `task_compute_rank_concordance` (line 470 of flow)
- `_resolve_factor_names` (line 453 of flow)

With `mock_config = MagicMock()`, any attribute access on unset fields returns
another MagicMock. If `_resolve_factor_names` returns a MagicMock list, and
`config.n_bootstrap // 20` triggers MagicMock arithmetic producing a huge
iteration count, the permutation loop in `_permutation_test()` could run
millions of iterations.

**Critical bug**: `MagicMock() // 20` returns a MagicMock, which when used as
`range(MagicMock)` should raise TypeError — BUT if Prefect's task wrapper
catches/retries this, or if the MagicMock is coerced to a large int via
`__index__`, the permutation loop could iterate indefinitely.

### H2: Accumulated Memory from `make test-staging` (CONTRIBUTING)

`make test-staging` runs the full staging test suite (~5000+ tests). If it
was killed by timeout but left zombie pytest worker processes, memory from
those processes would NOT be freed. The subsequent individual test run then
ran on a system already at high memory usage.

### H3: Specification Curve Permutation Loop Memory (STRUCTURAL)

Even with correct data, the permutation test has no memory guardrails:
- `_permutation_test()` has zero `gc.collect()` calls
- No `del` statements between iterations
- Creates fresh `permuted_data` dict each iteration (but old one isn't freed)
- With production data (24 conditions × 8 metrics × 500 permutations):
  ~2.2M Specification objects × 120 bytes = 265 MB minimum

### H4: DuckDB Import Side Effects (UNLIKELY but investigate)

The biostatistics flow imports `biostatistics_duckdb` at module level,
which imports DuckDB. DuckDB allocates memory pools on import. If multiple
test processes import DuckDB simultaneously, each gets its own memory pool.

## Foundation-PLR Precedent

The foundation-PLR project had the **identical failure** — batch accumulation
of 410 MLflow runs before DuckDB write consumed 62 GB RAM. Their fix:

1. **MemoryMonitor** class: warn at 12 GB, critical at 14 GB, force `gc.collect()`
2. **Per-run streaming writes** instead of batch accumulation
3. **Explicit `del` + `gc.collect()`** after each run
4. **CheckpointManager** for crash recovery via DuckDB state table
5. **Minimal test fixtures**: 5 runs × 5 samples (not production-scale)

## Prevention Rules

1. **ALL Prefect tasks in flow tests MUST be mocked** — no exceptions.
   Count the tasks in the flow, count the `@patch` decorators in the test.
   If they don't match, the test is broken.

2. **Never use bare `MagicMock()` for config** — use a real `BiostatisticsConfig()`
   instance with overridden fields, or create a proper test fixture.

3. **Add MemoryMonitor to permutation-heavy functions** — any function that
   loops >100 iterations with data allocation needs `gc.collect()` every N iterations.

4. **Test target: 32 GB laptop** — all unit tests must run on a 32 GB laptop.
   Any test that allocates >2 GB is a bug. Add `resource.setrlimit()` guards
   or `pytest-resource-usage` markers.

5. **Kill zombie processes** — `make test-staging` must use `--forked` or
   ensure clean process termination on timeout.

## Files Involved

- `tests/v2/unit/test_biostatistics_flow.py` — missing mocks
- `src/minivess/pipeline/biostatistics_specification_curve.py:284-358` — permutation loop with no GC
- `src/minivess/orchestration/flows/biostatistics_flow.py:454-474` — unmocked task calls
- `src/minivess/config/biostatistics_config.py` — `n_bootstrap=10_000` default

## Cross-References

- foundation-PLR streaming DuckDB: `foundation_PLR/src/data_io/streaming_duckdb_export.py`
- foundation-PLR planning doc: `foundation_PLR/docs/planning/prefect-mlflow-to-duckdb-test-suite-improvement.md`
- Report: `docs/planning/ram-issue-mock-data-biostatistics-duckup-report.md`
