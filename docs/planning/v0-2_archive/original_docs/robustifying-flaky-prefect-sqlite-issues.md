# Robustifying Flaky Prefect SQLite Issues

**Date**: 2026-03-19
**Branch**: fix/pre-debug-qa-verification

## Root Cause Analysis

The session-level `prefect_test_harness()` fixture in `tests/conftest.py` starts an
ephemeral SQLite-backed Prefect server shared across all tests. However, 4 test files
set `PREFECT_DISABLED=1` at **module level** (`os.environ[...]`), which mutates global
state at import time and conflicts with the running harness.

### Offending Files (Fixed 2026-03-19)

1. `tests/v2/unit/test_analysis_flow.py:15` — `os.environ["PREFECT_DISABLED"] = "1"`
2. `tests/v2/unit/test_acquisition_flow.py:14` — `os.environ["PREFECT_DISABLED"] = "1"`
3. `tests/v2/integration/test_analysis_flow_integration.py:27` — `os.environ["PREFECT_DISABLED"] = "1"`
4. `tests/v2/unit/pipeline/test_trainer_nan_handling.py:77` — `os.environ.setdefault("PREFECT_DISABLED", "1")`

### Why This Caused Flakiness

- `conftest.py` starts a session-level Prefect harness (SQLite server)
- Module-level `os.environ["PREFECT_DISABLED"] = "1"` runs at **import time**
- Depending on pytest collection order, this env var mutation happens:
  - **Before** the harness fixture runs → harness ignores the env var (session fixture)
  - **During** session → env var state leaks to subsequent test modules
- The contradiction (server running but client told to ignore it) causes intermittent
  connection errors and SQLite lock contention

### Fix Applied

Removed all module-level `PREFECT_DISABLED` mutations. The `conftest.py` comment
explicitly states: *"This replaces the old PREFECT_DISABLED=1 no-op approach."*

The session-level `prefect_test_harness()` is sufficient:
- Tests that call flow/task functions: harness provides a real server
- Tests that test pure Python logic: harness is invisible (no overhead)

### Additional Mitigations (Already Present)

- `pytest_configure()` in conftest.py isolates `PREFECT_HOME` per xdist worker
- `monkeypatch.setenv()` usage is acceptable for per-test env var changes (auto-restored)
- The `_allow_host_env()` fixture sets `MINIVESS_ALLOW_HOST=1` session-wide

## Verification

After fix: `make test-staging` and `make test-prod` must pass 3 consecutive times
with 0 failures.
