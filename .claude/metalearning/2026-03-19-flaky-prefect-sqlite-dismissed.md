# Metalearning: Flaky Prefect SQLite Tests Dismissed Without Root Cause

**Date**: 2026-03-19
**Pattern**: Zero Tolerance Violation (CLAUDE.md Rule 20)
**Category**: Silent failure dismissal

## What Happened

Flaky test failures involving Prefect SQLite were observed across multiple sessions
and dismissed as "probably SQLite locking" without investigation. The root cause was
never identified, violating Rule 20 (zero tolerance for observed failures).

## Root Cause (Identified 2026-03-19)

The session-level `prefect_test_harness()` fixture in `tests/conftest.py` starts a
SQLite-backed Prefect server. However, **4 test files** set `PREFECT_DISABLED=1` at
**module level** (before imports), which conflicts with the running harness:

1. `tests/v2/unit/test_analysis_flow.py:15` — `os.environ["PREFECT_DISABLED"] = "1"`
2. `tests/v2/unit/test_acquisition_flow.py:14` — `os.environ["PREFECT_DISABLED"] = "1"`
3. `tests/v2/integration/test_analysis_flow_integration.py:27` — `os.environ["PREFECT_DISABLED"] = "1"`
4. `tests/v2/unit/pipeline/test_trainer_nan_handling.py:77` — `os.environ.setdefault("PREFECT_DISABLED", "1")`

The conftest comment explicitly states: "This replaces the old PREFECT_DISABLED=1
no-op approach." These module-level env var mutations are LEGACY code that was never
cleaned up when the harness was introduced.

**Flakiness mechanism**: When pytest collects these files, the module-level
`os.environ["PREFECT_DISABLED"] = "1"` runs at import time and mutates global state.
Depending on collection order, this can affect subsequent tests that expect the
Prefect harness to be active, causing:
- Prefect API connection errors (server running but client told to ignore it)
- SQLite "database is locked" when the disabled/enabled state toggles mid-session

## Fix

1. Remove ALL module-level `os.environ["PREFECT_DISABLED"]` from test files
2. The session-level `prefect_test_harness()` handles all Prefect needs
3. Tests that don't use flows just call functions directly — the harness is invisible
4. For `monkeypatch.setenv("PREFECT_DISABLED", "1")` usage: keep only in tests that
   explicitly test the disabled-Prefect code path

## Lesson

"Probably SQLite locking" is not a root cause. Every flaky test has a specific,
identifiable mechanism. "It's probably X" without investigation is Rule 20 violation.
