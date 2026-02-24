# Great Expectations Validation Gates — Implementation Plan (Issue #45)

## Current State
- `great-expectations>=1.3` in pyproject.toml (quality extras)
- `expectations.py` has config-only suite builders (dict format)
- `gates.py` has Pandera-based gates but no GE execution
- No GE context, checkpoint, or batch validation runner

## Architecture

### New Module: `src/minivess/validation/ge_runner.py`

1. **run_expectation_suite()** — Execute a GE expectation suite dict against a pandas DataFrame
2. **GECheckpointResult** — Dataclass with pass/fail, statistics, failed expectations
3. **validate_nifti_batch()** — Pre-training gate using nifti_metadata_suite
4. **validate_metrics_batch()** — Post-training gate using training_metrics_suite

### Integration Points
- Uses GE v1 EphemeralDataContext (no filesystem state)
- Suite dicts from `expectations.py` are converted to GE ExpectationSuite
- Results mapped to existing `GateResult` dataclass for consistency
- Falls back to Pandera if GE is not installed (optional dependency)

## New Files
- `src/minivess/validation/ge_runner.py`

## Modified Files
- `src/minivess/validation/__init__.py` — add GE runner exports

## Test Plan
- `tests/v2/unit/test_ge_runner.py` (~15 tests)
