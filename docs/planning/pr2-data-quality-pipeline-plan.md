# PR-2: Data Quality Pipeline — Implementation Plan

**Branch**: `feat/data-quality-pipeline`
**Base**: `main`
**Status**: PLANNED
**Estimated tests**: 15-20 new (on top of ~60 existing)
**Estimated LOC**: ~400 new/modified

## Context

PR-2 wires the existing standalone data quality modules (`src/minivess/validation/`)
into the Prefect `data_flow.py` orchestration, making quality gates part of the
automated pipeline. The modules (Pandera schemas, GE expectations, DATA-CARE assessment,
DeepChecks config builders) exist and are tested individually (~60 tests), but the
data flow performs only minimal key-presence validation. This PR closes the 30%
integration gap.

### What exists (70% done)

| Module | File | Status | Tests |
|--------|------|--------|-------|
| Pandera schemas | `src/minivess/validation/schemas.py` | COMPLETE | 24 |
| Validation gates | `src/minivess/validation/gates.py` | COMPLETE | 8 |
| GE runner | `src/minivess/validation/ge_runner.py` | COMPLETE | 14 |
| GE expectations | `src/minivess/validation/expectations.py` | COMPLETE | 8 |
| whylogs profiling | `src/minivess/validation/profiling.py` | COMPLETE | 6 |
| DeepChecks Vision | `src/minivess/validation/deepchecks_vision.py` | STUBS (config builders + evaluator only) | 5 |
| DATA-CARE | `src/minivess/validation/data_care.py` | COMPLETE (not wired) | ~8 |
| Data flow | `src/minivess/orchestration/flows/data_flow.py` | MINIMAL (key validation only) | ~10 |

### What's missing (30% integration gap)

1. NIfTI metadata extraction task (read NIfTI headers into Pandera-compatible DataFrame)
2. Wire Pandera validation into `data_flow.py`
3. Wire GE batch expectations into `data_flow.py`
4. Wire DATA-CARE assessment into `data_flow.py`
5. DeepChecks 3D-to-2D slice adapter (~50 lines)
6. Wire DeepChecks into `data_flow.py`
7. Configurable gate enforcement via Dynaconf severity levels in `settings.toml`
8. Integration tests for the full pipeline

---

## User Decisions (Non-Negotiable)

These decisions were explicitly made by the user and override any defaults:

1. **Gate failure policy**: CONFIGURABLE PER-GATE via Dynaconf `settings.toml`
2. **Gate severities**:
   - `DATA_QUALITY_GATE_PANDERA = 'error'` (halt pipeline)
   - `DATA_QUALITY_GATE_GE = 'warning'` (tag MLflow + continue)
   - `DATA_QUALITY_GATE_DATACARE = 'warning'` (tag MLflow + continue)
   - `DATA_QUALITY_GATE_DEEPCHECKS = 'info'` (log only)
3. **Config location**: `configs/deployment/settings.toml` (extend existing, alongside drift config)
4. **DeepChecks 3D-to-2D adapter**: INCLUDED in PR-2 (Level 4 mandate)
5. **No ralph-loop**: Local TDD work, not autonomous plan execution

---

## Architecture

```
data_flow.py (Prefect @flow)
  ├── discover_data_task()          [existing]
  ├── extract_nifti_metadata_task() [NEW — T2]
  ├── pandera_gate_task()           [NEW — T3, calls validate_nifti_metadata()]
  ├── ge_gate_task()                [NEW — T3, calls validate_nifti_batch()]
  ├── datacare_gate_task()          [NEW — T4, calls assess_nifti_quality() → quality_gate()]
  ├── deepchecks_gate_task()        [NEW — T6, calls slice adapter → DeepChecks]
  ├── enforce_gate()                [NEW — T1, reads Dynaconf severity]
  ├── data_quality_gate()           [MODIFIED — orchestrates all gates]
  ├── split_data_task()             [existing]
  └── log_data_provenance_task()    [existing]
```

### Gate Enforcement Flow

```
GateResult(passed, errors, warnings, statistics)
     │
     ▼
enforce_gate(gate_name, result, severity_from_dynaconf)
     │
     ├── severity='error'   → if not passed: raise DataQualityError (halt)
     ├── severity='warning' → if not passed: mlflow.set_tag(f"quality_{gate_name}", "WARNING") + continue
     └── severity='info'    → log only, never halt
```

### Escape hatch

`MINIVESS_SKIP_QUALITY_GATE=1` bypasses ALL gates. For pytest only (enforced
by guard that checks `MINIVESS_ALLOW_HOST=1` is also set).

---

## Phase 0: Knowledge Graph Prerequisite

### KG-0: Create `data_quality_orchestration` decision node

**Rationale**: The KG `data.yaml` domain lists `data_validation_depth` (resolved:
`validation_onion_12_layer`) but has no node for the orchestration wiring of quality
gates into the Prefect flow. Add a decision node so the KG reflects this integration.

**Files**:
- `knowledge-graph/domains/data.yaml` — add `data_quality_orchestration` decision
- `knowledge-graph/decisions/L2-architecture/data_quality_orchestration.yaml` — new node

**Commit**: `chore(kg): add data_quality_orchestration decision node`

---

## Task T1: Dynaconf Gate Config + Enforcement Helper

### Goal

Add configurable gate severity to `settings.toml` and implement `enforce_gate()` that
reads the Dynaconf setting and takes the appropriate action (halt / warn / log).

### TDD Spec

**Test file**: `tests/v2/unit/test_gate_enforcement.py`
**Marker**: None (staging tier — no model loading, no slow)

#### RED phase — tests to write first

```python
# T1-R1: enforce_gate with severity='error' and failed gate raises DataQualityError
# T1-R2: enforce_gate with severity='error' and passed gate does NOT raise
# T1-R3: enforce_gate with severity='warning' and failed gate returns GateAction.WARN
# T1-R4: enforce_gate with severity='info' and failed gate returns GateAction.LOG
# T1-R5: enforce_gate reads severity from Dynaconf settings when not overridden
# T1-R6: MINIVESS_SKIP_QUALITY_GATE=1 bypasses all gates (returns GateAction.SKIP)
# T1-R7: get_gate_severity() returns configured severity from settings.toml
# T1-R8: get_gate_severity() falls back to 'warning' for unknown gate names
```

**Expected test count**: 8

#### GREEN phase — implementation

**File**: `src/minivess/validation/enforcement.py` (NEW)

```python
# Exports:
# - GateAction (enum: PASS, WARN, LOG, HALT, SKIP)
# - DataQualityError (exception, subclass of RuntimeError)
# - enforce_gate(gate_name: str, result: GateResult, severity: str | None = None) -> GateAction
# - get_gate_severity(gate_name: str) -> str
```

**File**: `configs/deployment/settings.toml` (MODIFY — append section)

```toml
# Data Quality Gates (PR-2)
# Severity levels: 'error' = halt pipeline, 'warning' = tag MLflow + continue, 'info' = log only
data_quality_gate_pandera = "error"
data_quality_gate_ge = "warning"
data_quality_gate_datacare = "warning"
data_quality_gate_deepchecks = "info"
```

**File**: `.env.example` (MODIFY — add env var)

```bash
# Data Quality Gate override (pytest only — skips ALL quality gates)
# MINIVESS_SKIP_QUALITY_GATE=1
```

#### VERIFY

```bash
uv run pytest tests/v2/unit/test_gate_enforcement.py -v
uv run ruff check src/minivess/validation/enforcement.py
uv run mypy src/minivess/validation/enforcement.py
```

#### Commit

```
feat(validation): T1 — configurable gate enforcement via Dynaconf severity
```

---

## Task T2: NIfTI Metadata Extraction Task

### Goal

Create a Prefect `@task` that reads NIfTI headers from discovered pairs and produces
a pandas DataFrame matching `NiftiMetadataSchema`. This DataFrame is the input to
Pandera, GE, and DATA-CARE gates.

### TDD Spec

**Test file**: `tests/v2/unit/test_nifti_metadata_extraction.py`
**Marker**: None (staging tier — uses synthetic NIfTI from nibabel, no real data)

#### RED phase — tests to write first

```python
# T2-R1: extract_nifti_metadata_task returns a pd.DataFrame
# T2-R2: DataFrame has all NiftiMetadataSchema columns
# T2-R3: shape values match actual NIfTI dimensions
# T2-R4: voxel_spacing values match NIfTI affine
# T2-R5: intensity_min / intensity_max match actual data
# T2-R6: has_valid_affine is True for standard affines
# T2-R7: num_foreground_voxels counts nonzero label voxels
# T2-R8: empty pairs list returns empty DataFrame with correct columns
```

**Expected test count**: 8

#### GREEN phase — implementation

**File**: `src/minivess/orchestration/flows/data_flow.py` (MODIFY — add task)

```python
@task(name="extract-nifti-metadata")
def extract_nifti_metadata_task(
    pairs: list[dict[str, str]],
) -> pd.DataFrame:
    """Read NIfTI headers and produce NiftiMetadataSchema-compatible DataFrame."""
    # Uses nibabel to read headers (nibabel is already a dependency)
    # Returns DataFrame with columns matching NiftiMetadataSchema
```

**Dependencies**: `nibabel` (already installed), `pandas` (already installed)

#### VERIFY

```bash
uv run pytest tests/v2/unit/test_nifti_metadata_extraction.py -v
uv run ruff check src/minivess/orchestration/flows/data_flow.py
uv run mypy src/minivess/orchestration/flows/data_flow.py
```

#### Commit

```
feat(data-flow): T2 — NIfTI metadata extraction task for quality gates
```

---

## Task T3: Wire Pandera + GE Gates into data_flow

### Goal

Add Prefect `@task` wrappers that call the existing `validate_nifti_metadata()` (Pandera)
and `validate_nifti_batch()` (GE) functions, then feed results through `enforce_gate()`.

### TDD Spec

**Test file**: `tests/v2/unit/test_data_flow_quality_gates.py`
**Marker**: None (staging tier)

#### RED phase — tests to write first

```python
# T3-R1: pandera_gate_task calls validate_nifti_metadata and returns GateResult
# T3-R2: pandera_gate_task with valid data returns passed=True
# T3-R3: pandera_gate_task with invalid data returns passed=False
# T3-R4: ge_gate_task calls validate_nifti_batch and returns GateResult
# T3-R5: ge_gate_task with valid data returns passed=True
# T3-R6: ge_gate_task with invalid data (out-of-range spacing) returns passed=False
```

**Expected test count**: 6

#### GREEN phase — implementation

**File**: `src/minivess/orchestration/flows/data_flow.py` (MODIFY — add tasks)

```python
@task(name="pandera-validation-gate")
def pandera_gate_task(metadata_df: pd.DataFrame) -> GateResult:
    """Run Pandera NiftiMetadataSchema validation."""
    from minivess.validation.gates import validate_nifti_metadata
    return validate_nifti_metadata(metadata_df)


@task(name="ge-validation-gate")
def ge_gate_task(metadata_df: pd.DataFrame) -> GateResult:
    """Run Great Expectations nifti_metadata_suite validation."""
    from minivess.validation.ge_runner import validate_nifti_batch
    return validate_nifti_batch(metadata_df)
```

The existing `data_quality_gate` task is MODIFIED to orchestrate all gates:

```python
@task(name="data-quality-gate")
def data_quality_gate(
    report: DataValidationReport,
    metadata_df: pd.DataFrame | None = None,
) -> bool:
    """Check whether data passes all quality gates.

    When metadata_df is provided, runs Pandera + GE + DATA-CARE + DeepChecks
    gates with configurable enforcement. When None, falls back to the legacy
    error-count check (backwards-compatible).
    """
```

#### VERIFY

```bash
uv run pytest tests/v2/unit/test_data_flow_quality_gates.py -v
uv run pytest tests/v2/unit/test_data_flow.py -v  # regression — existing tests still pass
uv run ruff check src/minivess/orchestration/flows/data_flow.py
uv run mypy src/minivess/orchestration/flows/data_flow.py
```

#### Commit

```
feat(data-flow): T3 — wire Pandera + GE validation gates into data_flow
```

---

## Task T4: Wire DATA-CARE Assessment into data_flow

### Goal

Add a Prefect `@task` that calls `assess_nifti_quality()` and converts via
`quality_gate()` to a `GateResult`, then enforces based on Dynaconf severity.

### TDD Spec

**Test file**: `tests/v2/unit/test_data_flow_quality_gates.py` (append to T3 file)
**Marker**: None (staging tier)

#### RED phase — tests to write first

```python
# T4-R1: datacare_gate_task returns GateResult
# T4-R2: datacare_gate_task with clean data returns passed=True
# T4-R3: datacare_gate_task with corrupted data returns passed=False
# T4-R4: datacare_gate_task includes DATA-CARE statistics in result
```

**Expected test count**: 4

#### GREEN phase — implementation

**File**: `src/minivess/orchestration/flows/data_flow.py` (MODIFY — add task)

```python
@task(name="datacare-validation-gate")
def datacare_gate_task(metadata_df: pd.DataFrame) -> GateResult:
    """Run DATA-CARE quality assessment on NIfTI metadata."""
    from minivess.validation.data_care import assess_nifti_quality, quality_gate
    report = assess_nifti_quality(metadata_df)
    return quality_gate(report)
```

#### VERIFY

```bash
uv run pytest tests/v2/unit/test_data_flow_quality_gates.py -v -k "datacare"
uv run ruff check src/minivess/orchestration/flows/data_flow.py
```

#### Commit

```
feat(data-flow): T4 — wire DATA-CARE assessment into data_flow
```

---

## Task T5: DeepChecks 3D-to-2D Slice Adapter

### Goal

DeepChecks Vision expects 2D images (H, W, C). Our data is 3D NIfTI (D, H, W).
Implement a thin adapter that extracts representative 2D slices (middle slice per axis,
or max-foreground slice from label) for DeepChecks validation.

This is a Level 4 mandate from the user: included in PR-2.

### TDD Spec

**Test file**: `tests/v2/unit/test_deepchecks_3d_adapter.py`
**Marker**: None (staging tier — uses synthetic numpy arrays, no DeepChecks import needed)

#### RED phase — tests to write first

```python
# T5-R1: extract_representative_slices returns list of 2D arrays
# T5-R2: default strategy extracts middle axial slice
# T5-R3: strategy='max_foreground' finds slice with most label voxels
# T5-R4: returned slices have shape (H, W) or (H, W, 1) for grayscale
# T5-R5: empty volume returns empty list (graceful degradation)
# T5-R6: build_deepchecks_dataset builds list of dicts with 'image' and 'label' keys
```

**Expected test count**: 6

#### GREEN phase — implementation

**File**: `src/minivess/validation/deepchecks_3d_adapter.py` (NEW, ~50 lines)

```python
"""3D-to-2D slice adapter for DeepChecks Vision integration.

DeepChecks Vision expects 2D images (H, W, C). MinIVess data is 3D NIfTI
(D, H, W). This adapter extracts representative 2D slices for property-based
validation checks (brightness, contrast, label distribution).

Strategy options:
- 'middle': Middle axial slice (default — fast, deterministic)
- 'max_foreground': Slice with most foreground voxels in label (slower, more informative)
"""

# Exports:
# - extract_representative_slices(volume: NDArray, label: NDArray | None, strategy: str) -> list[NDArray]
# - build_deepchecks_dataset(pairs: list[dict[str, str]], strategy: str) -> list[dict[str, NDArray]]
```

**Dependencies**: `numpy` (already installed), `nibabel` (already installed).
DeepChecks itself is optional — the adapter works with plain numpy.

#### VERIFY

```bash
uv run pytest tests/v2/unit/test_deepchecks_3d_adapter.py -v
uv run ruff check src/minivess/validation/deepchecks_3d_adapter.py
uv run mypy src/minivess/validation/deepchecks_3d_adapter.py
```

#### Commit

```
feat(validation): T5 — DeepChecks 3D-to-2D slice adapter
```

---

## Task T6: Wire DeepChecks into data_flow

### Goal

Add a Prefect `@task` that uses the slice adapter to convert 3D volumes to 2D,
runs DeepChecks data integrity checks, and returns a `GateResult`. Gracefully
degrades when DeepChecks is not installed (logs info, returns passing result).

### TDD Spec

**Test file**: `tests/v2/unit/test_data_flow_quality_gates.py` (append to T3/T4 file)
**Marker**: None (staging tier — mocks DeepChecks if not installed)

#### RED phase — tests to write first

```python
# T6-R1: deepchecks_gate_task returns GateResult
# T6-R2: deepchecks_gate_task gracefully degrades when deepchecks not installed
# T6-R3: deepchecks_gate_task calls slice adapter with correct strategy
```

**Expected test count**: 3

#### GREEN phase — implementation

**File**: `src/minivess/orchestration/flows/data_flow.py` (MODIFY — add task)

```python
@task(name="deepchecks-validation-gate")
def deepchecks_gate_task(
    pairs: list[dict[str, str]],
    *,
    slice_strategy: str = "middle",
) -> GateResult:
    """Run DeepChecks data integrity on 2D slices extracted from 3D volumes.

    Gracefully degrades to a passing result when DeepChecks is not installed.
    """
    from minivess.validation.gates import GateResult

    try:
        from minivess.validation.deepchecks_3d_adapter import build_deepchecks_dataset
        from minivess.validation.deepchecks_vision import (
            build_data_integrity_suite,
            evaluate_report,
        )
    except ImportError:
        logger.info("DeepChecks not installed — skipping vision validation")
        return GateResult(passed=True, warnings=["deepchecks not installed — skipped"])

    # Build 2D dataset, run integrity suite, evaluate
    ...
```

#### VERIFY

```bash
uv run pytest tests/v2/unit/test_data_flow_quality_gates.py -v -k "deepchecks"
uv run ruff check src/minivess/orchestration/flows/data_flow.py
```

#### Commit

```
feat(data-flow): T6 — wire DeepChecks vision gate into data_flow
```

---

## Task T7: Integration Test — Full Quality Pipeline

### Goal

End-to-end integration test that exercises the complete data flow with all quality
gates: discover -> extract metadata -> Pandera -> GE -> DATA-CARE -> DeepChecks ->
enforce -> split. Uses synthetic NIfTI files (nibabel), no real data required.

### TDD Spec

**Test file**: `tests/v2/integration/test_data_quality_pipeline.py`
**Marker**: `@pytest.mark.integration` (excluded from staging tier)

#### RED phase — tests to write first

```python
# T7-R1: Full pipeline with valid synthetic NIfTI passes all gates
# T7-R2: Full pipeline with corrupted NIfTI (bad spacing) triggers Pandera error gate
# T7-R3: Warning-severity gate failure tags MLflow and continues to split
# T7-R4: MINIVESS_SKIP_QUALITY_GATE=1 bypasses all gates
# T7-R5: DataFlowResult includes quality gate results in provenance
```

**Expected test count**: 5

#### GREEN phase — implementation

**File**: `src/minivess/orchestration/flows/data_flow.py` (MODIFY — update `run_data_flow`)

The main `run_data_flow` function is updated to insert quality gate tasks between
Step 2 (validate) and Step 4 (split):

```python
@flow(name=FLOW_NAME_DATA)
def run_data_flow(...) -> DataFlowResult:
    ...
    # Step 1: Discover
    pairs = discover_data_task(data_dir=data_dir)

    # Step 2: Basic validation (existing — key presence check)
    report = validate_data_task(pairs=pairs)

    # Step 2.5: NIfTI metadata extraction [NEW]
    metadata_df = extract_nifti_metadata_task(pairs)

    # Step 3: Quality gates [NEW — replaces old data_quality_gate]
    quality_passed = run_quality_gates(
        report=report,
        metadata_df=metadata_df,
        pairs=pairs,
    )

    # Step 4: Split (only if quality gate passed)
    ...
```

**File**: `tests/v2/integration/test_data_quality_pipeline.py` (NEW)

Uses `nibabel` to create synthetic NIfTI files in `tmp_path`, then runs
`run_data_flow()` with `MINIVESS_ALLOW_HOST=1` and `PREFECT_DISABLED=1`.

#### VERIFY

```bash
uv run pytest tests/v2/integration/test_data_quality_pipeline.py -v
uv run pytest tests/v2/unit/test_data_flow.py -v  # regression check
uv run ruff check src/minivess/orchestration/flows/data_flow.py
uv run mypy src/minivess/orchestration/flows/data_flow.py
```

#### Commit

```
feat(data-flow): T7 — integration test for full data quality pipeline
```

---

## TDD State File Integration

The `state/tdd-state.json` will be reset for this PR with the following initial state:

```json
{
  "plan_file": "docs/planning/pr2-data-quality-pipeline-plan.md",
  "plan_version": "1.0",
  "execution_mode": "interactive",
  "current_task_id": null,
  "current_phase": null,
  "inner_iteration": 0,
  "tasks": {
    "KG-0": { "status": "TODO", "phase": null, "iterations": 0 },
    "T1": { "status": "TODO", "phase": null, "iterations": 0 },
    "T2": { "status": "TODO", "phase": null, "iterations": 0 },
    "T3": { "status": "TODO", "phase": null, "iterations": 0 },
    "T4": { "status": "TODO", "phase": null, "iterations": 0 },
    "T5": { "status": "TODO", "phase": null, "iterations": 0 },
    "T6": { "status": "TODO", "phase": null, "iterations": 0 },
    "T7": { "status": "TODO", "phase": null, "iterations": 0 }
  },
  "convergence": {
    "reached": false,
    "tasks_done": 0,
    "tasks_total": 8,
    "tasks_stuck": 0
  },
  "session_inner_iterations": 0,
  "session_start": null
}
```

---

## File Inventory

### New Files

| File | Task | Purpose |
|------|------|---------|
| `src/minivess/validation/enforcement.py` | T1 | Gate enforcement helper (GateAction, enforce_gate, get_gate_severity) |
| `src/minivess/validation/deepchecks_3d_adapter.py` | T5 | 3D-to-2D slice adapter for DeepChecks Vision |
| `tests/v2/unit/test_gate_enforcement.py` | T1 | Tests for enforcement.py |
| `tests/v2/unit/test_nifti_metadata_extraction.py` | T2 | Tests for NIfTI metadata extraction task |
| `tests/v2/unit/test_data_flow_quality_gates.py` | T3,T4,T6 | Tests for wired quality gate tasks |
| `tests/v2/unit/test_deepchecks_3d_adapter.py` | T5 | Tests for 3D-to-2D slice adapter |
| `tests/v2/integration/test_data_quality_pipeline.py` | T7 | Integration test for full pipeline |
| `knowledge-graph/decisions/L2-architecture/data_quality_orchestration.yaml` | KG-0 | KG decision node |

### Modified Files

| File | Task | Change |
|------|------|--------|
| `configs/deployment/settings.toml` | T1 | Add data quality gate severity config |
| `.env.example` | T1 | Add MINIVESS_SKIP_QUALITY_GATE documentation |
| `src/minivess/orchestration/flows/data_flow.py` | T2,T3,T4,T6,T7 | Add quality gate tasks + wire into run_data_flow |
| `src/minivess/validation/__init__.py` | T1,T5 | Export enforcement + adapter modules |
| `knowledge-graph/domains/data.yaml` | KG-0 | Add data_quality_orchestration decision reference |

---

## Dependency Order

```
KG-0 (no code deps)
  │
  ▼
T1 (enforcement.py + settings.toml) ──────────────────┐
  │                                                     │
  ▼                                                     │
T2 (NIfTI metadata extraction task) ───┐                │
  │                                    │                │
  ▼                                    ▼                ▼
T3 (wire Pandera + GE) ← depends on T1+T2              │
  │                                                     │
  ▼                                                     │
T4 (wire DATA-CARE) ← depends on T1+T2                 │
  │                                                     │
T5 (DeepChecks adapter) ← independent                  │
  │                                                     │
  ▼                                                     │
T6 (wire DeepChecks) ← depends on T1+T5                │
  │                                                     │
  ▼                                                     │
T7 (integration) ← depends on ALL above ◄──────────────┘
```

**Parallelizable**: T5 can be done in parallel with T3/T4 (no shared files).
**Critical path**: KG-0 -> T1 -> T2 -> T3 -> T7

---

## Test Tier Assignment

| Test File | Tier | Markers | Rationale |
|-----------|------|---------|-----------|
| `test_gate_enforcement.py` | staging | (none) | Pure logic, no I/O |
| `test_nifti_metadata_extraction.py` | staging | (none) | Synthetic NIfTI via nibabel (~1s) |
| `test_data_flow_quality_gates.py` | staging | (none) | Mocked data, no real files |
| `test_deepchecks_3d_adapter.py` | staging | (none) | Pure numpy |
| `test_data_quality_pipeline.py` | prod | `@pytest.mark.integration` | Creates NIfTI files, runs full flow |

All unit tests target **< 3 seconds** individually. The integration test targets **< 30 seconds**.

---

## Backwards Compatibility

The existing `data_quality_gate` task signature is extended with an optional
`metadata_df` parameter. When `metadata_df is None`, the function falls back to the
legacy error-count check. This ensures:

1. All existing tests in `test_data_flow.py` continue to pass without modification
2. Callers that don't produce metadata (e.g., external dataset validation) still work
3. The `DataFlowResult` dataclass gains no new required fields

---

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| nibabel not installed in test env | nibabel is a core dependency (already in pyproject.toml) |
| DeepChecks not installed | Graceful degradation: `try/except ImportError` → passing GateResult with warning |
| Great Expectations not installed | `pytest.importorskip("great_expectations")` in tests; gate returns skip in flow |
| Dynaconf settings not loaded | `get_gate_severity()` has a sensible fallback ('warning') |
| Breaking existing data_flow tests | Backwards-compatible: metadata_df defaults to None |
| Gate adds latency to data flow | NIfTI header reading is fast (~ms per file); gates are DataFrame operations |

---

## Commit Sequence

```
1. chore(kg): add data_quality_orchestration decision node                    [KG-0]
2. feat(validation): T1 — configurable gate enforcement via Dynaconf severity [T1]
3. feat(data-flow): T2 — NIfTI metadata extraction task for quality gates     [T2]
4. feat(data-flow): T3 — wire Pandera + GE validation gates into data_flow    [T3]
5. feat(data-flow): T4 — wire DATA-CARE assessment into data_flow             [T4]
6. feat(validation): T5 — DeepChecks 3D-to-2D slice adapter                   [T5]
7. feat(data-flow): T6 — wire DeepChecks vision gate into data_flow           [T6]
8. feat(data-flow): T7 — integration test for full data quality pipeline      [T7]
```

Each commit is independently testable and passes `make test-staging`.

---

## Verification Checklist (Before PR)

```bash
# 1. All staging tests pass
make test-staging

# 2. Ruff clean
uv run ruff check src/minivess/validation/ src/minivess/orchestration/flows/data_flow.py tests/v2/

# 3. Mypy clean
uv run mypy src/minivess/validation/enforcement.py src/minivess/validation/deepchecks_3d_adapter.py

# 4. Existing data_flow tests still pass (regression)
uv run pytest tests/v2/unit/test_data_flow.py -v

# 5. Integration test passes
uv run pytest tests/v2/integration/test_data_quality_pipeline.py -v

# 6. New test count
uv run pytest tests/v2/unit/test_gate_enforcement.py tests/v2/unit/test_nifti_metadata_extraction.py tests/v2/unit/test_data_flow_quality_gates.py tests/v2/unit/test_deepchecks_3d_adapter.py tests/v2/integration/test_data_quality_pipeline.py --co -q | tail -1
# Expected: ~35 tests collected

# 7. Pre-commit hooks pass
uv run pre-commit run --all-files
```
