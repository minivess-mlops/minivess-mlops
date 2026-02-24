# Validation Pipeline Plan

**Issue**: #39 — Build validation pipeline (Pandera + Deepchecks Vision)
**Date**: 2026-02-24
**Status**: Draft → Implementation

---

## 1. Current State

Significant scaffolding exists in `src/minivess/validation/` but with **zero tests**:

| Module | Status | Content |
|--------|--------|---------|
| `schemas.py` | Code exists, untested | 3 Pandera DataFrameModels: NiftiMetadata, TrainingMetrics, AnnotationQuality |
| `expectations.py` | Code exists, untested | 2 GE suite builders (training_metrics, nifti_metadata) — returns config dicts |
| `deepchecks_vision.py` | Config stubs, untested | Suite config builders + ValidationReport dataclass (NO actual Deepchecks calls) |
| `drift.py` | Code exists, untested | KS + PSI prediction drift (DriftReport, detect_prediction_drift) |
| `__init__.py` | Exports all | 11 public symbols |

**Dependencies**: Pandera ≥0.22, Great Expectations ≥1.3, whylogs ≥1.5 in
`[quality]` optional deps. Deepchecks NOT in deps.

**Tests**: Zero validation tests exist.

---

## 2. Architecture Decision: Deepchecks

Deepchecks Vision is designed for 2D image classification/detection, not 3D
volumetric segmentation. Our MONAI CacheDataset returns 3D tensors (C×D×H×W)
which are incompatible with Deepchecks' VisionData format.

**Decision**: Keep the Deepchecks stubs as config-only (suite configuration
dicts). The actual validation for 3D imaging data is better handled by:
- Pandera for metadata validation (tabular)
- Great Expectations for batch quality checks (tabular)
- whylogs for statistical profiling (tabular + distributions)
- Our custom drift detectors (observability/drift.py) for image statistics

This avoids a heavyweight dependency that doesn't support our data format.

---

## 3. Implementation Plan

### T1: Test Pandera schemas (schemas.py)

Tests for all 3 DataFrameModels:
- `test_nifti_metadata_valid` — valid DataFrame passes
- `test_nifti_metadata_invalid_spacing` — negative spacing fails
- `test_nifti_metadata_invalid_affine` — false affine fails
- `test_nifti_metadata_intensity_range` — max < min fails
- `test_training_metrics_valid` — valid metrics pass
- `test_training_metrics_invalid_dice` — dice > 1.0 fails
- `test_annotation_quality_valid` — valid annotation data passes

### T2: Test GE expectation suites (expectations.py)

- `test_training_metrics_suite_structure` — correct keys/expectations
- `test_nifti_metadata_suite_structure` — correct keys/expectations
- `test_training_metrics_suite_against_data` — run GE validation on valid data
- `test_nifti_metadata_suite_against_data` — run GE validation on valid data

### T3: Test prediction drift (drift.py)

- `test_ks_no_drift` — same distribution → no drift
- `test_ks_drift_detected` — shifted distribution → drift
- `test_psi_no_drift` — same distribution → low PSI
- `test_psi_drift_detected` — shifted → high PSI
- `test_invalid_method` — unknown method raises ValueError

### T4: Test Deepchecks config builders (deepchecks_vision.py)

- `test_data_integrity_suite_config` — returns valid config dict
- `test_train_test_suite_config` — returns valid config dict
- `test_evaluate_report_all_pass` — all checks pass → report.passed=True
- `test_evaluate_report_with_failures` — failures → report.passed=False

### T5: whylogs profiling

Create `src/minivess/validation/profiling.py`:
- `profile_dataframe(df: pd.DataFrame) -> DatasetProfileView`
- `compare_profiles(reference, current) -> ProfileDriftReport`
- Thin wrapper around whylogs for integration with our pipeline

Tests:
- `test_profile_dataframe` — profiles a DataFrame
- `test_compare_profiles_no_drift` — same distribution passes
- `test_compare_profiles_with_drift` — shifted distribution detects drift

### T6: Validation gate function

Create `src/minivess/validation/gates.py`:
- `validate_nifti_metadata(df: pd.DataFrame) -> GateResult`
  - Runs Pandera schema validation
  - Returns pass/fail with details
- `validate_training_metrics(df: pd.DataFrame) -> GateResult`
  - Runs Pandera + GE checks
  - Returns pass/fail with details
- `GateResult` dataclass: passed, errors, warnings

Tests:
- `test_gate_passes_valid_metadata` — valid data passes gate
- `test_gate_fails_invalid_metadata` — invalid data fails gate
- `test_gate_passes_valid_metrics` — valid metrics pass gate
- `test_gate_fails_invalid_metrics` — invalid metrics fail gate

---

## 4. Execution Order

```
T1 (Pandera tests) → T2 (GE tests) → T3 (drift tests) → T4 (Deepchecks tests)
→ T5 (whylogs) → T6 (validation gates)
```

---

## 5. Out of Scope

- Actual Deepchecks Vision execution (incompatible with 3D data)
- Pipeline integration hooks in trainer.py (future issue)
- Grafana dashboards for validation metrics
