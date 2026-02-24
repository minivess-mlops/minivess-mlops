# Calibration-Under-Shift Framework — Implementation Plan (Issue #19)

## Current State
- CalibrationResult, ECE/MCE, temperature scaling exist in `ensemble/calibration.py`
- Drift detection in `observability/drift.py`
- Conformal prediction in `ensemble/conformal.py`
- No calibration evaluation under distribution shift

## Architecture

### New Module: `src/minivess/ensemble/calibration_shift.py`
- **ShiftType** — StrEnum: COVARIATE, LABEL, CONCEPT
- **ShiftedCalibrationResult** — Dataclass: source/target ECE, degradation, shift type
- **apply_synthetic_shift()** — Apply synthetic domain shift (intensity, noise, resolution)
- **evaluate_calibration_transfer()** — Evaluate calibration degradation under shift
- **CalibrationShiftAnalyzer** — Cross-domain calibration analysis:
  - add_domain() — register a domain's predictions
  - analyze_transfer() — evaluate all pairwise transfers
  - to_markdown() — generate transfer matrix report

## Test Plan
- `tests/v2/unit/test_calibration_shift.py` (~12 tests)
  - TestShiftType: enum values
  - TestSyntheticShift: intensity, noise, resolution shifts
  - TestCalibratedTransfer: ECE degradation, pairwise analysis
  - TestCalibrationShiftAnalyzer: multi-domain, markdown report
