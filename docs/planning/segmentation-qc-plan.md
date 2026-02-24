# Segmentation Quality Control — Implementation Plan (Issue #13)

## Current State
- No automated quality scoring for segmentation outputs
- No quality flags or thresholds
- Deepchecks Vision exists for data validation (separate concern)

## Architecture

### New Module: `src/minivess/pipeline/segmentation_qc.py`
- **QCFlag** — StrEnum: PASS, WARNING, FAIL
- **QCResult** — Dataclass: score, flag, metrics, reasons
- **SegmentationQC** — Quality control checker:
  - Connected component analysis (detect fragmentation)
  - Volume ratio check (prediction vs expected range)
  - Confidence score (mean softmax probability)
  - Border touching check (mask touching volume edges)
- **evaluate_segmentation_quality()** — Convenience function

## Test Plan
- `tests/v2/unit/test_segmentation_qc.py` (~12 tests)
  - TestQCFlag: enum values
  - TestQCResult: construction, score
  - TestSegmentationQC: connected components, volume ratio, confidence, border check
  - TestEvaluateQuality: end-to-end, pass/fail scenarios
