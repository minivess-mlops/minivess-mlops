# DATA-CARE Data Quality Assessment — Implementation Plan (Issue #11)

## Current State
- Pandera schemas validate metadata/metrics at DataFrame level
- Great Expectations handles batch quality gates
- whylogs profiles datasets for drift
- Deepchecks Vision handles image-level validation
- No multi-dimensional quality scoring framework

## Architecture

### New Module: `src/minivess/validation/data_care.py`

Maps DATA-CARE quality dimensions to MinIVess data pipeline:

1. **QualityDimension** — Enum of DATA-CARE quality dimensions
   - COMPLETENESS, CORRECTNESS, CONSISTENCY, UNIQUENESS, TIMELINESS, REPRESENTATIVENESS

2. **DimensionScore** — Per-dimension score (0-1) with details
   - dimension, score, max_score, issues list

3. **DataQualityReport** — Aggregate report across all dimensions
   - dimension_scores, overall_score, passed (bool), gate_threshold

4. **assess_nifti_quality()** — Score NIfTI metadata DataFrame
   - Completeness: missing value fraction
   - Correctness: valid ranges (spacing, intensity)
   - Consistency: affine validity, spacing uniformity
   - Uniqueness: duplicate file paths

5. **assess_metrics_quality()** — Score training metrics DataFrame
   - Completeness: null metric values
   - Correctness: metric ranges (Dice 0-1, loss >= 0)
   - Consistency: epoch monotonicity

6. **quality_gate()** — Convert DataQualityReport to GateResult

### Integration Points
- Consumes same DataFrames as Pandera schemas and GE suites
- Returns GateResult for pipeline integration
- Scores logged to MLflow via ExperimentTracker

## New Files
- `src/minivess/validation/data_care.py`

## Modified Files
- `src/minivess/validation/__init__.py` — add DATA-CARE exports

## Test Plan
- `tests/v2/unit/test_data_care.py` (~15 tests)
