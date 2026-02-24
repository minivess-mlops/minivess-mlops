# VessQC-Style Uncertainty-Guided Annotation Curation — Implementation Plan (Issue #10)

## Current State
- MC Dropout and Deep Ensemble UQ → `UncertaintyOutput` with uncertainty_map (B, 1, D, H, W)
- Conformal prediction → `ConformalResult` with prediction_sets
- `AnnotationQualitySchema` in validation/schemas.py (Pandera)
- No uncertainty-guided annotation curation workflow

## Architecture

### New Module: `src/minivess/validation/vessqc.py`

Bridges existing UQ outputs with annotation curation, following Terms et al. (2025).

1. **CurationFlag** — Dataclass for a flagged region in a single sample
   - sample_id, voxel_indices, mean/max uncertainty, volume fraction

2. **CurationReport** — Summary across a batch
   - flags list, totals, threshold used

3. **flag_uncertain_regions()** — Core flagging function
   - Takes uncertainty maps (B, 1, D, H, W) from MCDropout/DeepEnsemble
   - Threshold: explicit value or auto from percentile
   - Returns CurationReport with per-sample CurationFlags

4. **compute_error_detection_metrics()** — VessQC quality metrics
   - Takes CurationReport + ground truth error masks
   - Returns recall, precision, F1 for error detection

5. **rank_samples_by_uncertainty()** — Prioritization
   - Ranks samples by aggregate uncertainty for review queue

### Integration Points
- Consumes `UncertaintyOutput.uncertainty_map` from ensemble module
- Outputs flagging compatible with AnnotationQualitySchema
- Designed for Label Studio / napari integration (returns voxel coordinates)

## New Files
- `src/minivess/validation/vessqc.py`

## Modified Files
- `src/minivess/validation/__init__.py` — add VessQC exports

## Test Plan
- `tests/v2/unit/test_vessqc.py` (~15 tests)
  - TestCurationFlag/Report: dataclass construction
  - TestFlagUncertainRegions: threshold, percentile, empty maps
  - TestErrorDetectionMetrics: recall, precision, perfect/no overlap
  - TestRankSamples: ordering, top_k
