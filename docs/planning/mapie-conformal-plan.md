# MAPIE Conformal Prediction Integration — Implementation Plan (Issue #7)

## Current State
- `mapie>=1.3` installed (via pyproject.toml)
- Custom `ConformalPredictor` in ensemble/conformal.py (issue #41)
- No MAPIE integration
- No coverage metrics logging

## Architecture

### New Module: `src/minivess/ensemble/mapie_conformal.py`

1. **MapieConformalSegmentation** — Wraps MAPIE's SplitConformalClassifier for 3D segmentation
   - Flattens 3D voxel probabilities to tabular format for MAPIE calibration
   - Reshapes prediction sets back to 3D spatial format
2. **compute_coverage_metrics()** — Coverage score + mean prediction set width
3. **ConformalMetrics** — Dataclass for coverage, width, and set sizes

### Integration Points
- Calibrate on held-out validation volumes (softmax probs + labels)
- Produce voxel-wise prediction sets with coverage guarantees
- Coverage metrics logged via ExperimentTracker
- Configurable alpha (0.05, 0.1, 0.2)

### Key Design: Voxel-level conformal
- MAPIE SplitConformalClassifier operates on (N, C) probabilities
- 3D volumes flattened: (B, C, D, H, W) → (B*D*H*W, C) for calibration
- Prediction sets reshaped back: (B*D*H*W, C) → (B, C, D, H, W) boolean mask

## New Files
- `src/minivess/ensemble/mapie_conformal.py`

## Modified Files
- `src/minivess/ensemble/__init__.py` — add MAPIE exports

## Test Plan
- `tests/v2/unit/test_mapie_conformal.py` (~15 tests)
