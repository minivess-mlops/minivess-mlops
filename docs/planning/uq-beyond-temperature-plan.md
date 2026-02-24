# UQ Beyond Temperature Scaling — Implementation Plan (Issue #41)

## Current State
- Temperature scaling + ECE/MCE in `ensemble/calibration.py`
- `EnsemblePredictor` with mean/vote/weighted in `ensemble/strategies.py`
- SegResNet has `dropout_prob=0.2` (ready for MC Dropout)
- MAPIE and netcal in pyproject.toml dependencies
- `SegmentationOutput` returns both logits and softmax predictions

## Architecture: Three UQ Methods

### 1. MC Dropout (`ensemble/mc_dropout.py`)
- Enable dropout at inference via `model.train()` on dropout layers only
- Run N stochastic forward passes, collect predictions
- Compute mean prediction + voxel-wise uncertainty (predictive entropy or variance)
- Export uncertainty map as a separate tensor

### 2. Deep Ensembles (`ensemble/deep_ensembles.py`)
- Build on existing `EnsemblePredictor` infrastructure
- Add disagreement-based uncertainty: per-voxel variance across M members
- Return `UncertaintyOutput` with prediction + uncertainty_map
- 3-model ensemble as minimum viable (per PRD)

### 3. Conformal Prediction (`ensemble/conformal.py`)
- MAPIE-based wrapper for mask-level coverage guarantees
- Calibrate on held-out calibration set
- Produce prediction sets at user-specified alpha (default 0.1)
- Return coverage-guaranteed prediction masks

## Shared Output Dataclass
```python
@dataclass
class UncertaintyOutput:
    prediction: Tensor       # (B, C, D, H, W) mean probabilities
    uncertainty_map: Tensor   # (B, 1, D, H, W) voxel-wise uncertainty
    method: str               # "mc_dropout" | "deep_ensemble" | "conformal"
    metadata: dict            # method-specific info (n_samples, alpha, etc.)
```

## New Files
1. `src/minivess/ensemble/mc_dropout.py`
2. `src/minivess/ensemble/deep_ensembles.py`
3. `src/minivess/ensemble/conformal.py`
4. `tests/v2/unit/test_uq.py`

## Modified Files
- `src/minivess/ensemble/__init__.py` — add new exports

## Test Plan (~25 tests)
- TestUncertaintyOutput: dataclass fields, method values
- TestMCDropout: forward pass, uncertainty shape, dropout active, deterministic model
- TestDeepEnsembles: uncertainty from disagreement, single model, prediction shape
- TestConformalPrediction: calibration, prediction sets, coverage, alpha effect
- TestBenchmark: synthetic comparison of methods
