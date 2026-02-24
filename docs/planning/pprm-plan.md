# PPRM — Prediction-Powered Risk Monitoring (Issue #4)

## Current State
- Tier 1 (FeatureDriftDetector) and Tier 2 (EmbeddingDriftDetector) already implemented
- DriftResult dataclass exists
- No semi-supervised monitoring with formal guarantees

## Architecture

### New Module: `src/minivess/observability/pprm.py`
- **RiskEstimate** — Dataclass: risk, ci_lower, ci_upper, alarm, threshold
- **PPRMDetector** — Prediction-powered risk monitor
  - `calibrate(cal_predictions, cal_labels, risk_fn)` — compute rectifier from labeled set
  - `monitor(deployment_predictions)` — estimate risk with CI using unlabeled data
  - Semi-supervised: labeled calibration set + unlabeled deployment data
  - Configurable false alarm rate (α)
- **compute_prediction_risk()** — Risk function for segmentation (1 - Dice)

## Algorithm (Zhang et al., 2026)
1. On labeled calibration set: compute true risk R and predicted risk R̂
2. Rectifier δ = mean(R - R̂) per sample
3. On deployment: risk = mean(R̂_deploy) + δ
4. CI via CLT: risk ± z_α * se
5. Alarm if CI lower bound > threshold

## Test Plan
- `tests/v2/unit/test_pprm.py` (~14 tests)
  - TestRiskEstimate: construction, alarm logic, to_dict
  - TestPPRMDetector: calibrate, monitor, alarm thresholds
  - TestComputePredictionRisk: perfect/imperfect predictions
