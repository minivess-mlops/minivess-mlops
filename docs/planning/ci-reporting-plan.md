# Confidence Interval Reporting — Implementation Plan (Issue #6)

## Current State
- MetricResult has `values: dict[str, float]` — flat dict, no CIs
- SegmentationMetrics computes Dice + F1 via TorchMetrics
- PRD recommends percentile bootstrap (André et al., 2026)
- Conformal prediction already implemented (mapie_conformal.py)

## Architecture

### New Module: `src/minivess/pipeline/ci.py`
- **ConfidenceInterval** — Dataclass: point_estimate, lower, upper, confidence_level, method
- **MetricWithCI** — Dataclass wrapping metric name + ConfidenceInterval
- **bootstrap_ci()** — Percentile bootstrap for a metric function over samples
- **bca_bootstrap_ci()** — BCa bootstrap for small sample sizes
- **compute_metrics_with_ci()** — Compute Dice/HD95 with CIs from per-sample values

### Modified Module: `src/minivess/pipeline/metrics.py`
- No changes needed — CI computation is a post-hoc step over per-sample metrics

## Test Plan
- `tests/v2/unit/test_ci.py` (~14 tests)
  - TestConfidenceInterval: construction, validation, to_dict
  - TestBootstrapCI: known distribution, narrower with more samples, confidence levels
  - TestBcaBootstrapCI: known distribution, small samples
  - TestComputeMetricsWithCI: integration with metric functions
