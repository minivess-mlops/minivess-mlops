# CyclOps Healthcare ML Auditing Integration — Implementation Plan (Issue #12)

## Current State
- Drift detection in validation/drift.py (KS/PSI) and observability/drift.py (MMD)
- Model cards in compliance/model_card.py
- Audit trails in compliance/audit.py
- No fairness evaluation or subgroup analysis
- cyclops-query package unavailable (resolution failure with Python 3.13)

## Architecture

### New Module: `src/minivess/compliance/fairness.py`

CyclOps-inspired healthcare ML auditing with fairness evaluation.

1. **SubgroupMetrics** — Per-subgroup performance metrics
   - subgroup_name, size, metric values

2. **FairnessReport** — Aggregate fairness evaluation
   - subgroup_metrics, disparity_scores, passed/threshold

3. **evaluate_subgroup_fairness()** — Core evaluation
   - Applies metric function per subgroup
   - Computes max-min disparity across subgroups

4. **compute_disparity()** — Disparity metric
   - Max-min ratio for given metric across subgroups

5. **generate_audit_report()** — Markdown audit report
   - Healthcare-compliance formatted report for FDA/EU requirements

### Integration Points
- Consumes model predictions + subgroup labels (demographic metadata)
- Returns GateResult via fairness_gate() for pipeline integration
- Markdown reports compatible with ModelCard + RegulatoryDocGenerator

## New Files
- `src/minivess/compliance/fairness.py`

## Modified Files
- `src/minivess/compliance/__init__.py` — add fairness exports

## Test Plan
- `tests/v2/unit/test_fairness.py` (~15 tests)
