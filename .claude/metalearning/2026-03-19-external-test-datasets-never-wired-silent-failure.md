# Metalearning: External Test Datasets Never Wired — Silent Existing Failure

**Date:** 2026-03-19
**Severity:** P0 — PUBLICATION BLOCKER
**Trigger:** User asked if Analysis Flow evaluates external test datasets. Answer: NO.
**Pattern:** `2026-03-07-silent-existing-failures.md` — exact same class of failure.

---

## What Happened

1. `external_datasets.py` has a registry with 3 datasets (DeepVess, TubeNet, VesselNN)
2. `test_datasets.py` has a hierarchical DataLoader builder — fully functional
3. `evaluation_runner.py` accepts arbitrary datasets — works with any input
4. `data_flow.py` discovers external datasets and returns them
5. `analysis_flow.py` has a `dataloaders_dict` parameter for external evaluation
6. **BUT `_build_dataloaders_from_config()` returns `{}` (ALWAYS EMPTY)**
7. External datasets are discovered by data_flow, then **silently discarded**
8. Analysis flow evaluates ONLY MiniVess validation folds — NEVER external test data
9. Biostatistics has no train/val vs test separation
10. **Nobody noticed for months because no test verified the end-to-end path**

## Root Causes

### RC1: Stub Treated as Implementation
`_build_dataloaders_from_config()` has a comment: "builds loaders internally when
running in full pipeline mode." This was a TODO that was never implemented. The stub
returns `{}` and the calling code silently accepts the empty dict.

### RC2: No Integration Test for External Evaluation
Unit tests verify each component in isolation (registry, loader, runner). But no
integration test verifies the end-to-end path: data_flow → analysis_flow → evaluation
on external data. The components work individually but are never connected.

### RC3: Same Pattern as 2026-03-07 Silent Failures
This is Instance #9 of the silent failure pattern. Code exists, tests pass in
isolation, but the pipeline is broken because the wiring is missing. Nobody creates
an issue because "it's not related to current changes."

### RC4: VesselNN Role Confusion
Claude assumed VesselNN was a test dataset alongside DeepVess and TubeNet. User
corrected: VesselNN is ONLY for drift detection simulation, NOT for test evaluation.
VesselNN has data leakage to MiniVess (same PI, similar acquisition protocol).

**Correct dataset roles:**
- **DeepVess**: External TEST dataset (generalization evaluation)
- **TubeNet**: External TEST dataset (generalization evaluation)
- **VesselNN**: Drift detection ONLY (NOT for test evaluation, data leakage concern)

## Impact

- MiniVess metrics are train/val only — NOT generalizable
- Paper cannot make generalization claims without external test evaluation
- Biostatistics has no test data to compute the generalization gap
- FDA readiness requires external validation (IEC 62304)
- **All previous "evaluation results" are internal validation only**

## Corrective Actions

1. Wire data_flow external discovery → analysis_flow dataloaders_dict
2. Implement `test/deepvess/{metric}` and `test/tubenet/{metric}` prefix convention
3. Add split={trainval, test} dimension to biostatistics module
4. Update KG: VesselNN = drift detection ONLY, DeepVess + TubeNet = test sets
5. Download + DVC-version DeepVess and TubeNet
6. Add integration test: data_flow → analysis_flow → evaluation on external data
7. Add P0 task: audit ALL planned components for stub vs wired status

## The Rule

**Every planned pipeline component must have an INTEGRATION TEST that verifies
the end-to-end wiring, not just unit tests for individual components.**

A stub that returns `{}` and is silently accepted is NOT "code complete."
It is a publication-blocking bug.

## Related

- `2026-03-07-silent-existing-failures.md` — same failure pattern
- `2026-03-19-debug-run-is-full-production-no-shortcuts.md` — debug must test everything
- `paper_model_comparison.yaml` — defines which models need external evaluation
