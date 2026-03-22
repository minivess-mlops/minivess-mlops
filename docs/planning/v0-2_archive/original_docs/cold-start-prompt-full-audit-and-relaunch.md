# Cold-Start Prompt: Full Audit + 4th Pass Re-Launch

Copy-paste this into a new Claude Code session:

---

Continue from branch: `test/debug-factorial-4th-pass`

## CRITICAL CONTEXT: TRUST IS BROKEN

The previous agent sessions created 330 planning docs but failed to implement
critical components. 94 metalearning failure docs exist and error frequency is
ACCELERATING (6.7/day, up from 2.0/day). Issue #909 tracks this systemic failure.

**YOU MUST NOT:**
- Create planning docs when asked to implement
- Report "all tests pass" without listing what's NOT tested
- Assume any planned feature exists without verifying the source code
- Defer implementation to "next session" or "separate PR"

**YOU MUST:**
- Implement code FIRST, plan only when explicitly asked
- Verify every claim by reading actual source files
- Ask the user before choosing between plan vs implement
- List missing test coverage when reporting test results

## TASK 1 (BLOCKING): Full Audit of Planning Docs vs Implementation

Scan ALL 330 files in `docs/planning/` and cross-reference with source code.
For each plan:
- Does the source code it references actually exist?
- If status=planning: is there ANY implementation?
- If status=implemented: verify at least one referenced file exists

Focus first on the 10 XML plans with `status: planning`:

```
analysis-flow-debug-double-check.xml — P0
biostatistic-flow-debug-double-check.xml — P0
debug-factorial-local-post-analysis-biostats.xml — P0
mambavesselnet-implementation-plan.xml — P?
novel-loss-debugging-plan.xml — P?
post-training-flow-debug-double-check.xml — P0
run-debug-factorial-experiment-report-2nd-pass-fix-plan.xml — P0
run-debug-factorial-experiment.xml — P0
staging-prod-remote-test-suite-splits.xml — P1
test-sets-external-validation-deepvess-tubenet-analysis-flow-for-biostatistics.xml — P0 PUBLICATION BLOCKER
```

Report to user: which plans have code, which are phantom.

## TASK 2 (BLOCKING): DeepVess Data Pipeline

DeepVess is NOT on disk, NOT DVC-tracked, NOT on GCS.
This was requested on 2026-03-19 ("OBLIGATORY, no deferring").

Steps:
1. Download DeepVess from https://ecommons.cornell.edu/items/a79bb6d8-77cf-4917-8e26-f2716a6ac2a3
2. `dvc add data/raw/deepvess`
3. `dvc push data/raw/deepvess -r gcs`
4. Update `dvc.yaml` with deepvess download stage
5. Update `dvc.lock`
6. Wire into Analysis Flow for VesselFM zero-shot evaluation
7. Update `test_dvc_remote_sync.py` to verify deepvess hash

See: `docs/planning/test-sets-external-validation-deepvess-tubenet-analysis-flow-for-biostatistics.xml`
See: `src/minivess/data/external_datasets.py` (config exists, nothing wired)

## TASK 3: Re-Launch 4th Pass Factorial

After Tasks 1-2 are complete:
1. Run preflight: `uv run python scripts/preflight_gcp.py`
2. Launch: `./scripts/run_factorial.sh configs/factorial/debug.yaml`
3. Monitor: `.venv/bin/sky jobs queue -o json` (poll every 30s)
4. Report: `docs/planning/run-debug-factorial-experiment-report-4th-pass.md`

32 GPU jobs + 2 zero-shot = 34 total.
POST_TRAINING_METHODS=none,swag in each job.

## WHAT'S ALREADY IMPLEMENTED (verified, DO NOT re-do)

- Factorial configs consolidated to `configs/factorial/` (4 losses, 640 conditions)
- `bce_dice_05cldice` loss class (Khazem 2025)
- SWAG DataLoader wiring (both parent flow + standalone paths)
- SWAG mean model checkpoint (standard format)
- `scripts/preflight_gcp.py` (9 checks, all pass)
- `tests/v2/unit/deployment/test_skypilot_yamls.py` (17 tests)
- `tests/v2/unit/test_dvc_remote_sync.py` (7 tests)
- DVC pull fix (path-specific)
- SkyPilot YAML fixes (sky path, job_recovery, debug splits)
- Local 3-flow pipeline (8 conditions, 0 issues) — but only MiniVess val, NOT test data

## WHAT'S NOT IMPLEMENTED (phantom plans)

- External test set evaluation (DeepVess into Analysis Flow) — PUBLICATION BLOCKER
- DeepVess data download + DVC tracking
- Analysis Flow ensemble evaluation with 6-factor factorial
- Biostatistics Flow full implementation (v3)
- Post-training flow debug double-check
- Test suite tier redesign
- SkyPilot observability for /factorial-monitor

## ISSUES

- #907: Cloud GPU pipeline gaps (Prefect Cloud — deferred)
- #908: Local SkyPilot test suite (Tier 1 done, Tier 2 deferred)
- #909: Agent harness systemic failure (THIS issue — full audit needed)

## KEY FILES

- `scripts/run_factorial.sh` — launch script (with preflight)
- `scripts/preflight_gcp.py` — 9-check validation
- `deployment/skypilot/train_factorial.yaml` — per-job config
- `configs/factorial/debug.yaml` — 32-cell factorial
- `src/minivess/data/external_datasets.py` — DeepVess config (NOT WIRED)

## METALEARNING (read ALL before doing ANYTHING)

94 total docs. Most critical for this session:
- `2026-03-22-agent-harness-systemic-failure.md` — the meta-problem
- `2026-03-22-systematic-plan-without-implement-biggest-fuckup.md`
- `2026-03-22-planning-instead-of-implementing-silent-deferral.md`
- `2026-03-22-dvc-pull-untested-setup-script-failure.md`
- `2026-03-19-debug-run-is-full-production-no-shortcuts.md`
- `2026-03-07-silent-existing-failures.md` — original warning about stubs

## FACTORIAL DESIGN (DO NOT RE-DERIVE)

```
Layer A: 4 models × 4 losses × 2 aux_calib = 32 cells
Layer B: 2 post-training {none, swag} (same flow)
Layer C: 2 recalibration × 5 ensemble
Total: 640 conditions. Debug: 1 fold, 2 epochs, half data.
```
