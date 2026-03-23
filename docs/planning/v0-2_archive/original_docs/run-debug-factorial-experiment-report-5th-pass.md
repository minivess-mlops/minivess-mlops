# 5th Pass Debug Factorial — Live Execution Report

**Date**: 2026-03-23
**Branch**: `test/run-debug-gcp-5th-pass`
**Status**: IN PROGRESS — DynUNet conditions succeeding, remaining models pending
**XML Plan**: `run-debug-factorial-experiment-5th-pass.xml`

---

## What Was Attempted

Full debug factorial: 4 models × 4 losses × 2 aux_calib × 1 fold = 32 trainable
conditions + 2 zero-shot baselines on GCP L4 spot via SkyPilot.

Post-training: `none,swag` (comma-separated, parent flow iterates internally).

## Pre-Launch QA (Unprecedented Scope)

Before launching, 6 specialist reviewer agents audited the entire pipeline.
Fixes applied BEFORE any GCP spend:

| # | Fix | Category |
|---|-----|----------|
| 1 | CLI args: `--model-family`, `--loss-name`, `--experiment-name` | CRITICAL — 4th pass root cause |
| 2 | MLflow credentials in SkyPilot envs (USERNAME/PASSWORD) | CRITICAL — silent data loss |
| 3 | MLflow health check in setup (fail fast) | CRITICAL — silent data loss |
| 4 | MLflow retry config (10 retries, 300s timeout) | HIGH |
| 5 | DVC files in Docker image (dvc.yaml, dvc.lock, deepvess.dvc) | CRITICAL |
| 6 | DeepVess conditional data pull for zero-shot | HIGH |
| 7 | Zero-shot env vars (POST_TRAINING_METHODS=none) | HIGH |
| 8 | Checkpoint namespace: `{model}_{loss}/fold_{id}` | CRITICAL BUG |
| 9 | GPU preflight guard (exit on CUDA unavailable) | HIGH |
| 10 | `total_mem` → `total_memory` (PyTorch attribute) | RUNTIME — caught in first job |

57 new tests added before launch. Staging: 5819/0/0. Prod: 6152/0/0.

## Execution Timeline

```
2026-03-23T00:05 — Launch script started (Cycle 1, Attempt 1)
2026-03-23T00:07 — Job 69 submitted (dynunet-cbdice_cldice-calibtrue-f0)
2026-03-23T00:23 — Job 69 RUNNING (setup completed, training started)
2026-03-23T00:33 — Job 69 FAILED_SETUP (total_mem → AttributeError)
                   ROOT CAUSE: PyTorch CudaDeviceProperties uses 'total_memory'
                   not 'total_mem'. Typo in GPU preflight guard.
2026-03-23T00:34 — Kill-switch: launch script killed, job 70 cancelled
2026-03-23T00:35 — Fix committed: total_mem → total_memory + new test
2026-03-23T00:37 — Relaunch (Cycle 1, Attempt 2)
2026-03-23T00:38 — Job 71 submitted (dynunet-cbdice_cldice-calibtrue-f0)
2026-03-23T00:52 — Job 71 RUNNING (all preflight checks passed)
2026-03-23T01:22 — Job 71 SUCCEEDED (30m27s job duration)
                   MILESTONE: First fully clean GCP run with all fixes
2026-03-23T01:23 — Job 72 submitted (dynunet-cbdice_cldice-calibfalse-f0)
2026-03-23T01:53 — Job 72 SUCCEEDED (30m43s)
2026-03-23T02:00 — Job 73 submitted (dynunet-dice_ce-calibtrue-f0)
2026-03-23T02:30 — Job 73 SUCCEEDED (30m04s)
2026-03-23T02:40 — Job 74 submitted (dynunet-dice_ce-calibfalse-f0)
2026-03-23T03:10 — Job 74 SUCCEEDED (29m57s)
2026-03-23T03:20 — Job 75 submitted (dynunet-dice_ce_cldice-calibtrue-f0, STARTING)
```

## Job Status Matrix (Live)

### DynUNet (8 conditions)
| Loss | aux_calib=true | aux_calib=false |
|------|---------------|-----------------|
| cbdice_cldice | ✅ SUCCEEDED 30m27s (Job 71) | ✅ SUCCEEDED 30m43s (Job 72) |
| dice_ce | ✅ SUCCEEDED 30m04s (Job 73) | ✅ SUCCEEDED 29m57s (Job 74) |
| dice_ce_cldice | ⏳ STARTING (Job 75) | ⏳ PENDING |
| bce_dice_05cldice | ⏳ PENDING | ⏳ PENDING |

### MambaVesselNet (8 conditions)
| Loss | aux_calib=true | aux_calib=false |
|------|---------------|-----------------|
| cbdice_cldice | ⏳ PENDING | ⏳ PENDING |
| dice_ce | ⏳ PENDING | ⏳ PENDING |
| dice_ce_cldice | ⏳ PENDING | ⏳ PENDING |
| bce_dice_05cldice | ⏳ PENDING | ⏳ PENDING |

### SAM3 TopoLoRA (8 conditions)
| Loss | aux_calib=true | aux_calib=false |
|------|---------------|-----------------|
| cbdice_cldice | ⏳ PENDING | ⏳ PENDING |
| dice_ce | ⏳ PENDING | ⏳ PENDING |
| dice_ce_cldice | ⏳ PENDING | ⏳ PENDING |
| bce_dice_05cldice | ⏳ PENDING | ⏳ PENDING |

### SAM3 Hybrid (8 conditions)
| Loss | aux_calib=true | aux_calib=false |
|------|---------------|-----------------|
| cbdice_cldice | ⏳ PENDING | ⏳ PENDING |
| dice_ce | ⏳ PENDING | ⏳ PENDING |
| dice_ce_cldice | ⏳ PENDING | ⏳ PENDING |
| bce_dice_05cldice | ⏳ PENDING | ⏳ PENDING |

### Zero-Shot Baselines (2 conditions)
| Model | Dataset | Status |
|-------|---------|--------|
| sam3_vanilla | minivess | ⏳ PENDING |
| vesselfm | deepvess | ⏳ PENDING |

**Progress**: 4/34 SUCCEEDED | 0 FAILED | 1 STARTING | 29 PENDING

## Cost

| Item | Duration | Cost |
|------|----------|------|
| Job 69 (FAILED_SETUP — total_mem bug) | 16m setup | ~$0.06 |
| Job 70 (CANCELLED) | 10m starting | ~$0.04 |
| Job 71 (SUCCEEDED) | 44m total | ~$0.16 |
| Job 72 (SUCCEEDED) | 46m total | ~$0.17 |
| Job 73 (SUCCEEDED) | 46m total | ~$0.17 |
| Job 74 (SUCCEEDED) | 45m total | ~$0.17 |
| Job 75 (STARTING) | ~11m so far | ~$0.04 |
| **Total so far** | | **~$0.81** |

Estimated total for 34 conditions: ~$8-10 (at ~$0.17/job × 34 + setup overhead).

## Cloud Observations (Compound Learning)

### O1: Training + SWAG timing breakdown
DynUNet 2-epoch debug run takes ~30 min total:
- Setup (Docker pull + DVC + preflight): ~14 min
- Training (2 epochs): ~5 min (matches 4th pass)
- SWAG post-training: ~11 min
- **SWAG is 2.2x the training time** for a 2-epoch run. At production scale
  (50 epochs), SWAG overhead will be proportionally smaller.

### O2: Setup time dominates short debug runs
14 min setup / 30 min total = 47% of job time is setup overhead.
For production runs (50 epochs, ~2-3 hours), setup becomes <10%.
**Infrastructure test opportunity**: Measure and assert setup time < 20 min.

### O3: Consistent job durations across DynUNet conditions
All 4 DynUNet jobs: 29m57s–30m43s (±1.5%). Excellent reproducibility.
The loss function and aux_calib flags have negligible impact on DynUNet runtime.
**Infrastructure test opportunity**: Assert per-job cost < $0.50 for debug runs.

### O4: Sequential launch bottleneck
~20 min between job submissions. The RunPod SkyPilot controller is the bottleneck.
34 jobs × 20 min = ~11 hours just for submissions.
**Infrastructure improvement**: Investigate parallel `sky jobs launch` or
dedicated GCP controller (instead of RunPod).

### O5: GCP L4 spot provisioning time
~10-15 min from STARTING to RUNNING (includes Docker pull from same-region GAR).
4th pass had similar timing. L4 availability seems good in europe-north1.

### O6: `total_mem` vs `total_memory` — PyTorch attribute names
Discovered at runtime: `torch._C._CudaDeviceProperties` uses `total_memory`, not
`total_mem`. This is a classic "works locally, fails on cloud" bug because our
local tests never actually call this code path on a real GPU.
**Test added**: `test_setup_python_oneliner_is_valid` checks attribute names.

## Watchlist Verification (from XML plan)

| ID | Item | Status |
|----|------|--------|
| W1 | FAILED_SETUP within 5 min | ✅ Caught by kill-switch (job 69, total_mem bug) |
| W2 | SAM3 FP16 overflow → NaN | ⏳ Pending (SAM3 jobs not yet submitted) |
| W3 | Docker pull time from GAR | ✅ ~14 min total setup (acceptable) |
| W4 | DVC pull from GCS | ✅ Working (dvc.yaml/dvc.lock in image) |
| W5 | SWAG uncertainty estimates | ⏳ Pending (need to check MLflow metrics) |
| W6 | Spot preemption recovery | ⏳ Not triggered yet |
| W7 | MambaVesselNet CUDA compilation | ⏳ Pending (mamba jobs not yet submitted) |
| W8 | Cross-flow FlowContract on cloud | ⏳ Pending (post-training needs to complete) |

## New Test Opportunities Identified

### From O1: SWAG timing test
```
tests/v2/cloud/test_training_timing.py
  test_debug_dynunet_job_completes_under_45_min
  test_setup_overhead_under_20_min
```

### From O3: Cost estimation test
```
tests/v2/cloud/test_cost_estimation.py
  test_debug_job_cost_under_50_cents
  test_total_factorial_cost_under_15_dollars
```

### From O4: Launch bottleneck
```
tests/v2/unit/deployment/test_skypilot_launch_rate.py
  test_parallel_launch_feasibility (research spike)
```

### From O6: PyTorch attribute validation
```
tests/v2/unit/orchestration/test_checkpoint_namespace_isolation.py
  test_setup_python_oneliner_is_valid  (ALREADY ADDED ✅)
```

### From General: Setup script contract tests
```
tests/v2/cloud/test_setup_script_execution.py
  test_dvc_pull_produces_expected_file_count
  test_splits_json_created_after_setup
  test_mlflow_reachable_after_setup
  test_nvidia_smi_shows_gpu_after_setup
```

## Issues to Create

- [ ] Sequential launch bottleneck — investigate parallel sky jobs launch (#TBD)
- [ ] SWAG overhead measurement — expected vs actual for each model family (#TBD)
- [ ] Setup time baseline — measure and track across passes (#TBD)

---

*This report is updated live as jobs complete. Last update: 2026-03-23T03:25 UTC*
