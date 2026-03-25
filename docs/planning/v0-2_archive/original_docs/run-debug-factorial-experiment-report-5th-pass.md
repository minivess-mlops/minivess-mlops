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

**Progress**: 4/34 SUCCEEDED | 0 FAILED | 1 RUNNING | 29 PENDING
**Submission rate**: 5 submitted in 3 hrs (~36 min/submission via RunPod controller)
**ETA**: ~17-20 hours for all 34 (bottleneck: sequential launch through RunPod controller)

## Cost

| Item | Duration | Cost |
|------|----------|------|
| Job 69 (FAILED_SETUP — total_mem bug) | 16m setup | ~$0.06 |
| Job 70 (CANCELLED) | 10m starting | ~$0.04 |
| Job 71 (SUCCEEDED) | 44m total | ~$0.16 |
| Job 72 (SUCCEEDED) | 46m total | ~$0.17 |
| Job 73 (SUCCEEDED) | 46m total | ~$0.17 |
| Job 74 (SUCCEEDED) | 45m total | ~$0.17 |
| Job 75 (RUNNING) | ~36m so far | ~$0.13 |
| **Total so far** | | **~$0.90** |

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

### O4: Sequential launch bottleneck (MAJOR — 17-20 hr total runtime)
Measured: ~36 min between job submissions (worse than initial 20 min estimate).
34 jobs × 36 min = ~20 hours just for submissions.

**Root cause**: `run_factorial.sh` runs `sky jobs launch` sequentially. Each call
SSHes to the RunPod-hosted SkyPilot controller VM (EU-CZ-1), which then provisions
a GCP L4 spot VM. Cross-cloud SSH round-trip adds ~15-20 min overhead.

**Three improvement paths**:
1. **Move controller to GCP** — same-cloud = ~5 min/submission → 34×5 = 3 hrs (6x faster)
2. **Parallel submissions** — `&` + `wait` with N=4 → 34/4×10 = 1.5 hrs (13x faster)
3. **Pre-warm controller** — already happening, marginal improvement only

**Recommendation**: Path 2 (parallel) lowest effort. Path 1 (GCP controller) for production.

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

## CRITICAL FAILURE: RunPod Controller for GCP Jobs (O7)

### O7: SkyPilot controller on WRONG CLOUD — destroyed the entire run

**The single most damaging infrastructure failure in all 5 passes.**

At ~12:27 UTC, after 7 DynUNet jobs were submitted and 4 SUCCEEDED, the RunPod
network went down (`api.runpod.io` unreachable). This killed the SkyPilot jobs
controller (which was running on RunPod EU-CZ-1). Result:

- **25 of 34 trainable conditions** never submitted (LAUNCH_FAILED)
- **2 zero-shot baselines** never submitted (LAUNCH_FAILED)
- **Controller state lost** — `sky jobs queue` returns empty
- **Job history gone** — cannot verify if jobs 75-77 completed
- **~6 hours of wall-clock time wasted** (running on a broken topology)

### Root Cause

`~/.sky/config.yaml` had `cloud: runpod` for the jobs controller — a leftover
from the RunPod development phase (March 2026, RunPod debug profiling sessions).
Nobody checked this when switching to GCP for the factorial run.

```yaml
# ~/.sky/config.yaml (BEFORE fix)
jobs:
  controller:
    resources:
      disk_size: 40
      cloud: runpod   # ← RunPod controller for GCP jobs. WHY?
```

This created a pathological architecture:
```
Local machine → SSH → RunPod controller (EU-CZ-1) → SSH → GCP L4 VM
                       ↑ SINGLE POINT OF FAILURE ↑
```

### Why This Wasn't Caught by 6 Reviewer Agents

1. **Reviewers audited the task YAML** (`train_factorial.yaml`) — not `~/.sky/config.yaml`
2. **Reviewers audited train_flow.py** argparse — not the launch infrastructure
3. **Reviewers audited the Docker image** — not the SkyPilot controller placement
4. **Preflight checked `sky check gcp`** — which tests the GCP *backend*, not WHERE the controller runs
5. **The 4th pass "worked" despite this** — DynUNet/SAM3 Hybrid succeeded, so the misconfiguration appeared harmless (just slow)
6. **No test existed** for controller cloud vs. job cloud consistency

**This is a systemic code review failure**: 6 specialized agents with detailed
prompts, hundreds of lines of review output, and NONE checked the one config
file (`~/.sky/config.yaml`) that controls WHERE the entire infrastructure runs.
The agents reviewed the "what" (YAML, argparse, Docker) but not the "where"
(controller placement, network topology, single points of failure).

### Fix Applied

1. Changed `~/.sky/config.yaml`: `cloud: runpod` → `cloud: gcp`
2. Added preflight check #10: controller cloud must match job cloud
3. Metalearning doc: `2026-03-23-skypilot-controller-on-wrong-cloud.md`

### Impact on Launch Bottleneck (O4 revised)

The 36 min/submission was NOT inherent to SkyPilot. It was caused by cross-cloud
SSH through RunPod. With a GCP controller:
- Expected submission latency: ~5 min (same-cloud, no cross-provider SSH)
- Expected total for 34 jobs: ~3 hours (vs. 20 hours with RunPod controller)
- This is a **6x improvement** without any code changes to `run_factorial.sh`

Issue #913 Path 1 (GCP controller) is now implemented. Path 2 (parallel
submissions) remains valuable for further optimization.

## Issues Created

- [x] #912 — Experiment harness meta-skill (P0)
- [x] #913 — Factorial launch bottleneck — 36 min/submission (P1, root cause found: RunPod controller)

## Issues To Create (next session)

- [ ] SWAG overhead measurement — expected vs actual for each model family
- [ ] Setup time baseline — measure and track across passes
- [ ] Preflight: verify controller cloud matches job cloud (DONE in code, need test)

## Compound Learning Summary

| Pass | Jobs Attempted | Jobs Succeeded | Root Causes Found | New Tests | Cost |
|------|---------------|----------------|-------------------|-----------|------|
| 1st (local) | 24 | 0 | Docker, DVC, env vars | 0 | $0 |
| 2nd (local) | 24 | ~12 | MONAI ROI, MLflow tags | 0 | $0 |
| 3rd (GCP) | 32 | 14 | BF16, checkpoint, SWAG | 0 | ~$8 |
| 4th (GCP) | 32 | 14 | DVC bare pull, job_recovery, missing arg | 0 | ~$6 |
| **5th (GCP)** | **34** | **4** | **total_mem typo, RunPod controller, CLI args** | **57** | **~$1** |

The 5th pass was the first to systematically add tests BEFORE and DURING the run.
57 new tests. But the RunPod controller misconfiguration shows that infrastructure
topology review (not just code review) is essential.

## Next Pass (6th) Preparation

1. **Use `/experiment-harness` skill** (not ad-hoc) — Issue #912
2. **GCP controller verified** — `cloud: gcp` in config, preflight check #10
3. **Relaunch all 34 conditions** — DynUNet should complete in ~3 hrs (vs. 20 hrs)
4. **Focus watchlist**: MambaVesselNet (mamba-ssm import), SAM3 TopoLoRA (VRAM), zero-shot
5. **Carry forward**: All 8 watchlist items from 5th pass (W2-W8 unresolved)

---

*Report finalized: 2026-03-23T13:00 UTC*
*Next update: 6th pass launch (after GCP controller relaunch)*
