# 7th Pass Debug Factorial — Live Execution Report

**Date**: 2026-03-24
**Branch**: `test/run-debug-gcp-5th-pass`
**Status**: PRE-LAUNCH (report created BEFORE any sky jobs launch, per Rule H1)
**Harness version**: 0.2.1

---

## Cognitive Engagement: Why This Pass Is Different

**What changed since 6th pass:**
- 5 critical bugs fixed in run_factorial.sh (bottom-up review found wrong awk column,
  FAILED jobs never retried, integer overflow, silent preflight skip, silent sync failure)
- Fire-and-forget wrapper: run_factorial_resilient.sh with 1-week timeout
- job_recovery: max_restarts_on_errors: 3 added to all spot SkyPilot YAMLs
- 5-layer YAML contract enforcement (70 tests, pre-commit hook, preflight)
- AsyncCheckpointUploader designed (not yet implemented — research phase)

**What I expect to see:**
- GCP controller on europe-west1 provisions in ~3 min (verified in 6th pass)
- L4 spot may still queue (capacity dependent — 8+ hrs in 6th pass)
- With --resume working correctly now (fixed awk column), partial submissions recover
- With job_recovery, spot preemption + user code errors auto-retry (up to 3x)
- DynUNet should succeed (~30 min/job based on 5th pass data)
- MambaVesselNet/SAM3 TopoLoRA: FIRST TIME with all fixes — unknown outcome

**What might go wrong that prior passes haven't revealed:**
- MambaVesselNet mamba-ssm on L4 cloud VM (never completed a full run)
- SWAG scaling with max_epochs=2 (new formula, first cloud execution)
- Checkpoint namespace {model}_{loss}/fold_{id} on shared GCS mount (first cloud run)

---

## Pre-Launch QA Summary

| Session | Fixes Applied |
|---------|-------------|
| This session | 5 critical bash bugs (awk column, FAILED retry, integer, preflight, sync) |
| This session | fire-and-forget wrapper (run_factorial_resilient.sh) |
| Other session | job_recovery, retry, resume, signal handling, parallel submissions |
| Other session | AsyncCheckpointUploader, 4-flow validation tests, GCP quota preflight |
| Prior sessions | 4 CRITICAL silent failures, YAML contract, checkpoint namespace, SWAG scaling |

---

## Job Status Matrix (Live — updated after each poll)

### DynUNet (8 conditions)
| Loss | aux_calib=true | aux_calib=false |
|------|---------------|-----------------|
| cbdice_cldice | ⏳ | ⏳ |
| dice_ce | ⏳ | ⏳ |
| dice_ce_cldice | ⏳ | ⏳ |
| bce_dice_05cldice | ⏳ | ⏳ |

### MambaVesselNet (8 conditions)
| Loss | aux_calib=true | aux_calib=false |
|------|---------------|-----------------|
| cbdice_cldice | ⏳ | ⏳ |
| dice_ce | ⏳ | ⏳ |
| dice_ce_cldice | ⏳ | ⏳ |
| bce_dice_05cldice | ⏳ | ⏳ |

### SAM3 TopoLoRA (8 conditions)
| Loss | aux_calib=true | aux_calib=false |
|------|---------------|-----------------|
| cbdice_cldice | ⏳ | ⏳ |
| dice_ce | ⏳ | ⏳ |
| dice_ce_cldice | ⏳ | ⏳ |
| bce_dice_05cldice | ⏳ | ⏳ |

### SAM3 Hybrid (8 conditions)
| Loss | aux_calib=true | aux_calib=false |
|------|---------------|-----------------|
| cbdice_cldice | ⏳ | ⏳ |
| dice_ce | ⏳ | ⏳ |
| dice_ce_cldice | ⏳ | ⏳ |
| bce_dice_05cldice | ⏳ | ⏳ |

### Zero-Shot Baselines (2 conditions)
| Model | Dataset | Status |
|-------|---------|--------|
| sam3_vanilla | minivess | ⏳ |
| vesselfm | deepvess | ⏳ |

**Progress**: 0/34 SUCCEEDED | 0 FAILED | 0 RUNNING | 15 PENDING | 19 not yet submitted
**Resilient wrapper**: PID 21143, attempt 1 (still submitting), 161h remaining

---

## Cost Table (updated after each terminal job)

| Job ID | Condition | Duration | Cost | Status |
|--------|-----------|----------|------|--------|
| 3-17 | 8 DynUNet conditions (some duplicates) | 6-12h PENDING | ~$0 (not yet provisioned) | PENDING |
| **Total** | | | **~$0.00** | |

Note: PENDING jobs cost $0 — only RUNNING jobs consume GPU credits.

---

## Watchlist

| ID | Item | Severity | Source | Status |
|----|------|----------|--------|--------|
| W1 | L4 spot availability | **CRITICAL** | 6th+7th pass | ⚠️ 15 jobs PENDING 6-12 hrs, ZERO provisioned |
| W2 | MambaVesselNet mamba-ssm on L4 | HIGH | Never succeeded | ⏳ Not yet submitted |
| W3 | SWAG scaling (min formula) | MEDIUM | New code | ⏳ Not yet running |
| W4 | Checkpoint namespace on GCS | MEDIUM | New code | ⏳ Not yet running |
| W5 | --resume correctly skips existing | HIGH | Bug fixed this session | ⏳ Not yet tested (first pass still submitting) |
| W6 | job_recovery auto-retry on failure | MEDIUM | New feature | ⏳ Not yet triggered |
| W7 | Resilient wrapper fire-and-forget | HIGH | New script | ✅ Running (PID 21143, 6.5h alive) |
| W8 | Duplicate job submissions | MEDIUM | 7th pass observed | ⚠️ 3 conditions have 2-3 duplicates (see O2) |

---

## Cloud Observations (updated during execution)

### O1: GCP L4 Spot Market — Extended Unavailability (12+ hours)

**Data collected across passes 5-7:**

| Pass | Date | Region | Duration PENDING | Jobs Provisioned | Outcome |
|------|------|--------|-----------------|-----------------|---------|
| 5th | 2026-03-22 | europe-north1 (via RunPod ctrl) | ~15 min | 4/7 DynUNet | 4 SUCCEEDED (~30 min each) |
| 6th | 2026-03-23 | europe-west1 (GCP ctrl) | 8+ hrs | 0/5 | CANCELLED (A100 removed) |
| 7th | 2026-03-24 | auto (GCP ctrl) | **12+ hrs and counting** | **0/15** | Still PENDING |

**L4 spot availability patterns:**
- 5th pass provisioned within 15 min (2026-03-22 evening UTC)
- 6th pass ZERO provisioning in 8+ hours (2026-03-23 daytime UTC)
- 7th pass ZERO provisioning in 12+ hours (2026-03-24 early morning UTC)
- The 5th pass success may have been due to RunPod controller region (US) — SkyPilot
  may have selected US-based L4 zones. With europe-west1 controller, SkyPilot may
  be preferring European zones which have less L4 spot capacity.

**ROOT CAUSE FOUND**: `europe-north1` (Finland, where our GCS bucket lives) does NOT have
L4 GPUs AT ALL. It's not in GCP's L4 region/zone list. SkyPilot was searching zones that
physically cannot have L4s. EU L4 regions: europe-west1 (Belgium, 2 zones), europe-west3
(Frankfurt, 2 zones), europe-west4 (Netherlands, 3 zones), europe-west2 (London, 2 zones),
europe-west6 (Zurich, 2 zones) = 12 EU zones total. US has 18 L4 zones across 6 regions.

**Decision**: EU-preferred + US-fallback. SkyPilot `ordered:` preference list:
europe-west4 → europe-west1 → europe-west3 → us-central1 → us-east1 → us-west1.
Cross-continent GCS egress: ~$0.12/GB × 3 GB = $0.36/job (negligible vs 12h PENDING).
No GDPR concern (MiniVess = open-source mouse brain data from Cornell).

**Academic relevance**: This is a real-world constraint for biomedical researchers using
cloud GPUs. The spot market unpredictability should be documented in the Nature Protocols
paper as part of the cost/latency analysis. Recommendation for paper appendix:
- Table of spot provisioning latencies across GPU types and regions
- Comparison of spot vs on-demand for different experiment sizes
- Guidance for researchers: "expect spot queuing of 1-24 hours for L4 in EU regions"

### O2: Duplicate Job Submissions

15 jobs in queue but only 8 unique condition names. Some conditions submitted 2-3 times:
- `dynunet-cbdice_cldice-calibfalse-f0`: 3 copies (from passes 6 + 7)
- `dynunet-bce_dice_05cldice-calibfalse-f0`: 3 copies
- Others: 1-2 copies

**Root cause**: The --resume flag checks `sky jobs queue` for existing ACTIVE jobs,
but older PENDING jobs from previous passes are still in the queue. The idempotent
check works by NAME, so same-named jobs from different passes are detected. However,
the fix in this session (awk column 3 instead of 4) means --resume should now correctly
skip these. The duplicates are from BEFORE the fix was applied.

**Impact**: SkyPilot will run all copies independently. If L4 spots provision, we may
get 2-3 runs of the same condition (wasting ~$0.10-0.30 per duplicate). Not critical
but suboptimal.

**Fix for future**: Cancel duplicate jobs before next launch, OR add a `sky jobs cancel`
step in the resilient wrapper for jobs with duplicate names.

### O3: Resilient Wrapper Successfully Stays Alive

The `run_factorial_resilient.sh` wrapper (PID 21143) has been alive for 6.5+ hours
(since 04:29 UTC). It survived terminal close (nohup). The inner `run_factorial.sh`
is still in its first pass (submitting with parallel subshells). The wrapper has not
yet needed to retry (attempt 1 still in progress).

**Validation**: The fire-and-forget mechanism works. The PID file
(`outputs/resilient_factorial.pid`) contains the correct PID.

### O4: Parallel Submission Working (18 active processes)

18 active launch processes detected. The parallel submission (`wait -n` with
`PARALLEL_SUBMISSIONS=4` from config) is working. The script is actively
submitting conditions while previous ones queue for spots.

---

## GCP L4 Spot Market Analysis (for manuscript appendix)

### Methodology

Spot GPU availability data collected from 7 factorial experiment passes over 5 days
(2026-03-19 to 2026-03-24). Each pass submits 32-34 conditions as SkyPilot managed
spot jobs on GCP, requesting L4 GPUs (24 GB VRAM, Ada Lovelace architecture).

### Observations

1. **L4 spot provisioning is highly variable**: From <15 min (5th pass) to 12+ hours
   with no provisioning (7th pass). No correlation with time-of-day found in our
   limited sample.

2. **European regions appear more constrained**: All jobs target GCP with the controller
   in europe-west1. SkyPilot auto-selects the cheapest available zone. EU L4 capacity
   may be more limited than US.

3. **Spot pricing**: L4 spot at ~$0.22/hr (vs $0.70/hr on-demand, 3.2x savings).
   The savings are real but the latency is unpredictable.

4. **SkyPilot queuing behavior**: PENDING jobs wait indefinitely. SkyPilot periodically
   retries provisioning. `job_recovery.max_restarts_on_errors: 3` handles execution
   failures but NOT provisioning delays.

5. **Cost of waiting**: PENDING jobs cost $0 (no GPU allocated). The only cost is
   the SkyPilot controller VM (~$0.20/hr for n4-standard-4). Over 12 hours of
   waiting, controller cost is ~$2.40.

### Recommendations for Nature Protocols Paper

- **Table**: "Expected spot GPU provisioning latency by region and time"
- **Guidance**: "For time-sensitive experiments, use on-demand instances (+3.2x cost)"
- **Guidance**: "For cost-sensitive experiments, use spot + resilient launcher (fire-and-forget)"
- **Figure**: Gantt chart of job lifecycle across passes (PENDING → STARTING → RUNNING → SUCCEEDED)

---

## New Test Opportunities

### From O1: Spot market monitoring
```
tests/v2/cloud/test_spot_provisioning_latency.py
  test_log_pending_duration_to_mlflow — measure and log how long each job was PENDING
  test_controller_cost_during_waiting — estimate controller VM cost while jobs queue
```

### From O2: Duplicate detection
```
tests/v2/unit/config/test_resume_dedup.py
  test_resume_skips_duplicate_names — verify --resume handles same-named jobs from prior passes
  test_resume_retries_failed_jobs — verify FAILED jobs are retried, not skipped
```

---

*Report created: 2026-03-24T04:29 UTC. Last update: 2026-03-24T11:10 UTC*
