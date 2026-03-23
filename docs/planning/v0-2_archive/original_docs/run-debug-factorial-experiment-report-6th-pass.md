# 6th Pass Debug Factorial — Live Execution Report

**Date**: 2026-03-23
**Branch**: `test/run-debug-gcp-5th-pass` (continuing — 6th pass is a relaunch after fixes)
**Status**: PRE-LAUNCH (report created BEFORE any sky jobs launch, per Rule H1)
**XML Plan**: `run-debug-factorial-experiment-6th-pass.xml`
**Harness version**: 0.2.0

---

## Cognitive Engagement: Why These Watchlist Items?

**WHY I expect these specific items:**
1. GCP controller is NEW (first run without RunPod) — latency is a hypothesis, not a measurement
2. MambaVesselNet has NEVER succeeded in 5 passes — mamba-ssm import on L4 is untested
3. SWAG scaling formula is NEW code — `min(configured, max(2, max_epochs//5))` first cloud execution
4. Checkpoint namespace `{model}_{loss}/fold_{id}` is NEW — first time on shared GCS mount

**WHAT might go wrong that passes 1-5 haven't revealed:**
- GCP controller bootstrap time (first `sky jobs launch` creates it — could be 5 min or 15 min)
- MambaVesselNet CUDA ops may differ between local (RTX 2070) and cloud (L4)
- SWAG with 2 scaled epochs may produce degenerate posterior (too few weight samples)
- Concurrent checkpoint writes to GCS via MOUNT_CACHED could have ordering issues

---

## Pre-Launch QA

| # | Fix Since 5th Pass | Category |
|---|-------------------|----------|
| 1 | Controller: RunPod → GCP (same-cloud, ~5 min/submission expected) | CRITICAL |
| 2 | Preflight check #10: controller cloud matches job cloud | CRITICAL |
| 3 | Config architecture: controller + infrastructure in cloud YAMLs | HIGH |
| 4 | SWAG scaling: `min(configured, max(2, max_epochs // 5))` | HIGH |
| 5 | Perf metrics: 6 perf/* keys in MetricKeys | MEDIUM |
| 6 | SWAG timing instrumentation (perf/swag_seconds logged) | MEDIUM |
| 7 | smoke_local.yaml → local config (not gcp_spot) | LOW |
| 8 | sync_sky_config.py: generates ~/.sky/config.yaml from repo config | HIGH |

Total new tests since 5th pass: 32 (config chain validation + SWAG scaling + perf metrics)

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

**Progress**: 0/34 SUCCEEDED | 0 FAILED | 0 RUNNING | 34 PENDING

---

## Cost Table (updated after each terminal job)

| Job ID | Condition | Duration | Cost | Status |
|--------|-----------|----------|------|--------|
| — | — | — | — | — |
| **Total** | | | **$0.00** | |

---

## Watchlist (carried from 5th pass + new)

| ID | Item | Severity | Source | Status |
|----|------|----------|--------|--------|
| W1 | GCP controller bootstrap time | HIGH | NEW | ✅ ~3 min (europe-west1-b). Failed in europe-north1 (n4 quota=0). |
| W2 | SAM3 FP16 overflow → NaN | HIGH | Passes 1-5 | ⏳ Pending (SAM3 jobs not yet submitted) |
| W3 | MambaVesselNet mamba-ssm import on L4 | HIGH | Passes 1-5 | ⏳ Pending |
| W4 | SWAG scaling with 2 epochs | MEDIUM | NEW | ⏳ Pending (first job still PENDING) |
| W5 | Checkpoint namespace on shared GCS | MEDIUM | NEW | ⏳ Pending |
| W6 | Submission latency with GCP controller | MEDIUM | NEW | ✅ ~3 min/submission (was 36 min, 12x improvement) |
| W7 | Spot preemption recovery | LOW | Passes 1-5 | ⏳ Not yet triggered |
| W8 | DeepVess data pull for zero-shot | MEDIUM | 5th pass | ⏳ Pending |
| W9 | L4 spot availability | NEW | 6th pass | ⚠️ 27+ min PENDING, queue for spots |
| W10 | GCP n4 quota in europe-north1 | NEW | 6th pass | ❌ CPUS_PER_VM_FAMILY=0, switched to europe-west1 |

---

## Cloud Observations (compound learning — updated during execution)

### O1: GCP controller bootstrap — 3 min (12x improvement over RunPod)
Controller on GCP europe-west1-b provisioned in ~3 min. First job submitted almost
immediately after. Compare: RunPod controller was 36 min/submission.
**Impact**: 34 jobs × 3 min = ~1.7 hours submission time (vs 20 hours with RunPod).

### O2: europe-north1 has zero n4 CPU quota — NOT usable for controller
`CPUS_PER_VM_FAMILY` quota for n4 is 0 in europe-north1. The controller tried
europe-north1-a, failed, then the config was changed to europe-west1.
**Test opportunity**: Preflight check for CPU/GPU quota before provisioning.
**Issue**: Need to request n4 quota increase in europe-north1.

### O3: L4 spot availability — extended PENDING times
Jobs have been PENDING for 27+ min waiting for L4 spot VMs. This is a GCP capacity
issue, not a SkyPilot issue. Spot is correct for cost savings but can have wait times.
**Test opportunity**: Monitor PENDING duration, alert if > 30 min.
**Issue #914**: Add on-demand fallback option.

### O4: Project-level .sky.yaml works correctly
SkyPilot reads `.sky.yaml` from repo root and uses it as project-level config.
`allowed_clouds` warning appears (server vs client mismatch) but doesn't block.
**Compound learning**: Version-controlled SkyPilot config is the right architecture.

---

## New Test Opportunities (compound learning)

*(Empty — will be filled from observations)*

---

*Report created: 2026-03-23 pre-launch. Last update: pre-launch*
