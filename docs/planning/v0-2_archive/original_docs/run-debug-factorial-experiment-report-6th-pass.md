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
| W1 | GCP controller bootstrap time | HIGH | NEW (first GCP controller) | ⏳ |
| W2 | SAM3 FP16 overflow → NaN | HIGH | Passes 1-5 (T4 banned, BF16 on L4) | ⏳ |
| W3 | MambaVesselNet mamba-ssm import on L4 | HIGH | Passes 1-5 (never succeeded) | ⏳ |
| W4 | SWAG scaling with 2 epochs | MEDIUM | NEW (formula first cloud run) | ⏳ |
| W5 | Checkpoint namespace on shared GCS | MEDIUM | NEW (first cloud run with namespace) | ⏳ |
| W6 | Submission latency with GCP controller | MEDIUM | NEW (was 36 min with RunPod) | ⏳ |
| W7 | Spot preemption recovery | LOW | Passes 1-5 (never triggered) | ⏳ |
| W8 | DeepVess data pull for zero-shot | MEDIUM | 5th pass (added conditional pull) | ⏳ |

---

## Cloud Observations (compound learning — updated during execution)

*(Empty — will be filled during monitoring)*

---

## New Test Opportunities (compound learning)

*(Empty — will be filled from observations)*

---

*Report created: 2026-03-23 pre-launch. Last update: pre-launch*
