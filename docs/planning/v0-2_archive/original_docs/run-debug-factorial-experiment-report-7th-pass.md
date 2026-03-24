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

**Progress**: 0/34 SUCCEEDED | 0 FAILED | 0 RUNNING | 34 PENDING

---

## Cost Table (updated after each terminal job)

| Job ID | Condition | Duration | Cost | Status |
|--------|-----------|----------|------|--------|
| — | — | — | — | — |
| **Total** | | | **$0.00** | |

---

## Watchlist

| ID | Item | Severity | Source | Status |
|----|------|----------|--------|--------|
| W1 | L4 spot availability | HIGH | 6th pass (8+ hrs PENDING) | ⏳ |
| W2 | MambaVesselNet mamba-ssm on L4 | HIGH | Never succeeded in 6 passes | ⏳ |
| W3 | SWAG scaling (min formula) | MEDIUM | New code, first cloud run | ⏳ |
| W4 | Checkpoint namespace on GCS | MEDIUM | New code, first cloud run | ⏳ |
| W5 | --resume correctly skips existing | HIGH | Bug fixed this session (awk col) | ⏳ |
| W6 | job_recovery auto-retry on failure | MEDIUM | New feature, first cloud test | ⏳ |
| W7 | Resilient wrapper fire-and-forget | HIGH | New script, first use | ⏳ |

---

## Cloud Observations (updated during execution)

*(Empty — will be filled during monitoring)*

---

*Report created: 2026-03-24 pre-launch. Last update: pre-launch*
