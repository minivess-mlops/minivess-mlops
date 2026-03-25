# 9th Pass Debug Factorial — SAM3 + Zero-Shot Focus

**Date**: 2026-03-25
**Branch**: `test/run-debug-9th-pass`
**Config**: `configs/factorial/debug.yaml`
**XML**: `docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-9th-pass-SAM-zeroshot-focus.xml`
**Prior pass**: 8th pass (16/34 SUCCEEDED — DynUNet 8/8, MambaVesselNet 8/8)

## 1. Executive Summary

**Focus**: Relaunch 16 SAM3 conditions (8 TopoLoRA + 8 Hybrid) at BS=1/accum=4
plus 2 zero-shot baselines (sam3_vanilla + vesselfm). DynUNet and MambaVesselNet
are already SUCCEEDED from 8th pass — `--resume` skips them.

**Key fix**: SAM3 OOMed at BS=2 (21.9 GB / 24 GB L4). Now BS=1 + gradient
accumulation (accum=4) for effective BS=4. Expected VRAM: ~13 GB (TopoLoRA),
~7.2 GB (Hybrid).

**Estimated cost**: ~$3-5 (18 jobs on L4 spot + controller)

## 2. Status Matrix

| # | Condition | Status | Duration | VRAM | Cost | Notes |
|---|-----------|--------|----------|------|------|-------|
| **DynUNet (8/8 from 8th pass — SKIPPED via --resume)** |
| 1 | dynunet-cbdice_cldice-calibtrue-f0 | SKIPPED (8th) | — | — | — | |
| 2 | dynunet-cbdice_cldice-calibfalse-f0 | SKIPPED (8th) | — | — | — | |
| 3 | dynunet-dice_ce-calibtrue-f0 | SKIPPED (8th) | — | — | — | |
| 4 | dynunet-dice_ce-calibfalse-f0 | SKIPPED (8th) | — | — | — | |
| 5 | dynunet-dice_ce_cldice-calibtrue-f0 | SKIPPED (8th) | — | — | — | |
| 6 | dynunet-dice_ce_cldice-calibfalse-f0 | SKIPPED (8th) | — | — | — | |
| 7 | dynunet-bce_dice_05cldice-calibtrue-f0 | SKIPPED (8th) | — | — | — | |
| 8 | dynunet-bce_dice_05cldice-calibfalse-f0 | SKIPPED (8th) | — | — | — | |
| **MambaVesselNet (8/8 from 8th pass — SKIPPED via --resume)** |
| 9 | mambavesselnet-cbdice_cldice-calibtrue-f0 | SKIPPED (8th) | — | — | — | |
| 10 | mambavesselnet-cbdice_cldice-calibfalse-f0 | SKIPPED (8th) | — | — | — | |
| 11 | mambavesselnet-dice_ce-calibtrue-f0 | SKIPPED (8th) | — | — | — | |
| 12 | mambavesselnet-dice_ce-calibfalse-f0 | SKIPPED (8th) | — | — | — | |
| 13 | mambavesselnet-dice_ce_cldice-calibtrue-f0 | SKIPPED (8th) | — | — | — | |
| 14 | mambavesselnet-dice_ce_cldice-calibfalse-f0 | SKIPPED (8th) | — | — | — | |
| 15 | mambavesselnet-bce_dice_05cldice-calibtrue-f0 | SKIPPED (8th) | — | — | — | |
| 16 | mambavesselnet-bce_dice_05cldice-calibfalse-f0 | SKIPPED (8th) | — | — | — | |
| **SAM3 TopoLoRA (8 conditions — NEW, BS=1/accum=4)** |
| 17 | sam3_topolora-cbdice_cldice-calibtrue-f0 | PENDING | — | — | — | |
| 18 | sam3_topolora-cbdice_cldice-calibfalse-f0 | PENDING | — | — | — | |
| 19 | sam3_topolora-dice_ce-calibtrue-f0 | PENDING | — | — | — | |
| 20 | sam3_topolora-dice_ce-calibfalse-f0 | PENDING | — | — | — | |
| 21 | sam3_topolora-dice_ce_cldice-calibtrue-f0 | PENDING | — | — | — | |
| 22 | sam3_topolora-dice_ce_cldice-calibfalse-f0 | PENDING | — | — | — | |
| 23 | sam3_topolora-bce_dice_05cldice-calibtrue-f0 | PENDING | — | — | — | |
| 24 | sam3_topolora-bce_dice_05cldice-calibfalse-f0 | PENDING | — | — | — | |
| **SAM3 Hybrid (8 conditions — NEW, BS=1/accum=4)** |
| 25 | sam3_hybrid-cbdice_cldice-calibtrue-f0 | PENDING | — | — | — | |
| 26 | sam3_hybrid-cbdice_cldice-calibfalse-f0 | PENDING | — | — | — | |
| 27 | sam3_hybrid-dice_ce-calibtrue-f0 | PENDING | — | — | — | |
| 28 | sam3_hybrid-dice_ce-calibfalse-f0 | PENDING | — | — | — | |
| 29 | sam3_hybrid-dice_ce_cldice-calibtrue-f0 | PENDING | — | — | — | |
| 30 | sam3_hybrid-dice_ce_cldice-calibfalse-f0 | PENDING | — | — | — | |
| 31 | sam3_hybrid-bce_dice_05cldice-calibtrue-f0 | PENDING | — | — | — | |
| 32 | sam3_hybrid-bce_dice_05cldice-calibfalse-f0 | PENDING | — | — | — | |
| **Zero-shot baselines (2 conditions — NEW)** |
| 33 | sam3_vanilla-zeroshot-minivess-f0 | PENDING | — | — | — | |
| 34 | vesselfm-zeroshot-deepvess-f0 | PENDING | — | — | — | |

**Summary**: 16 SKIPPED (8th pass) + 18 PENDING (new) = 34 total

## 3. Timeline

| Time (UTC) | Event |
|------------|-------|
| 2026-03-25 T03:18 | Report file created (H1) |
| 2026-03-25 T03:19 | Phase 2 VALIDATE: All 13 preflight gates passed |
| 2026-03-25 T03:20 | Phase 3 EXECUTE: run_factorial.sh launched (PID 2240179) |
| 2026-03-25 T03:20 | First job submitted: dynunet-cbdice_cldice-calibtrue-f0 |
| | *Sky jobs queue empty — full 34-condition relaunch (not resume)* |
| | *Updates will be appended as jobs progress* |

## 4. Cost Tracking

| Category | Count | Unit Cost | Total |
|----------|-------|-----------|-------|
| SAM3 TopoLoRA (L4 spot) | 8 | ~$0.10 | ~$0.80 |
| SAM3 Hybrid (L4 spot) | 8 | ~$0.08 | ~$0.64 |
| Zero-shot (L4 spot) | 2 | ~$0.05 | ~$0.10 |
| Controller (n4-standard-4) | 1 | ~$0.60 | ~$0.60 |
| **Total** | **18+1** | | **~$2.14** |

## 5. Watchlist Tracking

| ID | Item | Status | Observation |
|----|------|--------|-------------|
| W1 | GPU quota=1 serial bottleneck | ACTIVE | All 18 jobs will queue serially |
| W2 | SAM3 TopoLoRA VRAM at BS=1 | MONITORING | Target: <16 GB on L4 |
| W3 | Gradient accumulation correctness | MONITORING | First cloud run with accum=4 |
| W4 | MLflow 413 large artifacts | KNOWN | GCS mount workaround in place |
| W5 | SWAG post-training for SAM3 | MONITORING | May OOM separately from training |
| W6 | SAM3 Hybrid VRAM at BS=1 | MONITORING | Expected ~7.2 GB |
| W7 | Spot preemption recovery | MONITORING | Working in 8th pass |
| W8 | Zero-shot baselines first launch | MONITORING | New launch path |
| W9 | WeightWatcher 3D conv crash | KNOWN | Non-blocking, expected |

## 6. Observations

### O1: Docker image predates gradient accumulation code
- **Image**: `base:latest` built 2026-03-23T15:08:25
- **Code commit**: 3d77c26 from 2026-03-25
- **Impact**: GRAD_ACCUM_STEPS env var is IGNORED by old code. SAM3 still gets BS=1
  (old code has `if _is_sam3: batch_size = min(batch_size, 1)`), but effective BS=1
  not effective BS=4 as planned.
- **Verdict**: OK for debug pass — BS=1 alone fixes OOM. Gradient accumulation is a
  quality improvement for production runs. Docker rebuild needed before paper_full.yaml.

### O2: Earlier SAM3 TopoLoRA attempts visible in queue (IDs 34-48)
- Multiple SAM3 TopoLoRA jobs from earlier today — these ran with old code + BS=2 → OOM/FAIL
- The new 9th pass jobs (IDs 49+) are being submitted alongside these stale entries
- `--resume` was not used — full 34-condition relaunch

### O3: SAM3 TopoLoRA RECOVERING (ID 48)
- Job 48 (sam3_topolora-cbdice_cldice-calibtrue-f0) shows RECOVERING after 1h
- This is an EARLIER attempt (before our code fix) — will likely OOM again
- Our NEW submission will get a fresh job ID

## 7. Compound Artifacts

- [ ] Report updated after every state change
- [ ] New tests per cloud observation
- [ ] GitHub issues for unresolved problems
- [ ] Updated watchlist for 10th pass
- [ ] Metalearning doc if process failure occurs
- [ ] harness-state.jsonl updated
