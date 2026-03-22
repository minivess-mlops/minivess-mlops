# 4th Pass Debug Factorial — Failure Report

**Date**: 2026-03-22
**Branch**: `test/debug-factorial-4th-pass`
**Status**: FAILED — all jobs cancelled, re-launch pending after fixes

---

## What Was Attempted

Launch full debug factorial: 4 models × 4 losses × 2 aux_calib × 1 fold = 32
trainable conditions + 2 zero-shot baselines on GCP L4 spot via SkyPilot.

## What Went Wrong

### Failure 1: `sky` binary not on PATH (pre-launch)

- **Impact**: First launch attempt submitted 0 jobs (32 `sky: command not found`)
- **Root cause**: `run_factorial.sh` calls `sky` directly but it's in `.venv/bin/`
- **Fix**: Added SKY_BIN fallback to check `.venv/bin/sky`
- **Prevention**: Local SkyPilot test suite (Issue #908)

### Failure 2: `job_recovery` field unsupported (pre-launch)

- **Impact**: Second launch attempt — all 32 jobs rejected at YAML parse time
- **Root cause**: SkyPilot v1.0 removed `job_recovery` field (was in earlier versions)
- **Fix**: Removed field from `train_factorial.yaml`
- **Prevention**: `sky.Task.from_yaml()` tests (Issue #908, Tier 1)

### Failure 3: DVC pull fails on `data/processed/minivess` (FAILED_SETUP)

- **Impact**: 8 of 10 submitted jobs FAILED_SETUP (~$5 wasted on VM provisioning)
- **Root cause**: `dvc pull -r gcs` pulls ALL DVC-tracked outputs, but
  `data/processed/minivess` was never pushed to GCS (tracked, never ran)
- **Fix**: Changed to `dvc pull data/raw/minivess -r gcs` (path-specific)
- **Prevention**: DVC test suite (DVC plan T1, T2), preflight script (T10)

### Failure 4: Job 55 stuck STARTING for 2.5 hours

- **Impact**: 1 job consumed VM time without starting training
- **Root cause**: Likely no L4 spot capacity in the zone SkyPilot selected
- **Fix**: Cancelled job. Consider adding region hints or fallback to on-demand
- **Prevention**: Monitor polling (30s interval) would have caught this in minutes

### Failure 5: No monitoring detected failures for 2+ hours

- **Impact**: All failures went undetected until manual `sky jobs queue` check
- **Root cause**: `ralph_monitor.py` watches ONE job at a time, not batch.
  Launch script stdout not monitored. No alerting on FAILED_SETUP.
- **Fix**: Upgrade /factorial-monitor with batch JSON polling
- **Prevention**: SkyPilot observability upgrade plan

## Cost

| Item | Cost |
|------|------|
| 8 FAILED_SETUP VMs (~2 min each) | ~$0.60 × 8 = ~$4.80 |
| 1 stuck STARTING VM (2.5 hrs) | ~$1.50 |
| 2 cancelled PENDING jobs | $0 |
| **Total wasted** | **~$6.30** |

## Timeline

```
05:08 — Launch script started (32 + 2 jobs)
05:08 — Job 55 submitted (dynunet-cbdice_cldice, PENDING)
05:08 — Job 56 submitted (mambavesselnet-dice_ce, STARTING)
05:20 — Jobs 57-60 FAILED_SETUP (DVC pull error)
05:30 — Jobs 61-63 FAILED_SETUP (same error)
05:40 — Job 64 submitted (sam3_topolora, PENDING)
07:20 — Manual sky jobs queue check — discovered 8 FAILED_SETUP
07:30 — Diagnosed: dvc pull -r gcs → data/processed not on GCS
07:35 — Fix committed: dvc pull data/raw/minivess -r gcs
07:40 — All jobs cancelled
07:50 — Metalearning doc + DVC test suite plan written
```

2 hours 20 minutes from first failure to detection. With 30s polling,
detection would have been within 3 minutes.

## Prevention: Three Interlocking Plans

| Plan | File | What it prevents |
|------|------|-----------------|
| **DVC test suite** | `docs/planning/dvc-test-suite-improvement.xml` | DVC pull failures, tracked-but-not-pushed, setup script bugs |
| **SkyPilot local tests** | `docs/planning/skypilot-fake-mock-ssh-test-suite-plan.md` | YAML schema errors, unsupported fields, missing env vars |
| **SkyPilot observability** | `docs/planning/skypilot-observability-for-factorial-monitor.md` | Silent failures, no alerting, single-job monitoring |

### Critical path to re-launch:

1. **Implement `scripts/preflight_gcp.py`** — validates Docker image, GCS access,
   DVC data, env vars BEFORE `sky jobs launch`
2. **Wire preflight into `run_factorial.sh`** — fail fast if prerequisites missing
3. **Test DVC pull locally** — `dvc pull data/raw/minivess -r gcs` on dev machine
4. **Re-launch with monitoring** — `sky dashboard` + batch JSON polling

## Issues Created

- **#907**: Cloud GPU pipeline gaps (Prefect, MLflow, orchestration parity)
- **#908**: Local SkyPilot test suite (YAML validation without credits)

## Lessons Learned

1. **Never launch without preflight.** A 10-second `gsutil ls` + `dvc status`
   would have prevented $6.30 of waste.
2. **Never trust setup scripts without testing them.** The DVC pull line
   "looked right" but was never run locally.
3. **Batch monitoring is mandatory.** Watching one job when 32 are running
   is negligent. The monitor must poll ALL jobs every 30s.
4. **SkyPilot YAML changes break across versions.** `job_recovery` worked
   in an earlier version but was removed in v1.0. `Task.from_yaml()` tests
   catch this instantly.
