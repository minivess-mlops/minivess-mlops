# Metalearning: SkyPilot Controller on Wrong Cloud — RunPod Controller for GCP Jobs

**Date**: 2026-03-23
**Severity**: CRITICAL — destroyed the 5th pass factorial run
**Category**: Stale configuration, infrastructure misconfiguration

## What Happened

The 5th pass debug factorial was a GCP run (34 jobs on GCP L4 spot). But
`~/.sky/config.yaml` had `cloud: runpod` for the jobs controller — a leftover
from the RunPod development phase (March 2026).

This caused:
1. **36 min/submission** instead of ~5 min (cross-cloud SSH: local → RunPod → GCP)
2. **RunPod network outage killed 25 of 34 jobs** — `api.runpod.io` became unreachable,
   the controller crashed, and all remaining submissions failed
3. **Controller state lost** — `sky jobs queue` returns empty after controller death
4. **~$0.90 wasted** on 7 DynUNet jobs that may or may not have completed

## Root Cause

```yaml
# ~/.sky/config.yaml
jobs:
  controller:
    resources:
      disk_size: 40
      cloud: runpod   # ← THIS LINE. Set during RunPod dev, never updated for GCP.
```

The comment in the file even says "Caps jobs controller disk to 40 GB (RunPod max)"
— acknowledging it was RunPod-specific. Nobody checked this when switching to GCP.

## Why This Wasn't Caught

1. **No test for controller cloud placement** — our SkyPilot YAML tests check the
   TASK YAML (accelerators, envs, Docker image) but never check `~/.sky/config.yaml`
2. **Preflight script doesn't check controller config** — `preflight_gcp.py` verifies
   the GCP backend is enabled but not WHERE the controller runs
3. **The 4th pass "worked"** — DynUNet and SAM3 Hybrid succeeded despite the RunPod
   controller, so the misconfiguration was never noticed (just slow)

## Fix Applied

Changed `~/.sky/config.yaml`:
```yaml
jobs:
  controller:
    resources:
      disk_size: 50
      cloud: gcp    # Same cloud as jobs — minimum latency
```

## Prevention

1. **Add preflight check**: `preflight_gcp.py` should verify that `~/.sky/config.yaml`
   controller cloud matches the job cloud. A GCP job with a RunPod controller is always wrong.
2. **Add test**: `test_skypilot_controller_matches_job_cloud` — parse both config.yaml
   and train_factorial.yaml, assert controller cloud == job cloud.
3. **Document in deployment/CLAUDE.md**: "Controller MUST be on the same cloud as jobs."

## Impact on Issue #913

The 36 min/submission bottleneck was NOT inherent to SkyPilot — it was caused by
cross-cloud SSH. With a GCP controller, submission latency should drop to ~5 min.
Path 1 from Issue #913 (move controller to GCP) is now implemented. The parallel
submission feature (Path 2) is still valuable for further speedup.
