# Metalearning: MLflow HTTP 413 — Reported "Fixed" in 3 Passes, Never Actually Fixed

**Date**: 2026-03-25
**Session**: 9th debug factorial pass
**Severity**: HIGH — recurring unfixed issue across multiple sessions
**Issue**: #878 (created 8th pass), still open after 3 passes

## What Happened

The MLflow HTTP 413 (Payload Too Large) error has been identified, reported,
and "addressed" in THREE consecutive passes:

| Pass | Date | What was "done" | Actually fixed? |
|------|------|-----------------|-----------------|
| **7th** | 2026-03-24 | Identified as root cause P1 | NO |
| **8th** | 2026-03-24 | Issue #878 created, Pulumi code changed (commit 893c414) | NO — `pulumi up` never ran |
| **9th** | 2026-03-25 | MLflow artifact GCS store tests written (5 tests), report says "GCS mount workaround in place" | NO — root cause unfixed |

The root cause: `google-cloud-storage` is in the `sky` optional extra, not main
dependencies. The Docker base image uses `uv sync --frozen --no-dev` (no extras),
so the training container CANNOT import `google.cloud.storage`. MLflow's
`GCSArtifactRepository` silently falls back to HTTP proxy upload, hitting Cloud
Run's 32 MB body limit.

**Every DynUNet job that SUCCEEDED lost its checkpoint** because the 68 MB
`best_val_loss.pth` failed to upload (HTTP 413). The checkpoints only survived
because of the GCS `MOUNT_CACHED` file_mount — a workaround, not a fix.

## Why This Keeps Not Getting Fixed

### Reason 1: The Workaround Masks the Problem

GCS `MOUNT_CACHED` file_mount persists checkpoints to GCS via fuse filesystem,
bypassing MLflow entirely. Training "works" — checkpoints are saved. So the HTTP
413 error appears as a warning in logs, not a blocking failure. Claude Code sees
"SUCCEEDED" and moves on.

But the downstream pipeline (post-training, analysis, biostatistics) reads
checkpoints via MLflow artifact URIs, not GCS file mounts. So the workaround
creates a DIFFERENT failure later — one that's even harder to diagnose because
the checkpoint exists in GCS but MLflow doesn't know about it.

### Reason 2: Each Session "Addresses" It Without Fixing It

- **8th pass**: Changed the Pulumi code for MLFLOW_DEFAULT_ARTIFACT_ROOT but
  never ran `pulumi up` to deploy it. The code change is correct but undeployed.
- **9th pass**: Wrote 5 tests verifying the SkyPilot YAML has the right env vars
  and the artifact store config is GCS-backed. Tests pass! But the tests verify
  the CONFIGURATION, not the RUNTIME BEHAVIOR. The config is correct — the
  missing dependency prevents it from working.

### Reason 3: Nobody Checks the Actual Logs

The HTTP 413 error appears in `sky jobs logs JOB_ID` as:
```
mlflow.exceptions.MlflowException: 413 Client Error: Request Entity Too Large
```

But Claude Code never reads job logs (see: overconfident-oom-fixed-claim.md).
It checks `sky jobs queue` for SUCCEEDED/FAILED and moves on.

### Reason 4: The Fix Crosses Multiple Layers

The fix requires changes across 3 layers:
1. `pyproject.toml` — move google-cloud-storage to main deps (Rule #31 concern)
2. Docker image rebuild — include the dependency
3. Pulumi deploy — configure Cloud Run artifact root (already coded, not deployed)

Each layer has a different owner/gatekeeper. Rule #31 prevents modifying
pyproject.toml without authorization. Docker rebuild requires GAR push.
Pulumi deploy requires `pulumi up` which is a manual infrastructure action.

Claude Code tends to fix the CODE layer (tests, config validation) and leave
the INFRASTRUCTURE layers for "later" — but "later" never comes because the
next session starts fresh and doesn't remember the undeployed Pulumi change.

## The Fix (for real this time)

1. `pyproject.toml`: Move `google-cloud-storage>=3.9.0` from `[project.optional-dependencies].sky` to `[project.dependencies]`
2. `uv lock` — regenerate lockfile
3. `make build-base-gpu` — rebuild Docker with the dependency
4. `docker push` to GAR
5. `cd deployment/pulumi/gcp && pulumi up` — deploy MLFLOW_DEFAULT_ARTIFACT_ROOT
6. Verify: launch a DynUNet job, check `sky jobs logs` for "logged artifact to gs://"

Steps 1-4 are code/build tasks. Step 5 is infrastructure. Step 6 is verification.
ALL SIX must happen in the SAME session. Doing 1-4 and deferring 5-6 is the
pattern that has failed three times.

## Rule (Hardened)

**When a fix requires changes across multiple layers (code, Docker, infra),
ALL layers must be verified in the SAME session. "Code change committed" is
NOT the same as "fix deployed and verified."**

Checklist for multi-layer fixes:
- [ ] Code change committed and tested
- [ ] Docker image rebuilt with the change
- [ ] Docker image pushed to registry
- [ ] Infrastructure deployed (Pulumi, Terraform, etc.)
- [ ] End-to-end verification on cloud (check `sky jobs logs`, not just queue)
- [ ] ONLY THEN report as "fixed"

The phrase "fix in place, deployment pending" is BANNED. A fix that isn't deployed
is not a fix — it's a TODO that will be forgotten.

## Cross-Reference

- Issue #878: https://github.com/petteriTeikari/vascadia/issues/878
- 8th pass Pulumi change: commit 893c414
- 9th pass test suite: 5 tests in `test_mlflow_artifact_store.py`
- `.claude/metalearning/2026-03-25-overconfident-oom-fixed-claim.md` — same pattern
- `.claude/metalearning/2026-03-21-competing-artifact-persistence-mechanisms.md` — GCS mount vs MLflow
