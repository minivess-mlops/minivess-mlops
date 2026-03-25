# Metalearning: Competing Checkpoint Mount STILL EXISTS

**Date**: 2026-03-24
**Severity**: CRITICAL — violates THE non-negotiable artifact persistence invariant
**Category**: Competing persistence mechanism, KG invariant violation

## What Exists

`train_factorial.yaml` lines 51-54:
```yaml
file_mounts:
  /app/checkpoints:
    source: gs://minivess-mlops-checkpoints
    mode: MOUNT_CACHED
```

This is a SEPARATE GCS bucket (`minivess-mlops-checkpoints`) mounted directly
by SkyPilot, bypassing MLflow entirely. Checkpoints written to `/app/checkpoints/`
go to this bucket via MOUNT_CACHED, not through `mlflow.log_artifact()`.

## Why This Is a Violation

KG invariant `mlflow_only_artifact_contract` (navigator.yaml line 191-200):
> "MLflow artifact store is THE ONLY persistence mechanism for checkpoints,
> model files, and experiment artifacts. NEVER build parallel persistence
> systems (SkyPilot file_mounts to separate GCS buckets, manual rsync,
> custom sync scripts)."

The metalearning doc this invariant references:
`.claude/metalearning/2026-03-21-competing-artifact-persistence-mechanisms.md`

## Why Claude Code Keeps Building This

1. **Practical convenience**: MOUNT_CACHED gives non-blocking writes during training.
   `mlflow.log_artifact()` is synchronous and blocks until upload completes.
2. **Spot recovery**: MOUNT_CACHED persists across preemption. MLflow artifacts
   are logged AFTER training completes — if preempted mid-training, no MLflow
   artifact exists but the MOUNT_CACHED checkpoint survives.
3. **Performance**: Writing to a GCS-backed FUSE mount is faster than uploading
   through MLflow's HTTP API to the artifact store.

All of these are real engineering concerns. But the invariant says: "When MLflow
persistence breaks, FIX MLFLOW CONFIG — don't work around it."

## The Correct Architecture

Per the KG invariant, checkpoints should be:
1. Written to LOCAL disk during training (fast, non-blocking)
2. Logged to MLflow artifact store AFTER each epoch or at checkpoints
3. On spot recovery, discovered via MLflow `artifact_uri` (not GCS mount)

The challenge: MLflow artifact upload is synchronous and slow for large
checkpoints (~900 MB for SAM3). This needs investigation:
- MLflow async artifact logging (if available)
- Smaller checkpoint formats (state_dict only, not full model)
- Periodic vs end-of-epoch artifact logging

## Fix Required

1. Remove `file_mounts` from `train_factorial.yaml`
2. Ensure `train_flow.py` logs checkpoints via `mlflow.log_artifact()`
3. Ensure spot recovery uses MLflow `artifact_uri` for checkpoint discovery
4. GCS bucket `minivess-mlops-checkpoints` becomes ORPHAN — document or delete

## See Also

- `knowledge-graph/navigator.yaml` invariant `mlflow_only_artifact_contract`
- `.claude/metalearning/2026-03-21-competing-artifact-persistence-mechanisms.md`
- `knowledge-graph/domains/cloud.yaml` line 94-103 (artifact persistence rule)
