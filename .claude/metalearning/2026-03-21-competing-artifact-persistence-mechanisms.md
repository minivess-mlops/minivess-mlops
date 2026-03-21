# Metalearning: Competing Artifact Persistence Mechanisms

**Date**: 2026-03-21
**Severity**: ARCHITECTURAL ANTI-PATTERN
**Trigger**: User flagged that suggesting SkyPilot `file_mounts` as a checkpoint
persistence solution competes with MLflow artifact store — the off-the-shelf tool
that already solves this problem.

## The Anti-Pattern

When MLflow artifact persistence broke (1st pass Glitch #8), Claude proposed a
**parallel persistence mechanism** (SkyPilot `file_mounts` to a separate GCS bucket)
instead of **fixing the off-the-shelf tool** (MLflow artifact store → GCS).

This created two competing artifact persistence paths:

```
PATH A (MLflow):   train_flow → mlflow.log_artifact() → Cloud Run → ???  (BROKEN)
PATH B (file_mounts): train_flow → /app/checkpoints → MOUNT_CACHED → gs://minivess-mlops-checkpoints

Both paths exist in train_factorial.yaml simultaneously.
```

## Why This Is Wrong

1. **Brittle**: Two mechanisms that must stay in sync. If one breaks, which is
   authoritative? If they disagree, which wins?

2. **Violates TOP-2**: "Automate everything. Nobody should ever manually…" — but
   now a researcher must understand BOTH mechanisms and know that checkpoints live
   in a different bucket than MLflow artifacts.

3. **Violates the inter-flow contract**: `orchestration/CLAUDE.md` says "Flows
   communicate through MLflow artifacts ONLY." A `file_mounts` GCS bucket is NOT
   MLflow — downstream flows can't discover it via `mlflow_client.search_runs()`.

4. **Off-the-shelf tools exist**: MLflow artifact store with a GCS backend is
   literally `--default-artifact-root gs://...`. It's one config flag. Building a
   parallel system is inventing a problem.

## Root Cause

Claude's 1st pass diagnoses Glitch #8 as "checkpoints lost because Cloud Run
proxied artifacts." The correct fix is:

```
FIX: mlflow server --default-artifact-root gs://minivess-mlops-mlflow-artifacts \
     --no-serve-artifacts
```

This makes MLflow write artifacts directly to GCS, bypassing the Cloud Run proxy.
The `--no-serve-artifacts` flag is the key — it tells the client to write to GCS
directly rather than through the server.

Instead, Claude suggested KEEPING the broken MLflow config and ADDING a parallel
`file_mounts` mechanism. This is the "duct tape over duct tape" anti-pattern.

## The Rule

> **When an off-the-shelf tool breaks, FIX THE TOOL — don't build a parallel system.**
>
> MLflow, DVC, SkyPilot, Prefect exist so we DON'T build custom infrastructure.
> If MLflow artifact storage doesn't persist to GCS, the answer is "fix MLflow config"
> not "add a second persistence path."

## What Should Have Been Done

1. Diagnose WHY MLflow artifacts weren't persisting (Cloud Run `--serve-artifacts`
   proxying instead of direct GCS write)
2. Fix the MLflow server config (`--no-serve-artifacts` + `--default-artifact-root gs://...`)
3. Update Pulumi to deploy the fix
4. Verify: `mlflow.log_artifact(test_file)` → appears in `gs://` bucket
5. Remove the competing `file_mounts` checkpoint bucket

## Correct Architecture (Single Path)

```
train_flow → mlflow.log_artifact(checkpoint.pth)
                    ↓
           MLflow client writes directly to gs://minivess-mlops-mlflow-artifacts
                    ↓
           Downstream flows: mlflow_client.download_artifacts(run_id, "checkpoints/")
                    ↓
           Spot recovery: mlflow_client.search_runs(filter="status='RUNNING'")
                         → download latest checkpoint → resume training
```

One path. One bucket. One discovery mechanism. Zero competing systems.

## Exception: Spot Resume Write-Ahead

The ONLY valid use of local disk during training is as an **ephemeral write-ahead
cache** — save checkpoint to local `/app/checkpoints/` THEN immediately
`mlflow.log_artifact()` to GCS. The local copy is a performance optimization
(avoid blocking training on GCS upload), NOT a persistence mechanism. If the VM
dies before the MLflow upload completes, the checkpoint is lost — and that's OK
because the PREVIOUS epoch's checkpoint is already in MLflow.

This is NOT a competing mechanism — it's a standard write-ahead pattern where
the local disk is explicitly ephemeral.

## How Claude Should Have Caught This

1. **Read orchestration/CLAUDE.md**: "Flows communicate through MLflow artifacts ONLY"
2. **Read cloud.yaml**: MLflow is the inter-flow contract
3. **Apply TOP-2**: Use off-the-shelf tools, don't build parallel infrastructure
4. **Apply library-first rule**: MLflow has GCS artifact support built-in

## Impact

- 1st pass: ALL 11 succeeded training jobs had LOST checkpoints
- Downstream flows (post-training, analysis, biostatistics, deploy) ALL BLOCKED
- Entire $2-3 GCP spend wasted — zero usable artifacts

## Tags

`#artifact-storage` `#mlflow` `#competing-mechanisms` `#off-the-shelf` `#gcs`
