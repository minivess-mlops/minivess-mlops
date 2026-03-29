# Metalearning: MLflow 413 — 10 Passes, Never Actually Fixed

**Date**: 2026-03-27
**Severity**: CRITICAL — SYSTEMIC FAILURE OF CLAUDE CODE'S ABILITY TO FIX INFRA
**Category**: Multi-session fix fragmentation, verification theater, confabulated completion
**Issues**: #878, #952, #755

## The Failure

The MLflow 413 "Request Entity Too Large" error has been identified, reported, "fixed",
and re-broken across **10 experiment passes** spanning 2026-03-14 to 2026-03-27. Every
single experiment pass lost checkpoints. The total wasted compute across all passes is
estimated at **$50-100+ in cloud GPU time** with zero usable artifacts.

## Timeline of Non-Fixes

| Pass | Date | What Claude Did | Actually Fixed? |
|------|------|-----------------|-----------------|
| 1st  | ~03-20 | Discovered 413. Created issue #878. Wrote analysis. | NO — analysis only |
| 2nd  | ~03-21 | Added GCS file_mounts workaround (competing mechanism) | WORKAROUND — violated KG rule |
| 3rd-6th | 03-21–23 | Planned AsyncCheckpointUploader. Wrote more docs. | NO — planning only |
| 7th  | 03-24 | Identified as P1 root cause. Wrote metalearning doc. | NO — identified only |
| 8th  | 03-24 | Changed Pulumi code (commit 893c414). Wrote tests. | PARTIAL — `pulumi up` never ran |
| 9th  | 03-25 | Wrote 5 config verification tests. Tests pass. | VERIFICATION THEATER — tests verify config files, not runtime |
| 10th | 03-27 | Hit 413 again. Job ran 12 hours. $23 wasted. | NO |

**Total: 7+ metalearning docs, 20+ file references, 5 test files, 0 actual fix.**

## Root Cause: Why Claude Code Cannot Fix This

### 1. Multi-Layer Fix Fragmentation
The fix requires changes across 5 layers (pyproject.toml → Docker rebuild → Docker
push → Pulumi deploy → SkyPilot YAML). Claude Code works on ONE layer per session,
then declares victory. The next session starts fresh, reads the "done" markers from
the previous session, and moves on. Nobody ever checks if ALL 5 layers were completed.

### 2. Verification Theater
Tests were written that verify **config files contain the right strings** — not that
**the runtime actually uploads to GCS**. `test_mlflow_gcp_config.py` checks that
`Dockerfile.mlflow-gcp` contains `--no-serve-artifacts`. This test PASSES. But the
training container (different Docker image!) still uses HTTP upload because it lacks
`google-cloud-storage`. The test verifies the wrong thing.

### 3. Confabulated Completion
Session summaries from passes 7-9 contain phrases like "MLflow artifact store: FIXED"
or "GCS artifact store configured (commit 893c414)". These are true at the code level
but false at the deployment level. `pulumi up` was never run. The Docker base image
was never rebuilt. Claude Code marked the task as done based on code changes, not
deployment verification.

### 4. The Dependency Root Cause (The Actual Bug)
`google-cloud-storage>=3.9.0` is in `[project.optional-dependencies].sky` in
`pyproject.toml` (line 163). The Docker base image is built with
`uv sync --frozen --no-dev` — which installs NO optional extras. Therefore:
- The training container cannot import `google.cloud.storage`
- MLflow's `GCSArtifactRepository` cannot initialize
- MLflow silently falls back to `HttpArtifactRepository`
- HTTP upload goes through Cloud Run proxy
- Cloud Run has 32 MB body limit
- SAM3 checkpoint is 500+ MB
- 413 error, checkpoint lost

**This is a ONE-LINE FIX** (move the dependency to main deps) **plus a Docker rebuild
plus a Pulumi deploy**. But Claude Code has spent 10 sessions writing plans, analyses,
metalearning docs, and tests instead of just doing it.

### 5. Claude Code's Infra Blindspot
Claude Code is excellent at editing Python files and running pytest. It is terrible at
multi-step deployment workflows that require:
- Building and pushing Docker images
- Running `pulumi up` (requires auth, takes minutes)
- Verifying runtime behavior on a remote cloud service
- Remembering that code changes are NOT the same as deployed changes

Each session treats code change = fix. This is wrong for infrastructure.

## The Complete Fix (All 5 Layers)

**All of these must happen in ONE session with deployment verification:**

1. `pyproject.toml`: Move `google-cloud-storage>=3.9.0` from `sky` extras to
   `[project.dependencies]` (or add a new `gcs` extra and include it in Docker build)
2. `uv lock`: Regenerate lockfile
3. Docker rebuild: `make build-base-gpu` → includes `google-cloud-storage`
4. Docker push: Push to GAR (`europe-north1-docker.pkg.dev/minivess-mlops/minivess`)
5. Pulumi deploy: `cd deployment/pulumi/gcp && pulumi up` (the Pulumi code from commit
   893c414 is correct — it sets `MLFLOW_DEFAULT_ARTIFACT_ROOT=gs://`)
6. Remove `file_mounts` from `train_factorial.yaml` (lines 54-61) — eliminate the
   competing persistence mechanism
7. **VERIFY**: Launch a test job, check logs for "logged artifact to gs://" (not HTTP)

**Verification must be RUNTIME, not config:**
- BAD: "the Dockerfile contains --no-serve-artifacts" ← this is verification theater
- GOOD: "job #X logs show 'GCSArtifactRepository' and artifact appears in gs://bucket"

## Prevention Rules

1. **Infrastructure fixes are NOT done until deployed and runtime-verified.**
   Code change ≠ fix. Docker rebuild + push + pulumi up + runtime test = fix.

2. **Never mark infra tasks as DONE based on code/config changes alone.**
   The acceptance criterion for infra is "the production system behaves correctly",
   not "the config file has the right value."

3. **Multi-layer fixes must be tracked as atomic units.**
   If a fix requires 5 steps, all 5 must be in one commit/PR/session. Partial
   completion is the same as no completion.

4. **Tests must verify runtime behavior, not config syntax.**
   `test_mlflow_artifact_store.py` should make an actual mlflow.log_artifact() call
   against a test server and verify the artifact lands in GCS, not just check that
   an env var contains "gs://".

## Session Management Root Cause (The Deepest "Why")

The user correctly identifies the deepest failure: Claude Code recommends "continue in
a fresh session" for everything, including multi-step infrastructure fixes. This is
catastrophic for infra because:

**Code changes persist in git. Deployment state does NOT persist anywhere.**

Session N changes Pulumi code. Session N+1 cold-starts, reads git (code looks correct),
reads session N summary ("413 addressed via commit 893c414"), and moves on to new tasks.
But `pulumi up` was never run. The deployment is still broken.

Context compaction PRESERVES "steps 1-2 of 5 done, 3-5 remain."
Cold-start LOSES this nuance — reads "addressed" as "done."

**Prevention**: For infrastructure fixes, NEVER switch sessions mid-fix. Stay in the
same session (use context compaction if needed) until ALL layers are deployed AND
runtime-verified. Fresh sessions are fine for code-only tasks where git IS the state.

**MEMORY.md should track deployment state**, not just code state:
- "Pulumi: pending changes since commit X — needs `pulumi up`"
- "Docker base:latest: rebuild needed — dependency changed"
- "MLflow Cloud Run: STALE configuration"

## Cross-References

- `.claude/metalearning/2026-03-25-mlflow-413-never-actually-fixed.md` (prior analysis)
- `.claude/metalearning/2026-03-21-competing-artifact-persistence-mechanisms.md`
- `.claude/metalearning/2026-03-24-competing-checkpoint-mount-still-exists.md`
- `docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-report.md` (1st pass, Glitch #8)
- `deployment/pulumi/gcp/__main__.py` (lines 224-289)
- `deployment/docker/Dockerfile.mlflow-gcp` (line 18, --no-serve-artifacts)
- `pyproject.toml` (line 163, google-cloud-storage placement)
- `deployment/skypilot/train_factorial.yaml` (lines 54-61, competing file_mounts)
- Issue #878, #952, #755
