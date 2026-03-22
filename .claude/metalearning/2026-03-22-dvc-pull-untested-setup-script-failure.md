# Metalearning: Untested SkyPilot Setup Script — DVC Pull Failure on GCP

**Date**: 2026-03-22
**Severity**: P0-CRITICAL — burned ~$5 in GCP credits on 8 FAILED_SETUP jobs
**Rule violated**: Rule 20 (Zero Tolerance for Observed Failures), Rule 24 (Tokens Upfront)

## What Happened

The SkyPilot setup script (`train_factorial.yaml`) ran `dvc pull -r gcs` which
pulls ALL DVC-tracked files. `data/processed/minivess` is DVC-tracked but was
never pushed to GCS. The raw training data (351 files) pulled successfully, but
DVC returned a non-zero exit code because of the missing processed data. The
`set -e` in the setup script killed the entire job.

8 out of 10 submitted jobs FAILED_SETUP with identical root cause. Each job
provisioned a GCP L4 spot VM (~$0.50-0.70 setup cost) before failing.

## Why This Is Unacceptable

1. **The setup script was NEVER tested locally.** Not once. We tested the
   Python training code extensively (3-flow pipeline, 46 min local run) but
   never ran the SHELL SCRIPT that actually executes on the VM.

2. **`dvc pull -r gcs` is a 3-second command** that could have been tested
   locally with `dvc pull data/raw/minivess -r gcs` to verify it works.

3. **The reviewer agent flagged the splits file bug** but did NOT check whether
   `dvc pull` would succeed. The review focused on YAML schema and Python code,
   not on the shell setup script.

4. **This is the SAME class of failure as the 1st pass** (Glitch #6: DVC partial
   pull). We "fixed" it by adding verification logic, but the fix was incomplete
   — it checked for existing data but didn't test the DVC pull path.

## Root Cause: AI Slop Implementation

The setup script was written to "look right" without being validated:

```bash
dvc pull -r gcs || {
    echo "FATAL: DVC pull from GCS failed."
    exit 1
}
```

This LOOKS correct — it pulls data and fails loudly. But:
- Nobody asked "what does `dvc pull -r gcs` actually pull?"
- Nobody checked if ALL DVC-tracked paths exist in GCS
- Nobody ran `dvc status -r gcs` to verify what's available
- The error message says "Ensure: gcloud auth application-default login"
  which is WRONG — the issue was missing data, not missing credentials

"AI slop" = code that passes superficial review but fails on first real use
because the implementation was never actually tested against real infrastructure.

## Prevention Rules

1. **NEVER deploy a setup script without running it locally first.**
   Even if the local environment differs from cloud, the DVC/pip/git commands
   can be tested. `dvc pull data/raw/minivess -r gcs` works on any machine
   with GCS credentials.

2. **ALWAYS run `dvc status -r <remote>` before any DVC pull in a script.**
   This shows exactly which files exist remotely. If files are missing,
   the script should log which ones and continue (not abort).

3. **Test setup scripts in Docker locally** before deploying to SkyPilot.
   ```bash
   docker run --rm -v $(pwd):/app minivess-base:latest bash -c "
     cd /app && source deployment/skypilot/train_factorial.yaml.setup.sh
   "
   ```

4. **The reviewer agent must check SHELL scripts, not just Python code.**
   Add "setup script validation" to the review checklist.

5. **Create a DVC test suite** that validates:
   - All DVC-tracked paths exist in the configured remote
   - `dvc pull` succeeds for each tracked path individually
   - Checkpoint directories are writable
   - GCS credentials work (ADC, service account)

## Occurrence

This is the 2nd DVC pull failure on GCP (1st pass Glitch #6 was the same class).
The "fix" from the 1st pass was incomplete.

## Cross-References

- `deployment/skypilot/train_factorial.yaml` lines 116-122
- `.claude/metalearning/2026-03-20-full-factorial-is-not-24-cells.md` (context amnesia)
- `docs/planning/run-debug-factorial-experiment-report.md` (1st pass Glitch #6)
- Issue #908 (local SkyPilot test suite)
