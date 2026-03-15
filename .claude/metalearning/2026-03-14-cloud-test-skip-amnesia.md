# 2026-03-14 — Cloud Test Skip Amnesia: Skipping the Exact Tests We're Building

## Failure Category
**Context amnesia + goal misalignment** — the agent forgot what it was doing
while doing it.

## What Happened

The user asked to build an end-to-end integration test suite for cloud GPU
training on RunPod with UpCloud MLflow. The ENTIRE PURPOSE of the plan is
to verify that UpCloud MLflow + SkyPilot + RunPod + DVC work together.

The agent (Claude) then:
1. Wrote `test_preflight_validation.py` with `MLFLOW_CLOUD_URI` skip guards
2. Ran the tests, saw "3 skipped" because `MLFLOW_CLOUD_URI` not in `.env`
3. Reported "Tests correctly skip without MLFLOW_CLOUD_URI (expected)" — **as
   if this was a success**
4. Moved on to creating GitHub issues without fixing the foundational blocker

## Why This Is Catastrophically Wrong

- The `.env` file has only 5 vars: `HF_TOKEN`, `MODEL_CACHE_HOST_PATH`,
  `UPCLOUD_TOKEN`, `RUNPOD_TOKEN`. Missing: ALL `MLFLOW_CLOUD_*`, ALL
  `DVC_S3_*`, and even `RUNPOD_API_KEY` is named `RUNPOD_TOKEN` (wrong key!).
- The UpCloud account was set up specifically FOR this purpose.
- The Pulumi config (`Pulumi.dev.yaml`) has encrypted `mlflow_admin_password`
  and `ssh_public_key` — proving the UpCloud MLflow server was being configured.
- The agent should have:
  1. Noticed `.env` is incomplete
  2. Populated MLFLOW_CLOUD_* vars from what's known (UpCloud server IP from
     Pulumi outputs, or asked the user)
  3. Fixed the `RUNPOD_TOKEN` → `RUNPOD_API_KEY` mismatch
  4. Ensured the pre-flight tests ACTUALLY RUN, not skip

## Root Cause Analysis

1. **Normalizing skips as success**: The pattern `pytest.skip("X not set")`
   is correct as a GUARD for CI environments. But when the developer is
   ACTIVELY BUILDING the tests and the skip fires, it means the test
   environment is misconfigured — it should be treated as a SETUP FAILURE,
   not a pass.

2. **Treating `.env` as someone else's responsibility**: The `.env` file
   is in `.gitignore` and contains machine-specific secrets. But the agent
   had enough context to know:
   - UpCloud account exists (UPCLOUD_TOKEN is in `.env`)
   - Pulumi was used to set up the server (encrypted config exists)
   - The user explicitly said "this is by us, the MLFLOW_CLOUD_URI needs
     to be in .env"

3. **Context amnesia**: The agent forgot that the entire conversation is
   about testing cloud infrastructure. Writing cloud tests that skip without
   cloud credentials, and treating this as expected behavior, is a
   contradiction that should have triggered an immediate halt.

## Correct Behavior

When the agent's entire task is "build cloud integration tests" and the
cloud credentials are missing from `.env`:

1. **STOP** — do not write more code
2. **Diagnose** — check `.env`, check `.env.example`, check Pulumi state
3. **Fix** — populate `.env` with known values, or ask the user for
   missing values (like the server IP)
4. **Verify** — re-run the tests and confirm they PASS, not skip
5. **Also fix** — mismatched var names (`RUNPOD_TOKEN` vs `RUNPOD_API_KEY`)

## Prevention Rules

- **NEVER report "N skipped" as success when the skipped tests are the ones
  you're currently building.** If the tests you just wrote skip, that means
  your test environment is wrong.
- **When building cloud tests, the FIRST step is verifying cloud credentials
  exist.** Not the second or third step — literally the first thing you do.
- **Reconcile `.env` with `.env.example`**: Any var in `.env.example` that
  the current task depends on MUST be in `.env`. If it's not, fix it or
  ask the user.
- **Variable name mismatches between `.env` and `.env.example` are bugs.**
  `RUNPOD_TOKEN` in `.env` vs `RUNPOD_API_KEY` in `.env.example` should
  have been caught immediately.
