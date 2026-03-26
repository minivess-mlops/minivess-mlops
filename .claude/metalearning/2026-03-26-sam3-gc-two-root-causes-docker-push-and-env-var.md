# SAM3 Gradient Checkpointing: Two Root Causes — Docker Push + Env Var

**Date**: 2026-03-26
**Impact**: ALL SAM3 TopoLoRA jobs (50+ attempts across 9 passes) FAILED with OOM
**Time wasted**: ~$15-20 in compute + 24+ hours of wall-clock time

## Root Cause 1: Docker Image Code Layer Not Pushed

**What happened**: The Docker image was rebuilt locally at 20:57 UTC with gradient
checkpointing code. The `docker push` was attempted but the CODE LAYER (`b0da7db20ecc`)
failed silently. All base layers showed "Layer already exists" but the app code layer
did not upload.

**Evidence**: Job 72 used `train_flow.py:590` (pre-autocast code). Job 73 used
`train_flow.py:623` (post-autocast, pre-GC-skip). The current code has these at
line 635. The line numbers prove the running Docker image was stale.

**Fix**: Re-ran `docker push` — the code layer uploaded after retry. Digest confirmed:
`sha256:08d6bfaab445699af3f493dfdf38849324f859c0565dcada7683ed8ca55cf88a`.

**Prevention**: After `docker push`, ALWAYS verify the code layer was pushed by
running the container and checking a known line:
```bash
docker run --rm <image> grep -n "skip_gradient_flow" /app/src/minivess/diagnostics/pre_training_checks.py
```

## Root Cause 2: Env Var Not Set in Old Job Submissions

**What happened**: Even after the Docker image was correctly pushed, SAM3 jobs
STILL OOMed. Investigation showed `skip_gradient_flow` was `False` despite the
code supporting it.

**Evidence**: Job 69 used `train_flow.py:635` (CORRECT code from new image) but
`check_gradient_flow` was still called at `pre_training_checks.py:257` (else branch).
The `GRADIENT_CHECKPOINTING` env var was `"false"` because the job was submitted
6 hours ago by an OLD version of `run_factorial.sh` that didn't parse `model_overrides`
for `gradient_checkpointing`.

**The chain**:
1. Job submitted at T=0 with OLD `run_factorial.sh` → `GRADIENT_CHECKPOINTING=false`
2. At T+3h, code committed + Docker image rebuilt with GC support
3. At T+6h, job gets VM, pulls NEW Docker image with GC code
4. BUT: env var is still `false` from original submission
5. `args.gradient_checkpointing.lower() == "true"` → `False`
6. `skip_gradient_flow` → `False`
7. `check_gradient_flow()` runs → OOM

**Fix**: Cancelled ALL PENDING SAM3 jobs. Re-ran `run_factorial.sh --resume` which
resubmits FAILED/CANCELLED conditions with the CORRECT env vars.

**Prevention**:
- SkyPilot `--env` values are IMMUTABLE after submission. They persist across
  spot recoveries but NEVER update from the YAML default.
- When adding a new env var to a SkyPilot YAML, ALL existing jobs with the old
  default MUST be cancelled and resubmitted.
- Consider adding a VERSION env var to the SkyPilot YAML that changes with each
  code change, so you can verify jobs are using the right version.

## Lessons

1. "Docker push succeeded" is not verified until you check the code layer
2. Env vars in SkyPilot jobs are set at submission time, NOT at VM start time
3. Updating code (Docker image) without updating env vars (job submission) = half a fix
4. When adding new features that require both code + config changes, you need to:
   - Push new Docker image (code)
   - Cancel all existing jobs (stale env vars)
   - Resubmit with updated launcher script (new env vars)
5. Job 69's stack trace was the key diagnostic: line 635 (new code) + line 257
   (gradient_flow called) = env var is wrong
