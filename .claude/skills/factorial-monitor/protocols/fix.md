# Protocol: Fix

## Prerequisites

- Aggregated diagnosis report from DIAGNOSE phase
- All failures grouped by root cause

## Fix Strategy Selection

For EACH root cause category, determine the fix approach:

| Category | Approach |
|----------|----------|
| AUTO-FIXABLE (DVC_NO_GIT, DISK_FULL, ENV_VAR_LITERAL) | Direct fix → test → rebuild |
| CONFIG-FIXABLE (OOM, TIMEOUT) | Edit config YAML → test → rebuild |
| CODE-FIXABLE (import error, loss function bug) | compose_with: self-learning-iterative-coder |
| UNRECOVERABLE (DATA_MISSING, REGISTRY_AUTH) | compose_with: issue-creator → report to user |

## Batch Fix Execution

Fix ALL root causes in a single pass. Do NOT fix one, re-launch, then fix the next.

1. **Plan with reviewer agents**: For each root cause, draft the fix strategy.
   Use parallel reviewer agents to evaluate the plan:
   - Does this fix address ALL affected jobs?
   - Could this fix break any currently-SUCCEEDED jobs?
   - Is the fix reproducible in Docker?

2. **Implement fixes**: Apply all fixes. For code changes, use the TDD skill's
   failure triage protocol (GATHER → CATEGORIZE → PLAN → FIX → VERIFY).

3. **Local verification**: `make test-staging` must pass.

4. **Docker rebuild** (Rule F3 — MANDATORY):
   ```bash
   # Build
   docker build -t <image:tag> -f deployment/docker/Dockerfile.train .
   # Push
   docker push <image:tag>
   # Verify digest
   docker manifest inspect <image:tag> | head -5
   ```

5. **Update SkyPilot YAML**: Ensure `image_id:` references the new image tag.

6. **Commit**: ONE batch commit for all fixes.
   ```
   fix: factorial run <experiment_id> — <N> root causes fixed

   Root causes:
   - DVC_NO_GIT: Added dvc init to entrypoint (4 jobs)
   - OOM: Reduced SAM3 patch_size to 96 (2 jobs)
   ```

## Code Freeze Verification

Before transitioning to RELAUNCH, verify:
- [ ] `make test-staging` passes
- [ ] Docker image rebuilt and pushed
- [ ] SkyPilot YAML references new image
- [ ] All fixes committed
- [ ] No uncommitted changes in `src/` or `configs/`

## Transition to RELAUNCH

Once all fixes are verified, transition to RELAUNCH.
