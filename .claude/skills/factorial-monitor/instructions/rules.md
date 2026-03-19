# Non-Negotiable Rules — Factorial Monitor

Five rules that prevent every known anti-pattern from the metalearning history.
These are CROSS-CUTTING — they apply to ALL phases.

## Rule F1: WAIT-FOR-TERMINAL

> No error diagnosis, no code fixes, no re-launches until ALL factorial jobs
> have reached terminal state (SUCCEEDED / FAILED / FAILED_SETUP / CANCELLED).

**Permitted during execution:**
- `sky jobs queue` (status polling)
- `sky jobs logs <id>` (read-only log inspection)
- Reading MLflow metrics
- `sky jobs cancel <id>` for a job PROVABLY wasting money (infinite loop, disk full)

**BANNED during execution:**
- `sky exec` (SSH into running pods)
- Code changes to `src/` or `configs/`
- Docker image rebuilds
- "This job looks like it will fail" as grounds for cancellation

**Kill-switch exception:** If 3+ jobs fail with the IDENTICAL error (same traceback,
same root cause category) within 5 minutes AND remaining running jobs haven't passed
the failure point → cancel remaining same-config jobs, begin batch diagnosis. Jobs
with DIFFERENT configurations continue running.

**Why:** Premature diagnosis on partial information causes Panic Fixing (Anti-Pattern 3)
and Whac-a-Mole (Anti-Pattern 2). Let jobs run to completion to get the full picture.

**Source:** `.claude/metalearning/2026-03-18-whac-a-mole-serial-failure-fixing.md`

---

## Rule F2: AGGREGATE-BEFORE-FIX

> After all jobs reach terminal state, collect ALL failure logs and categorize
> by root cause BEFORE writing any fix.

**Required output format:**
```json
{
  "root_cause_id": "DVC_NO_GIT",
  "description": ".dvc not initialized in container",
  "affected_jobs": [3, 7, 11],
  "fix_strategy": "Add dvc init to Docker entrypoint",
  "affected_files": ["deployment/Dockerfile.train"],
  "confidence": "high"
}
```

**BANNED:**
- Fixing the first failure you see
- Re-launching a failed job without a written root cause
- "Probably transient" without evidence (spot preemption exit code 24, or identical
  job succeeded on zero-code-change retry)
- "Pre-existing" as a failure classification
- "Not related to current changes" as an excuse to skip

**Why:** Silent dismissal (Anti-Pattern 1) and serial fixing (Anti-Pattern 2) are the
two most expensive failure modes. One root cause often explains 5+ job failures.
Batch fixing saves GPU-hours.

**Source:** `.claude/metalearning/2026-03-07-silent-existing-failures.md`

---

## Rule F3: REBUILD-BEFORE-RELAUNCH

> Every fix-relaunch cycle MUST execute this exact sequence:
> 1. Code fix with tests
> 2. `make test-staging` passes
> 3. Docker image rebuild + push with new tag
> 4. Update SkyPilot YAML to reference new image tag/digest
> 5. Re-launch ONLY the failed jobs

**Skipping ANY step is BANNED.**

Re-launching with a stale Docker image is the most expensive possible mistake —
it wastes the full GPU cost of every re-launched job while producing identical failures.

**Verification command:** Before re-launching, run:
```bash
docker manifest inspect <image:tag> | python3 -c "import sys,json; print(json.load(sys.stdin)['config']['digest'][:16])"
```
Compare with the digest in the SkyPilot YAML.

**Why:** Docker Image Staleness (Anti-Pattern 9) was identified by the anti-pattern
reviewer as having CRITICAL severity. Every re-launch without rebuild = wasted money.

---

## Rule F4: MAX-TWO-CYCLES

> A factorial run permits at most 2 fix-relaunch cycles (initial launch + 2 retries).
> If jobs still fail after the second cycle, STOP.

**On hitting the limit, produce a diagnostic report:**
- All root causes found across all cycles
- All fixes attempted and their outcomes
- Cost incurred so far (jobs x duration x hourly rate)
- Recommendation: redesign experiment, fix upstream dependency, or authorize
  additional cycles with explicit budget cap

**Why:** Without a hard stop, fix-relaunch loops can run indefinitely. Each cycle
costs real money. The user decides whether to spend more, not Claude.

---

## Rule F5: FACTORIAL-MANIFEST

> Before launch, create `factorial_manifest.json` that is the SINGLE SOURCE OF TRUTH
> for experiment state.

**Required fields:**
- `experiment_id`: Unique identifier (timestamp + config name)
- `config_file`: Path to the factorial YAML
- `factors`: Dict of factor names → levels
- `jobs`: List of `{condition, job_id, status, start_time, end_time, cost,
  mlflow_run_id, failure_category, relaunch_batch}`

**Invariants:**
- Re-launched jobs update the EXISTING entry (same condition), not create new entries
- `relaunch_batch` tracks which cycle: 0=original, 1=first retry, 2=second retry
- No job result is valid unless it appears in the manifest
- Manifest is saved to `outputs/factorial_manifest_<experiment_id>.json`

**Why:** Without a manifest, partial factorial completion leads to amnesia —
forgetting which conditions succeeded in batch 1 when fixing batch 2 failures.
