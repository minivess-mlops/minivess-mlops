# Retries for SkyPilot Spots & Auto-Resume Plan

**Date**: 2026-03-24
**Priority**: P0 — the system MUST be fire-and-forget
**Principle**: Zero manual intervention. Script submits, SkyPilot retries, spots
eventually provision. User comes back in 1 hour or 1 week — results are there.

---

## The Problem

Currently, the factorial launch requires manual intervention at multiple points:
1. Script gets killed → must manually restart
2. Spot unavailable → jobs PENDING indefinitely, no auto-escalation
3. Controller crashes → all job state lost, must relaunch
4. Network interruption → script dies, partially-submitted factorial
5. Preemption → SkyPilot handles recovery, but only for `sky jobs launch` (managed jobs)

**Unacceptable**: The user had to ask 4+ times to relaunch because the script
kept dying from various infrastructure issues. A production-grade system handles
ALL of these automatically.

---

## Architecture: What SkyPilot Already Provides

SkyPilot managed jobs (`sky jobs launch`) already handle:
- **Spot preemption recovery**: Controller detects preemption, provisions new VM, re-runs
- **Zone failover**: If a zone is full, tries next zone in the region
- **Checkpoint persistence**: `MOUNT_CACHED` uploads checkpoints to GCS asynchronously
- **Job queuing**: PENDING jobs wait indefinitely until resources are available

What SkyPilot does NOT handle:
- **Submitting the jobs in the first place** — that's `run_factorial.sh`
- **Resubmitting failed LAUNCH attempts** — script errors, network drops
- **Monitoring across sessions** — controller state persists, but monitoring doesn't

---

## Solution: 3 Layers of Resilience

### Layer 1: run_factorial.sh — Retry on Launch Failure

Current: launch fails → log "WARNING" → continue to next condition.
Fix: launch fails → retry up to 3 times with exponential backoff.

```bash
MAX_RETRIES=3
RETRY_DELAY=30  # seconds, doubles each retry

for retry in $(seq 1 $MAX_RETRIES); do
    if sky jobs launch ... -y; then
        break
    fi
    if [ "$retry" -eq "$MAX_RETRIES" ]; then
        echo "FAILED after $MAX_RETRIES retries"
        break
    fi
    sleep $((RETRY_DELAY * retry))
done
```

### Layer 2: Wrapper Script — Resume Incomplete Factorials

New: `scripts/run_factorial_resilient.sh` wraps `run_factorial.sh` with:
1. **Idempotent submissions**: Before launching condition X, check `sky jobs queue`
   for an existing job with the same name. Skip if already PENDING/RUNNING/SUCCEEDED.
2. **Resume from partial**: Parse the job log to find which conditions were LAUNCHED
   vs LAUNCH_FAILED, and only submit the missing ones.
3. **Infinite retry loop**: If ALL remaining conditions fail to launch (e.g., network
   down), wait 5 minutes and try again. Up to 1 week (configurable).

```bash
MAX_WAIT_HOURS=168  # 1 week
RETRY_INTERVAL=300  # 5 minutes

while [ remaining_conditions -gt 0 ] && [ elapsed_hours -lt MAX_WAIT_HOURS ]; do
    run_factorial.sh --resume  # Only submits missing conditions
    sleep $RETRY_INTERVAL
done
```

### Layer 3: SkyPilot Managed Jobs — Spot Recovery

Already works for submitted jobs. Key settings:
- `use_spot: true` — SkyPilot auto-recovers on preemption
- `MOUNT_CACHED` for checkpoints — survives preemption
- `check_resume_state_task()` in train_flow.py — loads latest checkpoint on resume

**No changes needed** at this layer — SkyPilot handles it.

---

## Implementation Tasks

### Task 1: Add `--resume` flag to run_factorial.sh
- Parse job log for previously LAUNCHED conditions
- Query `sky jobs queue` for existing jobs by name
- Skip conditions that are PENDING/RUNNING/SUCCEEDED
- Only submit LAUNCH_FAILED or never-attempted conditions

### Task 2: Add retry-with-backoff to individual launches
- MAX_RETRIES from config (default 3)
- Exponential backoff (30s, 60s, 120s)
- Log each retry attempt

### Task 3: Create run_factorial_resilient.sh wrapper
- Outer loop: call run_factorial.sh --resume every 5 minutes
- Configurable MAX_WAIT_HOURS (default 168 = 1 week)
- Progress reporting: "X/34 submitted, Y/34 succeeded, Z/34 remaining"
- Clean exit when all conditions are terminal (SUCCEEDED or max retries exhausted)

### Task 4: Add nohup/screen-free background execution
- `run_factorial_resilient.sh &` with proper signal handling
- PID file for status checks
- Log rotation (don't fill disk with retry logs)

---

## Configuration

Add to configs/factorial/debug.yaml:
```yaml
infrastructure:
  cloud_config: gcp_spot
  skypilot_yaml: deployment/skypilot/train_factorial.yaml
  # Retry settings (read by run_factorial.sh)
  max_launch_retries: 3
  launch_retry_backoff_seconds: 30
  # Resilient wrapper settings (read by run_factorial_resilient.sh)
  max_wait_hours: 168  # 1 week
  resume_interval_seconds: 300  # Check every 5 minutes
```

---

## What "Fire-and-Forget" Means

After the user runs:
```bash
nohup bash scripts/run_factorial_resilient.sh configs/factorial/debug.yaml &
```

They should be able to:
1. Close the terminal
2. Go to sleep
3. Come back in 1 day or 1 week
4. Run `uv run sky jobs queue` → see all 34 conditions SUCCEEDED
5. Run `cat outputs/*_factorial_job_ids.txt` → see the full log

Zero manual intervention at any point. If spots are unavailable for 3 days,
the script waits and retries. If the network drops, the script recovers.
If a job gets preempted, SkyPilot handles it.

---

## Related

- Issue #913: Launch bottleneck (parallel submissions — implemented)
- Issue #914: On-demand fallback (for users who need faster results)
- `.claude/metalearning/2026-03-24-yaml-is-the-contract-zero-improvisation.md`
