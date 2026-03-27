# Metalearning: No Job Duration Monitoring — 12h Debug Run Undetected

**Date**: 2026-03-27
**Severity**: CRITICAL
**Cost**: ~$2.50 wasted on a $0.02 job (125x overspend), plus 23 other jobs stuck PENDING
**Category**: Monitoring gap, experiment-harness skill deficiency

## What Happened

A debug factorial experiment was launched with 23 SkyPilot managed jobs on GCP L4 spot
instances. The configuration: 2 epochs, half data, 1 fold. Expected per-job runtime:
**5-10 minutes** (confirmed by actual training time of ~5 min in logs).

Job #154 (`sam3_hybrid-cbdice_cldice-calibtrue-f0`) ran for **12+ hours** with zero
alerts, zero flags, zero intervention from any monitoring skill.

## Root Cause Chain

1. **Training completed in ~5 minutes** (17:36 → 17:41) — as expected
2. **413 Request Entity Too Large** — MLflow Cloud Run rejected `last.pth` checkpoint
   upload. SAM3 Hybrid checkpoint exceeds Cloud Run's 32MB request body limit. Logged
   as WARNING, not raised. Training continued (correct — checkpoint upload failure
   shouldn't crash training).
3. **Post-training subflow started** — reloads SAM3 weights from scratch (all 1468
   params). This is the intended flow but adds 10-20 min per job.
4. **Spot preemption** during post-training — instance preempted, SkyPilot recovery
   triggered. Recovery = full setup again (Docker pull, DVC, HF weight download).
5. **Recovery instance slow** — weight loading at 148 it/s vs 815 it/s on original
   instance (5.5x slower). Possible disk I/O contention or degraded spot instance.
6. **Total: 5 min training + 20 min post-training + preemption + recovery + slow reload
   = hours.** Plus L4 spot queue time (21 jobs competing for 1-2 available instances).

## The Real Failure: No Duration Monitoring

The 12h runtime was **detectable and preventable**. A simple check:

```
if job_duration > 3 * expected_duration:
    ALERT("Job {id} running {duration}, expected {expected}. Investigate.")
```

This check exists in **zero** of our monitoring skills:

| Skill | Duration monitoring | Max duration | Stall detection |
|-------|-------------------|-------------|-----------------|
| `/experiment-harness` | None (delegates to factorial-monitor) | None | None |
| `/factorial-monitor` | Logs duration in table, never checks it | None | None |
| `/ralph-loop` | None | None (has $5 cost budget, never hit) | None |

**The factorial-monitor prints a live status table with duration columns but never
compares them against expected values.** It's like having a speedometer that displays
but has no speed limit.

## What Should Exist (Requirements)

### 1. Expected Duration Per Job Type
The experiment XML should declare expected per-job duration:
```xml
<expected-duration-minutes>15</expected-duration-minutes>
<max-duration-minutes>45</max-duration-minutes>  <!-- 3x safety margin -->
```

For debug runs: 2 epochs × ~2.5 min/epoch + 10 min setup + 10 min post-training = ~25 min.
Anything over 45 min (3x) is anomalous. Anything over 2h is broken.

### 2. Duration Anomaly Detection in factorial-monitor
The Phase 2 polling loop should check:
```
for job in running_jobs:
    if job.duration > max_duration_minutes:
        ALERT and cancel job
    elif job.duration > expected * 3:
        WARN and flag for investigation
```

### 3. Cost Anomaly Detection
Ralph-loop has a $5 cost budget but it's per-job, not per-experiment. Need:
- Per-experiment cost budget (e.g., debug experiment = $5 total)
- Per-job cost alarm (e.g., debug job > $0.50 = something is wrong)

### 4. Heartbeat Stall Detection
If a job's logs stop producing output for >10 minutes, flag it. SkyPilot shows
duration but doesn't distinguish "running and making progress" from "running and stuck."

## Bugs Found in Logs

### BUG 1: 413 Request Entity Too Large (MLflow Cloud Run)
```
413 Client Error: Request Entity Too Large for url:
https://minivess-mlflow-a7w6hliydq-lz.a.run.app/...
```
SAM3 checkpoint exceeds Cloud Run's 32MB default request body limit. Fix: increase
Cloud Run `--max-request-body-size` or use GCS artifact store (not HTTP upload).

### BUG 2: WeightWatcher crashes on Conv3d
```
AttributeError: 'NoneType' object has no attribute 'data'
```
WeightWatcher doesn't handle SAM3's Conv3d layers. Returns NaN metrics. Non-fatal
but wastes time and produces misleading metrics.

### BUG 3: Post-training reloads SAM3 from scratch
The post-training subflow loads SAM3 weights again (1468 params from HuggingFace).
In a single-container flow, the model is already in memory after training. This is
either a design choice (isolation) or a bug (unnecessary re-download).

### BUG 4: 21 jobs PENDING on spot
Only 1-2 L4 spot instances available at a time in europe-north1. 23 debug jobs
serialize to hours of wall-clock time even if each job takes 10 minutes. The
experiment-harness should estimate total wall-clock time:
`n_jobs × avg_duration / expected_spot_parallelism`.

## Prevention Rules

1. **MANDATORY**: Every experiment XML must declare `<expected-duration-minutes>` and
   `<max-duration-minutes>` per job. factorial-monitor MUST enforce these.

2. **MANDATORY**: factorial-monitor Phase 2 must implement duration anomaly detection:
   - `duration > max_duration → CANCEL + ALERT`
   - `duration > 3 × expected → WARN + investigate`

3. **MANDATORY**: experiment-harness Phase 2 VALIDATE must estimate total wall-clock
   time: `n_jobs × expected_duration / estimated_parallelism`. If >4h for a debug
   experiment, BLOCK launch and ask user.

4. **BUG FIX**: MLflow Cloud Run body size limit must be increased before production
   runs, or switch to GCS artifact store.

## Cross-References

- `.claude/skills/experiment-harness/SKILL.md` — needs duration monitoring in Phase 3
- `.claude/skills/factorial-monitor/SKILL.md` — needs duration anomaly detection in Phase 2
- `.claude/skills/ralph-loop/SKILL.md` — needs per-job max duration + stall detection
- `deployment/pulumi/gcp/CLAUDE.md` — Cloud Run body size limit
