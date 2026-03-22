# Factorial Monitor Skill Upgrade Plan

**Date**: 2026-03-22
**Branch**: `test/debug-factorial-4th-pass`
**Trigger**: 4th pass failure report — 2h 20m detection latency, $6.30 wasted
**Status**: Plan only. Implementation requires user authorization per `.claude/rules/no-unauthorized-infra.md`.

---

## 1. WHY the Current Monitoring Failed

The 4th pass debug factorial launched 10 jobs on GCP L4 spot. Eight FAILED_SETUP
(DVC pull error), one stuck STARTING for 2.5 hours, and none of these failures were
detected until a manual `sky jobs queue` check 2 hours 20 minutes after the first failure.

Five root causes explain why the monitoring infrastructure failed to catch this:

### 1.1 `run_factorial.sh` Does Not Capture Job IDs in a Machine-Readable Manifest

The launch script writes a human-readable `.txt` log:
```
# FORMAT: condition_id | model | loss | aux_calib | fold | status
1 | dynunet | cbdice_cldice | false | 0 | LAUNCHED
```

This text file has no SkyPilot `job_id`, no timestamp, no cost rate, and no way for a
downstream polling loop to match a SkyPilot queue entry back to a factorial condition.
The manifest schema defined in `templates/factorial-manifest.json` is never populated
by the actual launch script. The skill's SKILL.md says "Create `factorial_manifest.json`
mapping job_id to condition" (Phase 1), but `run_factorial.sh` does not do this.

**Gap**: The skill protocol exists on paper; the shell harness does not implement it.

### 1.2 `ralph_monitor.py` Watches ONE Job, Not a Batch

Ralph-loop (SKILL.md line 9) explicitly says: "Do NOT use for factorial multi-job
monitoring (use factorial-monitor)." But factorial-monitor has no monitoring script at
all. The skill describes a 60-second polling loop in `protocols/monitor.md`, but this
is a Claude Code behavior protocol (Claude polls manually), not an automated script.

When the user goes away, there is no process running that polls. The monitoring relies
entirely on the user or Claude Code being present and executing `sky jobs queue` commands.

**Gap**: No automated polling daemon. The skill assumes Claude Code is always present.

### 1.3 No Automatic Polling After Launch

`run_factorial.sh` ends with:
```
Monitor: sky jobs queue
Logs:    sky jobs logs <JOB_ID>
```

It prints instructions for the user to poll manually. There is no `--monitor` flag, no
backgrounded polling loop, no integration with `ralph_monitor.py`. The script launches
jobs and exits. Nobody watches.

**Gap**: Launch and monitor are completely disconnected. The script does not chain to
a monitoring phase.

### 1.4 No Alerting on FAILED_SETUP

SkyPilot's `FAILED_SETUP` state means the VM was provisioned ($) but the setup script
crashed before `run:` executed. This is the most expensive failure mode per wasted dollar
because the VM spins up, runs for 1-3 minutes during setup, and dies. At $0.60/instance,
eight FAILED_SETUP jobs cost $4.80 before anyone noticed.

Neither `ralph_monitor.py` nor the factorial-monitor skill has a specific detection
rule for FAILED_SETUP. The failure-patterns.md in ralph-loop lists `DATA_MISSING` and
`DVC_NO_GIT` as categories, but `FAILED_SETUP` as a SkyPilot-level state transition
is not tracked. The DVC pull error from the 4th pass would match `DATA_MISSING`, but
only if someone fetched the logs and ran the diagnosis — which nobody did for 2h 20m.

**Gap**: No early-exit detection for setup failures. FAILED_SETUP should trigger an
alert within 2 minutes (setup scripts run in ~60 seconds).

### 1.5 Detection Latency: 2 Hours 20 Minutes

Timeline from the failure report:
```
05:20 — Jobs 57-60 FAILED_SETUP (first failures)
07:40 — Manual sky jobs queue check (first detection)
```

With 30-second polling, the first FAILED_SETUP would have been detected at 05:21
(1 minute after failure). With the kill-switch rule (3+ identical failures within 5
minutes), remaining launches would have been cancelled at 05:25 (5 minutes, not 140
minutes). Cost saved: the 4 later FAILED_SETUP jobs ($2.40) plus the stuck STARTING
job ($1.50) = $3.90 of the $6.30 total.

**Gap**: 30-second polling would have reduced the detection window from 140 minutes to
1 minute — a 140x improvement.

---

## 2. WHAT the Upgraded Shell Harness Needs

### 2.1 `run_factorial.sh` Must Write `job_manifest.json` After Launch

The current `.txt` log must be replaced with (or supplemented by) a JSON manifest that
matches the schema in `templates/factorial-manifest.json`. The manifest must be written
incrementally — each successful `sky jobs launch` appends to the manifest immediately,
so a crash mid-launch does not lose already-launched job IDs.

**Implementation**: After each `sky jobs launch` call, capture the job ID from SkyPilot's
stdout. SkyPilot prints `Job ID: <int>` on successful launch. Parse this (using
`str.partition()`, not regex per CLAUDE.md Rule 16) and write to the manifest.

Required fields per job entry:
```json
{
  "condition": "dynunet-cbdice_cldice-calibfalse-f0",
  "model_family": "dynunet",
  "loss_name": "cbdice_cldice",
  "aux_calib": false,
  "fold": 0,
  "job_id": 55,
  "status": "SUBMITTED",
  "launched_at": "2026-03-22T05:08:12Z",
  "hourly_rate_usd": null,
  "cost_usd": null,
  "failure_category": null,
  "relaunch_batch": 0
}
```

The manifest file path: `outputs/factorial_manifest_<experiment_id>_<timestamp>.json`

**Acceptance criterion**: After `run_factorial.sh` completes, every launched job has a
`job_id` in the manifest. Zero null job_ids for LAUNCHED status jobs.

### 2.2 30-Second Polling Loop

A new script `scripts/monitor_factorial.py` (or a `--monitor` mode in `run_factorial.sh`)
must run after launch and poll continuously:

```
Every 30 seconds:
  1. Run `sky jobs queue --all -o json` (or parse sky jobs queue stdout)
  2. For each job_id in the manifest:
     a. Update status from SkyPilot queue
     b. Calculate elapsed time since launched_at
     c. Estimate cost: elapsed_hours * hourly_rate
  3. Print live status table (same format as protocols/monitor.md)
  4. Check FAILED_SETUP detection (see 2.3)
  5. Check kill-switch condition (see 2.5)
  6. If all jobs terminal: exit polling, write final manifest
```

**Note on `sky jobs queue` JSON output**: SkyPilot v1.0 may not have `-o json`. If not
available, parse the table output using `str.split()` per Rule 16. Verify the exact CLI
interface before implementation:

```bash
sky jobs queue --help 2>&1 | grep -i json
```

If JSON output is not available, use the SkyPilot Python API:
```python
import sky
sky.jobs.queue()  # Returns list of dicts
```

**Acceptance criterion**: Polling loop runs unattended. Status table updates every 30
seconds. No human intervention required between launch and final report.

### 2.3 Automatic FAILED_SETUP Detection Within 2 Minutes

FAILED_SETUP is the single most expensive failure mode per dollar because the VM is
provisioned and billed, but no useful work executes. Detection logic:

```
If a job transitions to FAILED_SETUP:
  1. Log: "ALERT: Job {job_id} ({condition}) FAILED_SETUP at {timestamp}"
  2. Fetch setup logs: `sky jobs logs {job_id} --no-follow`
  3. Run ralph-loop failure pattern matching on the logs
  4. Update manifest: status=FAILED_SETUP, failure_category=<matched category>
  5. Check kill-switch (3+ identical failures in 5 min)
```

Since setup scripts run in ~60 seconds, and polling runs every 30 seconds, the maximum
detection latency is 90 seconds (worst case: failure occurs 1 second after a poll, next
poll 30 seconds later, but the job was already FAILED_SETUP before the poll). Typical
detection latency: 15-45 seconds.

**New failure pattern for ralph-loop/instructions/failure-patterns.md**:

| Category | Pattern in Logs | Auto-fix? | Fix Action |
|----------|----------------|-----------|------------|
| `DVC_PULL_FAIL` | `FATAL: DVC pull from GCS failed` or `dvc pull` + non-zero exit | No | Fix DVC config, run preflight |
| `SETUP_TIMEOUT` | STARTING for >10 min without transitioning | Maybe | Check region capacity |

**Acceptance criterion**: FAILED_SETUP detected within 2 minutes of occurrence. Detection
latency logged in the manifest for each failure. Zero FAILED_SETUP jobs go undetected
for more than 2 minutes.

### 2.4 Cost Tracking Per Job

Each manifest entry tracks cost:

```json
{
  "hourly_rate_usd": 0.75,
  "started_at": "2026-03-22T05:08:12Z",
  "ended_at": "2026-03-22T05:10:45Z",
  "duration_seconds": 153,
  "cost_usd": 0.032
}
```

Hourly rate source: `sky show-gpus --cloud gcp` (do not hardcode per CLAUDE.md Rule 29).
The rate should be queried once at launch time and stored in the manifest header.

Running cost is displayed in the status table:
```
| Condition         | Job ID | Status       | Duration | Cost   |
|-------------------|--------|--------------|----------|--------|
| dynunet-dice_ce-f0| 55     | RUNNING      | 0:32:15  | $0.40  |
| sam3-cbdice-f0    | 57     | FAILED_SETUP | 0:02:33  | $0.03  |
| ...               | ...    | ...          | ...      | ...    |
| TOTAL             |        |              |          | $2.84  |
```

**Acceptance criterion**: Every terminal job has `cost_usd > 0`. Total cost matches
the sum of individual job costs within $0.01 rounding.

### 2.5 Kill-Switch: Cancel Remaining If >50% Fail

Two kill-switch triggers:

**Trigger A (rapid identical failures)**: Already defined in Rule F1 — 3+ jobs with
identical error within 5 minutes. This is the fast path.

**Trigger B (majority failure)**: New rule — if more than 50% of launched jobs have
reached a FAILED or FAILED_SETUP state, cancel all PENDING and STARTING jobs. Rationale:
if 17 out of 32 jobs fail, the remaining 15 are overwhelmingly likely to fail for the
same reason. Continuing wastes money.

Kill-switch execution:
```
1. Log: "KILL-SWITCH ACTIVATED: {reason}"
2. For each job with status PENDING or STARTING:
   a. `sky jobs cancel {job_id}`
   b. Update manifest: status=CANCELLED, failure_category="kill_switch"
3. For each job with status RUNNING:
   a. DO NOT CANCEL — let running jobs finish (they may have passed the failure point)
4. Transition to DIAGNOSE phase with current failures
```

**Acceptance criterion**: Kill-switch fires within 1 polling cycle (30s) after the
trigger condition is met. Cancelled jobs are marked in the manifest. RUNNING jobs are
NOT cancelled by the kill-switch.

---

## 3. HOW to Integrate with Ralph-Loop

### 3.1 Architectural Separation

| Concern | Ralph-Loop | Factorial-Monitor |
|---------|-----------|-------------------|
| **Scope** | One SkyPilot job | N SkyPilot jobs (batch) |
| **Polling** | Single job status + logs | Batch queue polling |
| **Diagnosis** | Per-job failure pattern matching | Aggregation across jobs by root cause |
| **Recovery** | Auto-fix + retry (max 3) | Selective re-launch of failed conditions (max 2 cycles) |
| **Kill-switch** | No (single job) | Yes (batch majority failure) |
| **Cost tracking** | Per-attempt JSONL event | Per-experiment manifest with running totals |
| **User presence** | Can run autonomously | Can run autonomously |

### 3.2 Composition Protocol

Factorial-monitor composes with ralph-loop at the per-job diagnosis level:

```
factorial-monitor (batch level)
  |
  |-- For each newly-terminal FAILED job:
  |     |-- Call ralph-loop's failure pattern matching
  |     |     (the 14 categories in failure-patterns.md)
  |     |-- Receive: {failure_category, auto_fixable, confidence}
  |     |-- Store in manifest entry for that job
  |
  |-- After all jobs terminal:
  |     |-- AGGREGATE failures by category (Rule F2)
  |     |-- Present ONE batch report
  |     |-- For each category: determine if auto-fixable at batch level
  |
  |-- FIX phase:
        |-- For code fixes: compose_with self-learning-iterative-coder
        |-- For infra fixes: apply directly
        |-- For unrecoverable: compose_with issue-creator
```

### 3.3 Shared Failure Patterns

Ralph-loop's `instructions/failure-patterns.md` is the single source of truth for
failure pattern definitions. Factorial-monitor IMPORTS these patterns — it never
defines its own parallel set. When a new failure pattern is discovered during a
factorial run (like `DVC_PULL_FAIL` from the 4th pass), it is added to ralph-loop's
pattern table and then available to both skills.

New patterns to add (from 4th pass analysis):

| Category | Pattern in Logs | Auto-fix? | Fix Action |
|----------|----------------|-----------|------------|
| `DVC_PULL_FAIL` | `FATAL: DVC pull` or `dvc pull` exit non-zero | No | Run preflight, fix DVC config, push data |
| `DVC_TRACKED_NOT_PUSHED` | `dvc pull` succeeds but directory is empty or missing | No | `dvc push <path> -r <remote>` before launch |
| `STUCK_STARTING` | STARTING state for >10 minutes | Maybe | Cancel + try different region/zone |

### 3.4 What Ralph-Loop Does NOT Need to Change

Ralph-loop's core loop (PRE-FLIGHT, LAUNCH, MONITOR, DIAGNOSE, FIX, REPORT) is correct
for single-job use. The factorial-monitor upgrade does NOT require modifying ralph-loop.
The integration point is purely at the diagnosis function — factorial-monitor calls
ralph-loop's pattern-matching logic and aggregates the results.

---

## 4. EXECUTABLE TASKS with Acceptance Criteria

### Task 1: Add Job ID Capture to `run_factorial.sh`

**File**: `scripts/run_factorial.sh`

**Change**: After each `sky jobs launch` call, capture the job ID from stdout. Write a
JSON manifest incrementally.

**Implementation details**:
- Capture `sky jobs launch` stdout to a temp variable
- Parse "Job ID: <int>" using bash string manipulation (`${output##*"Job ID: "}`)
  or Python `str.partition()` (no regex)
- Write each job entry to `outputs/factorial_manifest_<experiment_id>.json` using Python
  `json.dumps()` (incremental write: read → append → write)
- Include: job_id, condition, model, loss, aux_calib, fold, launched_at, relaunch_batch=0

**Acceptance criteria**:
- [ ] After a `--dry-run`, manifest contains all conditions with `job_id: null, status: "DRY_RUN"`
- [ ] After a live run, manifest contains all conditions with integer `job_id` values
- [ ] Manifest validates against `templates/factorial-manifest.json` schema
- [ ] If a launch fails mid-way (e.g., job 17 of 32), previously launched jobs still have correct IDs in the manifest
- [ ] No `import re` anywhere in the implementation

**Test**: `tests/v2/unit/scripts/test_run_factorial_manifest.py`
- Mock `sky jobs launch` to return predictable job IDs
- Verify manifest structure after 4 conditions

**Estimated effort**: 2-3 hours

---

### Task 2: Create `scripts/monitor_factorial.py` Polling Script

**File**: `scripts/monitor_factorial.py` (new)

**Purpose**: Standalone polling loop that reads a factorial manifest and monitors all
jobs until completion.

**Interface**:
```bash
# Start monitoring after launch:
python scripts/monitor_factorial.py outputs/factorial_manifest_<id>.json

# Or chain from run_factorial.sh (see Task 5):
./scripts/run_factorial.sh configs/experiment/debug_factorial.yaml --monitor
```

**Implementation details**:
- Read manifest JSON from argument
- Every 30 seconds:
  - Query SkyPilot job queue (Python API preferred: `sky.jobs.queue()`)
  - Match queue entries to manifest by job_id
  - Update status, duration, cost for each job
  - Print formatted status table to stdout
  - Check FAILED_SETUP detection (Task 3)
  - Check kill-switch conditions (Task 4)
  - Write updated manifest to disk
- Exit when all jobs are terminal
- Write final summary (total cost, success/failure counts)

**Acceptance criteria**:
- [ ] Script runs unattended for the full duration of a factorial experiment
- [ ] Status table updates every 30 seconds (tolerance: +/- 5 seconds)
- [ ] Manifest on disk always reflects the latest known state
- [ ] Script exits cleanly when all jobs are terminal
- [ ] No `import re` anywhere in the implementation
- [ ] Uses `datetime.now(timezone.utc)` for all timestamps (CLAUDE.md Rule 7)
- [ ] Uses `pathlib.Path` for all file operations (CLAUDE.md Rule 6)
- [ ] Encodes files with `encoding='utf-8'` (CLAUDE.md Rule 5)

**Test**: `tests/v2/unit/scripts/test_monitor_factorial.py`
- Mock `sky.jobs.queue()` returning staged failure sequences
- Verify manifest updates, status table output, terminal detection

**Estimated effort**: 4-6 hours

---

### Task 3: FAILED_SETUP Early Detection

**File**: `scripts/monitor_factorial.py` (within the polling loop)

**Implementation details**:
- On each poll, check for jobs that transitioned to `FAILED_SETUP` since last poll
- For each newly FAILED_SETUP job:
  1. Print alert: `ALERT: Job {job_id} ({condition}) FAILED_SETUP`
  2. Fetch logs: `sky jobs logs {job_id}` (or Python API equivalent)
  3. Run pattern matching from ralph-loop's failure categories
  4. Update manifest: `status=FAILED_SETUP, failure_category=<category>`
  5. Feed into kill-switch evaluation (Task 4)
- Track `first_failure_detected_at` and `detection_latency_seconds` in manifest

**New failure categories for ralph-loop** (added to `failure-patterns.md`):

| Category | Symptom | Log pattern (str match) |
|----------|---------|----------------------|
| `DVC_PULL_FAIL` | DVC pull exits non-zero | `"FATAL: DVC pull"` in log |
| `DVC_TRACKED_NOT_PUSHED` | DVC pull "succeeds" but data missing | `"No training data"` after DVC section |
| `STUCK_STARTING` | Job in STARTING for >10 min | Status = STARTING, elapsed > 600s |
| `SETUP_SCRIPT_CRASH` | Non-zero exit from setup block | `FAILED_SETUP` status with no matching pattern |

**Acceptance criteria**:
- [ ] Every FAILED_SETUP job triggers an ALERT log line within 60 seconds (2 poll cycles)
- [ ] Every FAILED_SETUP job has a `failure_category` in the manifest (never null)
- [ ] `detection_latency_seconds` is recorded and is always < 120
- [ ] `STUCK_STARTING` detected for jobs in STARTING state > 10 minutes

**Test**: `tests/v2/unit/scripts/test_failed_setup_detection.py`
- Mock a queue where 3 jobs go FAILED_SETUP at t=30s
- Verify detection happens at t=60s (next poll)
- Verify manifest has failure categories

**Estimated effort**: 2-3 hours (builds on Task 2)

---

### Task 4: Kill-Switch Implementation

**File**: `scripts/monitor_factorial.py` (within the polling loop)

**Two triggers**:

**Trigger A**: 3+ jobs fail with identical `failure_category` within 5 minutes.
Already specified in Rule F1 of `instructions/rules.md`. Implementation:
```python
# Track recent failures: list of (timestamp, failure_category) tuples
# After each new failure:
#   Count failures with same category in last 5 minutes
#   If count >= 3: activate kill-switch for same-config jobs
```

**Trigger B**: >50% of all launched jobs are FAILED or FAILED_SETUP.
```python
failed_count = sum(1 for j in manifest["jobs"] if j["status"] in ("FAILED", "FAILED_SETUP"))
total_count = len(manifest["jobs"])
if failed_count > total_count * 0.5:
    activate_kill_switch("majority_failure")
```

**Kill-switch action**:
```python
for job in manifest["jobs"]:
    if job["status"] in ("PENDING", "STARTING"):
        sky.jobs.cancel(job["job_id"])
        job["status"] = "CANCELLED"
        job["failure_category"] = "kill_switch"
    # RUNNING jobs are NOT cancelled — they may have passed the failure point
```

**Acceptance criteria**:
- [ ] Trigger A fires when 3+ jobs fail with same category within 5 minutes
- [ ] Trigger B fires when >50% of jobs are in failed states
- [ ] Only PENDING and STARTING jobs are cancelled — RUNNING jobs continue
- [ ] Kill-switch event is logged: timestamp, trigger reason, jobs cancelled
- [ ] Manifest records `"failure_category": "kill_switch"` for cancelled jobs
- [ ] Kill-switch fires within 1 polling cycle (30s) of trigger condition

**Test**: `tests/v2/unit/scripts/test_kill_switch.py`
- Scenario A: 3 FAILED_SETUP with same category in 2 minutes
- Scenario B: 17 of 32 jobs FAILED
- Verify correct jobs cancelled, correct jobs left running

**Estimated effort**: 2-3 hours (builds on Task 2)

---

### Task 5: Wire Monitoring into Launch Script

**File**: `scripts/run_factorial.sh`

**Change**: Add `--monitor` flag that chains to `scripts/monitor_factorial.py` after
all jobs are launched.

```bash
# Launch only:
./scripts/run_factorial.sh configs/experiment/debug_factorial.yaml

# Launch + monitor (recommended):
./scripts/run_factorial.sh --monitor configs/experiment/debug_factorial.yaml
```

When `--monitor` is passed:
1. Launch all jobs (existing behavior)
2. Write manifest (Task 1)
3. Exec into `python scripts/monitor_factorial.py <manifest_path>`

The script must handle the case where the user Ctrl-C's during monitoring — the
manifest on disk should still be valid (all writes are atomic: write to temp file,
then rename).

**Acceptance criteria**:
- [ ] `--monitor` flag accepted and documented in usage message
- [ ] After all launches complete, monitoring starts automatically
- [ ] Ctrl-C during monitoring leaves a valid manifest on disk
- [ ] `--dry-run --monitor` does NOT start monitoring (nothing to monitor)
- [ ] Exit code reflects experiment outcome: 0 = all succeeded, 1 = any failures

**Test**: `tests/v2/unit/scripts/test_run_factorial_monitor_flag.py`
- Verify flag parsing
- Verify exec handoff to monitor script

**Estimated effort**: 1-2 hours

---

### Task 6: Update Ralph-Loop Failure Patterns

**File**: `.claude/skills/ralph-loop/instructions/failure-patterns.md`

**Change**: Add the 4 new failure categories discovered during the 4th pass:

1. `DVC_PULL_FAIL` — DVC pull command exits non-zero
2. `DVC_TRACKED_NOT_PUSHED` — DVC pull "succeeds" but data directory is empty
3. `STUCK_STARTING` — Job stuck in STARTING state for >10 minutes
4. `SETUP_SCRIPT_CRASH` — FAILED_SETUP with no matching specific pattern

Each entry needs: symptom, log pattern (string match), auto-fix flag, fix action,
escalation path. Follow the existing format in `failure-patterns.md`.

**Acceptance criteria**:
- [ ] All 4 categories added with full entries matching existing format
- [ ] No regex patterns — only `str.partition()` and `in` checks
- [ ] Each category has a clear escalation path

**Estimated effort**: 30 minutes

---

### Task 7: Update SKILL.md and Protocols

**Files**:
- `.claude/skills/factorial-monitor/SKILL.md`
- `.claude/skills/factorial-monitor/protocols/launch.md`
- `.claude/skills/factorial-monitor/protocols/monitor.md`

**Changes**:
- SKILL.md: Update Phase 2 polling interval from 60s to 30s. Add reference to
  `scripts/monitor_factorial.py`. Add FAILED_SETUP detection to Phase 2. Add
  kill-switch Trigger B (majority failure) alongside Trigger A.
- `protocols/launch.md`: Document manifest creation via `run_factorial.sh` (not
  manual post-hoc creation). Reference Task 1 implementation.
- `protocols/monitor.md`: Document automated polling via `monitor_factorial.py`.
  Clarify that Claude Code presence is no longer required for monitoring. Add
  FAILED_SETUP detection protocol. Add kill-switch Trigger B.

**Acceptance criteria**:
- [ ] SKILL.md reflects the automated monitoring architecture
- [ ] Polling interval documented as 30s (not 60s)
- [ ] FAILED_SETUP explicitly mentioned in Phase 2
- [ ] Kill-switch has both Trigger A (identical errors) and Trigger B (majority failure)
- [ ] `scripts/monitor_factorial.py` referenced as the monitoring implementation

**Estimated effort**: 1 hour

---

### Task 8: Cost Rate Query at Launch Time

**File**: `scripts/run_factorial.sh` (or a helper called by it)

**Change**: Before launching, query the hourly rate for the configured GPU type and
store it in the manifest header.

```bash
# Query rate (no hardcoding — CLAUDE.md Rule 29)
sky show-gpus L4 --cloud gcp 2>/dev/null | ...
```

Or via Python:
```python
import sky
gpus = sky.list_accelerators(gpus_only=True, cloud='gcp')
# Extract L4 hourly rate
```

The rate is stored once in the manifest header:
```json
{
  "gpu_type": "L4",
  "cloud": "gcp",
  "hourly_rate_usd": 0.75,
  "spot_discount": 0.63,
  "effective_hourly_rate_usd": 0.28
}
```

**Acceptance criteria**:
- [ ] Hourly rate queried from SkyPilot at launch time, not hardcoded
- [ ] Rate stored in manifest header
- [ ] Cost calculations in monitor use this rate
- [ ] Spot vs. on-demand distinction tracked

**Estimated effort**: 1-2 hours

---

### Task 9: Preflight Integration Gate

**File**: `scripts/run_factorial.sh` (already partially implemented)

**Change**: The preflight check (`scripts/preflight_gcp.py`) already runs before launch.
Strengthen it with checks that would have caught the 4th pass failures:

1. **DVC pull path validation**: `dvc pull data/raw/minivess -r gcs` must succeed
   locally (or at least `dvc status data/raw/minivess -r gcs` must show no errors)
2. **SkyPilot YAML schema validation**: `sky.Task.from_yaml()` must parse without error
3. **Docker image freshness**: Manifest records the image digest at launch time; if the
   image is older than 24 hours, warn

These checks are already planned in Issue #908 (SkyPilot local test suite) and the
DVC test suite plan. This task ensures they are wired into the launch script as a
hard gate, not optional.

**Acceptance criteria**:
- [ ] `preflight_gcp.py` validates DVC pull path (not just GCS bucket access)
- [ ] `preflight_gcp.py` validates SkyPilot YAML via `Task.from_yaml()`
- [ ] Launch script refuses to proceed if preflight fails (already enforced, verify)
- [ ] `SKIP_PREFLIGHT=1` escape hatch is documented as dangerous

**Estimated effort**: 2-3 hours (overlaps with Issue #908)

---

## 5. Implementation Priority

Ordered by impact on preventing the 4th pass failure scenario:

| Priority | Task | Impact | Blocks |
|----------|------|--------|--------|
| **P0** | Task 1: Job ID capture in manifest | Without this, nothing else works | Tasks 2-5, 8 |
| **P0** | Task 6: Update ralph-loop patterns | New categories needed for diagnosis | Task 3 |
| **P1** | Task 2: Polling script | Core monitoring capability | Tasks 3, 4, 5 |
| **P1** | Task 3: FAILED_SETUP detection | Prevents 2h detection latency | Task 4 |
| **P1** | Task 4: Kill-switch | Prevents cascading waste | Task 5 |
| **P2** | Task 5: Wire --monitor flag | Usability: one command to launch + monitor | None |
| **P2** | Task 8: Cost rate query | Accurate cost tracking | None |
| **P2** | Task 7: Update SKILL.md | Documentation accuracy | None |
| **P3** | Task 9: Preflight integration | Defense in depth | None |

**Critical path**: Task 1 + Task 6 (parallel) -> Task 2 -> Task 3 -> Task 4 -> Task 5

**Total estimated effort**: 15-23 hours

---

## 6. How This Prevents the 4th Pass Failure

Mapping each 4th pass failure to the task that prevents it:

| 4th Pass Failure | Cost | Prevention Task | Detection Time |
|------------------|------|----------------|----------------|
| DVC pull fails (8 FAILED_SETUP) | $4.80 | Task 3 (FAILED_SETUP detection) + Task 4 (kill-switch) | < 2 min (was 2h 20m) |
| Job 55 stuck STARTING 2.5 hrs | $1.50 | Task 3 (STUCK_STARTING pattern) | < 10 min (was 2h 30m) |
| No monitoring at all | $0 (time waste) | Task 2 (automated polling) + Task 5 (--monitor flag) | 30s polling interval |
| No job ID tracking | $0 (confusion) | Task 1 (manifest with job IDs) | Manifest written at launch |
| Preflight missed DVC issue | $0 (would prevent launch) | Task 9 (preflight gate) | Pre-launch |

With this upgrade, the 4th pass scenario plays out differently:
```
05:08 — Launch starts, manifest written incrementally (Task 1)
05:08 — Monitoring starts automatically (Task 5)
05:20 — Jobs 57-60 FAILED_SETUP → detected at 05:20:30 (Task 3)
05:20 — Pattern match: DVC_PULL_FAIL (Task 6)
05:25 — Kill-switch: 4 identical failures in 5 min → cancel remaining (Task 4)
05:25 — Total cost: ~$2.40 (4 FAILED_SETUP, 0 stuck, 0 undetected)
         Saved: $3.90 vs. actual $6.30
```

---

## 7. Open Questions for User

1. **Polling mechanism**: Should `monitor_factorial.py` be a standalone Python script
   (recommended — can run in background via `nohup` or `tmux`) or integrated into
   `run_factorial.sh` as a bash loop? Python is preferred for JSON manipulation and
   SkyPilot API access.

2. **Notification method**: Beyond terminal output, should the monitor send
   notifications on failure? Options: desktop notification (`notify-send`), Slack
   webhook, email. Or is terminal + manifest sufficient for now?

3. **Kill-switch Trigger B threshold**: 50% is proposed. Should this be configurable
   in the factorial config YAML? Lower threshold (e.g., 30%) = more aggressive
   cancellation. Higher threshold (e.g., 75%) = more tolerant.

---

## References

- **4th pass failure report**: `docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-report-4th-pass-failure.md`
- **Current skill spec**: `.claude/skills/factorial-monitor/SKILL.md`
- **Ralph-loop skill**: `.claude/skills/ralph-loop/SKILL.md`
- **Ralph-loop failure patterns**: `.claude/skills/ralph-loop/instructions/failure-patterns.md`
- **Launch script**: `scripts/run_factorial.sh`
- **SkyPilot YAML**: `deployment/skypilot/train_factorial.yaml`
- **Factorial manifest schema**: `.claude/skills/factorial-monitor/templates/factorial-manifest.json`
- **Issue #908**: Local SkyPilot test suite
- **Issue #907**: Cloud GPU pipeline gaps
