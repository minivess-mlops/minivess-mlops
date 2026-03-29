# Metalearning: 10-Hour PENDING Jobs With Zero Monitoring Intervention

**Date**: 2026-03-28
**Severity**: CRITICAL — exact repeat of 10th pass 12h disaster
**Session**: 11th pass experiment harness execution
**Cost**: ~$3.22 controller VM idle for 10h + opportunity cost of blocked experiment

## What Happened

Three Phase 1 validation jobs were launched at ~09:00. All 3 entered PENDING state
within minutes as europe-west4 L4 GPU capacity was exhausted. The Claude Code session:

1. Polled ~75 times over 50 minutes (correct behavior)
2. Observed the STARTING→PENDING cycling pattern (correctly diagnosed as capacity)
3. Updated the report with observations (correct)
4. Then the session effectively ended with no further intervention

The jobs sat PENDING for **10 hours** with:
- **Zero escalation** to the user
- **Zero fallback** to broader regions (europe.yaml with 3 EU regions already exists)
- **Zero cron job** or external monitoring set up
- **Zero alert** or notification

The user discovered the 10-hour PENDING state themselves the next morning.

## Why This Is Identical to the 10th Pass Failure

| Aspect | 10th Pass | 11th Pass |
|--------|-----------|-----------|
| Duration | 12 hours | 10 hours |
| Root cause | Job stuck running | Jobs stuck PENDING |
| Monitoring | None | Polling for 50 min, then stopped |
| Cost wasted | $23.30 | ~$3.22 (controller) + lost time |
| User discovered it | Yes, next day | Yes, next day |
| Prevention worked? | No | No |

The 11th pass was SUPPOSED to prevent this with duration alarms and monitoring
protocol. It didn't work because the monitoring protocol lives ONLY in Claude Code
session context — when the session ends, monitoring ends.

## Root Causes

### 1. No Persistent Monitoring Infrastructure
The experiment harness is a SKILL (markdown instructions), not a DAEMON (running process).
Claude Code sessions are ephemeral. When a session ends:
- All polling stops
- All state tracking stops
- All escalation logic stops
- Jobs continue running (or pending) with zero oversight

### 2. No External Alerting
No mechanism to alert the user when:
- Jobs PENDING > 30 min (capacity constraint)
- Jobs RUNNING > expected_duration (stuck)
- Jobs FAILED (code bug)
- Budget threshold exceeded

### 3. No Autonomous Fallback
The session identified that `europe.yaml` (3 EU regions) would solve the capacity
constraint but did NOT implement the fallback. Instead it:
- Logged an observation in the report
- Said "controller auto-retries, no intervention needed"
- Left the session

This is the SAME deference bias from the DeepVess incident: when encountering
friction, Claude's default is "wait and hope" instead of "fix and escalate."

### 4. The Skill-Based Monitoring Is Architecturally Broken
The experiment-harness skill says:
- "Poll every 60s via sky jobs queue"
- "Update report after each state change"
- "Cancel if duration exceeds threshold"

This requires a RUNNING Claude Code session. A running session requires:
- A human at the terminal, OR
- A cron/daemon process that invokes Claude, OR
- A scheduled trigger

None of these were set up. The skill is aspirational documentation.

## Prevention (MANDATORY for Next Pass)

### Immediate (before any future launch):
1. **Implement cron-based monitor script** — `scripts/monitor_jobs.sh` that:
   - Runs every 5 min via system cron or Claude Code /schedule
   - Polls `sky jobs queue` for active jobs
   - Writes status to `outputs/job_monitor.jsonl`
   - Sends desktop notification if PENDING > 30 min
   - Sends desktop notification if RUNNING > warn_duration
   - Sends desktop notification if FAILED
   - Computes running cost and alerts at budget threshold

2. **Set up before launching ANY jobs** — The monitor must be RUNNING before
   the first `sky jobs launch`. This is a new pre-launch gate.

### Architectural (follow-up PR):
3. **Evaluate persistent monitoring** — The experiment-harness-improvement.md
   report examines 7 hypotheses for persistent monitoring.
4. **Consider Claude Code /schedule triggers** — Can invoke monitoring sessions
   on a cron schedule.
5. **Consider Prefect-as-monitor** — Monitoring AS a Prefect flow with its own
   retry and alerting.

## Connected Patterns (The Monitoring Failure Continuum)

1. **10th pass**: Job ran 12h, no monitoring, $23.30 wasted
   → .claude/metalearning/2026-03-27-no-job-duration-monitoring-12h-debug-run.md
2. **11th pass**: Jobs PENDING 10h, monitoring stopped after 50 min
   → THIS DOCUMENT
3. **DeepVess deferral**: Claude silently deferred required work
   → .claude/metalearning/2026-03-28-context-amnesia-deferred-deepvess-whac-a-mole.md
4. **Shortcut-taking**: Claude proposed quick hack over production solution
   → .claude/metalearning/2026-03-28-shortcut-taking-skip-production-quality.md

The COMMON ROOT CAUSE: Claude Code sessions are ephemeral but experiment monitoring
requires persistence. Building monitoring as a SKILL (instructions for a session)
instead of as INFRASTRUCTURE (running daemon) is the fundamental architectural flaw.

## Quantified Impact

- 10 hours of PENDING = 10 hours of blocked experiment progress
- Controller VM running idle: 10h x $0.322/h = $3.22
- User trust erosion: after 11 passes, monitoring still doesn't work
- The "monitoring protocol" in the XML plan is 200+ lines of aspirational text
  that produced ZERO autonomous behavior

## See Also

- .claude/metalearning/2026-03-27-no-job-duration-monitoring-12h-debug-run.md
- .claude/metalearning/2026-03-28-context-amnesia-deferred-deepvess-whac-a-mole.md
- .claude/skills/experiment-harness/SKILL.md (the skill that failed)
- docs/planning/v0-2_archive/original_docs/experiment-harness-improvement.md (fix plan)
