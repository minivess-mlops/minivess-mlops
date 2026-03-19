# Protocol: Monitor

## Polling Loop

Poll `sky jobs queue` every 60 seconds. For each poll:

1. Parse ALL job statuses from queue output
2. Update the manifest with current status for each job
3. Print a live status table:

```
┌───────────────────────────────────────────────────────────┐
│ Factorial Monitor — experiment_2026-03-19_debug           │
│ Poll #42 — 14:32:15 UTC                                  │
├─────────────┬──────────┬──────┬───────────┬───────────────┤
│ Condition   │ Job ID   │ Fold │ Status    │ Duration      │
├─────────────┼──────────┼──────┼───────────┼───────────────┤
│ dynunet/dice│ 101      │ 0    │ SUCCEEDED │ 0:45:12       │
│ dynunet/dice│ 102      │ 1    │ RUNNING   │ 0:32:05       │
│ sam3/cbdice │ 103      │ 0    │ FAILED    │ 0:12:33       │
│ ...         │ ...      │ ...  │ ...       │ ...           │
├─────────────┴──────────┴──────┴───────────┴───────────────┤
│ Progress: 8/24 SUCCEEDED │ 1 FAILED │ 15 RUNNING         │
│ Est. cost so far: $4.20                                   │
└───────────────────────────────────────────────────────────┘
```

## READ-ONLY Constraint (Rule F1)

During monitoring, the ONLY permitted actions are:
- `sky jobs queue` — status polling
- `sky jobs logs <id>` — read-only log inspection
- MLflow metric queries
- Updating the manifest file
- `sky jobs cancel <id>` — ONLY for provably wasted money

**BANNED:** `sky exec`, code edits, Docker rebuilds, SSH.

## Terminal State Detection

A job is terminal when its status is one of:
`SUCCEEDED`, `FAILED`, `FAILED_SETUP`, `CANCELLED`

Continue polling until ALL jobs are terminal.

## Kill-Switch (Rule F1 Exception)

If 3+ jobs fail with IDENTICAL error within 5 minutes:
1. Identify which remaining running jobs share the same configuration
2. Cancel those jobs: `sky jobs cancel <id>`
3. Let jobs with DIFFERENT configurations continue
4. Transition immediately to DIAGNOSE phase with available failures

## On Each New Failure

When a job transitions to FAILED/FAILED_SETUP:
1. Fetch logs: `sky jobs logs <job_id> --no-follow`
2. Run `ralph_monitor.analyze_logs(logs, status)` for preliminary categorization
3. Store the FailureInfo in the manifest entry for that job
4. Do NOT start fixing — continue monitoring other jobs

## Transition to DIAGNOSE

When ALL jobs are terminal (or kill-switch activated), transition to DIAGNOSE.
