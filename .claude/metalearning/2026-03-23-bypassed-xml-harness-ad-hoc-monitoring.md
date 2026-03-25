# Metalearning: Bypassed XML Harness — Ad-Hoc Monitoring Instead of SOP

**Date**: 2026-03-23
**Severity**: HIGH — systematic learning loss, compounding failure
**Category**: Process bypass, institutional memory failure

## What Happened

The 5th pass factorial experiment had a well-structured XML plan
(`run-debug-factorial-experiment-5th-pass.xml`) with:
- 6-phase monitoring protocol
- Dual mandate (platform validation + cloud test suite hardening)
- Watchlist of 8 items from passes 1-4
- Cloud test upgrade plan with specific new tests to write

Claude Code loaded the `/factorial-monitor` skill, acknowledged the protocol,
then IMMEDIATELY dropped into ad-hoc `sky jobs queue` polling — ignoring:
1. The XML plan's monitoring protocol
2. The report output specified in the XML (`run-debug-factorial-experiment-report-5th-pass.md`)
3. The dual mandate to capture test improvement opportunities
4. The watchlist items to verify
5. The cloud test upgrade plan

## Why This Keeps Happening

1. **Urgency bias**: The launch was happening in real-time, so Claude
   prioritized "watch the jobs" over "follow the process." The XML plan
   felt like overhead when jobs were actively running.

2. **Skill loading ≠ skill following**: Loading `/factorial-monitor` gives
   the illusion of following the protocol. But the skill is a reference
   document — Claude must actively execute each phase, not just read it.

3. **No enforcement mechanism**: The XML plan specifies `<report-output>`
   but nothing checks if the report file is actually created. The monitoring
   protocol has no automated checkpoints.

4. **Compounding value not internalized**: Each pass generates unique
   observations (timing, failure modes, resource usage) that should
   feed back into tests and documentation. Without the report, these
   observations evaporate with the conversation context.

## What Was Lost

From job 71 (SUCCEEDED, 30m27s) and job 69 (FAILED_SETUP), we observed:
- Docker pull time from GAR to GCP L4 spot: ~14 min setup total
- DVC pull from GCS: worked (dvc.yaml/dvc.lock in image fix validated)
- MLflow Cloud Run connectivity: working (health check passed)
- SWAG post-training: ~25 min on top of ~5 min training (6x overhead!)
- total_mem → total_memory: runtime bug discovered and fixed

NONE of these observations were written to the report file. They exist
only in conversation context, which will be lost.

## Prevention

1. **Write the report FIRST, poll SECOND**: Before the first `sky jobs queue`,
   create the report file with headers and the watchlist. Update it after
   each poll. The report is the primary artifact, not the terminal output.

2. **The XML plan is a checklist, not a reference**: Every `<phase>` and
   `<watchlist>` item must be explicitly checked off. Don't trust memory.

3. **Compound engineering principle**: Every run must leave the codebase
   measurably better than before. "Job succeeded" is not sufficient —
   "Job succeeded AND we wrote 3 new tests from cloud observations AND
   updated timing estimates AND filed 2 issues" is the bar.

## See Also

- `run-debug-factorial-experiment-5th-pass.xml` — the bypassed plan
- Previous reports: `report.md`, `report-2nd-pass-local.md`, `report-3rd-pass-local.md`, `report-4th-pass-failure.md`
- https://github.com/EveryInc/compound-engineering-plugin — compounding principle
