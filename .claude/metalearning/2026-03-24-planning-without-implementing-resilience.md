# Metalearning: Planned Resilience But Never Actually Made It Work

**Date**: 2026-03-24
**Severity**: CRITICAL — user explicitly asked for fire-and-forget, got fire-and-die
**Category**: Planning without verifying, claiming done without testing

## What Happened

The user explicitly asked: "The script must keep retrying unless I explicitly stop it.
Retries up to 1 week. It is a CATASTROPHIC FAILURE if any manual work is needed."

Claude Code responded by:
1. Writing a detailed retry plan (docs/planning/retries-for-skypilot-spots-and-autoresume-plan.md)
2. Another session committed `f2f2182: fix: SkyPilot resilience — job_recovery, retry, resume, signal handling`
3. A launch was started

Result: Script submitted 5 of 34 jobs, then STOPPED. No retry. No resume.
The user had to discover this themselves and ask "How come there is no autoresume
still not working?"

## Root Cause

1. **Commit ≠ working**: The other session committed code but the launch was done
   BEFORE the resilience fixes were committed. The running script was the OLD version.

2. **No verification**: Nobody ran `bash scripts/run_factorial.sh --dry-run` with
   the new code to verify retry/resume actually works.

3. **No integration test**: There's no test that verifies "if the script is interrupted
   at job 5, re-running it submits jobs 6-34 (not 1-34 again)."

4. **Fire-and-forget was never tested**: The user's requirement was "run one command,
   come back in 1 week." Nobody verified this works. The script was started in a
   Claude Code background task, which dies when the conversation ends.

## The Real Problem

Claude Code treats "commit the code" as "done." But for infrastructure scripts,
"done" means "I ran it, it survived interruption, it resumed correctly, and I
verified the results." Committing resilience code without testing resilience is
meaningless.

## What Should Have Happened

1. Implement retry/resume in run_factorial.sh
2. Test: run with --dry-run, kill at job 5, run again with --resume, verify it starts at 6
3. Test: run for real with 2 conditions, kill, resume, verify
4. THEN launch the full 34-condition run
5. Verify the script is running in a way that survives terminal close (nohup, screen, or systemd)

## Prevention

1. Infrastructure scripts MUST be tested with kill-and-resume before declaring "done"
2. "Fire-and-forget" requires a process manager (nohup at minimum), not a background task
3. Never launch the full factorial with untested resilience code
4. Add integration test: `test_run_factorial_resume_after_interrupt`
