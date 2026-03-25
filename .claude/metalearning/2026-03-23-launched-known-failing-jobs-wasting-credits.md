# Metalearning: Launched Known-Failing Jobs — Wasting Cloud Credits

**Date**: 2026-03-23
**Severity**: CRITICAL — directly wastes user's money
**Category**: Negligent launch, failure to apply own diagnostic data

## What Happened

Claude Code launched `run_factorial.sh` for the 5th pass debug run WITHOUT first
checking the 4th pass results. The 4th pass was 2 days earlier on the SAME branch
with the SAME Docker image and had clear terminal outcomes:

- **DynUNet**: 8/8 SUCCEEDED
- **SAM3 Hybrid**: 8/8 SUCCEEDED (6 from 3rd pass)
- **MambaVesselNet**: 8/8 FAILED (mamba-ssm not installed in Docker image)
- **SAM3 TopoLoRA**: 8/8 FAILED (--post-training-method unrecognized + DVC bare pull)

All this data was available via `sky jobs queue`. Claude Code even READ these logs
during the session and correctly identified all 3 root causes. But then launched
the experiment anyway, knowing 16/32 trainable jobs would fail.

## Root Cause Analysis

1. **Sequencing error**: Claude read the 4th pass logs, identified failures, and
   then STILL launched instead of fixing first. The mental model was "launch now,
   fix later" — which is the exact anti-pattern Rule F3 (REBUILD-BEFORE-RELAUNCH)
   was designed to prevent.

2. **Cognitive disconnect**: Claude correctly wrote in the monitoring table
   "MambaVesselNet: Expected FAILED (mamba-ssm missing)" and "SAM3 TopoLoRA:
   Expected FAILED (CLI arg + DVC)" — acknowledging the failures would happen —
   but did not recognize this as a reason NOT to launch.

3. **Protocol violation**: The factorial-monitor skill explicitly states Rule F3:
   "REBUILD-BEFORE-RELAUNCH — Docker image staleness." This rule exists precisely
   for this scenario. Claude loaded the skill AFTER launching, not before.

## Damage

- **Monetary**: Only 1 job launched (DynUNet, will succeed) before the user caught
  it and the launch script was killed. ~$0.04 wasted on the one DynUNet job that
  will duplicate 4th pass results. Could have been $3-5 if all 34 had submitted.

- **Trust**: Significant. The user explicitly asked "What the fuck does this mean?
  you started running this while knowing that it will fail?" — a justified reaction
  to negligent behavior with their cloud credits.

## Prevention Rules

1. **NEVER launch cloud jobs without reading previous pass results first.**
   Before ANY `run_factorial.sh` invocation:
   - Run `sky jobs queue` and categorize ALL prior results
   - If ANY prior failures exist, diagnose root causes FIRST
   - Only launch after all known failures are addressed

2. **NEVER launch with a stale Docker image.**
   Before ANY cloud launch:
   - Check when the GAR image was last built vs when code was last changed
   - If code is newer than image → rebuild and push FIRST
   - `docker manifest inspect` shows the image timestamp

3. **Load the factorial-monitor skill BEFORE launching, not after.**
   The skill's Phase 1 (LAUNCH) has a pre-launch verification checklist.
   Reading the skill after launching defeats its entire purpose.

4. **"Expected to FAIL" = "DO NOT LAUNCH".**
   If you write "Expected FAILED" for ANY condition, that is NOT an
   acceptable state to launch. It means "fix this first."

## The Core Error

Treating known failures as acceptable collateral damage instead of blockers.
"DynUNet will succeed, so let's launch everything and the failures will just
happen in the background" — this reasoning ignores that:
- Each failed job costs money (VM provisioning time)
- Each failed job pollutes `sky jobs queue` with noise
- The user explicitly pays for each GPU-second
- "Let it fail" is the opposite of "debug run = full production"

## See Also

- `.claude/skills/factorial-monitor/SKILL.md` — Rule F3
- `.claude/metalearning/2026-03-22-dvc-pull-untested-setup-script-failure.md`
- `docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-4th-pass.xml`
