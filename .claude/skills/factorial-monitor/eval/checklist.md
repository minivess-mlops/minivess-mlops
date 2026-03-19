# Eval Checklist — Factorial Monitor

Binary pass/fail criteria. Maximum 5 per autoresearch guidelines.

## Structural Criteria

1. **All-jobs-tracked**
   - Every condition in the factorial grid has a corresponding job_id in the manifest.
   - Pass: Manifest job count equals expected condition count from config YAML.
   - Fail: Any condition missing or untracked.

2. **Selective-relaunch**
   - Re-launch command launches ONLY failed conditions, not the entire grid.
   - Pass: Re-launched job count < total job count.
   - Fail: Full grid re-launched (wasting cloud spend on already-succeeded jobs).

3. **Cost-logged**
   - Total estimated cost recorded in manifest and final report.
   - Pass: Cost field present and non-zero for completed jobs.
   - Fail: Cost missing or zero.

## Behavioral Criteria

4. **Failures-aggregated-not-serial**
   - When multiple jobs failed, ONE aggregated diagnosis grouped by root cause was presented.
   - Pass: Single batch report with categories + counts before any fix attempt.
   - Fail: Separate per-job reports, or fixing started before all failures collected.

5. **No-silent-dismiss**
   - Every failed job resulted in: (a) fixed + relaunched, (b) GitHub issue created, or (c) explicitly reported to user.
   - Pass: Zero failures left unaccounted in the manifest.
   - Fail: Any failure classified as "pre-existing" or "transient" without evidence.

## Trigger Tests

**Should trigger:**
- "Monitor the factorial debug run"
- "Track all the SkyPilot jobs from the experiment"
- "Launch and monitor the factorial experiment"

**Should NOT trigger:**
- "Monitor this single SkyPilot job" (use ralph-loop)
- "Fix these test failures" (use self-learning-iterative-coder)
- "Create a GitHub issue" (use issue-creator)
