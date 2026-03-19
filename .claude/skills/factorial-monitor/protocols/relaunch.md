# Protocol: Relaunch

## Prerequisites

- All fixes committed and verified (Rule F3)
- Docker image rebuilt and pushed
- Manifest updated with fix information

## Selective Re-Launch (Rule F4)

Re-launch ONLY the failed conditions. Never re-launch the entire grid.

1. **Extract failed conditions from manifest:**
   Filter jobs where `status` is FAILED/FAILED_SETUP and `relaunch_batch < 2`.

2. **Generate re-launch commands:**
   For each failed condition, construct the `sky jobs launch` command with
   the same parameters as the original launch.

3. **Update manifest:**
   - Increment `relaunch_batch` for each re-launched condition
   - Reset `status` to `STARTING`
   - Record new `job_id` from SkyPilot

4. **Return to MONITOR phase.**

## Cycle Budget (Rule F4)

| Cycle | What Happens |
|-------|-------------|
| 0 (original) | Full factorial launch |
| 1 (first retry) | Re-launch only failed conditions from cycle 0 |
| 2 (second retry) | Re-launch only failed conditions from cycle 1 |
| 3+ | **HARD STOP** — escalate to user |

After cycle 2, if failures persist:
- Do NOT attempt cycle 3 without explicit user authorization
- Present the full diagnostic report (see protocols/report.md)
- Recommend specific next steps:
  - "Root cause X persists despite fix — likely a deeper issue in Y"
  - "Estimated cost for cycle 3: $Z — authorize?"

## Cost Tracking

Update manifest with cumulative cost after each cycle:
```
Cycle 0: 24 jobs × 0.7 hr × $1.30/hr = $21.84
Cycle 1:  6 jobs × 0.5 hr × $1.30/hr =  $3.90
Cycle 2:  2 jobs × 0.5 hr × $1.30/hr =  $1.30
Total: $27.04
```
