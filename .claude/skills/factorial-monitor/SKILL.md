---
name: factorial-monitor
version: 1.0.0
description: >
  Multi-job factorial experiment monitoring, aggregated diagnosis, and selective
  re-launch for SkyPilot jobs. Use when running factorial experiments (model x loss
  x calibration x fold) via SkyPilot and you need to track all jobs, aggregate
  failures by root cause, and re-launch only failed conditions.
  Do NOT use for: single-job monitoring (use ralph-loop), local test failures
  (use self-learning-iterative-coder), or sequential plan execution (use overnight-runner).
last_updated: 2026-03-19
activation: manual
invocation: /factorial-monitor
metadata:
  category: operations
  tags: [skypilot, monitoring, factorial, cloud, batch, gcp, experiment]
  relations:
    compose_with:
      - ralph-loop
      - self-learning-iterative-coder
      - issue-creator
    depend_on:
      - ralph-loop
    similar_to: []
    belong_to:
      - overnight-runner
---

# Factorial Monitor

> Multi-job factorial experiment monitoring with aggregated diagnosis, batch error
> resolution, and selective re-launch. The outer loop that orchestrates ralph-loop
> (per-job diagnosis) and self-learning-iterative-coder (batch code fixes).

## Non-Negotiable Rules

Five rules prevent ALL known anti-patterns. Read [instructions/rules.md](instructions/rules.md)
for full details with rationale. Summary:

| Rule | Name | Prevents |
|------|------|----------|
| F1 | WAIT-FOR-TERMINAL | Panic fixing, premature whac-a-mole |
| F2 | AGGREGATE-BEFORE-FIX | Silent dismissal, serial fixing |
| F3 | REBUILD-BEFORE-RELAUNCH | Docker image staleness |
| F4 | MAX-TWO-CYCLES | Infinite fix-relaunch loops, cost overrun |
| F5 | FACTORIAL-MANIFEST | Partial factorial amnesia |

**Kill-switch exception** (Rule F1): If 3+ jobs fail with IDENTICAL error within 5 min
AND remaining running jobs haven't passed the failure point → cancel same-config jobs,
begin batch diagnosis. Different-config jobs continue.

## Workflow (6 Phases)

```
Phase 1: LAUNCH       → protocols/launch.md
Phase 2: MONITOR      → protocols/monitor.md       (polling loop, READ-ONLY)
Phase 3: DIAGNOSE     → protocols/diagnose.md       (batch aggregation)
Phase 4: FIX          → protocols/fix.md            (reviewer-backed batch fix)
Phase 5: RELAUNCH     → protocols/relaunch.md       (selective, max 2 cycles)
Phase 6: REPORT       → protocols/report.md         (final summary + issues)
```

### Phase 1: LAUNCH
- Execute `run_factorial.sh <config.yaml>` or confirm already launched
- Create `factorial_manifest.json` mapping job_id → condition
- Verify SkyPilot YAML uses `image_id: docker:...` (Rule #17 — no bare VM)
- Record experiment_id, factors, levels, expected job count
- See: [protocols/launch.md](protocols/launch.md)

### Phase 2: MONITOR (polling loop)
- Poll `sky jobs queue` every 60s
- Print live status table: `| condition | job_id | status | duration |`
- For each newly-terminal failure: call `ralph_monitor.analyze_logs()`
- **READ-ONLY**: no code changes, no SSH, no `sky exec` while jobs run
- Continue until ALL jobs reach terminal state (Rule F1)
- See: [protocols/monitor.md](protocols/monitor.md)

### Phase 3: DIAGNOSE (all jobs terminal)
- Group failures by root cause category using ralph_monitor categories
- Present ONE aggregated report (Rule F2)
- Format: `{root_cause → [job_ids], fix_strategy, affected_files, confidence}`
- If 0 failures → skip to REPORT
- See: [protocols/diagnose.md](protocols/diagnose.md)

### Phase 4: FIX (with reviewer agents)
- For EACH root cause: plan batch fix strategy with reviewer agents
- If code fix needed → compose_with: self-learning-iterative-coder
- If config fix → edit YAML/env directly
- Execute Rule F3: `make test-staging` → Docker rebuild → push → verify digest
- Commit all fixes in ONE batch commit
- See: [protocols/fix.md](protocols/fix.md)

### Phase 5: RELAUNCH (max 2 cycles)
- Generate filtered re-launch command (ONLY failed conditions)
- Execute and return to MONITOR phase
- Update manifest with `relaunch_batch` number
- Rule F4: hard stop after 2 cycles → escalate to user
- See: [protocols/relaunch.md](protocols/relaunch.md)

### Phase 6: REPORT
- Summary: X succeeded, Y failed (root causes: A, B, C)
- Cost: total $ across all cycles
- If unrecoverable failures → compose_with: issue-creator
- Save to `outputs/factorial_run_<experiment_id>.jsonl`
- See: [protocols/report.md](protocols/report.md)

## Integration Points

| Skill | How It Integrates |
|-------|-------------------|
| `ralph-loop` | Per-job diagnosis via `analyze_logs()` — reuses failure pattern library |
| `self-learning-iterative-coder` | TDD loop when fixes require code changes |
| `issue-creator` | Unrecoverable failures after 2 cycles become GitHub issues |
| `overnight-runner` | Factorial runs are a type of batch execution |

## Quality Evaluation

See [eval/checklist.md](eval/checklist.md) for 5 binary pass/fail criteria.

## Manifest Schema

See [templates/factorial-manifest.json](templates/factorial-manifest.json) for the
experiment state tracking schema.
