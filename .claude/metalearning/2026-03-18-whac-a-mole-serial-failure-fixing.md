# Metalearning: Whac-a-Mole Serial Failure Fixing — The Worst Time Sink

**Date:** 2026-03-18
**Severity:** SYSTEMIC — Recurrence of the "silent existing failures" pattern (#2026-03-07)
  combined with a NEW anti-pattern: reactive one-by-one fixing without upfront diagnosis.
**Category:** AI behavioral pattern / process failure

## The Anti-Pattern

Claude Code encounters test failures, then:

1. Runs `make test-staging` (which has `-x` = stop after first failure)
2. Sees 1 failure
3. Fixes that 1 failure
4. Re-runs `make test-staging` (~2.5 min)
5. Sees a DIFFERENT failure (because the first one was masking it)
6. Fixes that 1 failure
7. Repeat 10+ times

**Total time wasted**: 10 cycles × 2.5 min = 25+ minutes of test runs alone,
plus investigation time. And with GitHub Actions CI, each cycle can take 15+ minutes
of wait time — a 10-cycle fix could take 4+ hours.

**The correct approach**: Run ALL tests WITHOUT `-x` FIRST, see ALL 69 failures at
once, categorize by root cause, then fix ALL instances of each root cause in one pass.

## This Session's Timeline

### What happened (2026-03-18)

1. Completed PR-D + PR-E (good work, 58 tests, 9 tasks)
2. Encountered `test_factorial_figures` failure in `make test-staging`
3. Dismissed it as "pre-existing and flaky" (BANNED per Rule #20)
4. User called it out — correctly
5. Started fixing one test at a time with `make test-staging -x`:
   - Fix matplotlib warning → re-run → find calibration test failure
   - Fix calibration (wrong: stale worktree) → re-run → find KG level values
   - Fix KG levels → re-run → find KG resolved_option
   - Fix KG options → re-run → find KG tripod_compliance missing
   - Fix tripod file → re-run → find observability old keys
   - Fix one observability assertion → re-run → find another
   - Fix another → re-run → another
6. User told me to stop and plan properly
7. Finally ran without `-x` → discovered **69 failures across 18 files**
8. Root cause: **ONE migration (#790)** broke ~50 tests, plus KG schema + matplotlib

### Time wasted on serial fixing: ~45 minutes
### Time it would have taken with proper diagnosis: ~10 minutes

## Root Cause Analysis

### 1. The `-x` flag hides the scope of the problem
`make test-staging` uses `addopts = ["-x"]` in `pyproject.toml`. This is correct for
development (fail fast), but WRONG for debugging multiple failures. Claude Code should
ALWAYS run without `-x` when investigating failures.

### 2. No "gather all failures first" protocol
There's no step in the workflow that says "before fixing anything, run the full suite
to see ALL failures." The instinct is to fix the first thing you see.

### 3. Missing root-cause categorization step
69 failures sounds terrifying. But they're actually just 3-4 root causes:
- Slash-prefix migration (#790): ~50 tests
- KG YAML schema: ~5 tests
- matplotlib warning: 1 test
- Stale worktree: 1 test

Categorizing FIRST would have revealed that most failures are mechanical find-replace.

### 4. Edits not persisting across branch merges
Multiple fixes were lost because squash merges from PR branches overwrote files.
The fixes were applied on `main` but `main` was then fast-forwarded past them.
Should have been on a dedicated fix branch from the start.

## The Protocol (Non-Negotiable)

### When encountering ANY test failure during `make test-staging`:

**STEP 1: GATHER** — Run without `-x` to see ALL failures:
```bash
uv run pytest tests/ -m "not model_loading and not slow and not integration" \
  --ignore=tests/gpu_instance --maxfail=200 -q --tb=line 2>&1 | grep "^FAILED"
```

**STEP 2: CATEGORIZE** — Group failures by file, then by root cause:
```bash
... | awk -F'::' '{print $1}' | sort | uniq -c | sort -rn
```

**STEP 3: PLAN** — For each root cause, determine the fix strategy:
- Mechanical migration → agent-driven batch find-replace
- Schema mismatch → single source file fix
- Flaky test → root cause the flakiness

**STEP 4: FIX** — Fix ALL instances of each root cause in one commit per cause.

**STEP 5: VERIFY** — Run full suite again without `-x` to confirm zero failures.

### NEVER:
- Fix one failure, re-run with `-x`, fix next failure (the whac-a-mole)
- Dismiss ANY failure as "pre-existing" without creating an issue
- Run `make test-staging` with `-x` when debugging (use `--maxfail=200` instead)

## Impact on Other Systems

### Self-Learning TDD Skill
The VERIFY phase should include a "gather all failures" step, not just "run -x".
Update: `.claude/skills/self-learning-iterative-coder/SKILL.md`

### Ralph Loop Skill
Infrastructure monitoring should periodically run the full test suite without `-x`
and report ALL failures, not just the first one.

### CLAUDE.md
Add explicit protocol: "When `make test-staging` fails, NEVER fix serially.
Run `--maxfail=200` first to see the full scope."

## References

- `.claude/metalearning/2026-03-07-silent-existing-failures.md` — the original sin
- `src/minivess/observability/metric_keys.py` — the MIGRATION_MAP that should have
  been applied to tests when it was applied to implementation
- Issue #790 — slash-prefix migration (implementation done, tests NOT migrated)
