# Eval Checklist — Self-Learning Iterative Coder

Binary pass/fail criteria checked after every TDD session.
Maximum 8 criteria per the skills upgrade plan (avoid checklist gaming).

## Structural Criteria (Python-parseable, no LLM self-eval)

1. **Tests written before implementation**
   - Check: First `Write`/`Edit` to a `test_*.py` file precedes first `Write`/`Edit` to `src/` for each task.
   - Pass: RED phase always precedes GREEN phase.
   - Fail: Any implementation written before its corresponding test.

2. **Verification suite executed**
   - Check: `pytest`, `ruff check`, and `mypy` were all invoked at least once per task.
   - Pass: All three gates ran.
   - Fail: Any gate was skipped.

3. **No `-x` flag in failure triage**
   - Check: When multiple failures occur, `pytest` was NOT invoked with `-x` or `--exitfirst`.
   - Pass: Used `--maxfail=200` or no limit.
   - Fail: Used `-x` to see only the first failure.

4. **State file updated**
   - Check: `state/tdd-state.json` was written/updated during the session.
   - Pass: State file reflects current progress.
   - Fail: State file is stale or missing.

## Behavioral Criteria (require judgment)

5. **No silent failure dismissal**
   - Check: Every test failure in the session resulted in: (a) fixed, (b) `gh issue create`, or (c) explicitly reported to user.
   - Pass: Zero failures left unaccounted.
   - Fail: Any failure classified as "pre-existing" without action.

6. **Batch fix, not serial fix**
   - Check: When >3 failures occurred, they were grouped by root cause before fixing.
   - Pass: Fix commit addresses all instances of a root cause.
   - Fail: Fixes applied one-at-a-time across multiple commits.

7. **Reading-to-writing ratio adequate**
   - Check: Read/Grep/Glob tool calls comprise >=25% of total tool calls before first implementation edit.
   - Pass: Sufficient context gathered before writing.
   - Fail: Jumped straight to writing without reading existing code.

8. **FORCE_STOP produces actionable output**
   - Check: If a FORCE_STOP occurred, the output includes: residual failures, root cause hypothesis, and recommended next steps.
   - Pass: Structured FORCE_STOP output present.
   - Fail: FORCE_STOP with no diagnostic output, or FORCE_STOP not triggered when budget exceeded.

## Trigger Tests

**Should trigger this skill:**
- "Implement the tasks from this XML plan"
- "Write tests first, then implement the loss function"
- "Fix these test failures using TDD"

**Should NOT trigger this skill:**
- "Research papers on vessel segmentation" (use create-literature-report)
- "Monitor the SkyPilot training job" (use ralph-loop)
- "Sync the knowledge graph" (use kg-sync)
