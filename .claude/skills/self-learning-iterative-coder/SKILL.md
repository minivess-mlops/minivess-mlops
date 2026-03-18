---
name: self-learning-iterative-coder
version: 3.0.0
description: >
  Production-grade TDD loop for plan-driven code implementation. Self-correcting
  RED→GREEN→VERIFY→FIX cycle with zero-tolerance failure triage, batch error
  resolution, and "tokens upfront" quality philosophy. Use when implementing from
  executable plans (XML/YAML/JSON), writing tests first, fixing code iteratively,
  or when ANY implementation task benefits from structured test-driven development.
last_updated: 2026-03-18
activation: manual
invocation: /tdd-iterate
revision_notes: >
  v2.1.0: Ralph Wiggum best-practices audit. Added accumulated learnings
  file (LEARNINGS.md), placeholder prevention rule, and FORCE_STOP
  escape hatch with structured next-steps output.
  v2.0.0: Post-execution retrospective after 27 tasks (163 tests).
  Replaced task-count budget with inner-iteration budget.
  Added spec-adaptation protocol. Made RED commit optional.
  Expanded dependency management. Scoped typecheck per-task.
  Simplified state updates and convergence reports.
  Added empirical data to self-correction principles.
---

# Self-Learning Iterative Coder

A self-correcting TDD loop that implements code from executable plans, acknowledging that LLMs are stochastic and cannot write production-grade code in a single pass.

> "Ralph is monolithic. Ralph works autonomously in a single repository as a single process that performs one task per loop." — Geoffrey Huntley

## When to Use

- You have an executable plan (XML, YAML, or JSON) with task definitions
- Each task has a TDD spec (tests to write first, then implementation)
- You want autonomous, self-correcting code implementation
- You want to minimize developer intervention between iterations

## Two-Loop Architecture

```
OUTER LOOP (Plan Execution):
  while plan has tasks with status != DONE:
    task = select_next_task(plan)         # protocols/task-selection.md

    INNER LOOP (TDD Red-Green-Refactor):
      Iteration N:
        1. RED:        Write failing tests   -> protocols/red-phase.md
        2. GREEN:      Implement code        -> protocols/green-phase.md
        3. VERIFY:     Run tests+lint+types  -> protocols/verify-phase.md
        4. FIX:        If failing, fix code  -> protocols/fix-phase.md
        5. CHECKPOINT: Git commit + state    -> protocols/checkpoint.md
        6. CONVERGE?   All green?            -> protocols/convergence.md
           If yes: Mark task DONE, continue outer loop
           If no:  Iteration N+1

    If FORCE_STOP: Log residual failures, move to next task
```

## Progress Visibility (Optional)

In **interactive mode**, print a progress banner at the start of each inner iteration for developer awareness. In **autonomous mode**, skip the banner — the state file and `git log --oneline` are the progress signals.

```
ITERATIVE TDD CODER - Task {task_id} - Inner Iteration {N}
----------------------------------------------------------
Step 1/6: RED        - Write failing tests    [{status}]
Step 2/6: GREEN      - Implement code         [{status}]
Step 3/6: VERIFY     - Run tests+lint+types   [{status}]
Step 4/6: FIX        - Analyze & fix failures [{status}]
Step 5/6: CHECKPOINT - Git commit + state     [{status}]
Step 6/6: CONVERGE   - Quality gate check     [{status}]
----------------------------------------------------------
Plan progress: {done}/{total} tasks DONE | {in_progress} IN_PROGRESS | {remaining} remaining
```

Status values: `[CURRENT]`, `[DONE]`, `[PENDING]`, `[SKIPPED]`

## Critical Rules

### 1. TESTS FIRST, ALWAYS
Never write implementation before tests. The RED phase is mandatory. Tests are the specification — they define what correct behavior looks like before any code exists.

### 2. VERIFY, DON'T ASSUME
Run the project's test, lint, and typecheck commands (e.g., `make test`, `make lint`, `make typecheck`) after every change. Never claim something works without running the verification suite. "Ghost completions" (claiming done without running tests) are the #1 failure mode of agentic coding.

### 3. STATE IS TRUTH
Load `state/tdd-state.json` at the start of every session. The state file is the crash-recovery mechanism. If context is lost, the state file tells you exactly where to resume. Update it after every checkpoint.

### 4. ONE TASK PER INNER LOOP
Don't try to implement multiple plan tasks simultaneously. Each task gets its own RED-GREEN-VERIFY-FIX-CHECKPOINT-CONVERGE cycle. Monolithic single-process per Huntley's Ralph philosophy.

### 5. SELF-CORRECT, DON'T SKIP
When tests fail, analyze the failure and fix it. Don't move on with broken tests. Iterations exist because LLMs process stochastically — self-correction is the norm, not a failure.

### 6. RESPECT PROJECT CONVENTIONS
Read the project's `CLAUDE.md` before starting. Follow all project-specific rules: package manager, file operations, code analysis approach, encoding, paths, timezone handling. The skill adapts to the project, not the other way around.

### 7. NO PLACEHOLDERS, NO STUBS
Never write `pass`, `TODO`, `NotImplementedError`, or stub implementations in the GREEN phase. Every function must have a real implementation that attempts to pass the tests. If you find yourself writing a placeholder, STOP — you are skipping the hard work. Escalating prompt: "DO NOT IMPLEMENT PLACEHOLDER CODE. FULL IMPLEMENTATIONS ONLY."

### 8. ACCUMULATE LEARNINGS
Maintain a `LEARNINGS.md` file in the project root. After each FORCE_STOP, STUCK resolution, or spec adaptation, append a dated entry describing what was discovered. This file persists across sessions and prevents re-discovering the same issues. Format:
```
## YYYY-MM-DD — Task {id}: {brief title}
- **Discovery**: What was learned
- **Resolution**: How it was resolved (or why it was deferred)
```

### 9. ZERO TOLERANCE FOR OBSERVED FAILURES
Every test failure seen during a session MUST result in: fixed, issue created, or reported to user. "Pre-existing" is NOT a classification. "Not related to current changes" is BANNED. "Separate issue" without creating the issue in the SAME response is a lie. Every failure in this repo was co-authored by Claude Code and is therefore Claude Code's responsibility. See: `.claude/metalearning/2026-03-07-silent-existing-failures.md`

### 10. FAILURE TRIAGE BEFORE FIXING
When the VERIFY phase reveals multiple failures, STOP. Do NOT fix one at a time. Invoke the Failure Triage Protocol (`protocols/failure-triage.md`) to GATHER all failures with `--maxfail=200`, CATEGORIZE by root cause, PLAN batch fixes, then FIX. One root cause often explains 50+ failures. Serial fixing wastes 25+ minutes on what should be 10 minutes. See: `.claude/metalearning/2026-03-18-whac-a-mole-serial-failure-fixing.md`

### 11. TOKENS UPFRONT — QUALITY OVER SPEED
Spend more tokens reading and understanding before writing code. The cost of sloppy initial code is always higher than the cost of careful initial code:
- **30% reading / 70% implementing** (not 5% / 95%)
- Read ALL relevant source files before writing tests
- Read ALL existing tests before writing new ones
- Understand the full interface before implementing
- "I'll just try this and see" without reading context is BANNED
- One careful implementation pass is cheaper than three sloppy-then-fix passes

## Activation

Before starting, run through the [ACTIVATION-CHECKLIST.md](ACTIVATION-CHECKLIST.md).

## Protocol Reference

| Protocol | Purpose | File |
|----------|---------|------|
| Task Selection | Parse plan, pick next eligible task | [protocols/task-selection.md](protocols/task-selection.md) |
| Spec Adaptation | Handle plan-vs-reality divergence | [protocols/spec-adaptation.md](protocols/spec-adaptation.md) |
| Red Phase | Write failing tests from TDD spec | [protocols/red-phase.md](protocols/red-phase.md) |
| Green Phase | Implement minimum code to pass tests | [protocols/green-phase.md](protocols/green-phase.md) |
| Verify Phase | Run tests + lint + typecheck | [protocols/verify-phase.md](protocols/verify-phase.md) |
| Fix Phase | Analyze failures, apply targeted fixes | [protocols/fix-phase.md](protocols/fix-phase.md) |
| **Failure Triage** | **Handle multiple failures systematically** | [protocols/failure-triage.md](protocols/failure-triage.md) |
| Checkpoint | Git commit + state update | [protocols/checkpoint.md](protocols/checkpoint.md) |
| Convergence | Per-task and per-plan stop conditions | [protocols/convergence.md](protocols/convergence.md) |

## State Management

| File | Purpose |
|------|---------|
| [state/tdd-state.schema.json](state/tdd-state.schema.json) | JSON Schema for state file validation |
| [state/example-state.json](state/example-state.json) | Reference example of a state file mid-execution |

## Philosophy

See [prompts/self-correction-principles.md](prompts/self-correction-principles.md) for the theoretical foundations: Ralph Wiggum loops, stochastic self-correction, boundary objects, and anti-patterns.

## Anti-Patterns

| Anti-Pattern | Why It Fails | Correct Approach |
|--------------|-------------|------------------|
| Ghost completion | Claiming tests pass without running them | Always run verification suite |
| Shotgun fix | Changing many things at once hoping something works | Analyze failure, make targeted fix |
| Test-after | Writing implementation first, tests second | RED phase is always first |
| Context hoarding | Trying to do too much in one session | Max 20 inner iterations per session, then new session |
| Skip the skip | Ignoring lint/type errors "because tests pass" | All three gates must be green |
| Infinite loop | Retrying the same fix without analyzing why it fails | After 2 identical failures, escalate |
| Convention ignorance | Using pip instead of uv, strings instead of Path | Read CLAUDE.md before starting |
| Placeholder code | Writing `pass`, `TODO`, `NotImplementedError` | Full implementations only |
| Knowledge amnesia | Re-discovering the same issues across sessions | Append to LEARNINGS.md |
| **Silent dismissal** | Classifying failures as "pre-existing" and moving on | Zero tolerance — fix, issue, or report (Rule #9) |
| **Whac-a-mole** | Fixing one test at a time with `-x` in a loop | Failure Triage Protocol — batch fix by root cause (Rule #10) |
| **Skim-and-code** | Writing code without reading existing implementation | Read 30%, implement 70% — tokens upfront (Rule #11) |

## Session Budget

- **Max inner iterations per session**: 20 (context budget — suggest new session after 20 cumulative iterations)
- **Max inner iterations per task**: 5 (FORCE_STOP — likely a design issue)
- **Max fix attempts per failure**: 3 (escalate if same failure persists)

Note: Simple tasks consume 1 iteration; complex tasks consume 3-5. The inner-iteration budget reflects actual context consumption better than a flat task count. In practice (27-task execution), ~70% of tasks completed in 1 iteration.

## Mapping to iterated-llm-council

This skill adapts the `iterated-llm-council` architecture (designed for manuscript refinement) to code implementation:

| iterated-llm-council (manuscripts) | self-learning-iterative-coder (code) |
|-------------------------------------|--------------------------------------|
| L3: Domain expert reviewers | Test suite = the reviewer |
| L2: Synthesize review findings | Aggregate failures (test + lint + types) |
| L1: Verdict (ACCEPT/REJECT) | Quality gate (ALL_GREEN / FAILING) |
| L0: XML action plan | Fix plan (what code to write/fix) |
| Execute: Apply .tex changes | Write/fix Python code |
| Checkpoint: Git commit + state | Git commit + state |
| Converge: Quality threshold met? | Tests pass + lint clean + types check? |
