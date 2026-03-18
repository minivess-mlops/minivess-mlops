# Protocol: Failure Triage (GATHER → CATEGORIZE → PLAN → FIX → VERIFY)

## Fixes TWO systemic bugs:

- **BUG-1**: Silent dismissal of "pre-existing" failures (CLAUDE.md Rule #20)
- **BUG-2**: Whac-a-mole serial failure fixing (CLAUDE.md Rule #23)

## TRIGGER

After ANY test suite run that shows failures — whether in the VERIFY phase,
ACTIVATION-CHECKLIST baseline check, or ad-hoc `make test-staging`.

## ZERO TOLERANCE RULE (BUG-1 prevention)

"Pre-existing" is NOT a valid classification.
"Not related to current changes" is BANNED.
"Separate issue" without creating the issue in the SAME response is a lie.

EVERY failure MUST result in one of:
1. **Fixed immediately** (if root cause is clear and fix is < 5 minutes)
2. **GitHub issue created** with root cause, affected files, and priority label
3. **Explicitly reported to user** with recommendation

There are no orphan failures. Period.

## MANDATORY STEPS (BUG-2 prevention)

### Step 1: GATHER (one pass, no stopping)

Run the full suite WITHOUT `-x`:

```bash
uv run pytest tests/ \
  -m "not model_loading and not slow and not integration and not gpu" \
  --ignore=tests/gpu_instance \
  --maxfail=200 -q --tb=line 2>&1 | grep "^FAILED"
```

**NEVER** use `-x` for failure investigation. Capture ALL failures in one run.

### Step 2: CATEGORIZE (group by root cause, not by test)

Count failures per file:
```bash
... | awk -F'::' '{print $1}' | sort | uniq -c | sort -rn
```

Then read failure messages. One root cause often explains 50+ failures.

Output a table:

| Root Cause | Files Affected | Failure Count | Fix Strategy |
|-----------|---------------|---------------|--------------|
| RC1: ... | file1, file2 | 15 | Batch replace X→Y |
| RC2: ... | file3 | 6 | Read source, update tests |

### Step 3: PLAN (one fix per root cause, not per test)

For each root cause:
1. Identify the **SOURCE OF TRUTH** (e.g., `MetricKeys` class for key names,
   `Trainer.fit()` for checkpoint naming, actual YAML for config values)
2. Read the source of truth FIRST
3. Plan a batch replacement/fix that handles ALL instances at once
4. Estimate how many tests will turn green

### Step 4: FIX (batch, one commit per root cause)

For each root cause:
1. Fix ALL instances in ONE pass (not one-by-one)
2. Run `ast.parse()` on every modified Python file to verify no syntax corruption
3. Run ONLY the affected test files to confirm GREEN
4. ONE commit per root cause with descriptive message

### Step 5: VERIFY (full suite again, WITHOUT -x)

Run the full staging suite again. If new failures appear → go back to Step 1.
Do NOT start fixing new failures immediately — triage them first.

**Exit criteria**: 0 failures in full suite.

## BANNED ANTI-PATTERNS

| Anti-Pattern | Why Dangerous | Detection |
|--------------|--------------|-----------|
| Running with `-x` and fixing one test at a time | 25+ min for 10-min work | If you're about to run pytest -x to investigate → STOP |
| Committing after each individual test fix | Noisy git history | If you have <5 fixes → they belong in one commit |
| Saying "pre-existing" or "not related to current changes" | Failures persist forever | These phrases are BANNED in the codebase |
| Using `replace_all` on strings without checking Python identifiers | Corrupts variable names | Only replace string literals used as dict keys |
| Proceeding to next plan task while failures exist | Compounding debt | VERIFY phase blocks progress on failures |
| Creating a "separate issue" without actually creating it | Promise without delivery | Issue must be created in the SAME response |

## ESCALATION

| Condition | Action |
|-----------|--------|
| 10+ files with same root cause | Launch an Agent for batch replacement |
| 3+ root causes | Write a mini XML plan before fixing |
| Same failure seen in 2+ sessions | Create a metalearning doc |
| >20% hallucination rate (citation verification) | Re-run Phase 1 with stricter prompts |

## INTEGRATION WITH TDD LOOP

This protocol is invoked automatically by the VERIFY phase when failures are
detected. The flow is:

```
VERIFY phase detects failures
    ↓
FAILURE GATE: "Are there multiple failures?"
    ├── YES → Invoke this protocol (GATHER→CATEGORIZE→PLAN→FIX→VERIFY)
    └── NO (single failure) → Standard fix-phase.md applies
```

After triage completes with 0 failures, control returns to the TDD inner loop.
