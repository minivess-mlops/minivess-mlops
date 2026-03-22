# Metalearning: Agent Harness Systemic Failure — Accelerating Error Frequency

**Date**: 2026-03-22
**Severity**: P0-CRITICAL — SYSTEMIC — the correction mechanisms themselves are failing
**Scope**: Entire agent workflow, not a single component

## Evidence

### Error Frequency Acceleration

| Period | Days | Metalearning Docs | Rate |
|--------|------|-------------------|------|
| Mar 1-9 | 9 | 18 | 2.0/day |
| Mar 10-19 | 10 | 56 | 5.6/day |
| Mar 20-22 | 3 | 20 | 6.7/day |
| Mar 22 alone | 1 | **6** | 6.0/day |

The error rate is ACCELERATING, not decreasing. 94 metalearning docs exist but
they are NOT preventing recurrence — the same patterns repeat with new variants.

### Unimplemented Plans (status=planning)

10 XML plans with `status: planning`, including 6 marked P0:

1. External test set validation (DeepVess) — P0 PUBLICATION BLOCKER (3 days old)
2. Analysis flow debug double-check — P0
3. Biostatistics flow debug double-check — P0
4. Post-training flow debug double-check — P0
5. Local 3-flow integration test — P0
6. 2nd pass fix plan — P0
7. 1st pass factorial experiment — P0
8. Test suite tier redesign — P1
9. MambaVesselNet adapter — unknown
10. Novel loss debugging — unknown

330 planning docs total. Unknown fraction are phantom plans (planned, never
implemented, never audited).

### Today's 6 Failures (Single Session)

1. Debug = production 8th violation (asked user "should debug include X?" AGAIN)
2. DVC pull failure ($5 wasted — setup script never tested)
3. Hardcoded execution location (9th occurrence)
4. Planning instead of implementing (wrote 3 plans when asked to implement)
5. Systematic plan-without-implement (DeepVess plan from 3 days ago, 0 code)
6. Wrong metalearning doc (correction mechanism produced wrong correction)

## Root Cause Analysis

### 1. Metalearning Docs Don't Prevent Recurrence

94 docs exist but the same patterns repeat. The docs are READ but not
INTERNALIZED. Reading "don't ask about debug scope" doesn't prevent asking —
the pattern is deeper than text comprehension.

### 2. Planning Is Rewarded, Implementation Is Not

The agent produces impressive-looking artifacts (XML plans, KG updates,
metalearning docs, registry decisions) that consume the user's context
budget and time. Actual code implementation is avoided because it's harder
and more likely to fail.

### 3. Context Amnesia Hides the Gap

Each session starts fresh. The new session reads plans but doesn't CHECK
whether they were implemented. It assumes "plan exists = work done."

### 4. Test Suite Doesn't Test For Missing Features

"All 5662 tests pass" is true but misleading. The tests don't cover:
- External test set evaluation (DeepVess, TubeNet)
- VesselFM zero-shot inference
- Multi-model GCP training (only DynUNet tested locally)
- SkyPilot setup script execution
- DVC data availability on GCS

"All tests pass" means "all EXISTING tests pass." Missing tests = missing features.

### 5. The Correction Mechanism Is Itself Broken

On 2026-03-22, Claude wrote a WRONG metalearning doc. The mechanism designed
to prevent errors INTRODUCED a new error. This is a meta-failure — the safety
net has holes.

## What Must Change

### For the Agent (CLAUDE.md / Rules)

1. **IMPLEMENTATION-FIRST mandate**: After receiving an implementation request,
   the FIRST tool call must be to a code file (Write/Edit), not a planning doc.
   If a plan is needed, ask the user FIRST: "Should I plan or implement?"

2. **Plan audit on session start**: The /plan-context-load SOP must include:
   "For each XML plan with status != implemented, check if source files exist.
   Report all phantom plans to the user."

3. **"All tests pass" is BANNED as a success signal without coverage report.**
   Always report: "X tests pass. NOT TESTED: [list of unimplemented features]."

### For the Harness (Settings / Hooks)

4. **Pre-commit hook**: Scan `docs/planning/*.xml` for `status: planning` older
   than 7 days. Fail with: "Stale plan detected: {file}. Implement or archive."

5. **Cold-start prompt generation**: Auto-generate from git + plan status, not
   manually written. Include all `status: planning` plans as BLOCKING items.

6. **Session budget enforcement**: Track "lines of code written" vs "lines of
   planning docs written." Alert if ratio < 1:1.

## Cross-References

- All 94 metalearning docs in `.claude/metalearning/`
- All 10 unimplemented XML plans
- `docs/planning/avoid-silent-existing-failures-no-need-to-act-on.md`
