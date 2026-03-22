# Metalearning: Silent Skip Report — Still Not Reporting Skip Details

**Date:** 2026-03-22
**Session:** 4th pass cold-start continuation
**Severity:** P1 — violates CLAUDE.md Rule 28 (Zero Silent Skips)
**Pattern:** 4th occurrence of this failure (2026-03-04, 2026-03-10, 2026-03-21, 2026-03-22)

## What Happened

Reported "5751 passed, 0 failed, 2 skipped" without investigating or reporting
what the 2 skips were. When the user called this out, investigation revealed 8
skips in the full suite:

| # | Test | Reason | Status |
|---|------|--------|--------|
| 1-6 | test_skypilot_mlflow.py | Cloud credentials not set | Acceptable (auto-skip) |
| 7 | test_mambavesselnet_construction.py | mamba-ssm IS installed | Acceptable (reverse skip) |
| 8 | test_compose_hardening.py:210 | Port bindings without explicit interface | Needs investigation |

## Root Cause

Same as every previous occurrence: the agent reads the test summary line,
sees 0 failures, and moves on without questioning skips. Rule 28 says
"ALWAYS report the skip count AND the skip reasons." The rule exists in
CLAUDE.md. The metalearning docs exist. The behavior persists.

## Why This Keeps Happening

1. Test output parsing focuses on PASSED/FAILED, not SKIPPED
2. "2 skipped" feels harmless when 5751 passed
3. Reporting skips requires an extra investigation step (re-running with verbose)
4. The agent optimizes for speed ("all green, let's commit") over thoroughness

## Prevention (Must Be Enforced By Hook)

The documentation-based approach has failed 4 times. The ONLY reliable fix is:
- A PostToolUse hook that blocks `git commit` when test output contains "skipped"
  without a corresponding skip analysis in the commit message
- OR: modify `make test-staging` to print skip reasons by default (add `-rs` flag)

## Connection to Rule 28

> "Every SKIPPED test is a bug hiding as a skip. When reporting test results,
> ALWAYS report the skip count AND the skip reasons."

4th violation in 18 days. Documentation alone does not work.
