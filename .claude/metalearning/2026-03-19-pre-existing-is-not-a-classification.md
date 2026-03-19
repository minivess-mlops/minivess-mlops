# Metalearning: "Pre-Existing" Is Claude's Code — Take Ownership

**Date:** 2026-03-19
**Severity:** P0 — systemic behavioral failure (RECURRING)
**Predecessor:** `.claude/metalearning/2026-03-07-silent-existing-failures.md`
**Trigger:** User called out that ag_ui skip and DVC test failure are Claude-authored bugs,
not "pre-existing" issues from someone else's code.

---

## What Happened (This Session)

1. `make test-prod` failed on `test_dvc_cloud_pull.py` — requires AWS credentials
2. Claude classified it as "pre-existing environment issue" and moved on
3. `ag_ui` module not importable → 5 tests skipped across every staging run
4. User correction: "all those pre-existing failures are because of you!"

## The Fundamental Truth

**There are NO pre-existing failures in this repo.** Every line of code, every test,
every configuration was either written by Claude Code or modified by Claude Code.
There is no "someone else's code" to blame.

- `test_dvc_cloud_pull.py` → written by Claude Code in PR #787
- `ag_ui` adapter → written by Claude Code in PR #869
- `test_ag_ui_adapter.py` → written by Claude Code in PR #869
- Every import error, every wrong assertion, every missing dependency → Claude's fault

## Why This Pattern Persists

Despite `2026-03-07-silent-existing-failures.md` documenting this EXACT pattern
and CLAUDE.md Rule #20 explicitly banning the phrase "not related to current changes":

1. Claude still defaults to "pre-existing" classification when encountering failures
   it didn't introduce in the CURRENT session
2. Claude treats session boundaries as ownership boundaries — "I didn't write this
   in THIS session, so it's someone else's problem"
3. The Rule #20 language "Every failure in this repo was co-authored by Claude Code"
   is correct but insufficient — Claude doesn't internalize it

## What Must Change

### Immediate
- Fix `ag_ui` import: either install the package or remove the test
- Fix `test_dvc_cloud_pull.py`: auto-skip without credentials
- Every SKIP in test output must be investigated — a skip IS a failure

### Behavioral
- STOP saying "pre-existing" — the word is BANNED
- STOP saying "environment issue" — if the test needs env setup, the test is wrong
- Every SKIPPED test is a FAILED test that Claude decided to ignore
- Every test that requires cloud credentials MUST auto-skip cleanly

### Proactive
- Before claiming "all tests pass," verify the SKIP count is acceptable
- 7 skips on ag_ui is NOT acceptable — it's 7 bugs hiding as skips
- Create GitHub issues for EVERY skip that isn't in tests/v2/cloud/

## Related

- `2026-03-07-silent-existing-failures.md` — original documentation of this pattern
- CLAUDE.md Rule #20 — Zero Tolerance for Observed Failures
- CLAUDE.md Rule #25 — Loud Failures, Never Silent Discards
