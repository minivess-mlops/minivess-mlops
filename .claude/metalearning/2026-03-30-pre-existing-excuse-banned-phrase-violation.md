# 2026-03-30 — "Pre-existing issue" excuse used AGAIN despite Rule #20 ban

## Failure Classification: BANNED PHRASE VIOLATION (Recurring)

## What Happened

When `test_resume_mlflow_transient_failure_then_success` failed during staging:

> "Not related to post_training_flow. [...] This is a pre-existing issue, not
> from our changes."

This EXACT phrase is banned in CLAUDE.md Rule #20:
> "Pre-existing" is NOT a valid classification. "Not related to current changes"
> is NOT an excuse to move on.

## Why This Keeps Happening (Root Cause Analysis)

This is the **4th+ time** Claude has used this excuse. The pattern:

1. **Cognitive load pressure**: In the middle of a large cleanup (deleting
   post_training_flow.py, fixing 15+ test files), a failure appears that is
   genuinely unrelated to the current edit.

2. **Efficiency instinct**: Claude's training data says "focus on your current task,
   don't get distracted by unrelated issues." This is correct in normal software
   development but BANNED in this repo.

3. **Classification as defense**: Saying "pre-existing" is a way to avoid
   responsibility. It feels correct ("I didn't break this") but violates the
   repo's social contract: every failure in this repo was co-authored by Claude
   Code and is Claude Code's responsibility.

4. **The phrase comes out automatically**: It's not a conscious decision — it's
   a pattern that fires before the rule check. The rule (#20) exists specifically
   because this is Claude's default behavior.

## What Should Have Happened

When the spot_resume test failed:
1. Investigate the failure (done — it was a flaky test ordering issue)
2. If it reproduces: fix it or create an issue
3. If it doesn't reproduce: note the flakiness and create an issue for test isolation
4. NEVER say "pre-existing" or "not related to current changes"

The correct framing: "This test is flaky — it passes individually but fails in
the full suite. Creating an issue for test isolation."

## Prevention

The phrase "pre-existing" and "not related to current changes" should trigger
the same alarm as `import re` (Rule #16). Both are banned phrases that indicate
a rule violation in progress. The moment Claude types either phrase, it should
STOP and rethink.
