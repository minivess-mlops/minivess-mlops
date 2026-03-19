# Anti-Patterns — Self-Learning Iterative Coder

Reference document for failure modes observed across 27+ task executions (163+ tests).
Read this at the START of every session alongside the 11 Critical Rules in SKILL.md.

## Anti-Pattern Catalog

| # | Anti-Pattern | Why It Fails | Correct Approach |
|---|-------------|-------------|------------------|
| 1 | **Ghost completion** | Claiming tests pass without running them | Always run verification suite |
| 2 | **Shotgun fix** | Changing many things at once hoping something works | Analyze failure, make targeted fix |
| 3 | **Test-after** | Writing implementation first, tests second | RED phase is always first |
| 4 | **Context hoarding** | Trying to do too much in one session | Max 20 inner iterations per session, then new session |
| 5 | **Skip the skip** | Ignoring lint/type errors "because tests pass" | All three gates must be green |
| 6 | **Infinite loop** | Retrying the same fix without analyzing why it fails | After 2 identical failures, escalate |
| 7 | **Convention ignorance** | Using pip instead of uv, strings instead of Path | Read CLAUDE.md before starting |
| 8 | **Placeholder code** | Writing `pass`, `TODO`, `NotImplementedError` | Full implementations only |
| 9 | **Knowledge amnesia** | Re-discovering the same issues across sessions | Append to LEARNINGS.md |
| 10 | **Silent dismissal** | Classifying failures as "pre-existing" and moving on | Zero tolerance — fix, issue, or report (Rule #9) |
| 11 | **Whac-a-mole** | Fixing one test at a time with `-x` in a loop | Failure Triage Protocol — batch fix by root cause (Rule #10) |
| 12 | **Skim-and-code** | Writing code without reading existing implementation | Read 30%, implement 70% — tokens upfront (Rule #11) |

## Detection Signals

How to recognize you are falling into an anti-pattern:

- **Ghost completion**: You typed "tests pass" but did not see pytest output in the conversation
- **Shotgun fix**: You changed >3 files in a single FIX phase iteration
- **Whac-a-mole**: You ran pytest with `-x` flag more than once in a FIX phase
- **Silent dismissal**: You used the phrase "pre-existing", "not related", or "separate issue" without a `gh issue create` call
- **Skim-and-code**: Your Read tool calls are <30% of total tool calls before the first Write/Edit

## Escalation Triggers

When you detect an anti-pattern, STOP the current phase and:
1. Name the anti-pattern explicitly in your response
2. Describe which detection signal triggered it
3. State the correct approach from the table above
4. Resume with the correct approach

## Source

- `.claude/metalearning/2026-03-07-silent-existing-failures.md` (Anti-Patterns #10)
- `.claude/metalearning/2026-03-18-whac-a-mole-serial-failure-fixing.md` (Anti-Pattern #11)
- Self-learning-iterative-coder v2.0.0 retrospective (Anti-Patterns #1-9)
