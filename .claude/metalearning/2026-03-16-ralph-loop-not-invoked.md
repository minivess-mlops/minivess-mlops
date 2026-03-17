# Metalearning: Ralph Loop Not Invoked During Plan Execution

**Date:** 2026-03-16
**Severity:** P0 — Critical process failure
**Trigger:** User had to remind Claude to use Ralph Loop during overnight plan execution

---

## What Happened

1. User asked to execute `mambavesselnet-overnight-optimized.xml` plan
2. The plan's metadata explicitly lists `ralph-loop` as a required skill
3. Claude executed P0-P3 manually (running shell commands one-by-one) without
   invoking the Ralph Loop skill for infrastructure monitoring
4. Job stuck in STARTING for 38+ minutes on RunPod — exact same issue from PR #756 QA
5. Without Ralph Loop, there was no automated diagnosis, no cost tracking, no
   structured JSONL logging, no retry automation
6. User had to manually intervene to point out this failure

---

## Root Cause

Claude treated the XML plan as a manual checklist, running each command individually,
instead of recognizing that the plan requires the `ralph-loop` skill for Phase 3-4
(Launch + Monitor + Diagnose). The plan explicitly specifies:

```xml
<skills>
  <skill>ralph-loop</skill>
  <skill>self-learning-iterative-coder</skill>
</skills>
```

And Phase 4 (Diagnose + Fix) is entirely structured around Ralph Loop's diagnosis
categories, JSONL logging, cost tracking, and retry logic. None of this was invoked.

---

## The Deeper Pattern

This is the same pattern as the overnight-runner-script-freeze: Claude knows ABOUT
the tooling but doesn't USE it when executing. The plan says "use Ralph Loop" and
Claude interprets it as "manually type the commands that Ralph Loop would run" instead
of actually invoking the skill.

**Skills listed in `<skills>` are not documentation — they are execution requirements.**

---

## Rule (Non-Negotiable)

When an XML plan lists skills in its `<skills>` block:
1. **Read the skill FIRST** before executing any phase that requires it
2. **Invoke the skill** — don't manually replicate its behavior
3. **Ralph Loop specifically**: When P3 (Launch + Monitor) begins, the Ralph Loop
   skill MUST be active for polling, diagnosis, cost tracking, and retry logic
4. If a skill cannot be invoked (e.g., not installed), STOP and tell the user

---

## Checklist

- [x] Metalearning doc written (this file)
- [ ] P0 GitHub issue created for the RunPod STARTING hang
- [ ] Ralph Loop invoked for remaining execution
