# Metalearning: Ralph Loop Not Invoked During Plan Execution

**Date:** 2026-03-16
**Severity:** P0 — Critical process failure

## What Happened

Plan metadata listed `ralph-loop` as a required skill. Claude executed P0-P3
manually without invoking Ralph Loop for monitoring, diagnosis, or cost tracking.
User had to remind Claude to use the skill.

## Rule

When an XML plan lists skills in `<skills>`, they are execution requirements.
Invoke the skill — don't manually replicate its behavior.
