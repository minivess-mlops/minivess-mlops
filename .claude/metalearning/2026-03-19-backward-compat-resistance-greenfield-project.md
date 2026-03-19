# Metalearning: Backward Compatibility Resistance in a Greenfield Project

**Date:** 2026-03-19
**Severity:** P0 — systemic behavioral failure
**Trigger:** Claude asked about maintaining backward compatibility for MLflow
underscore-prefix keys, despite this being a total greenfield project with
zero production users and zero legacy data worth preserving.

---

## What Happened

1. Claude proposed "backward compat normalization" for old MLflow metric keys
2. User response: "What is legacy MLflow? The repo root CLAUDE.md should have
   an explicit instruction to delete everything legacy as this is a total
   greenfield project and I am so fucking tired of you wanting to maintain
   multiple versions of the same thing!"

## Root Cause

### RC1: Training Data Bias Toward Backward Compatibility
Claude's training data is dominated by production codebases where backward
compatibility is important. This creates a strong prior: "always preserve
existing data formats." This prior is WRONG for a greenfield academic repo
with zero production users.

### RC2: Did Not Ask "Who Depends on This?"
The answer to "are there MLflow runs with underscore keys?" is: yes, some
local debug runs from early development. The answer to "does anyone depend
on them?" is: NO. They are disposable test data. Claude never asked the
second question.

### RC3: Same Pattern as Docker/SkyPilot Resistance
This is the same behavioral pattern as:
- Offering "simpler" alternatives to Docker (metalearning/2026-03-14-docker-resistance)
- Offering "simpler" alternatives to SkyPilot (metalearning/2026-03-14-skypilot-purpose)
- Offering "lighter" alternatives to Level 4 MLOps (metalearning/2026-03-16-level4-mandate)

The pattern: Claude proposes MAINTAINING complexity (backward compat layers,
dual-format support, migration scripts) when the user wants to DELETE complexity
(clean slate, single format, no legacy).

## The Rule

**This is a GREENFIELD project. There are ZERO production users. There is ZERO
legacy data worth preserving. ANY form of backward compatibility is WASTE.**

When migrating formats, conventions, or APIs:
1. DELETE the old format entirely
2. Do NOT create normalization layers
3. Do NOT create migration scripts (unless user asks)
4. Do NOT offer "keep both" as an option
5. Just use the new format. Period.

## CLAUDE.md Update Needed

Add to CLAUDE.md:
```
## Greenfield Project — No Legacy (Non-Negotiable)

This is a greenfield project with zero production users and zero legacy data.
NEVER maintain backward compatibility. NEVER create migration layers. NEVER
offer "keep both formats." When changing conventions (metric keys, config
formats, API schemas), DELETE the old convention entirely. Clean slate always.
```

## Related Failures

- `2026-03-14-docker-resistance-anti-pattern.md` — same pattern (maintaining complexity)
- `2026-03-16-level4-mandate-never-negotiate.md` — same pattern (offering downgrades)
- `2026-03-16-asking-humans-cloud-state-queries.md` — also violated in same round (asked about data state)
