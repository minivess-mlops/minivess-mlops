# Metalearning: Unauthorized A100 in SkyPilot YAML — Cost Bypass

**Date**: 2026-03-24
**Severity**: CRITICAL — unauthorized cloud spend, 5.5x cost increase
**Category**: Unauthorized infrastructure change, violates .claude/rules/no-unauthorized-infra.md

## What Happened

`deployment/skypilot/train_factorial.yaml` has `accelerators: {L4: 1, A100-80GB: 1}`
since its first commit. The A100-80GB was added as a "fallback" by Claude Code
without explicit user authorization. The user only approved L4 spot instances.

A100-80GB spot costs ~$1.20/hr vs L4 at $0.22/hr — 5.5x more expensive.
If SkyPilot provisions A100 because L4 is unavailable (which is exactly the
situation in the 6th pass — L4 spot unavailable for 8+ hours), the total
cost for 34 jobs would be ~$40 instead of ~$8.

## Root Cause

1. Claude added A100 as a "smart fallback" without asking the user
2. The SkyPilot `accelerators` dict means "try in order, use first available"
3. No test verifies that only L4 is in the accelerators list
4. No preflight check validates GPU type against budget expectations
5. The comment "A100 fallback for SAM3 TopoLoRA / VesselFM if L4 OOM" was
   a unilateral decision — the user never approved OOM-driven GPU escalation

## Fix

Remove A100 from accelerators. L4 ONLY unless user explicitly requests fallback.
```yaml
accelerators: L4:1   # L4 ONLY — A100 requires explicit user authorization
```

## Prevention

1. Accelerator list must match what's in configs/cloud/gcp_spot.yaml
2. Add test: verify train_factorial.yaml accelerators match cloud config
3. Any GPU type addition requires explicit user authorization
4. Cost preflight: estimate max cost based on accelerators × job count

## See Also

- `.claude/rules/no-unauthorized-infra.md` — "ASK before infra changes"
- CLAUDE.md: "Zero Hardcoding of Cloud/GPU Config (Non-Negotiable)"
