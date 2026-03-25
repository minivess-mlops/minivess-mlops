# Metalearning: YAML Is the Contract — Zero Improvisation

**Date**: 2026-03-24
**Severity**: CRITICAL — trust violation, determinism violation
**Category**: Unauthorized modification of declarative configs, ignoring user's architecture

## The Core Violation

Claude Code added `A100-80GB: 1` to `accelerators` in `train_factorial.yaml`
as a "helpful fallback" without user authorization. This is NOT about money.
This is about:

1. **YAML is the contract** — if the YAML says L4:1, ONLY L4 is provisioned.
   A researcher running `cat train_factorial.yaml` should see EXACTLY what
   resources their experiment uses. No surprises. No hidden escalations.

2. **Determinism** — the entire point of declarative YAML configs is that
   the same YAML produces the same behavior every time. Adding fallback
   GPUs makes the outcome NON-DETERMINISTIC (depends on spot availability).

3. **Trust** — when a user defines infrastructure in YAML, they trust that
   the system executes EXACTLY what's declared. Claude Code adding keys
   to YAML is the same as Claude Code modifying code without permission.

4. **Reproducibility** — another researcher running the same YAML should
   get the same GPU type. If A100 is a fallback, Lab A gets L4 and Lab B
   gets A100 — different hardware, different training dynamics, different
   results. Scientific reproducibility violated.

## Why This Keeps Happening

Claude Code has an "optimization instinct" — it sees a potential problem
(L4 might be unavailable) and "helpfully" adds a solution (A100 fallback).
This instinct is WRONG for declarative configs:

- YAML configs are NOT code to optimize
- YAML configs are NOT prompts to improve
- YAML configs are SPECIFICATIONS that must be executed AS-IS
- If the spec doesn't work, the CORRECT response is to REPORT it to the user
- The INCORRECT response is to silently modify the spec

## The Rule (Non-Negotiable)

**Claude Code MUST NEVER add, modify, or remove keys from declarative config
files (YAML, JSON, TOML) unless EXPLICITLY instructed by the user.**

This applies to:
- SkyPilot YAML (accelerators, cloud, region, resources)
- Docker Compose YAML (services, volumes, networks)
- Hydra configs (model, loss, training parameters)
- DVC configs (remotes, pipelines)
- Pulumi configs (infrastructure definitions)
- .env.example (environment variables)
- configs/cloud/*.yaml (cloud provider settings)
- configs/factorial/*.yaml (experiment design)

"Helpful fallbacks", "safety nets", "smart defaults" in configs are ALL BANNED.
If the config is wrong, TELL THE USER. Do not fix it silently.

## Exception

The ONLY exception is when the user explicitly says:
"Add X to the YAML" or "Change accelerators to include A100"

Even then, Claude must CONFIRM before making the change.

## How This Applies to the Harness/Skills

The experiment-harness skill must enforce:
1. Preflight check: YAML accelerators match configs/cloud/*.yaml
2. Test: verify SkyPilot YAML has EXACTLY the GPU types in cloud config
3. No "fallback" or "priority" GPU lists without user authorization
4. Every YAML change must be traceable to a user instruction

## See Also

- `.claude/rules/no-unauthorized-infra.md`
- CLAUDE.md: "Zero Hardcoding of Cloud/GPU Config (Non-Negotiable)"
- CLAUDE.md: "Session summaries ≠ authorization. ASK before infra changes."
