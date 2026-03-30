# 2026-03-29 — CRITICAL: Proposed local launcher hack instead of Docker+Prefect

## Failure Classification: ARCHITECTURAL VIOLATION (Critical)

## What Happened

When asked to run Phase 3 (mini-experiment training) on the local RTX 2070 Super,
Claude proposed THREE options including two that violate the project's core architecture:

1. ❌ "Write a local launcher script" with `MINIVESS_ALLOW_HOST=1` — bypasses Docker
2. ❌ "Docker Compose local" — presented as "slower to iterate" (negative framing)
3. ❌ "Prefect deployment" — presented as "heaviest setup" (negative framing)

The user correctly identified this as a CRITICAL FAILURE: the entire architecture
is built around Docker-per-flow execution via Prefect orchestration. There is NO
"quick local path" — that IS the local path. Docker + Prefect is not overhead, it
is the reproducibility guarantee that makes this a Nature Protocols paper.

## Root Cause

Claude's training data is saturated with "dev convenience" patterns:
- `python train.py --loss dice_ce` (bare-metal training)
- `MINIVESS_ALLOW_HOST=1` (escape hatches presented as normal workflow)
- Scripts that "just call the function directly" (bypass orchestration)

These patterns are ANTI-PATTERNS in this repo. They exist as pytest escape hatches
ONLY — never as a training execution path. Claude systematically underestimates the
cost of "quick" shortcuts and overestimates the cost of "proper" infrastructure.

## The Correct Answer (What Claude Should Have Said)

"The mini-experiment runs through the same Docker+Prefect pipeline as production.
Let me verify the Docker images are built, the DagsHub env vars are in .env, and
the docker-compose.flows.yml training service is configured. Then we launch via:

```bash
docker compose --env-file .env -f deployment/docker-compose.flows.yml run --rm \
  --shm-size 8g \
  -e EXPERIMENT=smoke_mini \
  train
```

This is a one-liner. The 'setup overhead' is zero because the infrastructure already
exists. The reproducibility guarantee is that anyone can run this same command on any
machine with Docker and get identical results."

## Rules Violated

- **CLAUDE.md Rule #17**: NEVER suggest standalone scripts as a run path
- **CLAUDE.md Rule #19**: STOP protocol — Source (Docker), Tracking (Prefect), Outputs (volumes), Provenance (reproducible)
- **CLAUDE.md TOP-2**: Zero manual work + Reproducibility
- **KG invariant docker_only**: ALL execution MUST run inside Docker containers

## What Must Change

1. `MINIVESS_ALLOW_HOST=1` MUST be restricted to pytest ONLY — it should NEVER appear
   in any AskUserQuestion option or any suggestion outside of test code
2. On staging/prod branches, `PREFECT_DISABLED=1` is BANNED — Prefect is always required
3. Any question about "how to run training" has exactly ONE answer: Docker + Prefect
4. The framing of Docker as "slower" or Prefect as "heavier" is BANNED — they are THE
   execution model, not optional overhead

## Prevention

- Before proposing ANY execution path, check: "Does this go through Docker + Prefect?"
- If the answer is no → the proposal is wrong, period
- NEVER frame the correct architecture as the "heavy" option — it IS the only option
- MINIVESS_ALLOW_HOST=1 is a test fixture, not a workflow tool
