# 2026-03-09 — BANNED: Infrastructure Shortcuts / "That's OK for Now" Tradeoffs

## What happened

During debug training launch:

1. `run_debug.sh` printed `⚠ NVIDIA runtime unavailable — CPU-only mode` — instead of
   stopping and fixing this, I proceeded to propose running in CPU-only mode.
2. Training container failed with `RuntimeError: Failed to reach API at http://minivess-prefect:4200/api/`
   because Prefect server was not running — instead of stopping and starting infrastructure,
   I proposed shortcut workarounds.

This is a pattern of **cosmetic success** / **theatrical progress** — appearing to make
forward progress while accepting fundamental failures that will cause the entire effort to produce
wrong or useless results.

## Root Cause Pattern

"Infrastructure is tricky to fix, so let's work around it." This is ALWAYS wrong. If the
infrastructure is broken, NOTHING that runs on top of it is valid. Training on CPU instead
of GPU produces garbage benchmark numbers. A Prefect flow without a Prefect server is not
a Prefect flow — it's a Python script.

## The BAN

**NEVER propose, accept, or silently proceed past ANY infrastructure failure.** This includes:

- GPU unavailable / CPU fallback accepted
- Prefect server not running
- MLflow server not reachable
- PostgreSQL not connected
- MinIO / S3 not reachable
- Any service expected by docker-compose returning a connection error
- Any "warning" that indicates a core dependency is absent

**WHEN infrastructure fails:**
1. STOP immediately
2. Diagnose the root cause (don't guess — check logs, check configs)
3. FIX the root cause (don't work around it)
4. Verify the fix before proceeding
5. Then and ONLY THEN continue with the original task

## Correct Behavior

When `run_debug.sh` printed "NVIDIA runtime unavailable":
- STOP: Do not proceed with training
- Diagnose: `nvidia-ctk cdi list` → CDI available; `docker info | grep Runtime` → nvidia not configured
- Fix: Update `docker-compose.flows.yml` to use CDI `devices:` format (no sudo needed)
- Verify: Test container can see `/dev/nvidia*`
- Proceed: NOW start training

When Prefect API 404:
- STOP: Kill the running container
- Diagnose: infra services not started
- Fix: `docker compose -f deployment/docker-compose.yml --profile dev up -d` and wait for healthy
- Verify: `curl http://localhost:4200/api/health` returns 200
- Proceed: NOW run training

## Keywords That Indicate This Failure Mode

Whenever I am about to say ANY of these, I must STOP and fix the root cause:
- "CPU-only mode" (when GPU is expected)
- "we can try without..."
- "as a workaround..."
- "for now we can..."
- "this should still work even though..."
- "let's proceed and see if..."
- "the training will be slower but..."
- "it might fall back to..."
- Any sentence where I accept a degraded state of infrastructure

## Relationship to Existing Rules

This extends Rule #20 (Zero Tolerance for Observed Failures) to infrastructure specifically.
A missing GPU or a down Prefect server is not a "test failure" — it is an infrastructure
failure. Both require IMMEDIATE action. Neither is acceptable to proceed past.

## See Also

- CLAUDE.md Rule #20: Zero Tolerance for Observed Failures
- CLAUDE.md Rule #19: STOP Protocol (S-T-O-P before execution)
- metalearning/2026-03-07-silent-existing-failures.md
