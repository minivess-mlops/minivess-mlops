# Metalearning: Docker Compose IS the Local Stack — Not Optional

**Date**: 2026-03-29
**Severity**: CRITICAL — fundamental misunderstanding of the dev environment
**Session**: 11th pass

## What Claude Got Wrong

Claude classified 165 test skips as "legitimate integration-tier skips" because
"Docker infrastructure isn't running." Claude proposed EXCLUDING integration tests
from `make test-prod` and creating a separate `make test-integration` target.

This is WRONG. The user's architecture is:

**Local development = Docker Compose running at all times.**

- MinIO, PostgreSQL, MLflow, Prefect, Grafana — all run via Docker Compose
- `docker compose -f deployment/docker-compose.yml up -d` starts the infra stack
- Integration tests that skip because "MinIO not reachable" are FAILURES
- The 165 skips mean the Docker stack is not running — FIX THE STACK, not the tests

## What CLAUDE.md Says

> "Docker is NOT optional. Docker is the execution model — the reproducibility guarantee."
> "The only Docker-free path is `uv run pytest` for fast unit tests."

The "Docker-free" path is ONLY for fast unit tests (staging tier). The prod tier
REQUIRES Docker Compose infrastructure. This is stated in the Three-Environment Model:

| Environment | Docker | Purpose |
|-------------|--------|---------|
| local | Docker-free **or Docker Compose** | Fast iteration, uv run pytest |

"Or Docker Compose" means: Docker Compose is the EXPECTED local setup. Tests that
require MinIO/PostgreSQL/MLflow are designed to run against the Docker Compose stack.

## The Correct Understanding

1. **Staging tier** (`make test-staging`): Fast unit tests. No Docker required.
   `uv run pytest` only. Zero skips expected.

2. **Prod tier** (`make test-prod`): Full suite INCLUDING tests that require
   Docker Compose infrastructure (MinIO, PostgreSQL, MLflow). Docker Compose
   MUST be running. Zero skips expected when stack is up.

3. **GPU tier** (`make test-gpu`): GPU-specific tests. Only on RunPod/cloud.

## Prevention

When encountering test skips that say "service not reachable":
1. DO NOT classify as "legitimate integration skip"
2. DO NOT propose excluding from test tiers
3. ASK: "Is Docker Compose supposed to be running?"
4. FIX: Start Docker Compose stack, re-run tests

## See Also

- CLAUDE.md TOP-2 principle: "Docker is NOT optional"
- deployment/CLAUDE.md: Docker Compose files and setup
- deployment/docker-compose.yml: Infrastructure services
