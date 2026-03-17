# Metalearning: OpenLineage Dev/Prod Confusion — Docker IS the Model

**Date:** 2026-03-17
**Severity:** P1 — RECURRING institutional knowledge failure
**Trigger:** User asked about OpenLineage integration depth. Claude proposed "H1: emit-only, local JSON" as recommended approach, suggesting Marquez/PostgreSQL was "additional complexity."

---

## What Happened

1. Claude wrote an OpenLineage report with 6 hypotheses (H1-H6)
2. Claude recommended H1 (local JSON only, no Marquez) as "zero complexity"
3. This **directly contradicts CLAUDE.md TOP-2**: "Docker is NOT optional. Docker is the execution model — the reproducibility guarantee."
4. User correctly pointed out: Docker-based services (Marquez + PostgreSQL in docker-compose) are NOT "complexity" — they ARE the standard execution model
5. Claude was treating docker-compose services as optional overhead, when they are the baseline

## Root Cause

### RC1: Conflating "dev" with "prod/staging"
The three-environment model is clear:
- **local**: Docker Compose for all services
- **env (RunPod)**: Docker image via SkyPilot
- **staging/prod (GCP)**: Docker image via SkyPilot

Claude keeps proposing "simpler" alternatives that bypass Docker, as if Docker is optional overhead. This is the SAME anti-pattern documented in:
- `.claude/metalearning/2026-03-14-docker-resistance-anti-pattern.md`
- `.claude/metalearning/2026-03-14-skypilot-bare-vm-docker-violation.md`

### RC2: Treating PostgreSQL as optional
Marquez needs PostgreSQL as its backend. PostgreSQL is ALREADY in docker-compose for Prefect and MLflow. Adding Marquez to the same PostgreSQL instance is zero additional complexity — it's a CREATE DATABASE statement.

### RC3: Underestimating the user's infrastructure vision
The user has repeatedly stated that ALL infrastructure runs in Docker. Suggesting "local JSON files" as an alternative to a proper database-backed service is exactly the kind of shortcut that the user explicitly forbids.

## Correct Approach

1. **Local execution**: Marquez + PostgreSQL in docker-compose (same PostgreSQL instance as Prefect/MLflow)
2. **GCP execution**: Marquez + Cloud SQL PostgreSQL (same instance as Prefect/MLflow)
3. **Sync**: A manual "Sync Flow" to replicate lineage data between local and remote PostgreSQL
4. **Never** propose "simpler" alternatives that bypass Docker infrastructure

## Rule

> When considering whether to implement a Docker-based service or a "simpler" alternative:
> **Always implement the Docker-based service.** The "simpler" alternative violates TOP-2.
> Docker services are NOT complexity — they ARE the execution model.

## Cross-References
- CLAUDE.md TOP-2: "Docker is NOT optional"
- `.claude/metalearning/2026-03-14-docker-resistance-anti-pattern.md`
- Three-environment model: local=Docker Compose, env=SkyPilot+Docker, staging/prod=SkyPilot+Docker
