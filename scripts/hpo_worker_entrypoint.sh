#!/usr/bin/env bash
# hpo_worker_entrypoint.sh — Entrypoint for parallel HPO workers (Issue #504).
#
# Each replica gets a unique GPU via REPLICA_INDEX (set by docker compose or orchestrator).
# The Optuna study is shared via PostgreSQL (OPTUNA_STORAGE_URL from .env.example).
#
# Usage (via docker compose):
#   docker compose -f deployment/docker-compose.flows.yml up --scale hpo-worker=4
#
# Each container gets REPLICA_INDEX 0..N-1 set by the orchestrator (or defaults to 0).
# CLAUDE.md Rule #18: no /tmp for artifacts. All outputs go to named volumes.

set -euo pipefail

# Assign GPU based on REPLICA_INDEX — set by docker compose scale or SkyPilot
export CUDA_VISIBLE_DEVICES="${REPLICA_INDEX:-0}"

echo "HPO worker starting (REPLICA_INDEX=${REPLICA_INDEX:-0}, GPU=${CUDA_VISIBLE_DEVICES})"

exec uv run python -m minivess.orchestration.flows.hpo_flow
