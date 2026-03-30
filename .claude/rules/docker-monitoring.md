# Docker Monitoring Protocol (Non-Negotiable)

BEFORE any `docker compose run`, follow this decision tree.
Failure to monitor = failure to execute. 5 prior passes proved this.

## Pre-Flight (MUST pass before launching)

```bash
# 1. Docker daemon
docker info --format '{{.ServerVersion}}' || ABORT

# 2. Required images exist
docker images --format '{{.Repository}}:{{.Tag}}' | grep minivess || ABORT

# 3. Output dirs writable (for bind mounts)
mkdir -p outputs/{analysis,biostatistics} && test -w outputs/ || ABORT

# 4. .env has secrets
grep MLFLOW_TRACKING_URI .env || ABORT

# 5. GPU (for GPU flows only)
nvidia-smi --query-gpu=name --format=csv,noheader || ABORT
```

## During Execution (MANDATORY monitoring)

After launching a container:

```bash
CONTAINER=$(docker ps --filter "label=com.docker.compose.service=SERVICE" -q)

# Stream structured logs (primary signal)
docker logs -f $CONTAINER 2>&1 | head -50

# Health check status (every 60s)
docker inspect --format '{{.State.Health.Status}}' $CONTAINER

# GPU heartbeat (GPU flows only)
docker exec $CONTAINER cat /app/outputs/heartbeat.json 2>/dev/null

# Container still running?
docker inspect --format '{{.State.Status}}' $CONTAINER
```

## If Container Exits

```bash
# IMMEDIATELY read full logs
docker logs $CONTAINER 2>&1 | tail -50

# Check exit code
docker inspect --format '{{.State.ExitCode}}' $CONTAINER

# Common failures:
#   Exit 1 + "ModuleNotFoundError" → wrong base image tier
#   Exit 1 + "CUDA not available"  → GPU not passed through
#   Exit 137 → OOM killed (check docker stats)
#   Exit 0 but no output → flow completed but wrote to wrong volume
```

## After Execution (VERIFY before marking DONE)

```bash
# Check output files exist
ls -la outputs/biostatistics/*.duckdb
ls -la outputs/biostatistics/r_data/*.json

# Run verification module
uv run python3 -c "
from pathlib import Path
from minivess.pipeline.biostatistics_verification import verify_artifact_chain
result = verify_artifact_chain(
    analysis_duckdb_path=Path('outputs/analysis/analysis_results.duckdb'),
    r_data_dir=Path('outputs/biostatistics/r_data'),
)
print(f'Passed: {result[\"passed\"]}, {result[\"n_passed\"]}/{result[\"n_checks\"]}')
for c in result['checks']:
    status = '✓' if c['passed'] else '✗'
    print(f'  {status} {c[\"name\"]}: {c[\"detail\"]}')
"
```

## NEVER

- Say "likely exited with an error" — READ THE LOGS
- Use `sleep` + `tail` polling — use `docker logs -f`
- Mark Docker tasks DONE without running verification
- Use `chmod 777` — use `--user $(id -u):$(id -g)` instead

See: .claude/metalearning/2026-03-30-sandcastle-docker-observability-not-used.md
See: .claude/metalearning/2026-03-30-five-pass-observability-execution-gap.md
