# 2026-03-30 — Building Docker Observability But Not Using It (Sandcastle Pattern)

## Failure

Claude Code implemented a comprehensive 5-layer Docker observability stack:
1. CUDA guard (fail-fast when GPU unavailable)
2. GPU heartbeat monitor (background pynvml thread, heartbeat.json)
3. Structured epoch logging (JSONL to /app/logs/)
4. Grafana LGTM stack (OpenTelemetry backend)
5. Docker HEALTHCHECK on all services

Then when actually running Docker containers in this session, **completely ignored all
of it**. Instead of:
- Checking `docker logs` for structured JSONL output
- Reading `heartbeat.json` from the container
- Using `docker inspect --format '{{.State.Health.Status}}'` for HEALTHCHECK
- Monitoring GPU utilization via the heartbeat

Claude Code resorted to:
- `tail -N` on background task output files
- `sleep` + re-check loops
- Manual `docker ps` status checks
- No verification that the observability code is actually being called

## Root Cause: "Building ≠ Using" Cognitive Disconnect

This is a variant of Rule #34 ("Import ≠ Done"). The observability code was written,
tested (AST enforcement), and committed — but never actually USED by the developer
(Claude Code itself) during Docker execution. The code exists in the codebase but the
operational workflow doesn't incorporate it.

This is the "sandcastle" phenomenon: building elaborate infrastructure that
collapses the moment real waves (actual Docker execution) hit it.

## Additional Failure: Root Permissions on Bind Mounts

The session repeatedly hit `chmod: Operation not permitted` errors because:
1. Docker containers run as root by default
2. Root creates files on bind mounts → host user can't modify them
3. `chmod -R 777` is a hack, not a solution
4. The Dockerfiles already define a non-root user (UID 1000) but the
   `docker compose run` invocations don't use `--user`

The correct approach:
- Base images already have `USER minivess` (UID 1000)
- `docker compose run` should respect this (it does by default)
- Bind mount paths should be pre-created with correct ownership
- NEVER use `chmod 777` — it's a security anti-pattern

## Prevention

### For Future Docker Execution Sessions

1. **BEFORE launching**: Verify observability is functional
   ```bash
   # Check HEALTHCHECK is configured
   docker inspect --format '{{.Config.Healthcheck}}' <container>
   # Check heartbeat path exists in compose volumes
   grep heartbeat deployment/docker-compose.flows.yml
   ```

2. **DURING execution**: Use observability tools
   ```bash
   # Structured logs
   docker logs -f <container> | python -m json.tool
   # GPU heartbeat
   docker exec <container> cat /app/outputs/heartbeat.json
   # HEALTHCHECK status
   docker inspect --format '{{.State.Health.Status}}' <container>
   ```

3. **AFTER execution**: Verify artifacts via verification module
   ```python
   from minivess.pipeline.biostatistics_verification import verify_artifact_chain
   result = verify_artifact_chain(analysis_duckdb_path=..., r_data_dir=...)
   ```

### For Bind Mount Permissions

- Pre-create output directories with correct ownership BEFORE Docker run
- Use `--user $(id -u):$(id -g)` for bind-mount Docker runs
- NEVER `chmod 777` — if permissions fail, fix the root cause

## How to Apply

When running ANY Docker command in a session:
1. Check Rule #34: is the observability code actually CALLED?
2. Monitor using the tools we built, not ad-hoc shell commands
3. Pre-create bind mount dirs with `mkdir -p && chown` if needed
