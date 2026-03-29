# Docker Health Check Stanzas for Training Containers

**Phase 6 deliverable** — Copy these stanzas into `deployment/docker-compose.flows.yml`.

Per CLAUDE.md Rule #31: Claude does NOT modify YAML config files without explicit user authorization.

## Train Service

```yaml
  train:
    # ... existing config ...
    healthcheck:
      test: ["CMD", "python", "scripts/healthcheck_training.py"]
      interval: 60s
      timeout: 10s
      retries: 3
      start_period: 600s  # 10 min grace for data loading
```

## HPO Service

```yaml
  hpo:
    # ... existing config ...
    healthcheck:
      test: ["CMD", "python", "scripts/healthcheck_training.py"]
      interval: 60s
      timeout: 10s
      retries: 3
      start_period: 600s
```

## Post-Training Service

```yaml
  post_training:
    # ... existing config ...
    healthcheck:
      test: ["CMD", "python", "scripts/healthcheck_training.py"]
      interval: 60s
      timeout: 10s
      retries: 3
      start_period: 300s  # 5 min grace (shorter — no data loading)
```

## Required Environment Variables

Add these to `x-common-env` in `docker-compose.flows.yml`:

```yaml
x-common-env: &common-env
  # ... existing vars ...
  STALL_THRESHOLD_MINUTES: ${STALL_THRESHOLD_MINUTES:-30}
  HEALTH_GRACE_PERIOD_MINUTES: ${HEALTH_GRACE_PERIOD_MINUTES:-10}
  GPU_HEARTBEAT_INTERVAL_S: ${GPU_HEARTBEAT_INTERVAL_S:-30}
  GPU_HEARTBEAT_LOW_UTIL_THRESHOLD_PCT: ${GPU_HEARTBEAT_LOW_UTIL_THRESHOLD_PCT:-5}
  GPU_HEARTBEAT_ALERT_AFTER_S: ${GPU_HEARTBEAT_ALERT_AFTER_S:-120}
  MLFLOW_STALL_THRESHOLD_MINUTES: ${MLFLOW_STALL_THRESHOLD_MINUTES:-15}
```

## Required .env.example Additions

```bash
# ── Observability ────────────────────────────────────────
STALL_THRESHOLD_MINUTES=30
HEALTH_GRACE_PERIOD_MINUTES=10
GPU_HEARTBEAT_INTERVAL_S=30
GPU_HEARTBEAT_LOW_UTIL_THRESHOLD_PCT=5
GPU_HEARTBEAT_ALERT_AFTER_S=120
MLFLOW_STALL_THRESHOLD_MINUTES=15
```

## How `docker ps` Will Show Health

After adding healthchecks:
```
NAMES                        STATUS
deployment-train-run-abc123  Up 45 minutes (healthy)    ← heartbeat fresh
deployment-train-run-def456  Up 4 hours (unhealthy)     ← heartbeat stale!
```
