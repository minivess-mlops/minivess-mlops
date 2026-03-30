# 2026-03-29 — CRITICAL: Hardcoded MLflow URI in docker-compose.flows.yml

## Failure Classification: HARDCODING VIOLATION (Critical)

## What Happened

`deployment/docker-compose.flows.yml` line 20 hardcodes:
```yaml
MLFLOW_TRACKING_URI: http://minivess-mlflow:${MLFLOW_PORT:-5000}
```

This IGNORES the `.env` variable `MLFLOW_TRACKING_URI=https://dagshub.com/...`.
The hardcoded value silently overrides what the user configured in `.env`,
making the DagsHub migration invisible to Docker containers.

## Why This Is Critical

1. `.env` is the single source of truth (CLAUDE.md Rule #22)
2. The user explicitly configured DagsHub MLflow in `.env`
3. Docker containers silently use the wrong MLflow server
4. Training metrics would go to a non-existent local MLflow instead of DagsHub
5. This wastes GPU hours with zero tracked results

## The Correct Pattern

```yaml
# CORRECT: Read from .env, provide local Docker fallback ONLY for local infra stack
MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI:-http://minivess-mlflow:${MLFLOW_PORT:-5000}}
```

This way:
- If `.env` has `MLFLOW_TRACKING_URI=https://dagshub.com/...` → DagsHub is used
- If `.env` doesn't set it → falls back to local Docker MLflow (infra stack)

## Rules Violated

- **CLAUDE.md Rule #22**: Single-source config via `.env.example`. BANNED: hardcoded URLs in Dockerfiles.
- **CLAUDE.md Rule #31**: Zero improvisation on declarative configs.

## Related

- `.claude/metalearning/2026-03-29-local-launcher-hack-proposed-instead-of-docker-prefect.md`
- Issue #971: Claude bypasses Docker+Prefect execution model
- CLAUDE.md Rule #22: "os.environ.get('VAR', 'fallback')" pattern is BANNED

## Prevention

Every environment variable in docker-compose MUST use `${VAR:-default}` syntax
where the variable name matches exactly what's in `.env.example`. NEVER construct
a URL from component parts when the user may have set the full URL in `.env`.
