# Docker Debug Training — Fix Plan

**Created**: 2026-03-09
**Status**: IN PROGRESS
**Branch**: `fix/prefect-docker-monai-optimization`

## Overview

The first full staging run (debug_all_models: 5 models × 3 folds × 2 epochs × 3 flows) revealed 17 distinct bugs in the Docker/orchestration infrastructure. This document catalogs all of them and provides a prioritized fix plan that avoids whac-a-mole one-at-a-time patching.

## What's Already Fixed (2026-03-09, commits 88b26ae → 6a222f4)

| # | Fix | Issue | Commit |
|---|-----|-------|--------|
| F1 | MLflow 3.x YAML folded scalar bug | mlflow server binding 127.0.0.1 | 88b26ae |
| F2 | MLflow `MLFLOW_SERVER_ALLOWED_HOSTS` env var | wrong env var name | 88b26ae |
| F3 | `minivess-network` added to infra containers | flow containers can't reach MLflow/Prefect | 88b26ae |
| F4 | CDI GPU check in run_debug.sh | `--runtime nvidia` always warned | 88b26ae |
| F5 | `deployment_` volume prefix in chown | wrong volume names in maintenance commands | 88b26ae |
| F6 | `MINIO_DOCKER_HOST` default → `minivess-minio` | service name vs container name | b1626a1 |
| F7 | `boto3` + `psycopg2-binary` in pyproject.toml | missing MLflow S3/Postgres deps | b1626a1 |
| F8 | `train_flow.py`: None MLflow tag → TypeError | upstream_data_run_id=None passed as tag | b1626a1 |
| F9 | `run_debug.sh`: `readarray` replaces `while` loop | docker compose consuming herestring stdin | ad79311 |
| F10 | `run_debug.sh`: Python summary SyntaxError | raw bash expansion into Python source | ad79311 |
| F11 | `export EXPERIMENT` after arg parse | Python subprocess couldn't see CLI arg | 2ec455c |
| F12 | `set +e ... set -e` around docker compose | one model failure aborts all remaining | 0a26c34 |
| F13 | `--env-file $REPO_ROOT/.env` in all docker compose calls | Docker Compose V2 .env discovery bug | 6a222f4 |
| F14 | `hf_auth.py` error message → `.env` (not `export`) | wrong user instruction | 6a222f4 |

## Remaining Known Bugs

### P0 — Blocking Current Debug Run

#### B1: SAM3 HF_TOKEN — Still Testing
The `--env-file` fix (F13) should resolve SAM3 authentication. Current run (`debug_all_models_20260309-042632.log`) is testing this. Verify outcome before marking resolved.

#### B2: `mlflow-artifacts` bucket not auto-created
MinIO does not auto-create S3 buckets. The first run requires a manual bucket creation step.
**Fix**: Add MinIO bucket initialization to infrastructure setup (see P1 section).

#### B3: Volume permissions for new volumes
Only the volumes listed in the one-time chown command (F5) are fixed. Any NEW volume added to docker-compose.flows.yml needs the same treatment.
**Fix**: Makefile target `make setup-volumes` that chowns ALL volumes.

### P1 — High Priority (Infrastructure Reliability)

#### B4: No `make setup-volumes` or `make init-infra` target
The initial setup (volume chown, MinIO bucket creation, data population) is undocumented manual work.
**Fix**: Add Makefile with:
```makefile
setup-volumes:   # chown all named volumes
init-minio:      # create mlflow-artifacts bucket
populate-data:   # copy local data into data_cache volume
init:            # = setup-volumes + init-minio
```

#### B5: Docker base image rebuild is not automatic
After any change to `src/` or `configs/`, the base image must be manually rebuilt with `--no-cache`. There is no guard to prevent stale images.
**Fix**:
1. Add image content hash to image label: `LABEL src_hash=$(sha256sum src/ | sort | sha256sum)`
2. Or: In `run_debug.sh`, always rebuild base before running: `docker build -t minivess-base:latest -f deployment/docker/Dockerfile.base .`

#### B6: `docker compose restart` vs `up --force-recreate` confusion
Many changes (env var, entrypoint, network) require `--force-recreate`. `restart` silently keeps stale config.
**Fix**: Document in `deployment/CLAUDE.md` (done) + add preflight check in `run_debug.sh` that warns if infra containers config is stale.

#### B7: MinIO bucket initialization not in docker-compose startup
`mlflow-artifacts` bucket is required but not auto-created.
**Fix**: Add a startup service to docker-compose.yml that runs `mc mb local/mlflow-artifacts`:
```yaml
minio-init:
  image: minio/mc
  depends_on: [minio]
  entrypoint: >
    /bin/sh -c "
    until mc ready local; do sleep 1; done;
    mc alias set local http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD};
    mc mb --ignore-existing local/mlflow-artifacts;
    "
```

#### B8: Multi-GPU PostgreSQL database multi-init
`POSTGRES_MULTIPLE_DATABASES` env var in docker-compose.yml is NOT a standard postgres feature. This requires a custom init script.
**Fix**: Either use a custom Docker init script (`docker-entrypoint-initdb.d/`) or create databases with `CREATE DATABASE IF NOT EXISTS`.

### P2 — Medium Priority (Training Correctness)

#### B9: MLflow tag name mismatch: `loss_function` vs `loss_name`
`orchestration/CLAUDE.md` documents: "Tag: `loss_function` (NOT `loss_name` — there's a mismatch to fix)".
**Fix**: Audit `train_flow.py`, `analysis_flow.py`, `post_training_flow.py` for consistent tag name. Use `loss_function` everywhere.

#### B10: `SAM3_LORA` in `ModelFamily` enum but not in model_builder registry
`models.py` has `SAM3_LORA = "sam3_lora"` but `model_builder.py` doesn't register it.
**Fix**: Either remove from enum OR add to registry with NotImplementedError stub + issue.

#### B11: SAM3 encoder freeze state hardcoded
`_build_sam3_vanilla` uses `encoder_frozen=True`, `_build_sam3_topolora` uses `encoder_frozen=False`. No way to override via config.
**Fix**: Add `encoder_frozen` to `ModelConfig.architecture_params` and pass through.

#### B12: Post-training flow has no pre-flight artifact validation
If training fails mid-fold, post_training silently skips.
**Fix**: Add explicit artifact availability check before running SWA/calibration plugins.

#### B13: MONAI UserWarnings still visible in container logs
MONAI multidimensional indexing warnings (`inferers/utils.py:226`) appear in container logs. CLAUDE.md Rule #7 requires suppression at entry point.
**Fix**: Add to `train_flow.py` entry point:
```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai")
```

### P3 — Low Priority (Technical Debt)

#### B14: Debug run timestamps use local time instead of UTC
`date +%Y%m%d-%H%M%S` in `run_debug.sh` uses local time. CLAUDE.md requires UTC.
**Fix**: Change to `date -u +%Y%m%d-%H%M%SZ`.

#### B15: No validation that model YAML configs exist for all registered models
Registered models (`comma_mamba`, `ulike_mamba`, etc.) may lack `configs/model/` YAML files.
**Fix**: Add preflight check in `run_debug.sh` that verifies each model has a config YAML.

#### B16: `set -euo pipefail` interacts badly with pipeline exit codes
Even with the current `set +e ... set -e` workaround, `PIPESTATUS[0]` behavior with `pipefail` is complex. The pipeline exit code may not reflect the actual model training outcome.
**Fix**: Use a more robust exit code capture pattern:
```bash
docker compose run ... train > "$MODEL_LOG" 2>&1; MODEL_STATUS=$?
```
(Remove tee, use output redirection instead of pipeline.)

#### B17: Orphan containers warning in every docker compose run
```
Found orphan containers ([minivess-mlflow minivess-postgres ...])
```
This appears on every `docker compose -f flows.yml run` because those containers belong to the infra compose project.
**Fix**: Add `--remove-orphans` or suppress the warning with `--quiet`.

## Fix Implementation Order

### Phase 1: Infrastructure Reliability (Do Now)

```
1. B7  — MinIO auto-bucket creation in docker-compose.yml
2. B4  — Makefile with setup-volumes + init-minio + populate-data
3. B5  — Auto-rebuild base image in run_debug.sh before runs
4. B17 — Suppress orphan containers warning
5. B8  — Verify POSTGRES_MULTIPLE_DATABASES with init script
```

### Phase 2: Training Correctness (This Session)

```
6. B1  — Verify SAM3 HF_TOKEN fix (running now)
7. B9  — Fix MLflow tag name: loss_function vs loss_name
8. B13 — Suppress MONAI UserWarnings in train_flow.py
9. B10 — Remove SAM3_LORA from ModelFamily OR implement
10. B12 — Post-training artifact pre-flight check
```

### Phase 3: Technical Debt (Next Sprint)

```
11. B11 — SAM3 encoder freeze configurable
12. B14 — UTC timestamps in run_debug.sh
13. B15 — Model YAML config validation preflight
14. B16 — Robust exit code capture (remove tee pipeline)
```

## Anti-Patterns That Created These Bugs

1. **Whac-a-mole debugging**: Fixing one bug at a time reveals the next. Need comprehensive preflight checks.
2. **Docker Compose V2 .env discovery**: Always use `--env-file` explicitly. Never assume auto-discovery.
3. **`docker compose restart` vs `up --force-recreate`**: Restart never applies config changes.
4. **`set -euo pipefail` without per-command error handling**: Kills the loop on first failure. Use `set +e ... set -e` brackets.
5. **Shell variables not exported**: Use `export VAR` or pass via `VAR=val cmd`. Subprocesses don't see unexported shell vars.
6. **Herestring stdin consumption in bash loops**: Use `readarray` or redirect stdin from `/dev/null`.
7. **Raw bash variable expansion into Python source**: Always pass via JSON file or environment variable.

## Key Lesson

**Comprehensive setup scripts prevent whac-a-mole.** A `make init` target that:
1. Creates `minivess-network`
2. Chowns all volumes
3. Creates MinIO buckets
4. Copies data
5. Validates `.env` exists with required keys

...would have prevented ALL of the P0 issues from the first staging run. This is the number one missing piece.

## Related Files

- `deployment/CLAUDE.md` — Docker infrastructure rules (updated with fixes)
- `.claude/metalearning/2026-03-09-dotenv-docker-compose-v2.md` — .env discovery bug
- `.claude/metalearning/2026-03-09-infrastructure-shortcut-ban.md` — No degraded infrastructure
- `LEARNINGS.md` — 12-entry catalog of Docker infrastructure failures
- `.claude/projects/.../memory/docker-infra-learnings.md` — Cross-session memory
