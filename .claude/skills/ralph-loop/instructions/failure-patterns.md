# Ralph Loop: Failure Categories (Known Patterns)

The skill matches log lines against known failure patterns.
**No regex** -- uses `str.partition()` and `in` checks per CLAUDE.md Rule #16.

## Failure Pattern Table

| Category | Pattern in Logs | Auto-fix? | Fix Action |
|----------|----------------|-----------|------------|
| `GPU_SOLD_OUT` | `insufficient-capacity` | Yes | Try next region/GPU |
| `DOCKER_PULL_FAIL` | `failed to pull image` | Maybe | Check auth, try public |
| `DOCKER_AUTH_FAIL` | `unauthorized` in pull | Yes | Refresh DockerLoginConfig |
| `DVC_NO_GIT` | `not a git repository` | Yes | `dvc init --no-scm` |
| `ENV_VAR_LITERAL` | `${VAR}` in error | Yes | Inline vars in Python API |
| `MLFLOW_ARTIFACT_500` | `too many 500 error` | Yes | Enable multipart upload |
| `MLFLOW_AUTH_FAIL` | `401 Unauthorized` | No | Check credentials |
| `OOM_CUDA` | `CUDA out of memory` | Maybe | Reduce patch/batch size |
| `OOM_CPU` | `Cannot allocate memory` | Maybe | Increase disk/shm |
| `TORCH_SAVE_IO` | `inline_container.cc` | Yes | Atomic save |
| `SPOT_PREEMPTION` | `preempted` | Yes | SkyPilot auto-recovery |
| `DATA_MISSING` | `No training data` | No | `dvc push` first |
| `DISK_FULL` | `No space left` | Yes | Increase disk_size |
| `TIMEOUT` | `timed out` | Maybe | Increase timeout |

## Detailed Resolution Steps

### GPU_SOLD_OUT
- **Symptoms**: SkyPilot provisioning fails with `insufficient-capacity` or similar availability error.
- **Auto-fix logic**: Rotate through regions starting from unpopular EU/Asia regions. For Lambda, there are 17 regions to try. For GCP, try alternative zones within the configured region first, then expand to other regions.
- **Escalation**: After exhausting all regions, report to user with availability summary.

### DOCKER_PULL_FAIL
- **Symptoms**: Container setup fails with `failed to pull image` in setup logs.
- **Auto-fix logic**: First check if the image tag exists (`docker manifest inspect`). If it does, the issue is likely auth or network. If it doesn't, the image needs to be built and pushed first.
- **Escalation**: If image doesn't exist, escalate -- user needs to build/push.

### DOCKER_AUTH_FAIL
- **Symptoms**: `unauthorized` appears during docker pull phase.
- **Auto-fix logic**: Refresh DockerLoginConfig in SkyPilot. For GHCR, regenerate token. For GAR, refresh ADC credentials.
- **Escalation**: If token refresh fails, user must re-authenticate manually.

### DVC_NO_GIT
- **Symptoms**: DVC commands fail with `not a git repository` inside the container.
- **Auto-fix logic**: Add `dvc init --no-scm` to the setup commands in SkyPilot YAML, or ensure the Docker image includes DVC initialization.
- **Escalation**: Rarely needs escalation -- auto-fix is reliable.

### ENV_VAR_LITERAL
- **Symptoms**: Error messages contain literal `${VAR}` strings instead of resolved values.
- **Auto-fix logic**: Switch from shell variable interpolation to Python API inline values. SkyPilot's shell env handling can miss variables; passing them via the Python API is more reliable.
- **Escalation**: If the variable is genuinely unset, escalate to check `.env` population.

### MLFLOW_ARTIFACT_500
- **Symptoms**: MLflow artifact upload fails with `too many 500 error responses`.
- **Auto-fix logic**: Enable multipart upload (`MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false`, configure chunked upload). Also check if the artifact store backend (GCS/S3) has quota issues.
- **Escalation**: If 500 errors persist after multipart fix, check MLflow server health.

### MLFLOW_AUTH_FAIL
- **Symptoms**: `401 Unauthorized` from MLflow tracking server.
- **Auto-fix logic**: None -- cannot auto-fix credential issues.
- **Escalation**: Immediately report to user. Check `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` in `.env`.

### OOM_CUDA
- **Symptoms**: `CUDA out of memory` during training or validation.
- **Auto-fix logic**: Reduce batch size by 50%, then reduce patch size. Check if the model profile YAML specifies VRAM requirements that exceed the current GPU.
- **Escalation**: If minimum viable batch/patch size still OOMs, need a bigger GPU.

### OOM_CPU
- **Symptoms**: `Cannot allocate memory` -- host RAM exhaustion.
- **Auto-fix logic**: Increase `--shm-size` for Docker, or increase disk size for swap. Reduce number of dataloader workers.
- **Escalation**: If the instance type genuinely lacks RAM, need a bigger instance.

### TORCH_SAVE_IO
- **Symptoms**: `inline_container.cc` error during checkpoint save.
- **Auto-fix logic**: Switch to atomic save pattern (save to temp file, then rename). This is typically a disk I/O or NFS issue.
- **Escalation**: If disk is healthy and atomic save still fails, check filesystem type.

### SPOT_PREEMPTION
- **Symptoms**: `preempted` in cluster status or logs.
- **Auto-fix logic**: SkyPilot handles this natively with spot recovery. Ensure `use_spot: true` and `spot_recovery: FAILOVER` are set in the YAML.
- **Escalation**: If preemption happens repeatedly (>3 times), suggest switching to on-demand.

### DATA_MISSING
- **Symptoms**: `No training data` or empty data directory at runtime.
- **Auto-fix logic**: None -- data must be pushed before launch.
- **Escalation**: Immediately report. User must run `dvc push -r <remote>` or upload data to the correct location.

### DISK_FULL
- **Symptoms**: `No space left on device` during training, checkpoint save, or Docker pull.
- **Auto-fix logic**: Increase `disk_size` in SkyPilot YAML (default: 100 GB, increase to 200 GB).
- **Escalation**: If 200 GB is still insufficient, investigate what's consuming space.

### TIMEOUT
- **Symptoms**: `timed out` -- job exceeds expected duration.
- **Auto-fix logic**: Increase timeout in SkyPilot YAML. Check if training is actually progressing (loss decreasing) vs. stuck.
- **Escalation**: If training is stuck (loss not decreasing), this is a training bug, not an infra issue.

## Rules

1. **Max 3 retries per failure category**: After 3 retries of the same category, escalate to user. Prevents infinite loops.
2. **No regex for log parsing**: Per CLAUDE.md Rule #16, all pattern matching uses `str.split()`, `in`, and `str.partition()`. `import re` is banned.
3. **Multi-region rotation**: For `GPU_SOLD_OUT`, rotate through all available regions starting from unpopular EU/Asia regions (implemented in `scripts/launch_smoke_test.py`).
4. **Diagnosis is mandatory**: Every failure MUST be classified into one of the 14 categories above. If a failure doesn't match any pattern, classify as a new category and document it.
