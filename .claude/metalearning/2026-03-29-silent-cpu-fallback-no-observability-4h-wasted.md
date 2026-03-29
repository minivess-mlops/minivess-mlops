# 2026-03-29 — CRITICAL: Silent CPU fallback, 4 hours wasted with zero observability

## Failure Classification: OBSERVABILITY FAILURE (Critical) + CUDA MISMATCH (Infrastructure)

## What Happened

After rebuilding the Docker base image (to include a BFloat16 fix for CbDice loss),
the cbdice_cldice training ran for **4 hours on CPU** with zero training epochs produced.
The container showed:
- 100% CPU utilization (appeared to be "working")
- Near-zero GPU utilization (appeared to be "loading data")
- Zero training metrics in MLflow
- Only 47 log lines in 4 hours

**Root cause**: PyTorch in the rebuilt image was compiled with CUDA 13.0, but the host
NVIDIA driver (560.35.05) only supports up to CUDA 12.6. PyTorch silently fell back to
CPU-only mode with a single warning buried in initialization output.

## Why This Is CRITICAL (Two Separate Failures)

### Failure 1: Silent CPU Fallback
PyTorch's behavior of silently falling back to CPU when CUDA is unavailable is
DANGEROUS for GPU training. The training flow should have:
1. **DETECTED** that CUDA was unavailable (`torch.cuda.is_available() == False`)
2. **LOGGED a loud ERROR** (not a warning buried in startup noise)
3. **RAISED an exception** — training on CPU is never intentional in this project
4. **FAILED FAST** — within seconds, not after hours of CPU-only preprocessing

### Failure 2: Zero Observability
For 4 hours, there was NO way to know the training was stuck:
1. Docker logs showed only 47 lines (all from startup)
2. Output buffering hid any progress
3. MLflow only had system metrics (no training metrics = silent failure)
4. GPU utilization at 0% was misinterpreted as "data loading" instead of "no GPU"
5. No heartbeat, no progress bar, no epoch counter, no ETA

## The Correct Behavior

```python
# In train_flow.py, BEFORE any model creation:
if not torch.cuda.is_available():
    logger.error("CUDA NOT AVAILABLE — training will NOT proceed on CPU")
    raise RuntimeError(
        f"CUDA not available. Host driver: {torch.version.cuda}, "
        f"PyTorch CUDA: {torch.version.cuda}. "
        f"Check NVIDIA driver compatibility."
    )
```

And the training loop should log:
```
[Epoch 1/20] train_loss=0.85 val_dice=0.32 | 3m12s | GPU: 98% 2.4GB
```

If zero epoch logs appear within 5 minutes, something is wrong.

## Rules Violated

- **CLAUDE.md Rule #25**: Loud failures, never silent discards
- **CLAUDE.md Rule #20**: Zero tolerance for observed failures
- **Observability principle**: If you can't see it, you can't debug it

## CUDA Version Mismatch Details

| Component | Version | Supports |
|-----------|---------|----------|
| Host NVIDIA driver | 560.35.05 | CUDA ≤ 12.6 |
| Base image CUDA toolkit | 12.6.3 | Compatible ✓ |
| PyTorch CUDA runtime | **13.0** | Requires driver ≥ 560.70 |

The mismatch was introduced by the base image rebuild which pulled a newer PyTorch
from PyPI with CUDA 13.0 bindings via `uv sync`. The `uv.lock` file should pin the
PyTorch CUDA version explicitly to prevent this.

## Required Fixes

1. **Pin PyTorch CUDA version** in `pyproject.toml` / `uv.lock` to match host driver
2. **Add CUDA availability guard** at the START of `training_flow()` — fail immediately
3. **Add GPU heartbeat** to training loop — if GPU util < 5% for > 2 minutes, log ERROR
4. **Add epoch progress logging** — force-flush stdout/stderr in Docker
5. **Add CUDA version check** to Docker preflight — compare container vs host
6. **Unbuffer Python output** in Docker: `ENV PYTHONUNBUFFERED=1` in Dockerfile.base

## Related

- `.claude/metalearning/2026-03-15-t4-turing-fp16-nan-ban.md` — GPU architecture issues
- `.claude/metalearning/2026-03-29-local-launcher-hack-proposed-instead-of-docker-prefect.md`
- Issue #971: Docker+Prefect bypass pattern
