# Metalearning: Warning Noise as DevEx Anti-Pattern

**Date**: 2026-03-21
**Trigger**: 3rd pass debug run flooded terminal with ONNX Runtime card0 warnings + cuda.cudart FutureWarning
**Severity**: DevEx degradation — researchers lose trust in terminal output when noise drowns signal

## Root Cause

Three layers of warning noise reach the terminal during normal `uv run python` or `uv run pytest`:

1. **ONNX Runtime C++ device discovery** (`device_discovery.cc:211`):
   - Fires at `.so` load time — BEFORE Python's `warnings` module is consulted
   - Reads `/sys/class/drm/card0/device/vendor` — fails on integrated GPUs (Intel/AMD iGPU)
   - Only suppressible via fd-level redirect or `ORT_LOGGING_LEVEL=3` env var
   - **Already solved** in `scripts/train_monitored.py` via `_suppress_fd2()` and Docker Compose via env var
   - **NOT solved** for bare `uv run python -c "..."` or `uv run pytest` without the wrapper

2. **cuda.cudart FutureWarning** (`<frozen importlib._bootstrap_external>:1325`):
   - Fires during torch import when CUDA toolkit deprecates cudart module
   - Already filtered in `train_flow.py` and `conftest.py` but only for those entry points
   - Any other entry point (scripts, notebooks, CLI one-liners) still shows it

3. **MetricsReloaded SyntaxWarning / MONAI UserWarning**:
   - Already well-handled in conftest.py and entry points

## Why This Matters

CLAUDE.md Design Goal #1: **Excellent DevEx for PhD researchers.**

> "Zero cosmetic noise (suppress warnings at entry point, NEVER tell user to 'just ignore')"

When a researcher runs ANY Python command and sees 5 lines of orange warning text, they:
1. Waste 30 seconds parsing whether it's actionable
2. Learn to ignore ALL terminal output (including real errors)
3. Lose trust in the platform's "zero-config" promise

The existing `train_monitored.py` has the RIGHT philosophy (route externals to DEBUG),
but it's only applied in ONE entry point. Every other entry point is unprotected.

## What Should Be Done

### Architecture: Centralized Warning Router

Create `src/minivess/observability/warning_router.py`:
- **Single function** `configure_warning_routing()` called by ALL entry points
- Sets `ORT_LOGGING_LEVEL=3` in os.environ BEFORE any imports
- Applies `_suppress_fd2()` for ONNX Runtime C++ noise
- Routes third-party warnings to `logging.DEBUG`
- Routes project warnings to `logging.WARNING`
- Suppresses MONAI sliding-window indexing warning entirely

### Entry Points That Need It

1. `train_flow.py` `__main__` block — partially done, needs centralization
2. `post_training_flow.py` `__main__` block — NOT done
3. `analysis_flow.py` `__main__` block — NOT done
4. `deploy_flow.py` `__main__` block — NOT done
5. `dashboard_flow.py` `__main__` block — NOT done
6. All `scripts/*.py` — some done, some not
7. `conftest.py` — already done for tests but should use the centralized function

### The card0 Specific Fix

For dual-GPU systems (iGPU + dGPU), the ONNX Runtime warning about card0 is:
- Caused by ORT trying to enumerate ALL GPU devices including the integrated one
- The iGPU doesn't have a vendor file at `/sys/class/drm/card0/device/vendor`
- Fix: `ORT_LOGGING_LEVEL=3` suppresses ALL ORT C++ warnings
- Better fix: `CUDA_VISIBLE_DEVICES=1` hides the iGPU entirely from CUDA runtime

## Key Principle

**Every entry point to the MinIVess pipeline must call `configure_warning_routing()`
as its FIRST action, before any library imports.** This is the "suppress at entry
point" mandate from CLAUDE.md DG1.7.
