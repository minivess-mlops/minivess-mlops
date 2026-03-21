# Warning Logger Level Analysis and DevEx Improvement Plan

**Date**: 2026-03-21
**Priority**: P1 — publication gate (DevEx)
**CLAUDE.md**: Design Goal #1 (Excellent DevEx), DG1.7 (zero cosmetic noise)

---

## Problem Statement

Running ANY Python command in MinIVess produces 3-5 lines of non-actionable warning
noise before any useful output. This violates CLAUDE.md DG1.7 ("suppress warnings at
entry point, NEVER tell user to 'just ignore'").

### Example (every `uv run python -c "..."` command):

```
2026-03-21 19:41:06 [W:onnxruntime:Default, device_discovery.cc:211 DiscoverDevicesForPlatform]
  GPU device discovery failed: device_discovery.cc:91 ReadFileContents Failed to open file:
  "/sys/class/drm/card0/device/vendor"
<frozen importlib._bootstrap_external>:1325: FutureWarning: The cuda.cudart module is deprecated
  and will be removed in a future release, please switch to use the cuda.bindings.runtime module
  instead.
```

**Impact**: Researchers waste time parsing non-actionable noise. Trust in terminal output erodes.

---

## Current Warning Sources (Ranked by Annoyance)

| # | Source | Category | Frequency | Actionable? | Current Fix |
|---|--------|----------|-----------|-------------|-------------|
| 1 | ONNX Runtime C++ device discovery (card0) | C++ stderr | Every import | No | `_suppress_fd2()` in train_monitored.py only |
| 2 | cuda.cudart FutureWarning | FutureWarning | Every import | No | Filtered in conftest.py + train_flow.py only |
| 3 | MetricsReloaded SyntaxWarning | SyntaxWarning | Every import | No | Filtered in conftest.py + train_flow.py |
| 4 | MONAI sliding-window indexing | UserWarning | 100s/epoch | No | Filtered in train_flow.py |
| 5 | PyParsing DeprecationWarning | DeprecationWarning | Every import | No | Filtered in conftest.py |
| 6 | torchmetrics UserWarning | UserWarning | Occasional | Maybe | Filtered in pyproject.toml |

---

## Current Suppression Architecture

### What's Already Done (Good)

1. **`scripts/train_monitored.py`** — Best implementation:
   - `_suppress_fd2()`: fd-level redirect for C++ warnings
   - `_route_warning()`: routes third-party to DEBUG, project to WARNING
   - Suppresses MONAI sliding-window entirely
   - Sets `ORT_LOGGING_LEVEL=3`

2. **`tests/conftest.py`** — Test-specific:
   - FutureWarning, DeprecationWarning, SyntaxWarning all filtered
   - Only applies during pytest collection

3. **Docker Compose** — Container-level:
   - `ORT_LOGGING_LEVEL=3` in `x-common-env`
   - Only applies inside Docker containers

4. **pyproject.toml** — pytest-specific:
   - 13 filterwarnings entries
   - Only applies during pytest

### What's Missing (The Gap)

**No centralized warning router.** Each entry point reimplements its own subset of
warning filters. The `train_monitored.py` pattern is the best but it's not reusable.

**Unprotected entry points:**
- `uv run python -c "..."` (quick CLI checks)
- All `scripts/*.py` except `train_monitored.py`
- `post_training_flow.py`, `analysis_flow.py`, `deploy_flow.py`, `dashboard_flow.py` `__main__` blocks
- Jupyter notebooks
- Any new entry point added in the future

---

## Proposed Solution

### Phase 1: Centralized Warning Router

Create `src/minivess/observability/warning_router.py`:

```python
"""Centralized warning routing for MinIVess.

MUST be called BEFORE any third-party imports. Sets up:
1. ORT_LOGGING_LEVEL=3 (suppresses C++ device discovery)
2. fd-level stderr redirect for ONNX Runtime .so load
3. Python warnings routing: externals → DEBUG, project → WARNING
4. Blanket suppression for known non-actionable warnings

Usage:
    from minivess.observability.warning_router import configure_warning_routing
    configure_warning_routing()  # FIRST LINE of every entry point
"""
```

### Phase 2: Apply to All Entry Points

Every `if __name__ == "__main__":` block and every Prefect flow entry point
calls `configure_warning_routing()` as its first action.

### Phase 3: .env.example + Documentation

Add `ORT_LOGGING_LEVEL=3` and `PYTHONWARNINGS=ignore::FutureWarning` to `.env.example`
with documentation explaining why.

### Phase 4: Review Agent Integration

Create a pre-commit hook or CI check that:
- Scans all `__main__` blocks for `configure_warning_routing()` call
- Flags new entry points that don't call it

---

## Dual-GPU Specific Fix (card0/card1)

For workstations with integrated GPU (card0) + discrete GPU (card1):

```bash
# In .env.example:
# Dual-GPU systems: hide integrated GPU from CUDA runtime
# CUDA_VISIBLE_DEVICES=1  # Uncomment if card0 is iGPU
```

This is more targeted than `ORT_LOGGING_LEVEL=3` but affects ALL CUDA operations.
Best approach: use `ORT_LOGGING_LEVEL=3` for ORT-specific noise, keep CUDA_VISIBLE_DEVICES
for users who want to explicitly select GPUs.

---

## Acceptance Criteria

1. `uv run python -c "import minivess"` produces ZERO warning lines
2. `uv run pytest tests/v2/unit/test_metrics.py` produces ZERO warning lines (before test output)
3. All Prefect flow entry points call `configure_warning_routing()`
4. New metalearning doc captures the pattern for future reference
5. `.env.example` documents `ORT_LOGGING_LEVEL` and `PYTHONWARNINGS`

---

## References

- CLAUDE.md Design Goal #1: Excellent DevEx
- `.claude/metalearning/2026-03-21-warning-noise-devex-antipattern.md`
- `scripts/train_monitored.py` (reference implementation of `_suppress_fd2` + `_route_warning`)
- ONNX Runtime issue: device_discovery.cc reads `/sys/class/drm/card*/device/vendor`
