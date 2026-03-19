# Metalearning: SAM3 Model Loading Tests Crept Back Into CI

**Date**: 2026-03-19
**Pattern**: Scope creep / ignoring existing architecture
**Category**: Test tier violation

## What Happened

SAM3 integration tests (14 training, 4 ONNX export, 3 MONAI Deploy, 2 BentoML,
3 MLflow serving = 26 tests) were placed in `tests/v2/integration/` instead of
`tests/gpu_instance/`. This caused them to run during `make test-prod` on the dev
machine (RTX 2070 Super, 8 GB VRAM), where they OOM or timeout — wasting 45+ minutes
per run.

The user had ALREADY instructed that SAM3 model loading should never be in CI.

## Root Cause

Claude Code added VRAM guards (`_insufficient_vram()`) as a band-aid instead of
moving the tests to the correct location (`tests/gpu_instance/`). This is wrong
because:

1. CI NEVER runs on a machine with SAM3-capable VRAM — these tests serve zero purpose in CI
2. The test tier architecture already has `tests/gpu_instance/` for exactly this use case
3. SAM3 model loading tests are instance-type validation, not regression tests
4. Adding skip guards instead of proper tier separation is architectural debt

## Fix Applied

1. MOVED all SAM3 model loading tests from `tests/v2/integration/` to `tests/gpu_instance/`
2. Added ban to CLAUDE.md: "SAM3 model loading tests NEVER in standard test suite"
3. The `conftest.py` `collect_ignore_glob = ["gpu_instance/*"]` already excludes them

## Rule

SAM3 model loading tests belong in `tests/gpu_instance/` ONLY. They are run once
per new instance type (e.g., L4 validation, A100 validation) via `make test-gpu`.
They are NEVER part of staging or prod test tiers. Period.
