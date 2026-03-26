# 10th Pass Debug Factorial Experiment Report

**Date**: 2026-03-26
**Branch**: `test/10th-pass-debug-run-qa`
**Plan**: `run-debug-factorial-experiment-10th-pass.xml`
**Scope**: 18 glitch-only cells (SAM3 TopoLoRA 8 + SAM3 Hybrid 8 + Zero-shot 2)

## Executive Summary

- **Total cells**: 18 (all submitted)
- **Succeeded**: 0 (jobs queuing — 1 L4 GPU quota, sequential execution)
- **Failed**: 0
- **Cost**: TBD (estimated: ~$5 per preflight)
- **Pre-launch issues found**: 4 (stale Docker, stale loop, network instability, duplicate submissions)
- **Test gaps identified**: 32 across 3 domains (5 CRITICAL, 10 HIGH)
- **Status**: MONITORING — jobs queuing on GCP L4 spot

## Pre-Launch Validation

| Gate | Status | Detail |
|------|--------|--------|
| GCP credentials | PASS | compute + storage enabled |
| L4 availability | PASS | L4: 1,2,4,8 on GCP |
| YAML contract | PASS | 10 files, 0 violations |
| DVC remote | PASS | `gcs -> gs://minivess-mlops-dvc-data` |
| GCS data | PASS | 199 files |
| Docker image | PASS | Rebuilt + pushed, PR #967 code verified (skip_gradient_flow + self.fusion) |
| Staging tests | PASS | 6470 passed, 0 failed, 0 skipped (298.7s) |
| Preflight (16 checks) | PASS | All 16/16 including YAML contract, cost estimate ~$5 |
| Dry-run | PASS | 32 training + 2 zero-shot; SAM3 grad_ckpt=true confirmed |

## What Changed Since 9th Pass

PR #967 (merged 2026-03-26): "fix: doughnut-hole bug fixes + 39 new tests (18 bugs, 5 phases)"

Key fixes being validated:
1. **T1**: Non-LoRA encoder params now frozen (LoRA actually efficient)
2. **T2**: `gradient_checkpointing` propagates from `config_dict`
3. **T4**: `get_volume_embeddings` uses correct MONAI dim order (B,C,H,W,D)
4. **T5**: SkyPilot exit codes 33/34 trigger EAGER_NEXT_REGION
5. **T6**: `check_gradient_flow` cleans up ~2.5 GB leaked gradients
6. **T7**: Config fingerprint consistent across call sites
7. **T8**: `get_eval_roi_size` (512,512,3) on TopoLoRA + Hybrid
8. **T15**: Single forward pass for inference checks (3x faster)
9. **T18**: `sam3_hybrid` uses `self.fusion()` (was dead code)
10. **T22**: CPU-safe bf16 detection

## Per-Cell Results

### SAM3 TopoLoRA (8 cells)

| Condition | Status | VRAM | Wall Time | DSC | Notes |
|-----------|--------|------|-----------|-----|-------|
| sam3_topolora-cbdice_cldice-calibtrue-f0 | PENDING | | | | |
| sam3_topolora-cbdice_cldice-calibfalse-f0 | PENDING | | | | |
| sam3_topolora-dice_ce-calibtrue-f0 | PENDING | | | | |
| sam3_topolora-dice_ce-calibfalse-f0 | PENDING | | | | |
| sam3_topolora-dice_ce_cldice-calibtrue-f0 | PENDING | | | | |
| sam3_topolora-dice_ce_cldice-calibfalse-f0 | PENDING | | | | |
| sam3_topolora-bce_dice_05cldice-calibtrue-f0 | PENDING | | | | |
| sam3_topolora-bce_dice_05cldice-calibfalse-f0 | PENDING | | | | |

### SAM3 Hybrid (8 cells)

| Condition | Status | VRAM | Wall Time | DSC | Notes |
|-----------|--------|------|-----------|-----|-------|
| sam3_hybrid-cbdice_cldice-calibtrue-f0 | PENDING | | | | |
| sam3_hybrid-cbdice_cldice-calibfalse-f0 | PENDING | | | | |
| sam3_hybrid-dice_ce-calibtrue-f0 | PENDING | | | | |
| sam3_hybrid-dice_ce-calibfalse-f0 | PENDING | | | | |
| sam3_hybrid-dice_ce_cldice-calibtrue-f0 | PENDING | | | | |
| sam3_hybrid-dice_ce_cldice-calibfalse-f0 | PENDING | | | | |
| sam3_hybrid-bce_dice_05cldice-calibtrue-f0 | PENDING | | | | |
| sam3_hybrid-bce_dice_05cldice-calibfalse-f0 | PENDING | | | | |

### Zero-Shot Baselines (2 cells)

| Condition | Status | Eval Dataset | DSC | Notes |
|-----------|--------|-------------|-----|-------|
| sam3_vanilla-none-calibfalse-f0 | PENDING | minivess | | |
| vesselfm-none-calibfalse-f0 | PENDING | deepvess | | |

## Fix Validation Matrix

| Fix | Expected Signal | Observed | Validated? |
|-----|----------------|----------|------------|
| T1: LoRA freeze | ~2M trainable params (not 648M) | TBD | |
| T2: GC propagation | `gradient_checkpointing: true` in config | TBD | |
| T4: Dim order | No NaN in encoder output | TBD | |
| T5: Exit codes | EAGER_NEXT_REGION on exit 33/34 | TBD | |
| T6: Gradient cleanup | No 2.5 GB VRAM leak | TBD | |
| T8: Eval ROI | Validation ~4 min (not ~6 hr) | TBD | |
| T18: Hybrid fusion | `self.fusion()` in forward pass | TBD | |
| T22: BF16 detection | No crash on CPU | TBD | |

## Cost Accounting

| Group | Cells | Est. Cost | Actual Cost | GPU-Hours |
|-------|-------|-----------|-------------|-----------|
| SAM3 TopoLoRA | 8 | ~$2.00 | TBD | TBD |
| SAM3 Hybrid | 8 | ~$2.00 | TBD | TBD |
| Zero-shot | 2 | ~$0.20 | TBD | TBD |
| **Total** | **18** | **~$4.20** | **TBD** | **TBD** |

## Observations

### Discovery 1: Docker Image Stale — GAR Missing PR #967

The GAR image was built from commit `2de98d5b` (PR #966) but HEAD was at `afeda231`
(3 commits ahead, including PR #967 doughnut-hole fixes). Rebuilt and pushed — 5 code
layers uploaded (confirmed: `skip_gradient_flow` at line 238, `self.fusion()` at line 241).

**Test opportunity**: Docker code layer verification test — after push, run container
and `grep` for a signature from the latest commit. Catches the silent push failure
documented in metalearning.

### Discovery 2: Stale Resilient Loop from Previous Session

A `while true` loop from a previous session (PID started 05:46) was still running
`SKIP_PREFLIGHT=1 bash scripts/run_factorial.sh --resume` every 5 minutes. It was
re-launching jobs with OLD env vars (pre-PR #967 `GRADIENT_CHECKPOINTING=false`).

**Test opportunity**: Lockfile-based process detection — test that `run_factorial.sh`
detects and warns about stale processes from prior sessions before launching.

### Discovery 3: Controller Network Instability

Intermittent `[Errno 101] Network is unreachable` to `oauth2.googleapis.com` caused
the SkyPilot controller to cycle between INIT and UP. The launch script's parallel
subprocesses got stuck in retry loops, preventing SAM3 conditions from being submitted.

**Test opportunity**: Network resilience test — verify `run_factorial.sh` can recover
from mid-launch network outage (submit remaining conditions after connectivity returns).

### Discovery 4: Duplicate Job Submissions from Retry

The first launch attempt submitted DynUNet/MambaVesselNet conditions (104-118) before
failing. The retry launched overlapping conditions, creating duplicates in the queue.
`--resume` checks by name but multiple background subprocesses submitted simultaneously.

**Test opportunity**: Race condition test — verify `run_factorial.sh` doesn't submit
duplicate conditions when parallel subprocesses overlap with the resume detection.

## Reviewer Agent Findings: Test Coverage Gaps

### SAM3 Adapter Gaps (10 issues, 3 CRITICAL)

| # | Gap | Priority | Impact |
|---|-----|----------|--------|
| 1 | GC propagation: only AST/mocks, no backward pass validation | CRITICAL | OOM if GC doesn't actually reduce activations |
| 2 | LoRA freeze: fake encoder only, no real SAM3 structure | CRITICAL | 648M params trainable instead of 2M |
| 3 | Fusion dead code: AST only, no output correctness | HIGH | SAM features ignored, hybrid degrades to DynUNet |
| 4 | Eval ROI: AST only, no actual call or integration | HIGH | 6-hour validation instead of 4-minute |
| 5 | skip_gradient_flow: config wiring only, no e2e | HIGH | OOM in pre-training checks despite GC |
| 6 | FPN neck freezing: not tested at all | MEDIUM | Extra VRAM + unnecessary optimization |
| 7 | GC disabled when frozen: side effects untested | LOW | Wasteful but harmless |
| 8 | GC backward pass: forward only, no recomputation | CRITICAL | Gradients may be incorrect or NaN |
| 9 | FP16/BF16 LoRA: dtype casting edge cases | MEDIUM | NaN from precision loss |
| 10 | NaN guard integration: not tested with injection | MEDIUM | Silent NaN propagation |

### Train Flow Resilience Gaps (14 issues, top 5)

| # | Gap | Priority | Impact |
|---|-----|----------|--------|
| 1 | Zero-shot early return: no e2e test | HIGH | GPU time wasted on frozen weights |
| 2 | Gradient checkpointing env var parsing | CRITICAL | "true"/"True"/"1" handling inconsistent |
| 3 | Resume duplicate detection race condition | HIGH | Duplicate jobs waste cloud credits |
| 4 | Model overrides parsing from YAML | MEDIUM | Wrong batch_size/grad_accum for SAM3 |
| 5 | Env var precedence (config vs argparse vs env) | HIGH | Config overridden by stale env var |

### Cloud Resilience Gaps (8 issues, top 5)

| # | Gap | Priority | Impact |
|---|-----|----------|--------|
| 1 | Exit code 33/34 actually used in setup script | HIGH | EAGER_NEXT_REGION never triggers |
| 2 | Docker push code layer verification | CRITICAL | Stale image runs, $15-20 wasted per pass |
| 3 | Env var immutability after submission | CRITICAL | Half-fix: new code + old config |
| 4 | YAML contract GPU allowlist enforcement | HIGH | A100 costs 5.5x more than L4 |
| 5 | Stale process detection from prior sessions | HIGH | Old loop re-launches with stale vars |

### Discovery 5: SAM3 HF Repo Name Merge Conflict Regression (CRITICAL)

`train_factorial.yaml` had `facebook/sam3-hiera-large` (404) instead of `facebook/sam3`.
This was ALREADY fixed in PR #940 but **silently reverted** during merge conflict
resolution in commit `993768f1` ("resolve conflicts with main"). The regression survived
through PR #967 because the 9th pass SAM3 jobs OOMed before reaching the weight download.

**Root cause chain**: PR #940 fixed it → PR #966 modified same file → merge conflict →
wrong resolution took old code → PR #967 merged the regression → ALL SAM3 setups 404.

**Test opportunity**: Cross-file HF repo consistency test — verify `train_factorial.yaml`
SAM3 repo matches `SAM3_HF_MODEL_ID` in `sam3_backbone.py`. Would have caught this
regression instantly.

### Discovery 6: HF Login Failure (All 3 Attempts)

HuggingFace CLI login failed all 3 attempts in job 120's setup. The `facebook/sam3`
repo is gated — requires accepted license + valid token. Even with the correct repo
name, downloads may fail if the HF token lacks access.

**HF token in plaintext logs**: `hf_DiJ...ptfdF` is visible in `sky jobs logs`.
Security concern — token should not appear in setup script output.

## Watchlist for Next Pass

1. **HF gated repo access**: Verify HF token has accepted `facebook/sam3` license
2. **Merge conflict regressions**: Add pre-commit hook or test for cross-file consistency
3. **HF token in logs**: Suppress `set -x` for sensitive commands or use `set +x` around them

## Timeline

| Time (UTC) | Event |
|------------|-------|
| 07:23 | Session started, cold-start context loaded |
| 07:35 | Docker image verified stale (PR #966 only) |
| 07:38 | Docker image rebuilt (all layers cached, PR #967 code verified) |
| 07:45 | Docker image pushed to GAR (5 code layers uploaded) |
| 07:50 | make test-staging passed (6470 passed, 0 failed, 298.7s) |
| 07:52 | All 16 preflight gates passed |
| 08:00 | 13 stale PENDING/RECOVERING jobs cancelled (old env vars) |
| 08:05 | Stale resilient loop from prior session killed (PID 3348789) |
| 08:10 | First launch attempt — controller network outage |
| 08:30 | Controller UP, 12 DynUNet/MambaVesselNet jobs submitted |
| 08:48 | 8 SAM3 TopoLoRA conditions manually submitted (119-126) |
| 08:53 | 8 SAM3 Hybrid conditions submitted (127-134) |
| 08:55 | Job 120 (TopoLoRA) briefly RUNNING, then preempted |
| 09:00 | sam3_vanilla zero-shot submitted (135), vesselfm pending |
| 09:05 | vesselfm zero-shot submitted (136) — all 18 cells in queue |
| 09:05 | 3 reviewer agents completed (32 test gaps identified) |
| 09:30 | CRITICAL: Job 120 logs show HF repo 404 (facebook/sam3-hiera-large) |
| 09:35 | Root cause: merge conflict regression (PR #940 fix reverted by 993768f1) |
| 09:40 | YAML fixed: facebook/sam3-hiera-large → facebook/sam3 |
| 09:42 | Jobs 119-136 cancelled (wrong YAML), 18 cells resubmitted (138-157) |
| 09:45 | HF repo consistency test written (18 tests, all passing) |
| 09:50 | Job 138 (sam3_vanilla zero-shot) RUNNING — setup phase succeeded |
| TBD | First SAM3 training job SUCCEEDED |
| TBD | All 18 target jobs terminal |

## Deep-Dive Findings (Second Wave — 4 Reviewer Agents)

### GC Propagation Chain — 11 Links, 4 Weak Points

Full chain traced from YAML config through 11 transformations to PyTorch model.
Key weak points identified:
1. **Link 2**: `str(settings.get('gradient_checkpointing', False)).lower()` — Python
   bool → string conversion in bash. If settings returns None → "none" → "none" ≠ "true" (safe, but confusing)
2. **Link 6**: Double-default — `os.environ.get("GRADIENT_CHECKPOINTING", "false")` AND
   argparse default. If both exist, argparse wins (correct, but undocumented)
3. **Link 7**: `bool(config_dict.get(...))` — uses Python bool() which makes ANY
   non-empty string truthy. Currently safe because upstream ensures True/False bool,
   but fragile if a string "true" leaks through.
4. **Link 9a**: Conditional injection `if _gc_config and "gradient_checkpointing" not in arch_params` —
   arch_params is populated by model profile YAML. If a profile sets
   `gradient_checkpointing: false`, the train_flow injection is skipped (correct
   but surprising interaction).

**New tests**: 26 tests in `test_gradient_checkpointing_chain.py` covering all 6 testable links.

### Cross-File Drift Vulnerabilities — 6 Classes Found

| Class | Files | Risk | Test Coverage |
|-------|-------|------|---------------|
| HF repo names | 12+ files | HIGH (broke this pass) | COVERED (18 tests) |
| Docker image path | 8+ files | MEDIUM | NOT COVERED |
| GCS bucket names | 5+ files | MEDIUM | NOT COVERED |
| Port numbers | 6+ files | LOW | NOT COVERED |
| Model family names | 10+ files | HIGH | PARTIALLY |
| Loss function names | 8+ files | MEDIUM | PARTIALLY |

**Action**: Write cross-file consistency tests for Docker image path and model family
names — these are the highest-risk unprotected classes.

### Setup Script Failure Modes — 5 Critical Gaps

1. **HF login silent failure**: 2>/dev/null silences errors, no exit 34 on total failure
2. **train_hpo.yaml + train_production.yaml NOT hardened**: Still use `|| true`, no retries
3. **No error categorization**: Can't distinguish 404 (permanent) from timeout (transient)
4. **Retries on permanent failures**: Wastes 90s retrying a 404 that will never succeed
5. **MLflow check has no retry**: Unlike DVC/HF, fails on first transient error

### Env Var Lifecycle Audit — Complete Table

14 training-critical env vars traced from .env.example → YAML → shell → Python.
Key findings:
- BATCH_SIZE and GRAD_ACCUM_STEPS have no YAML defaults (intentional: per-model overrides)
- 8 of 14 env vars have NO lifecycle test coverage (now covered by test_env_var_lifecycle.py)
- POST_TRAINING_METHODS has a fallback chain: `POST_TRAINING_METHODS` → `POST_TRAINING_METHOD`
  (backward compat, should be removed per Rule 26)
- INSTANCE_HOURLY_USD consumed in Python but NOT passed by run_factorial.sh (uses VM default)

**New tests**: 35 tests in `test_env_var_lifecycle.py` + 9 in `test_stale_process_detection.py`.

## Tests Written This Session

| Test File | Tests | Domain | Prevents |
|-----------|-------|--------|----------|
| `test_hf_repo_consistency.py` | 18 | Cross-file | Merge conflict regression (HF 404) |
| `test_gradient_checkpointing_chain.py` | 26 | Config→Model | GC not propagated → OOM |
| `test_env_var_lifecycle.py` | 35 | Env vars | Stale/missing env vars → wrong config |
| `test_stale_process_detection.py` | 9 | Process mgmt | Stale loop relaunching with old vars |
| **Total** | **88** | | |

## Go/No-Go Decision

<!-- After all 18 cells: ready for full production run? -->
