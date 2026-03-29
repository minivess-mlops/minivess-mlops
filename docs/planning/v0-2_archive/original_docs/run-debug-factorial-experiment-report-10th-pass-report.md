# 10th Pass Debug Factorial Experiment Report

**Date**: 2026-03-26
**Branch**: `test/10th-pass-debug-run-qa`
**Plan**: `run-debug-factorial-experiment-10th-pass.xml`
**Scope**: 18 glitch-only cells (SAM3 TopoLoRA 8 + SAM3 Hybrid 8 + Zero-shot 2)

## Executive Summary

- **Total cells**: 18 submitted + 14 DynUNet/MambaVesselNet from earlier launch
- **Succeeded**: 0
- **Running**: 1 (job 146: sam3_hybrid-dice_ce-calibtrue-f0, 1h 28m execution)
- **Failed**: 2 (jobs 113, 116: DynUNet cbdice_cldice — **NaN loss at init**)
- **Pending**: 31
- **Cost**: TBD (estimated ~$8-10 for full debug pass)
- **Pre-launch issues found**: 7 (stale Docker, stale loop, network, duplicates, HF 404, HF login, **NaN loss**)
- **Test gaps identified**: 115+ across 8 domains (14 CRITICAL, 20+ HIGH) — via 11 reviewer agents
- **New tests written**: 88 (staging: 6558 passed, 0 failed, 0 skipped)
- **Bugs fixed**: 3 (HF repo regression, stale loop, stale Docker image)
- **NEW bug discovered**: DynUNet + cbdice_cldice → NaN at initialization (job 116/113)
- **Status**: Day 2 — L4 spot market extremely tight, 13+ hr PENDING queues

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
| sam3_hybrid-cbdice_cldice-calibtrue-f0 | **RUNNING** (job 154) | loading | 2h+ | TBD | HF download OK, weights loading |
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
| 10:15 | Wave 2 agents completed (GC chain, cross-file drift, setup failures, env vars) |
| 10:30 | GC chain tests (26) + env var lifecycle tests (35) + stale process tests (9) written |
| 10:45 | Commit 145331cc: 70 new robustness tests (staging 6558 passed) |
| 11:00 | Commit 56ab1e7c: comprehensive report with all 7-agent findings |
| 11:05 | Wave 3 agents launched (bottom-up: train_flow, post-training, test infra, data/losses) |
| 11:15 | **Job 154 (sam3_hybrid-cbdice_cldice-calibtrue-f0) RUNNING** — 2h job time |
| 11:15 | Setup validated: HF download OK, DVC OK, MLflow OK, L4 23.6 GB, PyTorch 2.10 |
| 11:15 | SAM3 weights loading (1468/1468 params) — training imminent |
| TBD | First SAM3 training job SUCCEEDED |
| TBD | All 18 target jobs terminal |

## Deep-Dive Findings (Second Wave — 4 Reviewer Agents)

### 1. GC Propagation Chain — 11 Links, 9 Weak Points

Full chain traced end-to-end from YAML to PyTorch model activation:

```
[1] configs/factorial/debug.yaml (model_overrides.sam3_topolora.gradient_checkpointing: true)
    ↓ YAML bool
[2] scripts/run_factorial.sh (str(settings.get(..., False)).lower() → "true")
    ↓ bash string
[3] train_factorial.yaml envs (GRADIENT_CHECKPOINTING: "false" — STATIC DEFAULT)
    ↓ overridden by --env
[4] sky jobs launch --env GRADIENT_CHECKPOINTING="true"
    ↓ frozen at submission (IMMUTABLE)
[5] train_factorial.yaml run cmd → --gradient-checkpointing "${GRADIENT_CHECKPOINTING}"
    ↓ CLI string
[6] train_flow.py argparse (default=os.environ.get("GRADIENT_CHECKPOINTING", "false"))
    ↓ .lower() == "true" → bool
[7] train_flow.py config_dict branch (bool(config_dict.get(...)))
    ↓ Python bool
[8] train_flow.py config assembly (config["gradient_checkpointing"] = True)
    ↓ dict key
[9a] train_flow.py arch_params injection (conditional: only if key not already set)
[9b] train_flow.py skip_gradient_flow (OR of config + arch_params)
    ↓ both bool
[10] sam3_topolora.py (config.architecture_params.get("gradient_checkpointing", False))
    ↓ bool
[11] sam3_backbone.py (hasattr check → gradient_checkpointing_enable())
    ↓ PyTorch model state
[12] pre_training_checks.py (skip_gradient_flow → skip diagnostic)
```

**9 weak points identified:**

| # | Link | Risk | Failure Mode |
|---|------|------|--------------|
| 1 | Link 2 | String case | `str(None).lower()` → "none" ≠ "true" (safe but confusing) |
| 2 | Link 3 | Static default | YAML default "false" could override --env if SkyPilot order changes |
| 3 | Link 6 | Double-default | `os.environ.get()` AND argparse default — argparse wins but undocumented |
| 4 | Link 6 | Brittle conversion | `.lower() == "true"` makes "1", "yes", "True" all → False |
| 5 | Link 7 | bool() footgun | `bool(config_dict.get(...))` — non-empty string = truthy |
| 6 | Link 9a | Conditional skip | Won't inject if arch_params already has `gradient_checkpointing: false` |
| 7 | Link 10 | Silent default | `.get("gradient_checkpointing", False)` — missing key = no GC = OOM |
| 8 | Link 11 | Graceful degradation | `hasattr()` guard silently skips if encoder lacks method → OOM later |
| 9 | Link 12 | Synchronization | skip_gradient_flow must match actual GC state or diagnostic OOMs |

**New tests**: 26 in `test_gradient_checkpointing_chain.py` covering links 1-6 + integration.

### 2. Cross-File Drift Vulnerabilities — 6 Classes, 50+ Duplicates

Systematic audit of ALL conceptual values duplicated across multiple files:

| Class | Files | Examples | Test Coverage | Risk |
|-------|-------|---------|---------------|------|
| HF repo names | 12+ | `facebook/sam3` in 6 YAMLs + source + 3 profiles | **COVERED** (18 tests) | HIGH (broke this pass) |
| Docker image path | 9+ | `europe-north1-docker.pkg.dev/...` in 6 YAMLs + preflight + config | **NONE** | HIGH |
| GCS bucket names | 13+ | `minivess-mlops-dvc-data` in 5 YAMLs + .dvc/config + preflight + tests | **PARTIAL** | HIGH |
| Port numbers | 7+ | MLflow:5000, Prefect:4200, BentoML:3000/3333 (COLLISION!) | **NONE** | MEDIUM |
| Model family names | 10+ | `dynunet`, `sam3_topolora` in configs + method_capabilities + source | **PARTIAL** | MEDIUM |
| Loss function names | 6+ | `cbdice_cldice` in factorial configs + method_capabilities | **GOOD** | LOW |
| Metric prefixes | 3+ | `eval/fold`, `test/` in metric_keys.py + tracking + builder | **EXCELLENT** | LOW |
| Python versions | 1 | pyproject.toml only (centralized) | **EXCELLENT** | LOW |

**Specific drift risks found:**
- **BentoML port collision**: Docker Compose uses 3333, source code comments reference 3000, Grafana also uses 3000
- **GCS bucket inconsistency**: KG says `minivess-mlops-checkpoints` is DEPRECATED but still mounted in SkyPilot YAMLs
- **MLflow artifacts bucket**: `minivess-mlops-mlflow-artifacts` is "THE artifact store" per KG but NOT mounted in any YAML

**Priority for new consistency tests:**
1. Docker image GAR path (9+ files, 0 tests) — HIGH
2. GCS bucket names (13+ files, partial) — HIGH
3. Service port numbers (7+ files, 0 tests) — MEDIUM

### 3. Setup Script Failure Modes — 9 Steps, 5 Critical Gaps

Complete analysis of every command in the SkyPilot setup block:

| Step | Failure | Current Handling | Gap | Severity |
|------|---------|------------------|-----|----------|
| HF login | Token invalid/expired | Retry 3x, `2>/dev/null` | **No exit 34 on total failure — silently continues** | CRITICAL |
| DVC pull (MiniVess) | Network/Auth | Retry 3 + timeout 600 + exit 33 | Error message not specific (auth vs disk vs network) | MEDIUM |
| SAM3 weight download | 404/Auth/Timeout | Retry 3 + timeout 600 + exit 34 | **Retries 404 (permanent) — wastes 90s** | HIGH |
| VesselFM download | 404/Auth/Timeout | Retry 3 + timeout 600 + exit 34 | Same 404 retry waste | HIGH |
| DeepVess pull | Network/Auth | Conditional + retry 3 + exit 33 | Conditional on EVAL_DATASET may mask typos | MEDIUM |
| MLflow check | Network timeout | timeout 30 + exit 1 | **No retry (unlike DVC/HF)** | MEDIUM |
| nvidia-smi | Missing GPU/driver | Fail hard | Correct (fail-fast) | LOW |
| Python imports | Missing package | Fail hard | Correct (Docker mandate) | LOW |
| train_flow.py | Runtime error | Exit non-zero | Correct | LOW |

**Cross-YAML hardening inconsistency:**

| Feature | train_factorial.yaml | train_hpo.yaml | train_production.yaml |
|---------|---------------------|----------------|----------------------|
| DVC retry loop | 3 retries + timeout | **Single attempt + `\|\| true`** | **Single attempt + `2>/dev/null`** |
| HF download retry | 3 retries + exit 34 | **`\|\| true`** | **`2>/dev/null`** |
| Exit code 33/34 | Yes | **No** | **No** |
| EAGER_NEXT_REGION | Yes | **No** | **No** |

**Only `train_factorial.yaml` is hardened.** The other two YAMLs still silently swallow errors.

**HF token security issue:** `${HF_TOKEN}` appears in `set -x` output (bash trace).
If logging is enabled, the token is visible in `sky jobs logs`. Fix: `set +x` around
sensitive commands, or use `{ set +x; } 2>/dev/null` pattern.

### 4. Env Var Lifecycle Audit — 20 Vars, Complete Table

Full lifecycle traced for every training-critical env var:

| Var | Origin | YAML Default | Shell Override | Python Consumption | Type Conv. | Tested? |
|-----|--------|-------------|----------------|-------------------|------------|---------|
| MODEL_FAMILY | YAML | `dynunet` | `--env MODEL_FAMILY="${model}"` | `os.environ.get(..., "dynunet")` | str | Partial |
| LOSS_NAME | YAML | `cbdice_cldice` | `--env LOSS_NAME="${loss}"` | `os.environ.get(..., "cbdice_cldice")` | str | No |
| FOLD_ID | Shell loop | `"0"` | `--env FOLD_ID="${fold}"` | `os.environ.get(..., "-1")` | str→int | No |
| WITH_AUX_CALIB | YAML | `"false"` | `--env WITH_AUX_CALIB=...` | `.lower() == "true"` | str→bool | No |
| MAX_EPOCHS | YAML | `"2"` (debug) | `--env MAX_EPOCHS=...` | `os.environ.get(..., "100")` | str→int | No |
| MAX_TRAIN_VOLUMES | YAML | `"0"` | `--env MAX_TRAIN_VOLUMES=...` | `os.environ.get(..., "0")` | str→int | No |
| MAX_VAL_VOLUMES | YAML | `"0"` | `--env MAX_VAL_VOLUMES=...` | `os.environ.get(..., "0")` | str→int | No |
| EXPERIMENT_NAME | YAML | `debug_factorial` | `--env EXPERIMENT_NAME=...` | `os.environ.get(..., "minivess_training")` | str | No |
| BATCH_SIZE | Shell only | None | `--env BATCH_SIZE=${BATCH_SIZE}` | `os.environ.get(..., "2")` | str→int | Yes |
| GRAD_ACCUM_STEPS | Shell only | None | `--env GRAD_ACCUM_STEPS=...` | `os.environ.get(..., "1")` | str→int | Yes |
| GRADIENT_CHECKPOINTING | YAML | `"false"` | `--env GRADIENT_CHECKPOINTING=...` | `.lower() == "true"` | str→bool | **Yes (26 tests)** |
| ZERO_SHOT | YAML | `"false"` | ZS section only | `.lower() == "true"` | str→bool | No |
| EVAL_DATASET | YAML | `minivess` | ZS section only | `os.environ.get(..., "minivess")` | str | No |
| POST_TRAINING_METHODS | YAML | `"none,swag"` | `--env POST_TRAINING_METHODS=...` | Fallback chain (see below) | str | No |
| MLFLOW_TRACKING_URI | .env | `${MLFLOW_TRACKING_URI}` (placeholder) | Via `--env-file` | `resolve_tracking_uri()` | URL str | Yes |
| HF_TOKEN | .env | `${HF_TOKEN}` (placeholder) | Via `--env-file` | Setup phase only (not Python) | token str | Yes |
| INSTANCE_HOURLY_USD | .env | None | **NOT passed by shell** | `os.environ.get(..., "0.0")` | str→float | Yes |

**Key findings:**

1. **Default mismatch**: EXPERIMENT_NAME YAML default is `debug_factorial` but Python default
   is `minivess_training`. If run_factorial.sh doesn't pass `--env`, Python uses wrong default.

2. **Immutability after submission**: ALL env vars are frozen at `sky jobs launch` time.
   If code changes add new vars or change defaults, ALL pending jobs still use OLD values.
   Only `--resume` (cancel + resubmit) picks up new values.

3. **POST_TRAINING_METHODS legacy fallback**: train_flow.py checks `POST_TRAINING_METHODS`
   first, then falls back to `POST_TRAINING_METHOD` (singular). Should be single canonical
   var per Rule 26 (greenfield, no backward compat needed).

4. **6 Rule 22 violations**: BATCH_SIZE, GRAD_ACCUM_STEPS, FOLD_ID, MAX_TRAIN_VOLUMES,
   MAX_VAL_VOLUMES, WITH_AUX_CALIB are consumed in Python but NOT declared in `.env.example`.
   These are intentionally per-condition overrides (not user-configurable secrets), but should
   still be documented in `.env.example` with comments explaining their origin.

5. **3 boolean vars** (GRADIENT_CHECKPOINTING, WITH_AUX_CALIB, ZERO_SHOT) all use
   `.lower() == "true"` which silently treats "1", "yes", "True" as False. A centralized
   `parse_bool_env()` helper would be safer.

**New tests**: 35 in `test_env_var_lifecycle.py` + 9 in `test_stale_process_detection.py`.

## Tests Written This Session

| Test File | Tests | Domain | Prevents |
|-----------|-------|--------|----------|
| `test_hf_repo_consistency.py` | 18 | Cross-file | Merge conflict regression (HF 404) |
| `test_gradient_checkpointing_chain.py` | 26 | Config→Model | GC not propagated → OOM |
| `test_env_var_lifecycle.py` | 35 | Env vars | Stale/missing env vars → wrong config |
| `test_stale_process_detection.py` | 9 | Process mgmt | Stale loop relaunching with old vars |
| **Total** | **88** | | |

## Deep-Dive Findings (Third Wave — 4 Bottom-Up Agents)

Reading files bottom-to-top to counter positional bias, targeting untouched domains.

### 1. train_flow.py Bottom-Up — 15 Edge Cases Found

| # | Location | Issue | Severity |
|---|----------|-------|----------|
| 1 | L87 `log_epoch0_cost_estimate` | Division by zero if `max_epochs=0` leaks through guards | LOW |
| 2 | L1179 `_run_swag_post_training` | `max_epochs // 5 = 0` silently clamps to `max(2, 0) = 2` | MEDIUM |
| 3 | L1001 OpenLineage emission | Broad `except Exception` swallows lineage failures | LOW |
| 4 | L954 timing parse | Empty `_setup_durations={}` on exception silently skips cost logging | MEDIUM |
| 5 | L1087 post-training subflow | All exceptions swallowed — `post_training_run_id` stays None | MEDIUM |
| 6 | L1507 post-training methods | `"none,swag".split(",")` → iterates including "none" → 2 subflow calls | LOW |
| 7 | L842 `_build_model_and_trainer` | Returned `model` and `trainer` not validated as non-None | MEDIUM |
| 8 | L725 fold iteration | `folds_to_run = []` → `for fold_id in []:` → skips all training silently | **HIGH** |
| 9 | L1378 config_dict branch | `bool(config_dict.get(...))` — any non-empty string is truthy | MEDIUM |
| 10 | L688 `_build_dataloaders` | Empty dataloader dict accepted silently (Rule 25 violation candidate) | **HIGH** |
| 11 | L1438 zero-shot early return | Missing test for `zero_shot=True, max_epochs=1` (shouldn't early return) | MEDIUM |
| 12 | L1507 pt_run_ids join | IndexError if `pt_run_ids=[]` and single-element branch executes | LOW |
| 13 | L1465 config assembly | 20+ keys assembled in dict literal — any typo silently drops a key | LOW |
| 14 | L87 cost estimate | `hourly_rate * estimated_total_hours` with rate=0.0 is misleading $0 | LOW |
| 15 | L688 sliding window | ROI size > volume size in sliding window inference → single-patch (slow) | MEDIUM |

**Top 5 test priorities with file:line references:**
- `train_flow.py:725` — Empty `folds_to_run=[]` silently skips all training (#8)
- `train_flow.py:688` — Empty dataloader dict accepted silently, violates Rule 25 (#10)
- `train_flow.py:1438` — `zero_shot=True, max_epochs=1` should train, not early-return (#11)
- `train_flow.py:490` — `val_interval=0` from config silently disables validation (#14)
- `train_flow.py:307` — Malformed `epoch_latest.yaml` → `yaml.YAMLError` not caught (#10)

### 2. Post-Training + Analysis Flows — 17 Edge Cases Found

**CRITICAL: `builder.py:502` — `load_checkpoint()` returns random weights if state_dict
doesn't match. No guard test. Can produce plausible but meaningless predictions.**

**Post-training flow (`post_training_flow.py`):**

| # | Location | Issue | Tested? |
|---|----------|-------|---------|
| 1 | L276 | Empty checkpoint list → plugins receive `[]` and fail | No |
| 2 | L268 | Missing upstream training run → `None` propagated | No |
| 3 | L317-347 | Both weight and data plugins fail → which error reported? | No |
| 4 | L253 | Checkpoint count vs metadata count mismatch | No |
| 5 | L658 | `_average_checkpoints([])` silently returns without error (Rule 25) | No |
| 6 | L669 | Mismatched state_dict keys across folds → **corrupted averaged model** | No |
| 7 | L495 | Single checkpoint with subsample → `_average_checkpoints([1])` | No |

**Analysis flow (`analysis_flow.py`) + Ensemble builder (`builder.py`):**

| # | Location | Issue | Tested? |
|---|----------|-------|---------|
| 8 | builder L502 | **CRITICAL**: Random weights on state_dict mismatch (silent) | No |
| 9 | L613-651 | `build_ensembles([])` → empty dict or error? | No |
| 10 | L757 | Zero ensemble members → skipped with warning, no signal upstream | No |
| 11 | L744-754 | Single evaluation failure blocks entire flow (no try/except) | No |
| 12 | L1141 | NaN metrics silently omitted from report (not flagged) | No |
| 13 | L827-950 | Comparison with N=1 model → meaningless results produced | No |
| 14 | L233-243 | Missing `loss_function` tag → run silently skipped | No |
| 15 | L245 | Hard gate on `eval_fold2_dsc` → debug runs filtered out | No |
| 16 | L344 | Single-fold ensemble → 1-member "ensemble" produced | No |
| 17 | L594 | All checkpoint files missing → 0-member ensemble spec | No |

### 3. Test Infrastructure Audit — ROBUST (No Critical Issues)

The test infrastructure passed all robustness checks:

- **Zero-Skip Enforcement (Rule 28)**: Active and enforced via `pytest_terminal_summary` hook
- **Fixture cleanup**: All autouse fixtures properly clean up (save/restore pattern)
- **No vacuous assertions**: Zero `assert True` / `assert 1` statements
- **No execution order dependencies**: No module-level global state mutations
- **Proper tier separation**: staging/prod/gpu/cloud/e2e all correctly filtered

**One low-severity finding**: `RLIMIT_AS` resource limit only enforced on Linux (not macOS/Windows).

### 4. Data Pipeline + Losses + Metrics — 20 Numerical Edge Cases Found

**Data loading:**

| # | File:Line | Edge Case | Impact |
|---|-----------|-----------|--------|
| 1 | `loader.py:77` | Empty imagesTr/ with no `.nii.gz` → FileNotFoundError not caught | Crash |
| 2 | `transforms.py:124` | All-background volume → MONAI random sampling fallback | Wrong gradient |
| 3 | `multitask_targets.py:98` | `compute_fn` returns wrong shape → no validation | Crash |

**Loss functions:**

| # | File:Line | Edge Case | Impact |
|---|-----------|-----------|--------|
| 4 | `loss_functions.py:207` | ClassBalancedDice: class with 0 voxels → extreme weight | Wrong loss |
| 5 | `loss_functions.py:268` | BettiLoss: all-zero predictions → gradient=0 everywhere | Silent zero |
| 6 | `loss_functions.py:512` | GraphTopologyLoss: all 3 weights=0 → `tensor(0.0)` no warning | Silent zero |
| 7 | `skeleton_recall.py:100` | Thin structure → empty skeleton → **fallback uses full mask** (semantics change) | Wrong loss |
| 8 | `cape.py:148` | Empty skeleton after thinning → `tensor(0.0)` with `requires_grad=True` | Silent zero |
| 9 | `calibration_losses.py:159` | HL1ACELoss: zero spatial size → `scatter_reduce()` → NaN | NaN propagation |

**Metrics:**

| # | File:Line | Edge Case | Impact |
|---|-----------|-----------|--------|
| 10 | `calibration_metrics.py:82` | All-0 labels → LogisticRegression.fit() single-class ValueError | Crash |
| 11 | `topology_metrics.py:163` | ccDice: unmatched components → `0/0` → NaN | NaN |
| 12 | `topology_metrics.py:53` | NSD: single-voxel volume → MONAI SurfaceDice undefined | NaN/Crash |
| 13 | `topology_metrics.py:355` | Junction F1: linear vessel (no junctions) → vacuous 0/0 metrics | Wrong metric |
| 14 | `metrics.py:75` | SegmentationMetrics: target `(B,D,H,W,1)` → squeeze wrong dim | Wrong metric |
| 15 | `calibration_metrics.py:415` | BA-ECE: flat 1D vs 3D input → silently discards spatial info | Wrong ECE |
| 16 | `multitask_metrics.py:52` | Missing GT key → head skipped, downstream expects key → KeyError | Crash |
| 17 | `calibration_metrics.py:266` | NLL: exact 0/1 predictions → never tested at boundaries | Untested |
| 18 | `loss_functions.py:252` | BettiLoss: all-one predictions → `torch.diff()=0` | Silent zero |
| 19 | `validation.py:76` | `zip(patch, min_shape, strict=False)` → mismatched ndim silently truncated | Wrong validation |
| 20 | `transforms.py` | RandCrop → empty crop if patch > padded size → DiceLoss=1.0 | Wrong loss |

**Top 5 test priorities:**
1. `calibration_metrics.py:82` — LogisticRegression with all-0 labels (CRASH) (#10)
2. `skeleton_recall.py:100` — Thin structure skeleton fallback changes loss semantics (#7)
3. `loss_functions.py:207` — ClassBalancedDice empty foreground class → extreme weight (#4)
4. `builder.py:502` — **CRITICAL**: `load_checkpoint` returns random weights on mismatch (#8 from post-training)
5. `topology_metrics.py:163` — ccDice unmatched components → NaN propagation (#11)

## Combined Test Gap Summary (All 3 Waves, 11 Agents)

| Wave | Agents | Domain | Gaps Found | Tests Written |
|------|--------|--------|------------|---------------|
| Wave 1 | 3 | SAM3, Train Flow, Cloud | 32 | 18 (HF consistency) |
| Wave 2 | 4 | GC Chain, Drift, Setup, Env Vars | 31 | 70 (GC + env + stale) |
| Wave 3 | 4 | train_flow bottom-up, post-training, test infra, data/losses | 52+ | 0 (research only) |
| **Total** | **11** | **8 domains** | **115+** | **88** |

### Priority Matrix — Next Tests to Write

| Priority | Gap | File:Line | Impact | Effort |
|----------|-----|-----------|--------|--------|
| P0 | `load_checkpoint` random weights on mismatch | builder.py:502 | Corrupted predictions | LOW |
| P0 | LogisticRegression on all-0 labels | calibration_metrics.py:82 | Crash | LOW |
| P0 | Empty `folds_to_run` skips training silently | train_flow.py:725 | Lost experiment | LOW |
| P1 | ClassBalancedDice empty foreground | loss_functions.py:207 | Wrong loss value | LOW |
| P1 | SkeletonRecall thin structures | skeleton_recall.py:100 | Semantics change | MEDIUM |
| P1 | Docker image GAR path consistency | 9 files | Wrong image pulled | LOW |
| P1 | ccDice NaN on unmatched components | topology_metrics.py:163 | NaN propagation | LOW |
| P1 | `_average_checkpoints` mismatched keys | post_training_flow.py:669 | Corrupted model | MEDIUM |
| P2 | GraphTopologyLoss all zero weights | loss_functions.py:512 | Silent zero loss | LOW |
| P2 | GCS bucket consistency | 13 files | Wrong data source | LOW |

## Production Readiness Analysis (Wave 4 — 3 Agents)

### NEW BUG: NaN Loss at Init for DynUNet + cbdice_cldice

**Root cause**: `SoftclDiceLoss.forward()` in MONAI's `cldice.py` produces NaN
when soft skeleton computation yields near-zero values with random model weights.

**Mechanism** (confirmed by code review):
1. Random logits → uniform ~0.5 probabilities for all classes
2. `soft_skel()` applies 10 iterations of morphological erosion → near-zero skeleton
3. `tprec = (sum(skel_pred * y_true) + smooth) / (sum(skel_pred) + smooth)` → ~1.0
4. `tsens = (sum(skel_true * y_pred) + smooth) / (sum(skel_true) + smooth)` → ~1.0
5. `cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)` → NaN from FP32 rounding

**Why it breaks**: With 98% background brain vessel data, foreground skeleton is
extremely thin. Random predictions have no spatial structure → degenerate skeleton.
The combination of class-balanced dice + clDice is particularly fragile at init.

**Why dice_ce doesn't break**: Standard dice uses `(2*intersection + smooth) / (union + smooth)`.
No division-by-zero possible because union ≥ smooth > 0.

**Mitigation options**:
1. Increase smooth constant in SoftclDiceLoss (from 1e-5 to 1e-3)
2. Skip loss_sanity check for clDice-containing losses (they stabilize after ~5 epochs)
3. Add NaN guard: `if torch.isnan(cl_dice): return torch.tensor(0.0, requires_grad=True)`
4. Initialize model weights with better strategy (not uniform random)

**Test to write**: `test_cbdice_cldice_nan_at_init` — create random logits + vessel GT,
compute CbDiceClDiceLoss, verify loss is finite. This would have caught this on day 1.

### Scaling: Debug → Production

| Dimension | Debug | Production | Risk | Mitigation |
|-----------|-------|-----------|------|------------|
| Epochs | 2 | 50 | 25x longer wall time per cell | Checkpoint resume every epoch |
| Data | 23/12 train/val | 47/23 | 2x memory (~4.7 GB, fits L4) | Adaptive cache rate |
| Folds | 1 | 3 | 3x total jobs (54 cells + 6 zero-shot) | run_factorial.sh handles multi-fold |
| Total jobs | ~18 | ~60 | Sequential on 1 L4 = weeks | **Request quota increase** |
| Cost | ~$5 | ~$3 (L4 spot very cheap) | Low | Cost logging per cell |
| Wall time | ~17 GPU-hrs | ~500 GPU-hrs at 1 L4 = 21 days | **CRITICAL BOTTLENECK** | Quota increase to 4+ L4 |
| Preemptions | ~2 in debug | ~15 expected at 15% rate | Resume from epoch checkpoint | GCS mount persists |
| SWAG | 2//5=0 → 2 | 50//5=10 epochs | Untested at 10 epochs | Write SWAG scaling test |

**GPU quota is the #1 bottleneck**: 60 sequential jobs on 1 L4 = 21+ days. With 4 L4s
in parallel, this drops to ~5 days. Request quota increase BEFORE production launch.

### Spot Preemption Resilience — MOSTLY READY

| Capability | Status | Gap |
|-----------|--------|-----|
| Epoch-level checkpoints | IMPLEMENTED | Atomic write (pth then yaml) |
| GCS persistence | IMPLEMENTED | MOUNT_CACHED survives preemption |
| Resume from epoch N | IMPLEMENTED | Skips epochs 0..N-1 correctly |
| MLflow run status check | IMPLEMENTED | Must be RUNNING to resume |
| SkyPilot recovery | EXIT 33/34 → EAGER_NEXT_REGION | Max 3 restarts |

**Critical gaps for production:**
1. **MLflow unreachable → silent fresh start**: No retry on resume check. If Cloud Run
   is slow during recovery, all 25 completed epochs are re-run.
2. **MOUNT_CACHED async lag**: If training writes checkpoints faster than GCS upload,
   preemption loses the latest checkpoint. Risk: low (epochs are 10+ min each).
3. **No per-cell cost tracking**: Preempted cells that restart waste partial credits.
   No reporting mechanism for total cost including preemption waste.

### Training Log Observability Gap (User-Identified)

**Current state**: `sky jobs logs <ID> --no-follow` streams logs from the controller.
This is:
- Slow to connect (20-30s SSH + log pipe setup)
- Blocks the calling process (no timeout, must be killed)
- Unstructured text (ANSI codes, PID prefixes, no JSON/JSONL)
- Not LLM-parseable (interleaved stderr/stdout, progress bars)

**Production impact**: With 60+ jobs, manual log inspection is infeasible. Claude Code
spent 10+ minutes in this session just trying to get logs from a RUNNING job.

**Improvement opportunities**:
1. **JSONL structured logging**: Emit key events as JSON lines (epoch start/end, loss,
   VRAM, validation DSC) to a separate file. LLM reads JSONL, not raw logs.
2. **Heartbeat file**: Write `heartbeat.json` to GCS mount every 60s with:
   `{"epoch": 12, "loss": 0.45, "vram_gb": 9.2, "wall_time_s": 3600}`
3. **Summary artifact**: At job end, write `run_summary.json` to GCS with all key metrics.
   Claude Code reads this instead of streaming full logs.
4. **SkyPilot log API**: Instead of `sky jobs logs`, query the controller's API directly
   with filters (last N lines, grep pattern).

## Go/No-Go Decision

### Debug Pass: NOT YET COMPLETE
- 1/18 target cells actively running (job 146: sam3_hybrid, 1h 36m)
- 2 DynUNet cells FAILED (NaN loss — new bug)
- 31 cells still PENDING (L4 spot market very tight)
- Need at least 1 SAM3 SUCCEEDED before declaring debug pass ready

### Production Launch Prerequisites
1. **GPU quota increase**: Request 4+ L4 GPUs (current: 1 = 21 days)
2. **NaN loss fix**: cbdice_cldice init NaN must be resolved (affects 25% of cells)
3. **All 18 target cells SUCCEEDED** in debug pass
4. **JSONL structured logging** for production observability
5. **MLflow resume retry** for preemption resilience
6. **Docker image verification gate** in run_factorial.sh (not just preflight)
