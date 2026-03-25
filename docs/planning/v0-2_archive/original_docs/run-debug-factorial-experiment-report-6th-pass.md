# 6th Pass Debug Factorial — Live Execution Report

**Date**: 2026-03-23
**Branch**: `test/run-debug-gcp-5th-pass` (continuing — 6th pass is a relaunch after fixes)
**Status**: PRE-LAUNCH (report created BEFORE any sky jobs launch, per Rule H1)
**XML Plan**: `run-debug-factorial-experiment-6th-pass.xml`
**Harness version**: 0.2.0

---

## Cognitive Engagement: Why These Watchlist Items?

**WHY I expect these specific items:**
1. GCP controller is NEW (first run without RunPod) — latency is a hypothesis, not a measurement
2. MambaVesselNet has NEVER succeeded in 5 passes — mamba-ssm import on L4 is untested
3. SWAG scaling formula is NEW code — `min(configured, max(2, max_epochs//5))` first cloud execution
4. Checkpoint namespace `{model}_{loss}/fold_{id}` is NEW — first time on shared GCS mount

**WHAT might go wrong that passes 1-5 haven't revealed:**
- GCP controller bootstrap time (first `sky jobs launch` creates it — could be 5 min or 15 min)
- MambaVesselNet CUDA ops may differ between local (RTX 2070) and cloud (L4)
- SWAG with 2 scaled epochs may produce degenerate posterior (too few weight samples)
- Concurrent checkpoint writes to GCS via MOUNT_CACHED could have ordering issues

---

## Pre-Launch QA

| # | Fix Since 5th Pass | Category |
|---|-------------------|----------|
| 1 | Controller: RunPod → GCP (same-cloud, ~5 min/submission expected) | CRITICAL |
| 2 | Preflight check #10: controller cloud matches job cloud | CRITICAL |
| 3 | Config architecture: controller + infrastructure in cloud YAMLs | HIGH |
| 4 | SWAG scaling: `min(configured, max(2, max_epochs // 5))` | HIGH |
| 5 | Perf metrics: 6 perf/* keys in MetricKeys | MEDIUM |
| 6 | SWAG timing instrumentation (perf/swag_seconds logged) | MEDIUM |
| 7 | smoke_local.yaml → local config (not gcp_spot) | LOW |
| 8 | sync_sky_config.py: generates ~/.sky/config.yaml from repo config | HIGH |

Total new tests since 5th pass: 32 (config chain validation + SWAG scaling + perf metrics)

---

## Job Status Matrix (Live — updated after each poll)

### DynUNet (8 conditions)
| Loss | aux_calib=true | aux_calib=false |
|------|---------------|-----------------|
| cbdice_cldice | ⏳ | ⏳ |
| dice_ce | ⏳ | ⏳ |
| dice_ce_cldice | ⏳ | ⏳ |
| bce_dice_05cldice | ⏳ | ⏳ |

### MambaVesselNet (8 conditions)
| Loss | aux_calib=true | aux_calib=false |
|------|---------------|-----------------|
| cbdice_cldice | ⏳ | ⏳ |
| dice_ce | ⏳ | ⏳ |
| dice_ce_cldice | ⏳ | ⏳ |
| bce_dice_05cldice | ⏳ | ⏳ |

### SAM3 TopoLoRA (8 conditions)
| Loss | aux_calib=true | aux_calib=false |
|------|---------------|-----------------|
| cbdice_cldice | ⏳ | ⏳ |
| dice_ce | ⏳ | ⏳ |
| dice_ce_cldice | ⏳ | ⏳ |
| bce_dice_05cldice | ⏳ | ⏳ |

### SAM3 Hybrid (8 conditions)
| Loss | aux_calib=true | aux_calib=false |
|------|---------------|-----------------|
| cbdice_cldice | ⏳ | ⏳ |
| dice_ce | ⏳ | ⏳ |
| dice_ce_cldice | ⏳ | ⏳ |
| bce_dice_05cldice | ⏳ | ⏳ |

### Zero-Shot Baselines (2 conditions)
| Model | Dataset | Status |
|-------|---------|--------|
| sam3_vanilla | minivess | ⏳ |
| vesselfm | deepvess | ⏳ |

**Progress**: 0/34 SUCCEEDED | 0 FAILED | 0 RUNNING | 34 PENDING

---

## Cost Table (updated after each terminal job)

| Job ID | Condition | Duration | Cost | Status |
|--------|-----------|----------|------|--------|
| — | — | — | — | — |
| **Total** | | | **$0.00** | |

---

## Watchlist (carried from 5th pass + new)

| ID | Item | Severity | Source | Status |
|----|------|----------|--------|--------|
| W1 | GCP controller bootstrap time | HIGH | NEW | ✅ ~3 min (europe-west1-b). Failed in europe-north1 (n4 quota=0). |
| W2 | SAM3 FP16 overflow → NaN | HIGH | Passes 1-5 | ⏳ Pending (SAM3 jobs not yet submitted) |
| W3 | MambaVesselNet mamba-ssm import on L4 | HIGH | Passes 1-5 | ⏳ Pending |
| W4 | SWAG scaling with 2 epochs | MEDIUM | NEW | ⏳ Pending (first job still PENDING) |
| W5 | Checkpoint namespace on shared GCS | MEDIUM | NEW | ⏳ Pending |
| W6 | Submission latency with GCP controller | MEDIUM | NEW | ✅ ~3 min/submission (was 36 min, 12x improvement) |
| W7 | Spot preemption recovery | LOW | Passes 1-5 | ⏳ Not yet triggered |
| W8 | DeepVess data pull for zero-shot | MEDIUM | 5th pass | ⏳ Pending |
| W9 | L4 spot availability | NEW | 6th pass | ⚠️ 27+ min PENDING, queue for spots |
| W10 | GCP n4 quota in europe-north1 | NEW | 6th pass | ❌ CPUS_PER_VM_FAMILY=0, switched to europe-west1 |

---

## Cloud Observations (compound learning — updated during execution)

### O1: GCP controller bootstrap — 3 min (12x improvement over RunPod)
Controller on GCP europe-west1-b provisioned in ~3 min. First job submitted almost
immediately after. Compare: RunPod controller was 36 min/submission.
**Impact**: 34 jobs × 3 min = ~1.7 hours submission time (vs 20 hours with RunPod).

### O2: europe-north1 has zero n4 CPU quota — NOT usable for controller
`CPUS_PER_VM_FAMILY` quota for n4 is 0 in europe-north1. The controller tried
europe-north1-a, failed, then the config was changed to europe-west1.
**Test opportunity**: Preflight check for CPU/GPU quota before provisioning.
**Issue**: Need to request n4 quota increase in europe-north1.

### O3: L4 spot availability — extended PENDING times
Jobs have been PENDING for 27+ min waiting for L4 spot VMs. This is a GCP capacity
issue, not a SkyPilot issue. Spot is correct for cost savings but can have wait times.
**Test opportunity**: Monitor PENDING duration, alert if > 30 min.
**Issue #914**: Add on-demand fallback option.

### O4: Project-level .sky.yaml works correctly
SkyPilot reads `.sky.yaml` from repo root and uses it as project-level config.
`allowed_clouds` warning appears (server vs client mismatch) but doesn't block.
**Compound learning**: Version-controlled SkyPilot config is the right architecture.

### O5: Unauthorized A100 in SkyPilot YAML (CRITICAL process failure)
`train_factorial.yaml` had `accelerators: {L4: 1, A100-80GB: 1}` since its first
commit — Claude Code added A100 as "helpful fallback" WITHOUT user authorization.
A100 spot = $1.20/hr vs L4 = $0.22/hr (5.5x cost). If SkyPilot provisioned A100
because L4 was unavailable, 34 jobs = ~$40 instead of ~$8.

**This is NOT about money. It is about determinism.** The YAML is the contract.
If the YAML says L4:1, ONLY L4 is provisioned. Claude Code does not improvise.

**Fix applied**: Removed A100 from ALL SkyPilot YAMLs. Built 5-layer YAML contract
enforcement system (golden contract, pre-commit hook, 70 tests, preflight, CLAUDE.md
Rule 31). See: metalearning/2026-03-24-yaml-is-the-contract-zero-improvisation.md

### O6: L4 spot — 8+ hours PENDING, zero capacity
L4-only jobs PENDING 8+ hours (European daytime + evening). GCP spot capacity
exhausted in europe-north1 region. This is normal spot behavior — the tradeoff
for $0.22/hr vs $0.70/hr on-demand.
**Key insight**: The system MUST be fire-and-forget. User runs one command, comes
back in 1 hour or 1 week — results are there. Current script dies on network
interruptions and needs manual restart. UNACCEPTABLE.

### O7: No auto-resume, no auto-retry (CRITICAL architecture gap)
The launch script (`run_factorial.sh`) has ZERO resilience:
- Script killed → must manually restart
- Network interruption → script dies, partial submission
- All jobs fail to launch → no retry, just logs "32 failed"
- No `--resume` flag to continue from partial state
- No wrapper for fire-and-forget execution

This is the MOST IMPORTANT infrastructure gap. Every other fix (controller,
parallel, contract) is pointless if the script can't survive a network blip.

**Plan**: `docs/planning/retries-for-skypilot-spots-and-autoresume-plan.md`
**3 layers**: retry-with-backoff per launch, --resume for partials,
resilient wrapper with 1-week max wait.

### O8: sync_sky_config.py infra/cloud conflict killed 32 jobs
`sync_sky_config.py` wrote `cloud: gcp` to `~/.sky/config.yaml` while
`.sky.yaml` had `infra: gcp/europe-west1`. SkyPilot v1.0 rejects both.
ALL 32 jobs LAUNCH_FAILED. Fix: sync script now skips when project
`.sky.yaml` exists (project config takes precedence).

**Test opportunity**: `test_sync_sky_config_no_conflict_with_project_config`

---

## New Test Opportunities (compound learning)

### From O1: Controller bootstrap timing
```
tests/v2/cloud/test_controller_bootstrap.py
  test_gcp_controller_bootstraps_under_10_min
  test_controller_region_matches_config
```

### From O2: GCP quota validation
```
tests/v2/cloud/test_gcp_quota_preflight.py
  test_cpu_quota_sufficient_for_controller
  test_gpu_quota_sufficient_for_l4_jobs
  test_quota_check_runs_before_launch (preflight #11)
```

### From O3 + O5: Spot availability monitoring
```
tests/v2/cloud/test_spot_availability.py
  test_pending_duration_logged_to_mlflow
  test_spot_fallback_timeout_configurable (Issue #914)
```

### From O4: Project-level SkyPilot config
```
tests/v2/unit/deployment/test_sky_project_config.py
  test_sky_yaml_exists_in_repo_root
  test_sky_yaml_has_controller_region
  test_sky_yaml_allowed_clouds_matches_claude_md
```

---

## Deep Exploration: 3 Reviewer Agents (while jobs queue for spots)

### Reviewer 1: Untested Cloud Execution Paths — 10 Critical Gaps

| # | Component | Risk | Test File Needed |
|---|-----------|------|-----------------|
| 1 | DVC init --no-scm on fresh container | HIGH | tests/v2/unit/deployment/test_dvc_setup_commands.py |
| 2 | DVC remote add -r gcs | HIGH | tests/v2/cloud/test_dvc_remote_setup.py |
| 3 | DVC data pull from GCS | CRITICAL | tests/v2/cloud/test_dvc_pull_on_gcs.py |
| 4 | HF_TOKEN → login → weight download chain | HIGH | tests/v2/cloud/test_hf_token_setup.py |
| 5 | GPU verification (nvidia-smi + torch.cuda) on L4 | HIGH | tests/v2/cloud/test_gpu_preflight.py |
| 6 | MLflow auth credential injection end-to-end | CRITICAL | tests/v2/cloud/test_mlflow_uri_credential_injection.py |
| 7 | GCS MOUNT_CACHED checkpoint persistence | CRITICAL | tests/v2/cloud/test_gcs_mount_checkpoint.py |
| 8 | SWAG checkpoint discovery on GCS mount | HIGH | tests/v2/cloud/test_swag_checkpoint_discovery.py |
| 9 | Splits file copy on cloud VM | HIGH | tests/v2/cloud/test_splits_file_setup.py |
| 10 | Spot preemption → checkpoint resume | CRITICAL | tests/v2/cloud/test_preemption_resume.py |

**Key insight**: The YAML structure is validated (40+ tests), but the **runtime behavior
of each setup command is completely untested** on cloud VMs. We test the WHAT, not the HOW.

### Reviewer 2: Error Handling Audit — 11 Silent Failure Patterns

**CRITICAL (production data loss):**
1. Post-training plugin failures → empty model_paths (SWAG silently fails)
2. External test DataLoader failure → falls back to raw pairs (metrics on wrong scale)
3. Empty external test dataset → returns {} (DeepVess evaluation never runs)

**HIGH (training waste):**
4. Metadata collection (git hash, system info) logged as warning, continues
5. MLflow run creation fails → `mlflow_run_id=None`, downstream blind
6. Post-training MLflow logging fails → biostatistics has no data
7. DataLoader fallback hides test dataset failures
8. Post-training discovery returns [] → SWAG variants missing from ensemble

**Systemic pattern**: `except Exception: logger.warning(...)` + continue.
Caller gets empty dict/None and doesn't know if result is legitimately empty or failed.

### Reviewer 3: Cross-Flow Integration — 6 Contract Gaps

**GAP 1**: Checkpoint naming convention — train saves `best_val_loss.pth`,
post_training looks for `best.ckpt` first. No formal spec. Format drift = silent fallback.

**GAP 2 (SYSTEMIC)**: Metric key naming inconsistent across ALL flows:
| Component | Pattern | Example |
|-----------|---------|---------|
| train_flow | `fold/{id}/{metric}` | `fold/0/val_loss` |
| evaluation_runner | `{prefix}/{ds}/{subset}/{metric}` | `eval/minivess/fold_0/dsc` |
| biostatistics | `eval/{fold}/{metric}` | `eval/0/dsc` |
| builder | `eval_fold{id}_{metric}` | `eval_fold2_dsc` |

**GAP 3**: SWAG/post-training checkpoint format undefined. Analysis doesn't know
how to load SWAG models vs regular checkpoints.

**GAP 4**: deploy_flow hardcodes `experiment_name="minivess_training"` (line 346)
instead of using `resolve_experiment_name()`.

**GAP 5**: No end-to-end test: train logs metrics → analysis queries → biostatistics parses.

### Reviewer 4: Configuration Consistency — 5 ORPHAN, 3 PHANTOM, 2 DRIFT

| Category | Count | Key Examples | Risk |
|----------|-------|-------------|------|
| **ORPHAN** (YAML defined, never read) | 5 | `infrastructure.parallel_submissions`, `splits_file`, `mlflow.*`, `MLFLOW_ARTIFACT_BUCKET`, `GCS_*` | MEDIUM |
| **PHANTOM** (code hardcodes instead of reading config) | 3 | `sleep 5` instead of `rate_limit_seconds`, `infrastructure.cloud_config` never loaded, `MINIVESS_ALLOW_HOST` hardcoded | MEDIUM |
| **DRIFT** (code default ≠ YAML value) | 2 | `tracking_uri` fallback to `"mlruns"` (Rule #22 violation), `PREFECT_DISABLED` hardcoded | MEDIUM |

**Critical finding**: `infrastructure.parallel_submissions` and `rate_limit_seconds` are
defined in configs/cloud/*.yaml, validated by tests, but **NEVER read by run_factorial.sh**.
The script hardcodes `sleep 5`. The config-to-code chain is broken at the last mile.

**Rule #22 violation**: `train_flow.py:534` uses `config.get("tracking_uri", "mlruns")` —
hardcoded fallback. Should use `resolve_tracking_uri()` (fail loudly, no defaults).

---

## Expanded Test Opportunities (from deep exploration)

### Cloud Tests (tests/v2/cloud/) — 10 new files
```
test_dvc_setup_commands.py           — DVC init + remote add on fresh container
test_dvc_pull_on_gcs.py              — actual DVC pull from GCS bucket
test_hf_token_setup.py               — HF_TOKEN → login → weight download
test_gpu_preflight.py                — nvidia-smi + torch.cuda on L4
test_mlflow_uri_credential_injection.py — URI + auth end-to-end
test_gcs_mount_checkpoint.py         — MOUNT_CACHED write/read persistence
test_swag_checkpoint_discovery.py    — SWAG finds checkpoints on GCS
test_splits_file_setup.py            — splits.json copy in Docker container
test_preemption_resume.py            — simulated preemption → checkpoint recovery
test_gcp_quota_preflight.py          — CPU/GPU quota before provisioning
```

### Unit/Integration Tests — 6 new files
```
test_metric_key_consistency.py       — train keys match analysis/biostat queries
test_checkpoint_format_consistency.py — naming convention across flows
test_error_path_coverage.py          — exercise all except Exception: paths
test_deploy_flow_experiment_name.py  — no hardcoded experiment names
test_post_training_status_validation.py — check status != "error" before using
test_docker_image_content.py         — splits, cache dirs, DVC files in image
```

### Config Consistency Tests — 4 new files
```
tests/v2/unit/config/test_config_consumption.py  — ORPHAN keys have consumers or are documented
tests/v2/unit/config/test_env_var_documented.py   — every os.environ.get() has .env.example entry
tests/v2/unit/config/test_no_hardcoded_defaults.py — no config.get("key", HARDCODED) for Rule #22 vars
tests/v2/unit/config/test_infra_params_wired.py   — run_factorial.sh reads parallel_submissions from config
```

### Issues to Create
- [ ] Metric key naming standardization across all 5 flows
- [ ] Checkpoint format spec (canonical naming + format)
- [ ] deploy_flow hardcoded experiment name → resolve_experiment_name()
- [ ] Post-training plugin error propagation (status="error" → raise)
- [ ] External test DataLoader fallback → should raise, not degrade
- [ ] GCP quota preflight check (#11 in preflight script)
- [ ] Wire infrastructure.parallel_submissions into run_factorial.sh (currently ORPHAN)
- [ ] Remove tracking_uri hardcoded fallback "mlruns" (Rule #22 violation)
- [ ] Clean up ORPHAN .env.example vars (MLFLOW_ARTIFACT_BUCKET, GCS_*)

---

*Report created: 2026-03-23 pre-launch. Last update: 2026-03-23T18:00 UTC (deep exploration while spot-queuing)*
