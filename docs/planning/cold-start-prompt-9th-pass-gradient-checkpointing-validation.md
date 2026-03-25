# Cold-Start Prompt: 9th Pass Gradient Checkpointing Validation

> **MANDATORY READ BEFORE ANY ACTION**: `.claude/metalearning/2026-03-25-overconfident-oom-fixed-claim.md`
> BANNED: Claiming "SAM3 training works" without `sky jobs logs` showing Epoch progress + loss values.

## WHAT IS RUNNING RIGHT NOW

The `run_factorial.sh --resume` loop submitted conditions with a Docker image
containing **SAM3 gradient checkpointing** (`gradient_checkpointing_enable(use_reentrant=False)`).

| Status | Count | Details |
|--------|-------|---------|
| SUCCEEDED | 5+ | DynUNet conditions (IDs 53-56, 75+) |
| PENDING (no VM) | ~10 | SAM3 + DynUNet + MambaVesselNet — waiting for L4 spot capacity |
| FAILED | 2+ | SAM3 GC-enabled IDs 72-73 (**INVESTIGATE THESE FIRST**) |
| STARTING/RECOVERING | 2 | SAM3 IDs 70-71 |

**No SAM3 job has SUCCEEDED yet.** The two FAILED GC-enabled SAM3 jobs (72, 73)
are the FIRST PRIORITY — check their logs to determine if gradient checkpointing
activated and whether the failure was OOM, spot preemption, or something else.

**WARNING**: Multiple `run_factorial.sh` processes may be running from prior launch
attempts. Only the most recent matters — earlier ones submit duplicate conditions that
SkyPilot `--resume` will skip. Consider killing stale processes:
`kill $(ps aux | grep run_factorial | grep -v grep | awk '{print $2}')` and
relaunching a single clean instance.

**Branch**: `test/run-debug-9th-pass-report`
**Latest commit**: `2de98d5b` — gradient checkpointing implementation
**Docker image**: GAR `base:latest` digest `sha256:08d6bfaa` (2026-03-25 ~20:50 UTC)
**Config**: `configs/factorial/debug.yaml` with `gradient_checkpointing: true` in SAM3 model_overrides

## IMMEDIATE ACTION: Check SAM3 Job Logs

This is the **critical validation** — has SAM3 TopoLoRA EVER completed a training
iteration? Across 9 passes and 50+ SAM3 job attempts, zero training iterations have
completed. Gradient checkpointing should reduce VRAM from ~22 GiB to ~10 GiB
(unvalidated estimate) on L4.

```bash
# Step 1: Find SAM3 jobs from this launch (IDs >= 69)
.venv/bin/sky jobs queue | grep sam3

# Step 2: For SUCCEEDED jobs — VERIFY ACTUAL TRAINING (not just job duration)
.venv/bin/sky jobs logs <JOB_ID> | grep -E "gradient checkpointing|Epoch|train/loss|VRAM"
# MUST see: "SAM3 encoder gradient checkpointing ENABLED (non-reentrant)"
# MUST see: "Epoch 1/2" with loss values
# MUST NOT see: "CUDA out of memory"

# Step 3: For FAILED jobs — check error
.venv/bin/sky jobs logs <JOB_ID> | tail -30
```

**METALEARNING RULE**: Do NOT claim "fixed" until `sky jobs logs` shows actual
training output (Epoch progress + loss values). Job duration ≠ training time.
See: `.claude/metalearning/2026-03-25-overconfident-oom-fixed-claim.md`

## WHAT WAS DONE IN THE 9TH PASS SESSION

### SAM3 OOM Debugging (the core struggle)

| Pass | What happened | Root cause | Fix applied |
|------|--------------|------------|-------------|
| 9th v1 | All SAM3 OOM, claimed "fixed" | FP32 forward in `check_gradient_flow()` | Added autocast to all 4 pre-training check functions |
| 9th v2 | Still OOM WITH autocast | Model too large for L4 even in AMP (21.67/21.96 GiB) | Gradient checkpointing + skip diagnostic gradient flow check |
| 9th v3 | **CURRENTLY RUNNING** — awaiting validation | N/A | HF `gradient_checkpointing_enable(use_reentrant=False)` |

### Session Deliverables (all committed and pushed)

| Category | What | Tests |
|----------|------|-------|
| SAM3 BS=1 + gradient accumulation | model_overrides, trainer accum loop, OOM detection, VRAM estimator | 80+ |
| Cloud robustness (23 issues #942-#964) | Preflight gates, YAML hardening, lockfile, resume fix, checkpoints, monitoring | 159 |
| Security hardening | Trivy→Grype, pip-audit, SHA-256 weight pinning, torch.load audit | 44 |
| SAM3 pre-training autocast | autocast in 4 check functions, FP32 upcast for gradient flow | 6 |
| SAM3 gradient checkpointing | HF native gradient_checkpointing_enable(), skip diagnostic, full wiring | 16 |
| **Total new tests** | | **6398 staging, 0 fail, 0 skip, 0 xfail** |

### Key Metalearning Documents

1. `.claude/metalearning/2026-03-25-overconfident-oom-fixed-claim.md` — **CRITICAL**: Job duration ≠ training time. BANNED: "confirmed training" without checking logs.
2. `.claude/metalearning/2026-03-25-mlflow-413-never-actually-fixed.md` — MLflow HTTP 413 "addressed" in 3 passes, never deployed. BANNED: "fix in place, deployment pending."
3. `.claude/metalearning/2026-03-25-stale-docker-image-launch.md` — Launched with old Docker image. Docker freshness gate now implemented.
4. `.claude/metalearning/2026-03-25-xfail-dismissal-as-pre-existing.md` — xfails are bugs, not features.

### Issues Created and Closed

- **#940**: SAM3 OOM fix (closed — BS=1 + gradient accumulation implemented)
- **#942-#964**: 23 cloud robustness issues (all closed — implemented)
- **#966**: A100 upgrade option (OPEN — P2, with decision matrix)

### Unresolved P0 Blockers (depends on gradient checkpointing validation)

| Blocker | Status | What's needed |
|---------|--------|--------------|
| SAM3 TopoLoRA → SUCCEEDED | **AWAITING** | Check sky jobs logs after GC jobs run |
| SAM3 Hybrid → SUCCEEDED | **AWAITING** | Never submitted in any pass |
| Zero-shot baselines → SUCCEEDED | **AWAITING** | Never submitted in any pass |
| MLflow HTTP 413 | **UNRESOLVED** | google-cloud-storage IS installed (via dvc[gs]). Root cause unknown — investigate Pulumi deploy state and MLflow server config |
| SWAG artifact upload | **BLOCKED BY 413** | Depends on MLflow artifact store fix |

## WHAT TO DO NEXT

### If SAM3 TopoLoRA SUCCEEDED with gradient checkpointing:

1. Verify training logs show actual epochs + loss values
2. Record VRAM peak from logs (expected ~10 GiB)
3. Update model profile VRAM data in `configs/model_profiles/sam3_topolora.yaml`
4. Update experiment report: `docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-report-9th-pass-v2-SAM-zeroshot-swag-focus.md`
5. Investigate MLflow 413 (the next blocker):
   - `cd deployment/pulumi/gcp && pulumi stack output` — check MLFLOW_DEFAULT_ARTIFACT_ROOT
   - `sky jobs logs <DynUNet_JOB_ID> | grep "413\|artifact\|mlflow"` — find actual error
6. Run full debug factorial to completion (32 training + 2 zero-shot = 34 conditions)
7. Create PR → merge to main → promote to prod

### If SAM3 TopoLoRA STILL OOMs with gradient checkpointing:

1. Check logs: `sky jobs logs <JOB_ID> | grep -E "gradient checkpointing|CUDA out of memory"`
2. Verify "gradient checkpointing ENABLED" appears in logs (if not, the feature didn't activate)
3. If activated but still OOM: the model is genuinely too large for L4 even with GC
   - Options: A100 (Issue #966), smaller patch_size, DeepSpeed ZeRO
   - Decision matrix in Issue #966

### If SAM3 TopoLoRA jobs are STILL PENDING after hours:

1. Check `sky jobs queue` — if TOT. DURATION >> JOB DURATION (or JOB DURATION is `-`), the VM never provisioned
2. This means L4 spot capacity is exhausted in all configured regions
3. Options:
   - Wait (resilient wrapper will keep retrying)
   - Check if on-demand is acceptable: modify SkyPilot YAML `use_spot: false` (costs 3x more)
   - Check if A100 spot is available: Issue #966 has the full decision matrix
4. If MambaVesselNet jobs are ALSO stuck PENDING → capacity issue, not SAM3-specific

### If SAM3 TopoLoRA FAILED but NOT OOM:

1. Check actual error in logs — could be HF download issue, data issue, etc.
2. The HF repo was fixed: `facebook/sam3-hiera-large` → `facebook/sam3`
3. Check if the SkyPilot YAML has the correct HF repo name

## FILES TO READ FIRST

1. `docs/planning/v0-2_archive/original_docs/run-debug-factorial-experiment-report-9th-pass-v2-SAM-zeroshot-swag-focus.md` — current report
2. `.claude/metalearning/2026-03-25-overconfident-oom-fixed-claim.md` — MANDATORY before claiming anything about SAM3
3. `docs/planning/v0-2_archive/original_docs/sam3-gradient-checkpointing-plan.xml` — the plan being validated
4. `docs/planning/cold-start-prompt-8th-pass-backlog-remaining-18-tasks.md` — 18 backlog tasks still pending

## GIT STATE

```
Branch: test/run-debug-9th-pass-report
Latest: 2de98d5b feat: SAM3 gradient checkpointing — VRAM ~22 GiB → ~10 GiB on L4 (#966)
Main:   21d227a7 (PR #965 merged — 203 cloud robustness tests + security hardening)
Prod:   21d227a7 (reset to main)
```

Note: The gradient checkpointing commits are NOT yet on main — they're on the report branch.
After validation, create PR → merge to main.
