# Cold-Start Prompt: Debug Factorial Run on GCP

## Date: 2026-03-20
## Branch: test/debug-factorial-run (branched from main after PR #871 merge)
## Execution Mode: INTERACTIVE (user available for decisions)

---

## To run in a new session:

```bash
claude -p "Read and execute the plan at:
/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/cold-start-prompt-debug-factorial-run.md
Branch: test/debug-factorial-run." --dangerously-skip-permissions
```

---

## MANDATORY READING BEFORE ANY ACTION

Read these files before starting:

1. `/home/petteri/Dropbox/github-personal/minivess-mlops/CLAUDE.md` — All rules
2. `/home/petteri/Dropbox/github-personal/minivess-mlops/docs/planning/run-debug-factorial-experiment.xml` — The execution plan (7 phases)
3. `/home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/factorial-monitor/SKILL.md` — The monitoring skill
4. `/home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/factorial-monitor/protocols/launch.md` — Launch protocol
5. `/home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/factorial-monitor/protocols/monitor.md` — Monitor protocol
6. `/home/petteri/Dropbox/github-personal/minivess-mlops/.claude/skills/factorial-monitor/protocols/diagnose.md` — Diagnosis protocol
7. `/home/petteri/Dropbox/github-personal/minivess-mlops/scripts/run_factorial.sh` — The deterministic launch script
8. `/home/petteri/Dropbox/github-personal/minivess-mlops/deployment/skypilot/train_factorial.yaml` — SkyPilot per-condition YAML
9. `/home/petteri/Dropbox/github-personal/minivess-mlops/configs/experiment/debug_factorial.yaml` — Debug experiment config
10. `/home/petteri/Dropbox/github-personal/minivess-mlops/knowledge-graph/decisions/L3-technology/paper_model_comparison.yaml` — 6-model lineup
11. `/home/petteri/Dropbox/github-personal/minivess-mlops/docs/reports/2026-03-20-prod-test-suite-report.md` — Latest test report

---

## CONTEXT: What This Branch Does

This branch runs the **debug factorial experiment** on GCP via SkyPilot.
The debug run is the FULL production experiment with ONLY 3 differences:

| Parameter | Debug | Production |
|-----------|-------|-----------|
| Epochs | 2 | 50 |
| Data | Half (~23 train / ~12 val) | Full (~47 / ~23) |
| Folds | 1 (fold-0) | 3 |
| **Everything else** | **IDENTICAL** | **IDENTICAL** |

### Experiment Design

- **Trainable factorial**: 4 models × 3 losses × 2 aux_calib = **24 conditions** on fold-0
  - Models: dynunet, mambavesselnet, sam3_topolora, sam3_hybrid
  - Losses: cbdice_cldice, dice_ce, dice_ce_cldice
  - Aux calibration: with/without hL1-ACE
- **Zero-shot baselines**: SAM3 Vanilla (MiniVess fold-0) + VesselFM (DeepVess only)
- **Total**: 26 conditions
- **Estimated cost**: ~$6-10 on GCP L4 spot

### Two-Layer Architecture

```
Layer 1 (Deterministic): scripts/run_factorial.sh
  - Pure sky jobs launch calls in a loop
  - ANY researcher can run this WITHOUT Claude Code
  - NO claude -p, NO screen, NO nohup

Layer 2 (Monitoring): Claude Code with /factorial-monitor Skill
  - The "Claude Harness" around the deterministic .sh script
  - Monitors SkyPilot job status, diagnoses failures
  - Plans fixes with reviewer agents if bugs found
  - After training: triggers downstream flows (post-training, analysis, biostatistics, deploy)
```

---

## EXECUTION PLAN

### Phase 0: Pre-Flight Verification

Run ALL of these checks before spending cloud credits:

```bash
# 1. Verify GCP access
sky check gcp

# 2. Verify GAR image exists
docker manifest inspect europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest

# 3. Verify GCP resources via Pulumi
cd deployment/pulumi/gcp && pulumi stack output

# 4. Verify code is clean
make test-staging

# 5. Verify .env has required variables
grep -q "HF_TOKEN" .env && echo "HF_TOKEN: set" || echo "ERROR: HF_TOKEN missing"
grep -q "MLFLOW_TRACKING_URI" .env && echo "MLFLOW_TRACKING_URI: set" || echo "ERROR: missing"

# 6. Verify debug config parses
python3 -c "import yaml; c=yaml.safe_load(open('configs/experiment/debug_factorial.yaml', encoding='utf-8')); print(f'Models: {len(c[\"factors\"][\"model_family\"])}, Losses: {len(c[\"factors\"][\"loss_name\"])}, Conditions: {len(c[\"factors\"][\"model_family\"]) * len(c[\"factors\"][\"loss_name\"]) * len(c[\"factors\"][\"aux_calibration\"])}')"

# 7. Verify SkyPilot YAML tests pass
uv run pytest tests/v2/unit/test_train_factorial_yaml.py -v

# 8. Dry-run the launch script
./scripts/run_factorial.sh --dry-run configs/experiment/debug_factorial.yaml
```

**GATE**: ALL checks must pass. If ANY fails, STOP and fix before proceeding.

### Phase 1: Launch Training Conditions

```bash
# Launch all 26 conditions (24 trainable + 2 zero-shot)
./scripts/run_factorial.sh configs/experiment/debug_factorial.yaml
```

The script will:
- Parse `debug_factorial.yaml` for all factorial factors
- Launch each condition via `sky jobs launch` with `--name` per condition
- Rate-limit with `sleep 5` between launches
- Log job IDs to `outputs/{timestamp}_factorial_job_ids.txt`
- Handle per-condition launch failures (one failure doesn't abort all)

### Phase 2: Monitor with /factorial-monitor

After launching, invoke the `/factorial-monitor` skill to monitor all 26 conditions:

```
/factorial-monitor
```

The skill will:
- Poll `sky jobs queue` every 60s
- Print live status table: `| condition | job_id | status | duration |`
- Wait until ALL jobs reach terminal state (Rule F1: WAIT-FOR-TERMINAL)
- For each failure: capture logs with `sky jobs logs JOB_ID`
- Expected: ~5-10 min per condition on L4 (2 epochs, half data)

Watch for:
- **STARTING timeout** (>15 min): Docker pull or quota issue
- **FAILED_SETUP**: DVC pull failure, HF login failure, data not found
- **FAILED**: OOM, NaN, config error, infrastructure issue
- **Spot preemption**: SkyPilot auto-recovers, no action needed

### Phase 3: Diagnose Failures (if any)

If any conditions fail, the `/factorial-monitor` skill aggregates by root cause:

```
{root_cause → [job_ids], fix_strategy, affected_files, confidence}
```

Common failure categories:
- **OOM**: Check model VRAM profile, adjust patch_size or batch_size
- **NaN**: Check BF16/FP16 dtype, loss function, learning rate
- **Config error**: Fix config, rebuild Docker, re-launch failed conditions
- **Infrastructure**: Docker pull, DVC pull, HF auth, quota

### Phase 4: Fix + Re-Launch (if needed)

If code fixes are needed:
1. Use `/self-learning-iterative-coder` for TDD fix
2. Rebuild Docker image: `make build-base-gpu && make push-gar`
3. Re-launch ONLY failed conditions (not all 26)
4. Maximum 2 fix-relaunch cycles (Rule F4)

### Phase 5: Verify Training Artifacts

After all 26 conditions succeed, verify MLflow contains ALL expected data:

For each of 24 trainable conditions:
- MLflow run exists with status=FINISHED
- Params: model/family, train/loss_name, train/max_epochs, with_aux_calib
- Metrics: train/loss, val/loss, val/dice (per epoch)
- Artifacts: config/resolved_config.yaml, checkpoints/best_*.pth

For 2 zero-shot baselines:
- SAM3 Vanilla: eval metrics on MiniVess fold-0
- VesselFM: test/deepvess/* metrics on DeepVess

Critical check: `with_aux_calib=true` vs `false` should produce DIFFERENT loss values.

### Phase 6: Downstream Flows (Local Docker Compose)

After training artifacts are verified, run downstream flows locally:

```bash
# Start infrastructure
docker compose -f deployment/docker-compose.yml --profile dev up -d

# Post-training (SWA, calibration)
docker compose --env-file .env -f deployment/docker-compose.flows.yml run --rm post_training

# Analysis (evaluation on MiniVess + DeepVess external test)
docker compose --env-file .env -f deployment/docker-compose.flows.yml run --rm --shm-size 8g analyze

# Biostatistics (ANOVA, pairwise, specification curve — both trainval and test splits)
docker compose --env-file .env -f deployment/docker-compose.flows.yml run --rm biostatistics

# Deploy (champion discovery, ONNX export, BentoML import — no cloud deploy)
docker compose --env-file .env -f deployment/docker-compose.flows.yml run --rm deploy
```

### Phase 7: Debug Run Report

Generate report at `docs/reports/debug-factorial-run-report.md`:
- Total cost (from SkyPilot job logs)
- Wall-clock time per condition
- Failures encountered and fixes applied
- MLflow artifact completeness
- Biostatistics output quality
- Production run cost estimate

---

## CRITICAL RULES

1. **The .sh is the product** — `run_factorial.sh` is deterministic and LLM-free. Claude monitors but never wraps it.
2. **Debug = production** — 26 conditions, no shortcuts (CLAUDE.md Rule 27)
3. **T4 BANNED** — L4 only (CLAUDE.md)
4. **Docker image_id ONLY** — No bare VM setup (CLAUDE.md Rule 17)
5. **No screen/nohup** — Pure `sky jobs launch` calls (metalearning 2026-03-09, 2026-03-16)
6. **WAIT-FOR-TERMINAL** — Don't panic fix while jobs are running (Rule F1)
7. **AGGREGATE-BEFORE-FIX** — Group failures by root cause, then batch fix (Rule F2)
8. **MAX-TWO-CYCLES** — Hard stop after 2 fix-relaunch cycles (Rule F4)
9. **VesselFM = DeepVess ONLY** — Data leakage on MiniVess
10. **Session summaries ≠ authorization** — ASK before infrastructure changes

---

## WHAT NOT TO DO

- Do NOT wrap `run_factorial.sh` in `claude -p` or `screen`
- Do NOT fix one job at a time (whac-a-mole) — aggregate all failures first
- Do NOT re-launch all 26 conditions — only failed ones
- Do NOT use T4 GPU
- Do NOT use bare VM setup in SkyPilot YAML
- Do NOT bypass Docker — all execution through containers
- Do NOT modify Pulumi/cloud infrastructure without asking user
- Do NOT run more than 2 fix-relaunch cycles without user approval
