# Cold-Start Prompt: 4th Pass Debug Factorial Re-Launch

Copy-paste this into a new Claude Code session:

---

Continue from branch: `test/debug-factorial-4th-pass`

## TASK: Re-launch 4th pass debug factorial on GCP after fixing all blockers

### WHAT HAPPENED (previous session, 2026-03-22)

First launch attempt burned ~$6.30 on 8 FAILED_SETUP jobs + 1 stuck job because:
1. `sky` binary not on PATH → FIXED (SKY_BIN fallback)
2. `job_recovery` field unsupported in SkyPilot v1.0 → FIXED (removed)
3. `dvc pull -r gcs` failed on `data/processed/minivess` (tracked, never pushed) → FIXED (path-specific pull)
4. No monitoring detected failures for 2+ hours → ISSUE (need batch monitoring)

All fixes committed and pushed. Preflight script implemented and passes (9/9 checks).
5662 staging tests pass. Local 3-flow dry run completed with 0 issues (8/8 conditions).

### BLOCKING TASKS (must do before re-launch)

1. **DeepVess data** — NOT on disk, NOT DVC-tracked, NOT on GCS.
   - Source: https://ecommons.cornell.edu/items/a79bb6d8-77cf-4917-8e26-f2716a6ac2a3
   - Config: `src/minivess/data/external_datasets.py` has the metadata
   - Need: download → `dvc add data/raw/deepvess` → `dvc push -r gcs`
   - Blocks: VesselFM zero-shot baseline (job 34 of 34)
   - Alternative: launch 32 trainable jobs first, add VesselFM later

2. **Re-launch factorial** with preflight:
   ```bash
   ./scripts/run_factorial.sh configs/factorial/debug.yaml
   ```
   - Preflight runs automatically (9 checks)
   - 32 trainable + 2 zero-shot = 34 jobs
   - Monitor with: `.venv/bin/sky jobs queue` or `sky dashboard`

3. **Write 4th pass report** during/after run:
   `docs/planning/run-debug-factorial-experiment-report-4th-pass.md`

### ALREADY DONE (do NOT re-implement)

- Factorial configs consolidated to `configs/factorial/` (4 losses, 640 conditions)
- `bce_dice_05cldice` loss (Khazem 2025 TopoLoRA paper weights)
- SWAG DataLoader wiring (both parent flow + standalone paths)
- SWAG mean model checkpoint (standard format for inference)
- `scripts/preflight_gcp.py` (9 checks, wired into run_factorial.sh)
- `tests/v2/unit/deployment/test_skypilot_yamls.py` (17 tests)
- `tests/v2/unit/test_dvc_remote_sync.py` (7 tests)
- DVC pull fix (path-specific: `dvc pull data/raw/minivess -r gcs`)
- SkyPilot YAML fixes (sky path, job_recovery removed, debug splits)
- Local DynUNet training via parent flow (training + SWAG sub-flows)
- Local 3-flow pipeline (8/8 conditions evaluated, 0 issues)

### OPEN ISSUES

- #907: Cloud GPU pipeline gaps (Prefect Cloud — deferred by user)
- #908: Local SkyPilot test suite (Tier 1 implemented, Tier 2 KinD deferred)

### KEY FILES

- `scripts/run_factorial.sh` — launch script (with preflight)
- `scripts/preflight_gcp.py` — 9-check pre-launch validation
- `deployment/skypilot/train_factorial.yaml` — per-job SkyPilot config
- `configs/factorial/debug.yaml` — 32-cell factorial (4×4×2)
- `docs/planning/run-debug-factorial-experiment-4th-pass.xml` — plan
- `docs/planning/run-debug-factorial-experiment-report-4th-pass-failure.md` — failure report

### FACTORIAL DESIGN (DO NOT RE-DERIVE — registry decision)

```
Layer A: 4 models × 4 losses × 2 aux_calib = 32 cells (GPU)
Layer B: 2 post-training methods {none, swag} (same GPU job)
Layer C: 2 recalibration × 5 ensemble (CPU)
Total: 640 conditions. Debug: 1 fold, 2 epochs, half data.
```

### METALEARNING (read before doing ANYTHING)

- `2026-03-22-dvc-pull-untested-setup-script-failure.md` — DVC incident
- `2026-03-22-planning-instead-of-implementing-silent-deferral.md` — NEVER plan when asked to implement
- `2026-03-22-hardcoded-execution-location-in-factorial-configs.md` — flows are location-agnostic
- `2026-03-22-debug-equals-production-8th-violation.md` — debug = production (Rule 27)
