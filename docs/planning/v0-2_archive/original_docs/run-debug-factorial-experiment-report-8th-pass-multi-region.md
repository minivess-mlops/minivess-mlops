# Debug Factorial Experiment Report — 8th Pass (Multi-Region)

**Date**: 2026-03-24
**Branch**: `test/run-debug-gcp-5th-pass`
**Config**: `configs/factorial/debug.yaml`
**Commit**: `b8bc912` (dynamic region injection + zero-skip enforcement)

## Key Change: Multi-Region L4 Failover

This is the FIRST pass using dynamic `ordered:` region injection.
Previous passes (5th–7th) were stuck on europe-north1 (NO L4 GPUs) for 12+ hours.

**Region priority (from `configs/cloud/regions/europe_us.yaml`)**:
1. europe-west4 (Netherlands, 3 zones) — closest to GCS data
2. europe-west1 (Belgium, 2 zones) — SkyPilot controller location
3. europe-west3 (Frankfurt, 2 zones)
4. us-central1 (Iowa, 3 zones) — US fallback
5. us-east1 (South Carolina, 3 zones)
6. us-west1 (Oregon, 3 zones)

**US egress cost**: ~$0.12/GB × 3 GB = ~$0.36/job (acceptable).

## Factorial Design

| Factor | Levels | Count |
|--------|--------|-------|
| Models | dynunet, mambavesselnet, sam3_topolora, sam3_hybrid | 4 |
| Losses | cbdice_cldice, dice_ce, dice_ce_cldice, bce_dice_05cldice | 4 |
| Aux calibration | true, false | 2 |
| Post-training | none, swag (iterated inside each job) | 2 |
| Folds | 0 (debug: 1 fold only) | 1 |

**Training cells**: 4 × 4 × 2 × 1 = 32 GPU jobs
**Zero-shot baselines**: sam3_vanilla (minivess) + vesselfm (deepvess) = 2 GPU jobs
**Total GPU jobs**: 34

### Debug reductions (Rule 27: same as production except these 3):
- **Epochs**: 2 (production: 50)
- **Data**: 23 train / 12 val (production: 47/23)
- **Folds**: 1 (production: 3)

## Infrastructure

- **GPU**: L4:1 (spot), multi-region failover via `ordered:` block
- **Image**: `europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest`
- **Controller**: GCP europe-west1 (n4-standard-4, already UP)
- **MLflow**: Cloud Run (`https://minivess-mlflow-a7w6hliydq-lz.a.run.app`)
- **Checkpoints**: GCS mount (`gs://minivess-mlops-checkpoints`, MOUNT_CACHED)
- **Data**: DVC pull from `gs://minivess-mlops-dvc-data` in setup phase
- **Resilient wrapper**: `run_factorial_resilient.sh` with --resume

## Previous Passes Summary

| Pass | Date | Result | Root cause |
|------|------|--------|-----------|
| 1st | 2026-03-23 | CANCELLED | Controller on RunPod (wrong cloud) |
| 2nd | 2026-03-23 | CANCELLED | Unauthorized A100-80GB added |
| 3rd | 2026-03-23 | CANCELLED | Controller zone-hopping (no region pin) |
| 4th | 2026-03-23 | CANCELLED | sync_sky_config.py wrote conflicting cloud: key |
| 5th–7th | 2026-03-23–24 | CANCELLED (12+ hr PENDING) | europe-north1 has NO L4 GPUs |
| **8th** | **2026-03-24** | **LAUNCHING** | **Multi-region failover (this pass)** |

## Launch Log

### Pre-launch checks
- [x] `make test-prod`: 6378 passed, 0 skipped, 0 failed
- [x] Zero-skip enforcement active in conftest.py
- [x] Region injection generates valid SkyPilot YAML (16 tests)
- [x] `sky.Task.from_yaml()` validates generated YAML
- [x] GCP ADC credentials active
- [x] MLflow server reachable (Cloud Run)
- [x] SkyPilot controller UP in europe-west1
- [x] Dry-run shows correct region injection
- [x] All previous jobs CANCELLED (zero SUCCEEDED)

### Launch command
```bash
nohup bash scripts/run_factorial_resilient.sh configs/factorial/debug.yaml &
```

### Job submission timeline

| Time (UTC) | Event | Details |
|------------|-------|---------|
| 14:20 | Launch started | Resilient wrapper PID 1187022 |
| 14:20 | Controller INIT | Stale API server from repo rename, needed fresh start |
| 14:26 | Controller UP | europe-west1-b, n4-standard-4 |
| 14:28 | First batch submitted | Jobs 18-21 (4 parallel, 5s rate limit) |
| 14:29 | Job 18 provisioning | europe-west4-a: `insufficientCapacity` → failover working |
| ~14:35 | Job 18 RUNNING | Successfully provisioned via multi-region failover |
| ~14:40 | Job 18 RECOVERING | Preempted after ~5 min, `job_recovery` handling it |
| TBD | All 34 submitted | Wrapper continues submitting as slots open |

## Observations

### Region provisioning (CONFIRMED WORKING)
- **Multi-region failover works**: SkyPilot correctly loads all 6 regions from `ordered:` block
- **europe-west4-a** had `insufficientCapacity` on first try → SkyPilot moved to next region
- **Job 18 actually ran for ~5 min** before preemption → proves L4 spots ARE available
- **RECOVERING state active** → `job_recovery.max_restarts_on_errors: 3` is working
- Previous passes: 12+ hr stuck PENDING on europe-north1 (no L4). Now: provisioned in <10 min

### SkyPilot API server stale path (fixed)
- After repo rename (`minivess-mlops` → `vascadia`), SkyPilot API server cached old venv path
- Had to `rm -rf ~/.sky/api_server/` and restart fresh
- Same root cause as the stale pycache skip issue — repo rename doesn't clear cached state

### Controller re-provision
- Controller (europe-west1-b) was UP from previous session but needed re-provisioning
- Network unreachable errors during startup (transient, self-resolved)
- Total controller re-provision time: ~6 minutes

### Training metrics
- Do any conditions OOM on L4? → TBD (first job preempted before completion)
- MLflow connectivity from non-EU regions? → TBD (waiting for US region provision)
- Checkpoint persistence via GCS mount? → TBD

## Cost Tracking

| Item | Estimate | Actual |
|------|----------|--------|
| L4 spot (per job) | ~$0.22/hr | TBD |
| Total GPU hours | ~34 × 0.5 hr = 17 hr | TBD |
| Total cost | ~$3.74 | TBD |
| Controller (n4) | ~$0.15/hr × runtime | Running since 14:26 |
| GCS egress (US) | ~$0.36/job × US jobs | TBD |

## Optimization Opportunities

1. **Stale state cleanup after repo rename**: Need a `make clean-rename` target that
   clears `.venv`, `__pycache__`, `~/.sky/api_server/`, and any other cached paths.
2. **SkyPilot `allowed_clouds` warning**: Server and client configs differ. Should sync
   `.sky.yaml` `allowed_clouds` with server-side config to suppress warnings.
3. **Parallel submission limit**: Currently 4. Could increase if controller handles it —
   but 4 is conservative and prevents API quota issues.
