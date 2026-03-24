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
*(Updated as jobs are submitted)*

| Time (UTC) | Event | Details |
|------------|-------|---------|
| TBD | Launch started | Resilient wrapper running |
| TBD | First batch submitted | 4 parallel, 5s rate limit |
| TBD | All 34 submitted | Or retry on partial failure |

## Observations

*(Updated during the run)*

### Region provisioning
- Which regions actually provision L4 spots?
- How long does cross-region GAR pull take?
- Does US fallback trigger, and what's the actual egress cost?

### Training metrics
- Do any conditions OOM on L4?
- MLflow connectivity from non-EU regions?
- Checkpoint persistence via GCS mount?

## Cost Tracking

| Item | Estimate | Actual |
|------|----------|--------|
| L4 spot (per job) | ~$0.22/hr | TBD |
| Total GPU hours | ~34 × 0.5 hr = 17 hr | TBD |
| Total cost | ~$3.74 | TBD |
| Controller (n4) | ~$0.15/hr × runtime | TBD |
| GCS egress (US) | ~$0.36/job × US jobs | TBD |

## Optimization Opportunities

*(Discovered during the run)*

1. TBD
