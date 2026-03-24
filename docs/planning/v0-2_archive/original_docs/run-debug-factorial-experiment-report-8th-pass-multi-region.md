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
| 14:29 | Job 18 provisioning | europe-west4-a: `insufficientCapacity` → failover to -b |
| ~14:35 | Job 18 RUNNING | Provisioned in europe-west4-b |
| ~14:40 | Job 18 preempted | Spot preemption after ~5 min → RECOVERING |
| ~16:20 | Job 20 SUCCEEDED | dynunet/cbdice_cldice/calib=true — first SUCCEEDED job ever! |
| ~16:30 | Job 21 SUCCEEDED | dynunet/dice_ce/calib=false |
| ~16:40 | Job 22 SUCCEEDED | dynunet/dice_ce_cldice/calib=true |
| ~17:00 | Job 24 SUCCEEDED | dynunet/bce_dice_05cldice/calib=true |
| ~17:30 | Job 18 SUCCEEDED | dynunet/cbdice_cldice/calib=false (1 recovery from preemption) |
| ~17:30 | Job 19 SUCCEEDED | dynunet/dice_ce/calib=true |
| 17:48 | Job 27 STARTING | **MambaVesselNet first cloud submission** |
| 18:01 | Job 27 RUNNING | **MambaVesselNet mamba-ssm compiles on L4!** (8 min job duration) |
| ongoing | 6/34 SUCCEEDED | Sequential execution, GPUS_ALL_REGIONS=1 |

## Critical Infrastructure Findings

### FINDING 1: GPUS_ALL_REGIONS quota = 1 (CRITICAL BOTTLENECK)

```
gcloud compute project-info describe → GPUS_ALL_REGIONS: 1.0/1.0
europe-west4 → PREEMPTIBLE_NVIDIA_L4_GPUS: 1.0/1.0
```

The GCP project has a **global GPU quota of 1**. Only ONE GPU job can be provisioned at
a time across ALL regions. Parallel submission is correct (queues them), but provisioning
is strictly sequential. This means:
- 34 jobs × ~10 min each = ~5.6 hours minimum (ignoring provisioning overhead)
- Actual: ~15 min/job (10 min run + 5 min provisioning) = ~8.5 hours total
- **Action**: Request quota increase to 4-8 GPUs to enable true parallelism

### FINDING 2: 413 Request Entity Too Large — MLflow artifact uploads fail (HIGH)

**ALL checkpoint uploads to MLflow via Cloud Run fail with HTTP 413.**

- `best_val_loss.pth`: 67.7 MB (includes optimizer state)
- Cloud Run default max request body: 32 MB
- Affects: checkpoint artifacts AND `swag_model.pt` (post-training)
- Metrics (small JSON payloads) log correctly — only large binary artifacts fail

**Checkpoints ARE safe** — the GCS `MOUNT_CACHED` file_mount works correctly:
```
gs://minivess-mlops-checkpoints/dynunet_cbdice_cldice/fold_0/best_val_loss.pth  67.7 MB ✓
gs://minivess-mlops-checkpoints/dynunet_cbdice_cldice/fold_0/epoch_latest.pth   22.6 MB ✓
gs://minivess-mlops-checkpoints/dynunet_cbdice_cldice/fold_0/last.pth           67.7 MB ✓
gs://minivess-mlops-checkpoints/dynunet_cbdice_cldice/fold_0/metric_history.json  1.9 KB ✓
```

**Root cause**: MLflow on Cloud Run uses HTTP artifact uploads. Cloud Run's ingress
proxy rejects bodies > 32 MB. Two fix paths:
1. **Increase Cloud Run max request size** (Pulumi: `max_instance_request_concurrency`
   won't help — need `--max-request-body-size` or nginx config)
2. **Configure MLflow to use GCS-backed artifact store** instead of HTTP uploads.
   This is the correct long-term fix — artifacts go directly to GCS, bypassing Cloud Run.

### FINDING 3: Calibration metrics crash (MEDIUM)

```
WARNING | minivess.pipeline.metrics - Failed to compute calibration metrics
torch.cat(self._cal_probs).numpy().ravel()  →  MonAI MetaTensor error
```

All 3 succeeded jobs show this warning. `val/nll` is `nan` in all runs.
Root cause: MONAI's `MetaTensor` doesn't support `.numpy()` directly after `torch.cat()`.
Fix: `.cpu().detach().numpy()` or convert to plain tensor before cat.

### FINDING 4: WeightWatcher crash on 3D convolutions (LOW)

```
WARNING | minivess.diagnostics.weight_diagnostics - WeightWatcher crashed on model layers
(likely 3D conv with weight=None). Returning NaN metrics.
```

Non-blocking. WeightWatcher doesn't support 3D convolutions natively.
The diagnostic returns NaN metrics which is the correct fallback behavior.

### FINDING 5: MLflow URL auth encoding (COSMETIC)

MLflow artifact URLs contain literal `%24%7BMLFLOW_TRACKING_USERNAME%7D` (URL-encoded
`${MLFLOW_TRACKING_USERNAME}`). This happens because `MLFLOW_TRACKING_USERNAME` and
`MLFLOW_TRACKING_PASSWORD` are not in `.env` — SkyPilot passes the literal YAML
placeholder `${MLFLOW_TRACKING_USERNAME}` as the env var value.

Since the MLflow Cloud Run instance doesn't require auth, the fix is to either:
- Remove `MLFLOW_TRACKING_USERNAME`/`PASSWORD` from the SkyPilot YAML envs
- Or add empty values to `.env`: `MLFLOW_TRACKING_USERNAME=` / `MLFLOW_TRACKING_PASSWORD=`

## Observations

### Region provisioning (CONFIRMED WORKING)
- **Multi-region failover works**: SkyPilot correctly loads all 6 regions from `ordered:` block
- **europe-west4-b** was the first successful provision (europe-west4-a had insufficientCapacity)
- **3 jobs SUCCEEDED** — first successful training runs in 8 passes!
- **quotaExceeded** errors when >1 job tries to provision simultaneously (GPUS_ALL_REGIONS=1)
- SkyPilot correctly handles quota exhaustion by retrying after the running job finishes
- All jobs so far ran in **europe-west4** (closest to GCS data in europe-north1)

### Training performance (DynUNet, 2 epochs, L4 GPU)
| Condition | train/loss | val/loss | val/dice | VRAM (MB) | Epoch time (s) |
|-----------|-----------|----------|----------|-----------|----------------|
| cbdice_cldice/calib=true | 1.127 | 1.112 | 0.486 | 4017 | 67 |
| dice_ce/calib=false | 0.823 | 0.903 | 0.304 | 2895 | 56 |
| dice_ce_cldice/calib=true | 1.260 | 1.315 | 0.301 | 4013 | 66 |

Key observations:
- **No OOM** — DynUNet uses 2.9-4.0 GB on L4 (24 GB). Plenty of headroom.
- **cbdice_cldice has best val/dice** (0.486 vs 0.30) after just 2 epochs
- **Calibration adds ~1.1 GB VRAM** (4017 vs 2895 MB)
- **Epoch time**: 56-67 seconds (includes validation)
- **Inference latency**: 1.9-2.8 seconds per volume
- **GPU utilization**: 28-100% (varies by condition — clDice computation is CPU-bound)

### GCS checkpoint persistence (CONFIRMED WORKING)
- All 3 succeeded jobs have checkpoints in `gs://minivess-mlops-checkpoints/`
- `MOUNT_CACHED` mode works correctly — async upload, non-blocking writes
- SHA256 checksums present for integrity verification
- `metric_history.json` logged per fold

### SkyPilot API server stale path (fixed)
- After repo rename (`minivess-mlops` → `vascadia`), SkyPilot API server cached old venv path
- Had to `rm -rf ~/.sky/api_server/` and restart fresh

## Cost Tracking

| Item | Estimate | Actual |
|------|----------|--------|
| L4 spot (per job, europe-west4) | ~$0.34/hr | $0.34/hr (g2-standard-4) |
| Job duration (DynUNet, 2 epochs) | ~10 min | 9-10 min |
| Cost per job | ~$0.06 | ~$0.06 |
| Total GPU jobs | 34 | 3 succeeded, 31 remaining |
| Estimated total GPU cost | ~$2.04 | Ongoing |
| Controller (n4-standard-4) | ~$0.15/hr | Running since 14:26 |
| Provisioning overhead | negligible | ~5 min/job (quota contention) |

## Optimization Opportunities (Prioritized)

### P0: Request GCP GPU quota increase
`GPUS_ALL_REGIONS=1` forces serial execution. With quota=4, we'd provision 4 jobs
simultaneously and cut total runtime from ~8.5 hr to ~2.5 hr. The cost is identical
(same total GPU-hours), but wall-clock time drops 3.5x.
```bash
gcloud compute project-info add-metadata --project=minivess-mlops \
  --metadata=GPUS_ALL_REGIONS=4  # This doesn't work — need console quota request
```
**Action**: Request quota increase via GCP Console → IAM & Admin → Quotas.

### P1: Configure MLflow GCS artifact store
The 413 error is a dead end with Cloud Run's HTTP proxy. The correct fix is
`--default-artifact-root gs://minivess-mlops-mlflow-artifacts` on the MLflow server.
This makes artifact uploads go directly to GCS (no HTTP body size limit).
Currently: checkpoints persist via GCS mount (working), but MLflow can't reference them.
**Action**: Update Pulumi MLflow Cloud Run config with GCS artifact root.

### P2: Fix calibration metrics crash
`torch.cat()` on MONAI `MetaTensor` → `.numpy()` fails. Need `.cpu().detach().numpy()`.
This causes `val/nll=nan` in all runs.
**Action**: Fix `src/minivess/pipeline/metrics.py:123`.

### P3: Clean up MLflow auth env vars
Remove `MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` from SkyPilot YAML since
the Cloud Run MLflow instance doesn't require auth. The literal `${...}` placeholders
create ugly URL-encoded auth in artifact URLs.
**Action**: Remove from `deployment/skypilot/train_factorial.yaml` envs section.

### P4: Stale state cleanup after repo rename
Need a `make clean-rename` target that clears `.venv`, `__pycache__`,
`~/.sky/api_server/`, and any other cached paths.

### P5: SkyPilot `allowed_clouds` warning suppression
Server and client configs differ. Should sync `.sky.yaml` `allowed_clouds`
with server-side config to suppress the warnings.
