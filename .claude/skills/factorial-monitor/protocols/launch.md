# Protocol: Launch

## Pre-Launch Verification

Before launching ANY factorial experiment:

1. **Verify Docker image exists and is fresh:**
   ```bash
   docker manifest inspect <image:tag>
   ```

2. **Verify SkyPilot YAML uses Docker (Rule #17):**
   ```bash
   grep "image_id:" deployment/skypilot/train_factorial.yaml
   ```
   Must contain `docker:` prefix. Bare VM setup is BANNED.

3. **Verify .env populated:**
   Required: `HF_TOKEN`, `MLFLOW_TRACKING_URI`, GCP credentials.

4. **Verify DVC data accessible:**
   ```bash
   dvc status -r gcs
   ```

5. **Estimate cost:**
   ```
   N_jobs x estimated_hours x hourly_rate = total_estimate
   ```
   Report to user before launching. Get explicit approval if >$20.

## Launch Execution

```bash
./scripts/run_factorial.sh <config.yaml>
```

The script loops through all factorial conditions and calls `sky jobs launch` for each.

## Manifest Creation

After launch, parse SkyPilot output to build `factorial_manifest.json`:

```bash
sky jobs queue
```

Map each job_id to its condition (model, loss, aux_calib, fold) using the launch
order from the config YAML.

Save manifest to: `outputs/factorial_manifest_<experiment_id>.json`

## Transition to MONITOR

Once all jobs are launched and the manifest is populated, enter the MONITOR phase.
From this point, the workspace is READ-ONLY (Rule F1).
