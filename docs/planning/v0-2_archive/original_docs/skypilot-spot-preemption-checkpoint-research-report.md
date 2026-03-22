# SkyPilot Spot Preemption and Checkpoint Resumption Research Report

**Date**: 2026-03-21
**Scope**: Factual research on SkyPilot spot recovery mechanisms, applicable to MinIVess MLOps
**Sources**: SkyPilot v0.5.0 docs, SkyPilot latest docs, GitHub discussion #1653, SkyNomad paper (arXiv 2601.06520v1)

---

## Q1: Does SkyPilot Automatically Restart Spot Jobs After Preemption? What Is the Mechanism?

**Yes, but ONLY when using `sky jobs launch` (managed jobs).** `sky launch --use-spot` does NOT provide auto-recovery.

### Mechanism

1. A **jobs controller** (small on-demand CPU VM) launches automatically with the first managed job.
2. The controller monitors the running spot instance.
3. On preemption, the controller **tears down the old cluster and provisions a new one** in another region/cloud.
4. The job restarts from scratch by default -- both `setup` and `run` re-execute on the new VM.
5. The controller autostops after 10 minutes of idle time. Cost: ~$0.4/hr running, <$0.004/hr stopped.

**Key quote from docs**: "Tear down the old temporary cluster and provision a new one in another region, then restart the job."

### Recovery Configuration (newer API)

The `job_recovery` field in `resources:` controls recovery behavior:

```yaml
resources:
  accelerators: A100:8
  use_spot: true
  job_recovery:
    strategy: EAGER_NEXT_REGION  # or FAILOVER
    max_restarts_on_errors: 3
    recover_on_exit_codes: [33, 34]
```

- `FAILOVER`: Restarts in same region first, moves to next region if unavailable.
- `EAGER_NEXT_REGION` (default): Moves to next region immediately on failure.
- `max_restarts_on_errors`: Limits restarts from non-zero exit codes (user code failures).
- `recover_on_exit_codes`: Exit codes that ALWAYS trigger recovery (not counted against max_restarts_on_errors).
- **Warning**: Exit code 137 should NOT be in `recover_on_exit_codes`.

### Launch Command Implications

| Command | Auto-Recovery | SSH Access | Use Case |
|---------|--------------|------------|----------|
| `sky jobs launch` | Yes | No | Long-running jobs, production |
| `sky launch --use-spot` | No | Yes | Interactive dev on spot instances |

**Our YAMLs currently use comments mentioning `sky jobs launch`** (correct), but the YAML itself does not include the `job_recovery` field.

---

## Q2: Does SkyPilot Support MOUNT_CACHED for GCS Buckets? How Does It Interact with file_mounts?

**Yes.** SkyPilot supports three storage modes, all compatible with GCS:

### Mode Comparison

| Aspect | MOUNT | COPY | MOUNT_CACHED |
|--------|-------|------|--------------|
| **Technology** | FUSE | Pre-fetch to disk | rclone VFS + local cache |
| **Read performance** | Slow (stream from remote) | Fast (local disk) | Fast (local cache + remote fallback) |
| **Write performance** | Slow | N/A (read-only) | Fast (async upload) |
| **Write support** | Yes (limited -- no random writes, no append) | No | Yes (all operations) |
| **Provisioning speed** | Fast | Slow | Fast |
| **Disk requirement** | None | Disk > bucket size | Cache must fit on disk |
| **Consistency** | Close-to-open (immediate cross-node) | Snapshot | Eventually consistent |
| **Upload timing** | Synchronous on close() | N/A | Asynchronous, guaranteed before task completion |

### MOUNT_CACHED Details (from latest docs)

```yaml
file_mounts:
  /app/checkpoints:
    source: gs://minivess-mlops-checkpoints
    mode: MOUNT_CACHED
    type: MODEL_CHECKPOINT_RW
```

**Workload type presets for MOUNT_CACHED**:
- `MODEL_CHECKPOINT_RO` -- Loading weights/checkpoints (read-only)
- `MODEL_CHECKPOINT_RW` -- Saving and loading checkpoints (read-write)
- `DATASET_RO` -- Reading datasets
- `DATASET_RW` -- Reading and writing datasets

**Custom configuration**:
```yaml
file_mounts:
  /app/checkpoints:
    source: gs://minivess-mlops-checkpoints
    mode: MOUNT_CACHED
    type: MODEL_CHECKPOINT_RW
    config:
      mount_cached:
        transfers: 16
        vfs_cache_max_size: "100G"
```

**Key behavior**:
- Writes are cached locally before asynchronous upload to bucket.
- `close()` does NOT guarantee the file is in the bucket yet.
- SkyPilot waits for all cached data to upload **before marking the task as complete**.
- If writes outpace uploads, the cache grows until disk space is exhausted.
- Write performance depends on `disk_tier` -- faster disks = better performance.

### MOUNT Limitations (our current mode)

MOUNT mode uses FUSE directly:
- **No random writes or append operations** -- this is a significant limitation for checkpoint files.
- File permissions are not guaranteed to be preserved.
- Synchronous: every `close()` uploads the entire file before returning.
- **Critical**: This means checkpoint saves block training until the upload completes.

---

## Q3: What Is the Best Practice for Checkpoint Resumption with SkyPilot Spot Jobs?

From the SkyPilot docs and GitHub discussion #1653:

### Architecture

```
Training Code
    ├── Periodic save → /checkpoint/ (mounted GCS bucket)
    └── On startup → Load latest from /checkpoint/ (if exists)

SkyPilot Controller
    ├── Monitors spot instance
    ├── On preemption → provisions new VM
    └── Re-mounts same GCS bucket → /checkpoint/
```

### Implementation Steps

1. **Mount a persistent cloud bucket** via `file_mounts` with `MOUNT` or `MOUNT_CACHED`.
2. **Application saves checkpoints periodically** to the mount point.
3. **Application reloads latest checkpoint on startup** -- this is the user's responsibility.
4. **Same bucket is re-mounted** on the recovery VM automatically.
5. **Use `$SKYPILOT_TASK_ID`** to group MLflow runs across preemption recoveries (stays constant).

### Official Example (from docs)

```yaml
file_mounts:
  /checkpoint:
    name: my-checkpoint-bucket
    mode: MOUNT_CACHED  # or MOUNT

run: |
  python train.py \
    --output_dir /checkpoint/model/ \
    --save_steps 1000 \
    --run_name $SKYPILOT_TASK_ID
```

### GitHub Discussion #1653 Key Points

- SkyPilot maintainers recommend using the Storage feature to mount cloud buckets.
- "The user program can save the checkpoint to the directory periodically and the checkpoint will stay in the cloud bucket even after preemption."
- "After the recovery of the spot VMs, the same cloud bucket will be mounted to the new VMs again."
- Cloud providers offer preemption warnings (AWS: 2 minutes, AliCloud: 5 minutes) that can trigger an emergency checkpoint save.

---

## Q4: Should Our Setup Block Detect Resume vs Fresh Start? Or Does SkyPilot Handle That?

**SkyPilot does NOT handle checkpoint loading -- that is 100% the application's responsibility.**

SkyPilot's role:
- Provisions new VM after preemption.
- Re-runs `setup` and `run` from scratch on the new VM.
- Re-mounts the same cloud bucket (checkpoints persist in GCS).
- Sets `$SKYPILOT_TASK_ID` to the same value across recoveries.

Application's role:
- Check for existing checkpoint in the mounted directory at the start of `run`.
- If found, resume from that checkpoint.
- If not found, start fresh.

### Our Current Implementation

`train_flow.py` already has `check_resume_state_task()` (line 263) that:
1. Looks for `epoch_latest.yaml` in `checkpoint_dir`.
2. Reads it with `yaml.safe_load()`.
3. Validates the referenced MLflow run is still `RUNNING`.
4. Returns the state dict if valid, `None` otherwise.

**Gap**: The resume detection is in the `run` command (train_flow.py), NOT in the `setup` block. This is **correct** -- setup should only prepare data/environment, not make training decisions.

**Gap**: The MLflow run status check (`run.info.status == "RUNNING"`) may be incorrect for spot recovery. After preemption, the old MLflow run may be stuck in `RUNNING` state (the process was killed, not gracefully terminated). This is actually **desirable** -- it indicates the run should be resumed. However, if the MLflow server is unreachable during recovery, this check will fail silently and start fresh.

---

## Q5: Does Our Current train_factorial.yaml Correctly Use SkyPilot's Spot Recovery Features?

**Partially. There are gaps.**

### What Is Correct

1. `use_spot: true` -- spot instances enabled.
2. `file_mounts` with GCS bucket -- checkpoint persistence exists.
3. Docker image via `image_id: docker:` -- compliant with Docker mandate.
4. `setup` is idempotent (checks if data already present).
5. Launch comment says `sky jobs launch` -- correct command for managed jobs.

### What Is Missing

| Gap | Current | Recommended | Priority |
|-----|---------|-------------|----------|
| **`job_recovery` field** | Not present | `job_recovery: {strategy: FAILOVER, max_restarts_on_errors: 3}` | P0 |
| **Storage mode** | `mode: MOUNT` | `mode: MOUNT_CACHED` with `type: MODEL_CHECKPOINT_RW` | P1 |
| **`$SKYPILOT_TASK_ID` tagging** | Not used in YAML | Should be passed to train_flow for MLflow grouping | P1 |
| **`disk_tier`** | Not set | `disk_tier: best` (SSD for MOUNT_CACHED performance) | P2 (only for MOUNT_CACHED) |
| **Preemption signal handler** | Not wired | SIGTERM handler to trigger emergency checkpoint | P2 |

### Recommended train_factorial.yaml Changes

```yaml
resources:
  image_id: docker:europe-north1-docker.pkg.dev/minivess-mlops/minivess/base:latest
  accelerators: {L4: 1, A100-80GB: 1}
  cloud: gcp
  use_spot: true
  disk_size: 100
  disk_tier: best  # SSD for MOUNT_CACHED performance
  job_recovery:
    strategy: FAILOVER
    max_restarts_on_errors: 3

file_mounts:
  /app/checkpoints:
    source: gs://minivess-mlops-checkpoints
    mode: MOUNT_CACHED
    type: MODEL_CHECKPOINT_RW
```

---

## Q6: What Is the Recommended Way to Persist Checkpoints During Spot Jobs?

**MOUNT_CACHED with a GCS bucket is the recommended approach.**

### Option Comparison

| Approach | Pros | Cons |
|----------|------|------|
| **MOUNT** (current) | Simple, immediate consistency | Slow (blocks on every write), no random writes, no append |
| **MOUNT_CACHED** (recommended) | Fast writes, async upload, all operations supported | Eventually consistent, cache can fill disk |
| **rsync (manual)** | Full control | Not automatic, requires scripting, no protection during preemption |
| **Periodic rsync to bucket** | Flexible | Complex, error-prone, duplicates SkyPilot's built-in capability |

### Why MOUNT_CACHED Is Better for Checkpoints

1. **Non-blocking writes**: Training doesn't pause while checkpoint uploads to GCS.
2. **All write operations supported**: Unlike MOUNT, supports random writes and appends (relevant for YAML state files).
3. **Guaranteed persistence before task completion**: SkyPilot waits for all cached data to upload before marking the task as finished.
4. **SSD caching**: With `disk_tier: best`, local writes are SSD-speed; upload happens in background.

### Critical Caveat for Preemption

With MOUNT_CACHED, writes are asynchronous. If preemption happens between a checkpoint `close()` and the rclone async upload completing, **the last checkpoint may be lost**. The previous checkpoint (already uploaded) will still be available.

**Mitigation**: Save checkpoints frequently enough that losing the most recent one is acceptable. For a 50-epoch training run with checkpoints every 5 epochs, losing at most 5 epochs of progress is acceptable.

With MOUNT mode, this caveat does not apply -- writes are synchronous and guaranteed to be in GCS after `close()`. But the performance cost is significant (blocking training during upload).

### Our Checkpoint Integrity Module

`src/minivess/pipeline/checkpoint_integrity.py` already implements SHA256 sidecar verification. This correctly handles the scenario where a spot preemption interrupts a write:
- `write_sha256_sidecar()` writes the hash atomically after the checkpoint.
- `verify_checkpoint_sha256()` validates integrity before loading.
- If the checkpoint was corrupted by preemption, the hash won't match, and the system falls back to the previous valid checkpoint.

---

## Q7: Are There Any SkyPilot-Specific Environment Variables or Signals for Preemption?

### Environment Variables

SkyPilot automatically sets these variables. Available in both `setup` and `run`:

| Variable | Available In | Description |
|----------|-------------|-------------|
| `SKYPILOT_TASK_ID` | setup + run | Unique task ID, **constant across recoveries** |
| `SKYPILOT_NUM_NODES` | setup + run | Total cluster nodes |
| `SKYPILOT_CLUSTER_INFO` | setup + run | JSON: `{"cluster_name": "...", "cloud": "...", "region": "...", "zone": "..."}` |
| `SKYPILOT_USER` | setup + run | Username of job launcher |
| `SKYPILOT_SETUP_NODE_RANK` | setup only | Node rank during setup |
| `SKYPILOT_SETUP_NODE_IPS` | setup only | Node IPs during setup |
| `SKYPILOT_SETUP_NUM_GPUS_PER_NODE` | setup only | GPU count per node (may not be available) |
| `SKYPILOT_NODE_RANK` | run only | Node rank during run |
| `SKYPILOT_NODE_IPS` | run only | Node IPs during run |
| `SKYPILOT_NUM_GPUS_PER_NODE` | run only | Reserved GPU count per node |

**Note**: `SKYPILOT_TASK_ID` is the most critical -- it stays the same across preemption recoveries, allowing MLflow run grouping. Our `cost_logging.py` already reads this variable.

### Preemption Signals

**SkyPilot itself does NOT provide preemption signals to the application.** However, cloud providers do:

| Provider | Warning Mechanism | Lead Time |
|----------|-------------------|-----------|
| **GCP** | Metadata server (`/computeMetadata/v1/instance/preempted`) | 30 seconds |
| **AWS** | Instance metadata (`/latest/meta-data/spot/instance-action`) | 2 minutes |
| **AliCloud** | Instance metadata | 5 minutes |
| **RunPod** | No warning mechanism documented | N/A |

GitHub discussion #1653 confirms: these provider-specific warnings can be used by the application to trigger an emergency checkpoint save, but SkyPilot does not abstract them into a unified signal.

**SIGTERM**: Cloud providers typically send SIGTERM to processes before terminating spot instances. A SIGTERM handler in the training loop could trigger an emergency checkpoint save. This is already planned in our codebase (referenced in `skypilot-compute-offloading-plan-for-vesselfm-sam3-and-synthetic-generation.xml`).

---

## Summary of Actionable Items for MinIVess

| # | Action | Files | Priority |
|---|--------|-------|----------|
| 1 | Add `job_recovery` to all SkyPilot YAMLs | `train_factorial.yaml`, `train_production.yaml`, `smoke_test_gcp.yaml` | P0 |
| 2 | Upgrade `mode: MOUNT` to `mode: MOUNT_CACHED` with `type: MODEL_CHECKPOINT_RW` | Same files | P1 |
| 3 | Add `disk_tier: best` to all YAMLs using MOUNT_CACHED | Same files | P1 |
| 4 | Wire `$SKYPILOT_TASK_ID` into MLflow tags for run grouping across recoveries | `train_flow.py` | P1 |
| 5 | Add SIGTERM handler for emergency checkpoint on preemption | `train_flow.py` or training loop | P2 |
| 6 | Update `test_skypilot_production.py` to test for `job_recovery` field | `test_skypilot_production.py` | P1 |
| 7 | Add `disk_tier: best` to `train_factorial.yaml` (already present in `smoke_test_gcp.yaml`) | `train_factorial.yaml` | P2 |

### SkyPilot Version Note

Our `pyproject.toml` pins `skypilot-nightly[runpod]>=1.0.0.dev0`. The `MOUNT_CACHED` mode and `job_recovery` field are available in the latest SkyPilot API (the managed jobs system that replaced the older `sky spot` commands). Our nightly pin should include these features, but this should be verified before implementation.

---

## Appendix: SkyNomad Paper (arXiv 2601.06520)

The SkyNomad paper describes a system built ON TOP of SkyPilot (~6,000 LoC) that adds proactive migration between regions based on:
- **Survival analysis** (Nelson-Aalen estimator) to predict spot lifetime.
- **Availability probing** every 2 hours to candidate regions.
- **Two-stage migration pipeline**: checkpoint to object store in target region while target instance provisions.

This is complementary to SkyPilot's built-in spot recovery (which is reactive -- waits for preemption, then finds new resources). SkyNomad proactively migrates BEFORE preemption is likely. Not directly actionable for MinIVess now, but relevant if spot preemption becomes a significant cost/delay issue at scale.
