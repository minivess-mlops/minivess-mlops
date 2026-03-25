# MLflow Async Checkpoint Architecture — Multi-Hypothesis Decision Matrix

**Date**: 2026-03-24
**Context**: KG invariant `mlflow_only_artifact_contract` violated by SkyPilot
`file_mounts` to `gs://minivess-mlops-checkpoints`. Need MLflow-only persistence.

---

## The Engineering Problem

Training writes checkpoints every epoch (~50MB DynUNet, ~500MB SAM3).
Current: MOUNT_CACHED writes to GCS via rclone (fast, async, non-blocking).
Required: MLflow artifact store is THE ONLY persistence mechanism.
Challenge: `mlflow.log_artifact()` is synchronous and blocks training.

## Key Finding: MLflow 3.x Async Logging Does NOT Cover Artifacts

- `MLFLOW_ENABLE_ASYNC_LOGGING=true` enables async for metrics/params only
- `mlflow.log_artifact()` has NO async variant (open FR: GitHub #14153)
- BUT `mlflow.log_artifact()` IS thread-safe in MLflow 3.x (contextvars)
- Solution: wrap in `ThreadPoolExecutor` manually

## Cloud Instance Local Storage

| Cloud | Instance | GPU | Local NVMe | IOPS (write) | Ephemeral? |
|-------|----------|-----|-----------|-------------|------------|
| **GCP** | g2-standard-4 | 1x L4 | 375 GiB | 90K | YES |
| **AWS** | g5.xlarge | 1x A10G | 250 GB | ~90K (est) | YES |
| **Azure** | NC24ads_A100_v4 | 1x A100 | 960 GiB | ~100K (est) | YES |

- Local NVMe write latency: ~62 μs (4K avg) — **50-100x faster** than network disk
- pd-ssd (GCP persistent): 1-5 ms latency, survives preemption
- SkyPilot `disk_tier: best` = pd-ssd (NOT local NVMe)

## Checkpoint Write/Upload Timing

| Checkpoint | Size | Local NVMe Write | GCS Upload (100-200 MB/s) |
|-----------|------|-----------------|--------------------------|
| DynUNet | ~50 MB | ~50 ms | 0.3-0.5 s |
| SAM3 Vanilla | ~500 MB | ~400 ms | 2.5-5 s |
| SAM3 ViT-32L (full) | ~2.5 GB | ~2 s | 12-25 s |

## Multi-Hypothesis Decision Matrix

### H1: Keep MOUNT_CACHED (status quo) + MLflow Thin Wrapper

**How**: Keep `file_mounts` for fast writes. Add background thread that polls
GCS for newly-uploaded files and registers them as MLflow artifacts.

| Criterion | Score (1-5) |
|-----------|-------------|
| Training speed impact | 5 (zero — writes are already async) |
| MLflow integration | 3 (indirect — MLflow knows about ckpts after upload, not during) |
| Spot recovery | 4 (MOUNT_CACHED handles it, MLflow discovers after) |
| KG invariant compliance | **2 (VIOLATES — two persistence mechanisms)** |
| Code complexity | 4 (thin wrapper, ~50 lines) |
| Multi-cloud portability | 3 (SkyPilot MOUNT_CACHED works on GCP/AWS/Azure) |

### H2: Local SSD + ThreadPoolExecutor + mlflow.log_artifact()

**How**: Write to local SSD (fast sync), background thread uploads via MLflow.

```python
executor = ThreadPoolExecutor(max_workers=1)
# After torch.save():
executor.submit(mlflow.log_artifact, str(ckpt_path), "checkpoints")
```

| Criterion | Score (1-5) |
|-----------|-------------|
| Training speed impact | 5 (local SSD write ~50ms, upload in background) |
| MLflow integration | **5 (native — mlflow.log_artifact() is the upload)** |
| Spot recovery | 3 (lose current epoch ckpt if preempted during upload) |
| KG invariant compliance | **5 (MLflow IS the only mechanism)** |
| Code complexity | 4 (~100 lines for upload manager + flush) |
| Multi-cloud portability | 5 (MLflow handles GCS/S3/Azure Blob via artifact backends) |

### H3: Local SSD + ProcessPoolExecutor + GCS Direct + MLflow Tag

**How**: Write to local SSD, process-based worker uploads to GCS directly
(faster than MLflow proxy), then tags the MLflow run with the artifact URI.

| Criterion | Score (1-5) |
|-----------|-------------|
| Training speed impact | 5 (same as H2) |
| MLflow integration | 4 (MLflow knows via tags, but artifact not in MLflow store) |
| Spot recovery | 3 (same as H2) |
| KG invariant compliance | **3 (GCS is the store, MLflow is the index — debatable)** |
| Code complexity | 3 (~200 lines, more moving parts) |
| Multi-cloud portability | 3 (GCS-specific, needs adapters for S3/Azure) |

### H4: PyTorch DCP async_save() + GCS StorageWriter

**How**: Use PyTorch's native async checkpoint with Google's GCS connector.

| Criterion | Score (1-5) |
|-----------|-------------|
| Training speed impact | 5 (staging ~0.78s, then fully non-blocking) |
| MLflow integration | 2 (DCP doesn't know about MLflow, manual integration) |
| Spot recovery | 4 (future-based, explicit upload completion signal) |
| KG invariant compliance | 3 (DCP → GCS is the store, MLflow must be added separately) |
| Code complexity | 2 (requires DCP state dict format, GCS connector dependency) |
| Multi-cloud portability | 3 (GCS connector is GCP-specific, S3 connector exists separately) |

### H5: pd-ssd Persistent Disk + Periodic MLflow Sync

**How**: Use `disk_tier: best` (pd-ssd, persists across preemption).
Background cron syncs to MLflow every N minutes.

| Criterion | Score (1-5) |
|-----------|-------------|
| Training speed impact | 3 (pd-ssd is 50-100x slower than NVMe for writes) |
| MLflow integration | 4 (periodic sync, not real-time) |
| Spot recovery | **5 (pd-ssd survives preemption — no data loss)** |
| KG invariant compliance | 3 (pd-ssd is persistent but not MLflow) |
| Code complexity | 3 (cron + sync script) |
| Multi-cloud portability | 3 (pd-ssd is GCP-specific, EBS on AWS) |

## Recommendation

**H2: Local SSD + ThreadPoolExecutor + mlflow.log_artifact()** — RECOMMENDED

| Metric | Value |
|--------|-------|
| KG compliance | **FULL** — MLflow is the only artifact store |
| Training impact | **Zero** — 50ms local write, upload in background |
| Data loss window | ~0.5s (DynUNet) to ~25s (SAM3) per checkpoint |
| Code complexity | ~100 lines (`AsyncCheckpointUploader` class) |
| Multi-cloud | Works on GCP/AWS/Azure (MLflow handles backends) |

**Implementation sketch**:
```python
class AsyncCheckpointUploader:
    def __init__(self, mlflow_run_id: str, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[Future] = []

    def upload(self, local_path: Path, artifact_subdir: str = "checkpoints") -> None:
        future = self._executor.submit(
            mlflow.log_artifact, str(local_path), artifact_subdir
        )
        self._futures.append(future)

    def flush(self) -> None:
        for f in self._futures:
            f.result()  # Wait for all uploads, raise on failure
        self._futures.clear()

    def shutdown(self) -> None:
        self.flush()
        self._executor.shutdown(wait=True)
```

**Remove from train_factorial.yaml**:
```yaml
# DELETE these lines:
file_mounts:
  /app/checkpoints:
    source: gs://minivess-mlops-checkpoints
    mode: MOUNT_CACHED
```

**Update spot recovery**: `check_resume_state_task()` discovers checkpoints from
MLflow `artifact_uri` instead of local disk path.

---

## References

- MLflow async logging: `MLFLOW_ENABLE_ASYNC_LOGGING` (metrics only, not artifacts)
- MLflow thread safety: GitHub #9235 (fixed in 3.x with contextvars)
- MLflow async artifact FR: GitHub #14153 (closed without merge)
- GCP Local SSD: 375 GiB, 90K write IOPS, 62μs latency
- SkyPilot MOUNT_CACHED: rclone VFS, 9.6x speedup (SkyPilot blog)
- PyTorch DCP async_save: 6x faster (Meta blog), GCS connector available
- MosaicML Composer RemoteUploaderDownloader: ProcessPoolExecutor pattern
