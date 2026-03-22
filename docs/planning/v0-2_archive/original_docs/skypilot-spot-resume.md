# SkyPilot Spot Resume Double-Check

**Date**: 2026-03-21
**Purpose**: Verify our spot preemption resume works optimally with SkyPilot

---

## How SkyPilot Spot Recovery Works

SkyPilot managed jobs (`sky jobs launch`) use a **controller VM** that monitors
the spot instance. On preemption, the controller provisions a new VM in another
region/cloud, re-runs `setup`, then re-runs `run`. Both blocks execute FROM SCRATCH.

**Key insight**: SkyPilot does NOT manage checkpoint state. It provides:
1. **Infrastructure recovery** (new VM provisioning)
2. **Storage persistence** (GCS bucket re-mounted)
3. **Environment variables** (`$SKYPILOT_TASK_ID` constant across recoveries)

The APPLICATION is responsible for:
1. Detecting resume vs fresh start
2. Finding the latest checkpoint
3. Loading state and resuming from the correct epoch

---

## Our Current Implementation

### What Works

| Component | File | Status |
|-----------|------|--------|
| `check_resume_state_task()` | `train_flow.py:262-302` | Reads `epoch_latest.yaml`, validates MLflow run RUNNING |
| Resume wiring | `train_flow.py:607-622` | Loads `epoch_latest.pth`, sets `start_epoch` |
| Atomic checkpoint writes | `base.py:save_checkpoint()` | sync→rename pattern prevents corruption |
| SHA256 sidecar | `base.py:save_checkpoint()` | Verifies integrity on load |
| GCS file_mounts | `train_factorial.yaml` | `/app/checkpoints` mounted from GCS bucket |

### What's Missing (5 Gaps from Research)

| # | Gap | Severity | Fix |
|---|-----|----------|-----|
| 1 | **`job_recovery` field missing** from train_factorial.yaml | **P0** | Add `job_recovery: {strategy: FAILOVER, max_restarts_on_errors: 3}` |
| 2 | **MOUNT vs MOUNT_CACHED** | P1 | Switch from `MOUNT` to `MOUNT_CACHED` for async checkpoint upload |
| 3 | **`$SKYPILOT_TASK_ID` not tagged** in MLflow | P1 | Tag MLflow runs with task_id for cross-recovery grouping |
| 4 | **No `disk_tier: best`** | P2 | Add for faster local checkpoint I/O |
| 5 | **No SIGTERM handler** | P2 | Catch GCP 30s preemption warning, save final checkpoint |

---

## Gap 1 (P0): `job_recovery` Field

**Current**: train_factorial.yaml has NO `job_recovery` field.
**Impact**: SkyPilot will NOT auto-recover from preemption.

**Fix**:
```yaml
# Add after resources block
job_recovery:
  strategy: FAILOVER       # Try another region/cloud on preemption
  max_restarts_on_errors: 3 # Max recovery attempts
```

**SkyPilot docs**: "For managed spot jobs, SkyPilot automatically handles recovery
by reprovisioning the cluster in a different zone or cloud."

---

## Gap 2 (P1): MOUNT_CACHED Instead of MOUNT

**Current**: `file_mounts` uses `mode: MOUNT` (FUSE, synchronous writes).
**Problem**: MOUNT blocks training during checkpoint upload (~50MB checkpoint
blocks for ~5-10s depending on network).

**Fix**:
```yaml
file_mounts:
  /app/checkpoints:
    source: gs://minivess-mlops-checkpoints
    mode: MOUNT_CACHED     # Async upload, non-blocking writes
```

**Caveat**: If preemption occurs between close() and async upload completing,
the last checkpoint may be lost. The PREVIOUS checkpoint survives (it was
fully uploaded). Our `check_resume_state_task()` handles this correctly —
it looks for `epoch_latest.yaml` which only exists after a complete save.

---

## Gap 3 (P1): SKYPILOT_TASK_ID Tagging

**Current**: MLflow runs are not tagged with the SkyPilot task ID.
**Impact**: On recovery, a new MLflow run is created without linking to
the previous run. Cross-recovery provenance is lost.

**Fix**: In `train_flow.py`, tag MLflow runs with:
```python
import os
task_id = os.environ.get("SKYPILOT_TASK_ID", "")
if task_id:
    mlflow.set_tag("skypilot_task_id", task_id)
```

---

## Gap 4 (P2): disk_tier

```yaml
resources:
  disk_tier: best  # SSD for faster checkpoint I/O
```

Already present in `smoke_test_gcp.yaml` but missing from `train_factorial.yaml`.

---

## Gap 5 (P2): SIGTERM Handler

GCP gives 30 seconds warning before spot preemption via SIGTERM.
A handler could save a final checkpoint in those 30 seconds:

```python
import signal

def _preemption_handler(signum, frame):
    logger.warning("SIGTERM received — saving emergency checkpoint")
    save_checkpoint(model, checkpoint_dir / "emergency.pth")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, _preemption_handler)
```

Not critical — our periodic checkpoint saving (every epoch) means we lose
at most 1 epoch of work. But for long epochs (SAM3 TopoLoRA ~38h total),
losing 1 epoch = losing 1+ hours.

---

## Verification Checklist

- [x] `check_resume_state_task()` correctly reads `epoch_latest.yaml`
- [x] Resume wiring loads `epoch_latest.pth` and sets `start_epoch`
- [x] Atomic checkpoint writes prevent corruption (sync→rename)
- [x] SHA256 sidecar verifies integrity on load
- [x] GCS file_mounts in train_factorial.yaml
- [ ] `job_recovery` field added (Gap 1)
- [ ] MOUNT_CACHED mode (Gap 2)
- [ ] SKYPILOT_TASK_ID tagging (Gap 3)
- [ ] disk_tier: best (Gap 4)
- [ ] SIGTERM handler (Gap 5)

---

## References

- [SkyPilot Spot Jobs Docs](https://docs.skypilot.co/en/v0.5.0/examples/spot-jobs.html)
- [SkyPilot Discussion #1653](https://github.com/skypilot-org/skypilot/discussions/1653)
- [SkyPilot Paper (arXiv:2601.06520)](https://arxiv.org/html/2601.06520v1)
- Internal: `docs/planning/skypilot-spot-preemption-checkpoint-research-report.md`
