# MLflow vs TensorBoard Status Report — Epoch-Level Metrics

**Date**: 2026-03-21
**Status**: MLflow ONLY — no TensorBoard. Per-epoch metrics ARE preserved.

---

## Answer: All Per-Epoch Metrics Go to MLflow

**No TensorBoard** — grep for `SummaryWriter`, `tensorboard`, `TensorboardLogger`
returns 0 results across the entire codebase. Training metrics go EXCLUSIVELY to MLflow.

**Per-epoch metrics ARE saved** with step information via two mechanisms:

### Mechanism 1: MLflow `log_metrics(step=epoch+1)`

`trainer.py` line 798:
```python
self.tracker.log_epoch_metrics(epoch_log, step=epoch + 1)
```

`tracking.py` line 293:
```python
def log_epoch_metrics(self, metrics: dict[str, float], *, step: int, prefix: str = "") -> None:
    prefixed = dict(sorted(...))
    mlflow.log_metrics(prefixed, step=step)  # ← step parameter preserves epoch
```

This creates per-metric files in `mlruns/<exp_id>/<run_id>/metrics/<metric_key>` with format:
```
<timestamp> <value> <step>
1234567890 0.5432 1
1234567900 0.4521 2
1234567910 0.3890 3
```

Each line is one epoch. Training curves are fully reconstructable.

### Mechanism 2: `metric_history.json` artifact

`multi_metric_tracker.py` (MetricHistory class, lines 321-369):
```python
self._metric_history.record_epoch(
    epoch=epoch,
    metrics=all_metrics,
    wall_time_sec=epoch_wall_time,
    checkpoints_saved=improved_metrics,
)
```

Saved to `artifacts/history/metric_history.json`:
```json
{
  "epochs": [
    {"epoch": 0, "metrics": {"train/loss": 0.54, "val/loss": 0.67, ...}, "wall_time_sec": 45.2},
    {"epoch": 1, "metrics": {"train/loss": 0.45, "val/loss": 0.58, ...}, "wall_time_sec": 43.1}
  ]
}
```

### Metrics Logged Per Epoch

| Prefix | Metrics |
|--------|---------|
| `train/` | `loss`, `dice`, `f1_foreground`, `patience_counter` |
| `val/` | `loss`, `dice`, `f1_foreground`, `cldice`, `masd`, `compound_masd_cldice` |
| `optim/` | `lr`, `grad_scale` |
| `prof/` | `train_seconds`, `val_seconds`, `first_epoch_seconds`, `steady_epoch_seconds` |
| `gpu/` | `utilization_pct`, `mem_bw_util_pct`, `temperature_c`, `power_w` |
| `checkpoint/` | `size_mb` |

### How to Read Training Curves

**Via MLflow API (recommended):**
```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
history = client.get_metric_history(run_id, "val/loss")
# Returns list of Metric(key, value, timestamp, step)
epochs = [(m.step, m.value) for m in history]
```

**Via metric files directly:**
```python
from pathlib import Path
metric_file = Path(f"mlruns/{exp_id}/{run_id}/metrics/val/loss")
lines = metric_file.read_text(encoding="utf-8").strip().splitlines()
epochs = [(int(parts[2]), float(parts[1])) for parts in (l.split() for l in lines)]
```

**Via metric_history.json:**
```python
import json
history = json.loads(Path(f"mlruns/{exp_id}/{run_id}/artifacts/history/metric_history.json").read_text())
for epoch_data in history["epochs"]:
    print(f"Epoch {epoch_data['epoch']}: val_loss={epoch_data['metrics']['val/loss']}")
```

### Epoch Indexing

| Location | Index Start | Example (100 epochs) |
|----------|------------|---------------------|
| MLflow step parameter | 1-indexed | step=1 to step=100 |
| metric_history.json | 0-indexed | epoch=0 to epoch=99 |
| Checkpoint naming | N/A (best only) | `best_val_loss.pth` (no epoch in name) |

### What Will Change with Calibration Metrics

When we add Tier 1 calibration metrics (ECE, pECE, BA-ECE, MCE, Brier, NLL, BUC, CECE,
OE, D-ECE, smECE) to the validation loop, they will be logged per-epoch via the same
`log_epoch_metrics(step=epoch+1)` mechanism. This means:

- `val/ece`, `val/pece`, `val/ba_ece` will have full epoch-level history in MLflow
- Training curves for calibration metrics will be available alongside segmentation metrics
- The `metric_history.json` artifact will include calibration metrics
- The biostatistics flow can read per-epoch calibration from metric_history.json

**OBLIGATORY**: Every fold in every mlrun gets epoch-level metrics for ALL tracked metrics
(segmentation + calibration). This is already the architecture — `log_epoch_metrics` is
called unconditionally after each epoch.

### No Gaps Found

The current architecture is correct for epoch-level metric preservation:
- `mlflow.log_metrics(prefixed, step=step)` — preserves epoch in MLflow metric files
- `metric_history.json` — preserves epoch in artifact
- Both mechanisms log EVERY epoch, not just best or final
- Adding calibration metrics requires only extending the `epoch_log` dict — the logging
  infrastructure handles it automatically
