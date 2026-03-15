# Metafailure: Chasing Phantom Root Causes While Using Wrong Config

**Date**: 2026-03-15
**Category**: Debugging methodology failure, confirmation bias
**Severity**: HIGH — 5 GCP runs + 2 RunPod runs, ~$1.50 cloud spend, ~10 hours debugging
**Root cause of metafailure**: GCP SkyPilot YAML selected wrong experiment config

## What Happened

sam3_hybrid val_loss=NaN was reported on RunPod RTX 4090. The investigation:
1. Identified 8 hypotheses (H1-H8)
2. **Confirmed H1 (sentinel NaN) in Run 2** — val_loss=0.7264 (finite) with correct config
3. Then spent 5 more GCP runs testing H2b (AMP) and H8 (BF16)
4. Every GCP run showed NaN — interpreted as evidence for AMP/BF16 hypotheses
5. **Reality**: Every GCP run used `smoke_sam3_hybrid.yaml` which SKIPS validation
6. The NaN was always the sentinel, never a numerical failure

## Why It Took So Long

### Failure 1: Did not verify which config was used on GCP

The GCP smoke test YAML set `EXPERIMENT="smoke_${MODEL_FAMILY}"` → `smoke_sam3_hybrid`.
Nobody checked that this resolves to the LOCAL config (validation skipped) instead of
the CLOUD config (validation enabled). A single `grep val_interval` on the resolved
config would have caught this in 30 seconds.

### Failure 2: Confirmation bias toward complex hypotheses

After Run 2 confirmed H1, the logical next step was to reproduce the fix on GCP.
Instead, we jumped to H8 (BF16 dtype) and H2b (AMP+3D) — more interesting hypotheses
involving real numerical analysis. The boring answer (wrong config file) was not
re-examined because H1 was "already fixed."

### Failure 3: Confusing two separate fixes

H1 was fixed in `train_flow.py` (config values override code heuristics). This fix
works when the config SETS `val_interval: 1`. But the GCP runs used a different config
(`smoke_sam3_hybrid.yaml`) that has `val_interval: 3` — the train_flow fix was
irrelevant because the wrong config was loaded in the first place.

### Failure 4: Docker rebuild cycles as debugging cargo cult

Three Docker rebuilds (BF16, AMP-val-off, cache-bust) consumed ~45 min each.
Each rebuild was motivated by the belief that the previous image "didn't have the fix."
Every image was correctly built — the Python code was fine. The config selection
in the SkyPilot YAML was the bug, not the Docker image contents.

### Failure 5: No experiment config name in MLflow logs

The MLflow runs logged `mixed_precision: True` and 40+ other params, but NOT the
experiment config filename. If `EXPERIMENT=smoke_sam3_hybrid` had been logged as
a param, the mismatch would have been immediately visible in the MLflow UI.

## What Should Have Happened

1. After Run 2 (H1 confirmed), verify EXACTLY which config GCP uses:
   `grep EXPERIMENT smoke_test_gcp.yaml` → sees `smoke_${MODEL_FAMILY}`
2. `cat configs/experiment/smoke_sam3_hybrid.yaml` → sees `val_interval: 3`, `max_epochs: 2`
3. Recognize 3 > 2 → sentinel skip → same H1 mechanism
4. Fix: prefer `_cloud.yaml` variant for GCP launches
5. **Total time: 5 minutes instead of 10 hours**

## Prevention

1. **Log the experiment config name to MLflow**: `mlflow.log_param("experiment_config", EXPERIMENT)`
2. **Replace NaN sentinel with a distinct marker**: `float("-inf")` or a status enum.
   NaN should ALWAYS mean "something broke numerically."
3. **After confirming a root cause, re-verify it on every new platform** before
   testing more complex hypotheses.
4. **Add pre-flight config validation to cloud launches**: Print `val_interval`,
   `max_epochs` to the log and assert validation will actually run.

## Cost of This Metafailure

| Item | Cloud cost | Time |
|------|-----------|------|
| 5 GCP spot runs (T4/L4) | ~$0.80 | ~50 min GPU |
| 2 RunPod runs (RTX 4090) | ~$0.70 | ~30 min GPU |
| 3 Docker rebuilds + GAR pushes | $0 (local) | ~90 min |
| Investigation + code changes | $0 | ~8 hours |
| **Total** | **~$1.50** | **~10 hours** |

The BF16 auto-detect and AMP-val-off mitigations are genuinely useful preventive
improvements. But they were NOT the answer to the immediate NaN, and treating them
as the active root cause delayed the actual fix by hours.

## Cross-References

- [sam3-val-loss-final-report.md](../../docs/planning/sam3-val-loss-final-report.md) — Full incident report
- [sam3-nan-loss-fix.md](../../docs/planning/sam3-nan-loss-fix.md) — Original investigation
- [sam3-bf16-fp16-fuckup.md](2026-03-15-sam3-bf16-fp16-fuckup.md) — BF16 knowledge loss (valid finding, not root cause)
- [t4-turing-fp16-nan-ban.md](2026-03-15-t4-turing-fp16-nan-ban.md) — T4 ban (preventive, based on theoretical risk)
- [amp-validation-nan-3d.md](2026-03-15-amp-validation-nan-3d.md) — AMP+3D NaN (community evidence, not observed in our runs)
