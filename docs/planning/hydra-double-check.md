# Hydra Config Single-Source-of-Truth Audit
## Are We Whac-a-Moling or Fixing Root Causes?

*Generated 2026-03-15 · Prompted by sam3_hybrid val_loss=NaN across 4 GCP jobs*

---

## Executive Summary

**Short answer: yes, we have been partly whac-a-moling.** The immediate `++val_interval=1`
fix is correct and necessary. But there is a deeper architectural gap: the `configs/cloud/`
config group exists and is composed by Hydra, but it only carries **infrastructure**
params (GPU type, Docker image, region). **Training-behavior params** (val_interval,
mixed_precision) that differ by platform are scattered across:

1. Individual experiment YAMLs (`smoke_sam3_hybrid.yaml` → `val_interval: 3`)
2. Bash shell overrides in SkyPilot YAML (`CLOUD_OVERRIDES=...`)
3. Hardcoded heuristics in Python (`train_flow.py:349-354`)

This is three separate parallel config systems for the same concern.

---

## Current State: What IS Working

### The Hydra pipeline is structurally sound

```
configs/base.yaml
  defaults:
    - data: minivess          ✅ exists, correct
    - model: dynunet          ✅ exists, 5 model variants
    - training: default       ✅ exists, seed/compute/epochs
    - checkpoint: standard    ✅ exists, lightweight variant
    - cloud: local            ✅ exists, 5 cloud variants (local/gcp_spot/runpod_dev/lambda/gcp_quotas)
    - lab: default            ✅ exists, override per-lab
    - user: default           ✅ exists, override per-user
```

`compose_experiment_config()` in `src/minivess/config/compose.py` is well-designed:
- Uses Hydra Compose API when available
- Falls back to manual YAML merge (now with `+`/`++` prefix stripping — fixed 2026-03-15)
- Returns a flat dict that `train_flow.py` consumes

`train_flow.py` DOES respect config values from Hydra when they're present:
```python
# Lines 346-358: config-first, heuristics as fallback
_config_val_interval = config.get("val_interval")
if _config_val_interval is not None:
    val_interval = int(_config_val_interval)   # ← HYDRA PATH (correct)
elif _is_sam3_hybrid and debug:
    val_interval = max_epochs + 1              # ← FALLBACK heuristic
```

**The Hydra path works correctly when val_interval IS in the config.**

---

## Root Cause Analysis: Why Glitches Happen

### Root Cause 1: Cloud configs carry only infrastructure, not training behavior

`configs/cloud/gcp_spot.yaml` (and all cloud configs) contain:
```yaml
provider: gcp
accelerators: [T4:1, L4:1, A100:1]
docker_image: ...
mlflow_env_var: MLFLOW_GCP_URI
```

They do NOT contain training-behavior differences like:
```yaml
val_interval: 1   # L4 has 24 GB — always validate (not like 8 GB RunPod)
```

**This is the gap.** `configs/cloud/` exists but doesn't encode the fact that cloud
GPUs have enough VRAM to run validation every epoch.

### Root Cause 2: `mixed_precision` is not in SAM3 model configs

SAM3 models require `mixed_precision: false` due to MONAI #4243
(`sliding_window_inference + autocast = NaN`). But `configs/model/sam3_hybrid.yaml`
and `configs/model/sam3_vanilla.yaml` do NOT set this. It should be:

```yaml
# configs/model/sam3_hybrid.yaml
model: sam3_hybrid
mixed_precision: false   # ← MISSING — should be here, not in bash overrides
```

Because `mixed_precision: false` is MODEL-specific (SAM3 architecture constraint),
not cloud-specific. DynUNet runs fine with AMP on GCP.

### Root Cause 3: Experiment configs are GPU-specific, not composable

`configs/experiment/smoke_sam3_hybrid.yaml` contains:
```yaml
val_interval: 3   # val_interval > max_epochs=2 → validation sentinel
```

This is correct for RunPod RTX 4090... but WRONG for GCP L4. The experiment
config encodes a hardware assumption instead of being hardware-agnostic.

The fix was `smoke_sam3_hybrid_cloud.yaml` — but this creates N×M explosion:
- N experiment configs × M cloud variants = N×M files
- Currently: 1 experiment × 2 clouds = 2 files
- At scale: 20 experiments × 3 clouds = 60 files

### Root Cause 4: The bash CLOUD_OVERRIDES pattern

`smoke_test_gcp.yaml` run section:
```bash
CLOUD_OVERRIDES="+mixed_precision=false,++val_interval=1"
export HYDRA_OVERRIDES="experiment_name=...,${CLOUD_OVERRIDES}"
```

This is a 3rd config injection system (alongside Hydra and Python heuristics).
The `+`/`++` Hydra syntax in bash strings is fragile:
- `+key` fails if key exists → exception → fallback → `+` prefix not stripped → silent failure
- This caused 4 consecutive val_loss=NaN jobs before the `++` fix

### Root Cause 5: Docker image staleness as a trigger

The `run:` section in `smoke_test_gcp.yaml` checks:
```bash
if [ -f "configs/experiment/smoke_${MODEL_FAMILY}_cloud.yaml" ]; then
    export EXPERIMENT="smoke_${MODEL_FAMILY}_cloud"
fi
```

The Docker image is built from the repo. If `smoke_sam3_hybrid_cloud.yaml` is
added to the repo AFTER the last Docker build, the file doesn't exist in the
container, triggering the CLOUD_OVERRIDES fallback path.

---

## Architecture Fix: Hydra as Single Source of Truth

### Fix 1 (HIGH PRIORITY): Add `val_interval` to `configs/cloud/gcp_spot.yaml`

```yaml
# configs/cloud/gcp_spot.yaml  (addition)

# Training behavior on cloud GPUs (L4: 24 GB, A100: 40-80 GB).
# Override the OOM-prevention sentinel (val_interval=max_epochs+1) from
# 8 GB configs. Cloud GPUs have enough VRAM to validate every epoch.
val_interval: 1
```

Then in `smoke_test_gcp.yaml` run section, replace CLOUD_OVERRIDES with:
```bash
export HYDRA_OVERRIDES="experiment_name=...,cloud=gcp_spot"
```

This is the architecturally correct Hydra way: override the cloud config group.

### Fix 2 (HIGH PRIORITY): Add `mixed_precision: false` to SAM3 model configs

```yaml
# configs/model/sam3_hybrid.yaml  (addition)
mixed_precision: false   # BF16 encoder: disable AMP (MONAI #4243, sliding_window+autocast=NaN)

# configs/model/sam3_vanilla.yaml  (addition)
mixed_precision: false   # Same encoder architecture, same AMP constraint
```

This moves `mixed_precision` from a bash override to where it belongs: the MODEL
config that encodes the architectural constraint. No more bash overrides needed.

### Fix 3 (MEDIUM PRIORITY): Make experiment configs cloud-agnostic

`smoke_sam3_hybrid.yaml` should NOT have `val_interval: 3`. Instead:
- Remove `val_interval` from experiment configs (let cloud config provide it)
- Remove `_cloud.yaml` experiment variants (they become redundant)
- The cloud config group handles the hardware-specific adjustments

**Before** (N×M explosion):
```
configs/experiment/smoke_sam3_hybrid.yaml         (RunPod)
configs/experiment/smoke_sam3_hybrid_cloud.yaml   (GCP)
```

**After** (composable):
```
configs/experiment/smoke_sam3_hybrid.yaml   (hardware-agnostic)
configs/cloud/gcp_spot.yaml                 (cloud-specific adjustments)
→ compose: +experiment=smoke_sam3_hybrid cloud=gcp_spot
```

### Fix 4 (MEDIUM PRIORITY): Remove CLOUD_OVERRIDES from smoke_test_gcp.yaml

Once Fixes 1+2 are applied, the run section simplifies to:
```bash
export EXPERIMENT="smoke_${MODEL_FAMILY}"
export EXPERIMENT_UUID=$(python -c "import uuid; print(uuid.uuid4().hex[:8])")
export HYDRA_OVERRIDES="experiment_name=smoke_test_${EXPERIMENT_UUID}_${MODEL_FAMILY},cloud=gcp_spot"
python -m minivess.orchestration.flows.train_flow
```

No bash string manipulation of Hydra override syntax. No `+`/`++` fragility.

---

## What We Should NOT Do (Whac-a-Mole Anti-Patterns)

| Anti-Pattern | Why Bad |
|---|---|
| Adding more `_cloud.yaml` experiment variants | N×M file explosion; doesn't scale |
| Putting training params in SkyPilot YAML run: section | 3rd config system, not Hydra |
| Using `HYDRA_OVERRIDES` for architecture constraints | Arch constraints belong in model config |
| Adding if/elif model-name heuristics to train_flow.py | Hardcoded, not config-driven |
| Bash `+`/`++` manipulation | Fragile syntax, silent failures |

---

## Implementation Plan

### Phase 1: Model configs (1-2 hours, no cloud run needed)
1. Add `mixed_precision: false` to `configs/model/sam3_hybrid.yaml`
2. Add `mixed_precision: false` to `configs/model/sam3_vanilla.yaml`
3. Tests: verify compose produces `mixed_precision=False` for SAM3 models

### Phase 2: Cloud config (1-2 hours, requires Docker image rebuild)
1. Add `val_interval: 1` to `configs/cloud/gcp_spot.yaml`
2. Update `smoke_test_gcp.yaml` run section: remove CLOUD_OVERRIDES, use `cloud=gcp_spot`
3. Rebuild Docker image to include updated configs
4. Tests: verify smoke test on GCP uses val_interval=1

### Phase 3: Experiment config cleanup (1-2 hours)
1. Remove `val_interval: 3` from `smoke_sam3_hybrid.yaml` (let cloud config provide it)
2. Deprecate `smoke_sam3_hybrid_cloud.yaml` (it becomes redundant)
3. Update test assertions for smoke config structure

### Phase 4: Enforcement (optional but valuable)
1. Add pre-commit check: `configs/experiment/smoke_*.yaml` must NOT contain `val_interval`
2. Add test: SAM3 model configs must have `mixed_precision: false`
3. Add test: SkyPilot run sections must NOT contain Hydra override strings in bash

---

## Is Hydra Composability Optional Here?

**No, but it's also not the problem.** Hydra IS being used. The issue is that
the config groups don't capture ALL the relevant variation:

- `configs/cloud/` captures infrastructure variation ✅
- `configs/model/` should capture architecture constraints (mixed_precision) ❌ MISSING
- `configs/cloud/` should also capture training-behavior differences (val_interval) ❌ MISSING

The composability was always there. We just didn't put the right values in the
right config groups.

---

## Current State vs Intent (Gap Summary)

| Parameter | Where it should be | Where it is now |
|---|---|---|
| `mixed_precision: false` | `configs/model/sam3_hybrid.yaml` | Bash override in SkyPilot YAML |
| `val_interval: 1` (cloud) | `configs/cloud/gcp_spot.yaml` | Bash override in SkyPilot YAML |
| `val_interval: max+1` (8GB) | Nowhere (use model default logic) | `smoke_sam3_hybrid.yaml` |

---

## Related Issues and Context

- [#192: Hydra config composability audit](https://github.com/petteriTeikari/minivess-mlops/issues/192) — P1 audit, deferred
- [#302: Update tests for Hydra-only config loading](https://github.com/petteriTeikari/minivess-mlops/issues/302)
- [#708: Hydra cloud config groups](https://github.com/petteriTeikari/minivess-mlops/issues/708) — CLOSED: configs/cloud/ created (infra params only)
- [#288: Hydra Compose API bridge](https://github.com/petteriTeikari/minivess-mlops/issues/288) — CLOSED: compose_experiment_config() implemented
- [#295: Full Hydra migration](https://github.com/petteriTeikari/minivess-mlops/issues/295) — CLOSED: experiment configs migrated

**The gap**: #708 created `configs/cloud/` but only put infrastructure params in it.
The training-behavior params (val_interval, mixed_precision) were never added.
This is the missing step that caused 4 consecutive GCP jobs to produce val_loss=NaN.

---

## Immediate Next Steps (Before Closing This Issue)

1. ✅ Fix `++val_interval=1` syntax in smoke_test_gcp.yaml (done, committed)
2. ✅ Fix `_compose_with_manual_merge` to strip `+`/`++` prefixes (done, committed)
3. 🔲 Add `mixed_precision: false` to SAM3 model configs (Phase 1)
4. 🔲 Add `val_interval: 1` to `configs/cloud/gcp_spot.yaml` (Phase 2)
5. 🔲 Update smoke_test_gcp.yaml to use `cloud=gcp_spot` override instead of CLOUD_OVERRIDES
6. 🔲 Rebuild Docker image and re-run GCP smoke tests
7. 🔲 Deprecate `_cloud.yaml` experiment variants

The `++val_interval` fix is correct for now. But Phases 1-3 are the real fix that
eliminates the entire class of "cloud param not propagated" bugs.
