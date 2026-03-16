# Metalearning: Hydra Override `+`/`++` Prefix Bug in Manual Merge Fallback

**Date**: 2026-03-15
**Issue**: GCP smoke test `sam3_hybrid` produced `val_loss=NaN` across 4 consecutive jobs
**Root cause**: Two compounding bugs in Hydra override syntax + fallback logic

---

## What Happened

`smoke_test_gcp.yaml` run section sets CLOUD_OVERRIDES for SAM3 models:
```bash
CLOUD_OVERRIDES="+mixed_precision=false,+val_interval=1"
```

This should:
1. Disable AMP (MONAI #4243: `sliding_window + autocast = NaN`)
2. Set `val_interval=1` (was `3` in `smoke_sam3_hybrid.yaml`, skipping validation on L4)

### Bug 1: `+val_interval=1` throws Hydra ConfigCompositionException

In Hydra:
- `+key=value` = append NEW key (fails if key already exists in resolved config)
- `++key=value` = force override (upsert: adds if missing, overrides if present)

`smoke_sam3_hybrid.yaml` has `val_interval: 3`. When loaded via
`+experiment=smoke_sam3_hybrid`, the key `val_interval` is in the resolved config.
Using `+val_interval=1` throws:
> "Could not add 'val_interval' — key already exists. Use '++' to force override."

### Bug 2: Manual merge fallback doesn't strip `+`/`++` prefixes

When Hydra compose fails (due to Bug 1), `compose_experiment_config()` catches the
exception and falls back to `_compose_with_manual_merge()`. The fallback did:
```python
key, value = override.split("=", 1)
merged[key] = _parse_value(value)   # key = "+mixed_precision" → stored under WRONG KEY!
```

So `merged["+mixed_precision"] = False` instead of `merged["mixed_precision"] = False`.
Both overrides silently fail. Result: `mixed_precision=True`, `val_interval=3` (sentinel
→ validation skipped → `val_loss=NaN`).

---

## Fixes Applied

### Fix 1: `deployment/skypilot/smoke_test_gcp.yaml`
```bash
# Before (BROKEN):
CLOUD_OVERRIDES="+mixed_precision=false,+val_interval=1"

# After (FIXED):
CLOUD_OVERRIDES="+mixed_precision=false,++val_interval=1"
```
- `mixed_precision` is NOT in the base struct → `+` is correct (add new key)
- `val_interval` IS in `smoke_sam3_hybrid.yaml` → `++` is required (force override)

**General rule**: When unsure if a key exists, use `++` — it's always safe.

### Fix 2: `src/minivess/config/compose.py` — `_compose_with_manual_merge()`
```python
# Added key prefix stripping:
key = key.lstrip("+")   # strips "+" and "++"
```
The manual fallback should strip `+`/`++` markers since it applies all overrides
unconditionally. Without this, any Hydra prefix causes the wrong key to be stored.

---

## Tests Added

`tests/v2/unit/test_config_composition.py::TestComposeBridge`:
- `test_plus_prefix_override_applies_new_key` — `+mixed_precision=false` → `mixed_precision=False`
- `test_double_plus_prefix_override_existing_key` — `++val_interval=1` → `val_interval=1`

Both pass after fixes (58/58 in test_config_composition.py).

---

## Impact

4 GCP managed jobs (IDs 1-4 in this session's queue) had `val_loss=NaN` before fix.
Job 5 (submitted 2026-03-15) uses the corrected override syntax and should produce
finite `val_loss` and `mixed_precision=False` in MLflow.

---

## Lessons

1. **Hydra override prefixes are semantic**:
   - `key=value` — override existing struct key (fails if key missing)
   - `+key=value` — add new key (fails if key already exists)
   - `++key=value` — upsert (always safe)
   - **When in doubt: use `++`**

2. **Fallback paths must handle upstream syntax**: If Hydra's override syntax is
   passed through to a non-Hydra code path, strip the Hydra-specific prefixes.

3. **Test override behavior explicitly**: Config composition tests should cover
   the `+`/`++` prefix cases specifically — they're non-obvious failure modes.

4. **One broken override blocks all overrides**: In Hydra, if any override in the
   list throws, the whole compose fails. A single bad override can silently
   prevent all other overrides from applying.
