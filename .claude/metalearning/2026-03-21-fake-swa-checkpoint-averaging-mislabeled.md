# Metalearning: "SWA" Implementation Is NOT SWA — Post-Hoc Checkpoint Averaging Mislabeled

**Date**: 2026-03-21
**Severity**: HIGH — scientific integrity issue
**Root cause**: Claude Code invented its own definition of "SWA" instead of implementing the algorithm from the literature
**Pattern**: Same anti-pattern as SAM3 confabulation (2026-03-02) — confident implementation of something that doesn't match the paper

## What Happened

The codebase contains `SWAPlugin`, `MultiSWAPlugin`, `uniform_swa()`, and extensive
config/test infrastructure all labeled "SWA" (Stochastic Weight Averaging). **None of
it implements actual SWA.**

What the code ACTUALLY does:
1. Takes N already-trained checkpoints (best_val_loss, best_val_dice, etc. from ONE run)
2. Computes arithmetic mean of their state dicts
3. Saves the averaged weights

What REAL SWA does (Izmailov et al. 2018, UAI):
1. **During training**: Switch to cyclic or constant LR after warmup (`swa_start_epoch`)
2. **During training**: Periodically collect weight snapshots via `AveragedModel`
3. **After training**: Recalibrate batch normalization via `update_bn()`
4. Produces a **single model** that finds flatter optima with better generalization

PyTorch provides all of this in `torch.optim.swa_utils`:
```python
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=0.05)
# ... during training loop ...
swa_model.update_parameters(model)
update_bn(train_loader, swa_model)
```

**Zero imports of `swa_utils` exist anywhere in the codebase.**

## Why This Is Dangerous

1. **Scientific fraud by another name**: If a paper says "we applied SWA" but the code
   just averages best-metric checkpoints, that's a reproducibility lie. Anyone trying to
   reproduce the results by implementing actual SWA will get different numbers.

2. **Misleading metrics**: The debug run showed <0.001 DSC improvement from "SWA" — which
   is expected because averaging 7 checkpoints from the same trajectory (selected by
   different metrics) produces nearly identical weights. Real SWA explores different loss
   basins via cyclic LR and shows 0.5-2% improvement typically.

3. **Wasted engineering**: Tests, configs, plugins, MLflow tags — all built around a
   misnamed concept. The factorial experiment has `post_training_method: swa` as a factor,
   but it's not testing what anyone thinks it's testing.

## Correct Terminology

| What the code does | Correct name | Reference |
|--------------------|-------------|-----------|
| Average N best-metric checkpoints | **Checkpoint averaging** or **Uniform model soup** | Wortsman et al. (2022) |
| Average random subsets of checkpoints | **Subsampled checkpoint ensemble** | — |
| Cyclic LR + periodic weight collection + BN update | **SWA** | Izmailov et al. (2018) |
| Track second moments during training | **SWAG** | Maddox et al. (2019) |

## The Broader Rule This Violates

> **Everything as it is in the literature!** (User directive, 2026-03-21)
>
> This is a platform repo for developing MLOps infrastructure. We do NOT create
> our own SOTA implementations or our own definitions. Every algorithm must match
> the paper it claims to implement. If the paper says "cyclic LR + periodic weight
> averaging during training," that is what the code must do.

This is the SAME failure pattern as:
- SAM3 stub encoder (2026-03-02): Invented a fake encoder instead of using real weights
- TopoLoRA Conv2d bug: Applied LoRA to wrong layer types without reading the paper

## Required Actions

### Immediate (before any publication)

1. **Rename ALL "SWA" references** to "checkpoint averaging":
   - `SWAPlugin` → `CheckpointAveragingPlugin`
   - `MultiSWAPlugin` → `SubsampledEnsemblePlugin`
   - `uniform_swa()` → `uniform_checkpoint_average()`
   - Config keys: `swa:` → `checkpoint_averaging:`
   - Factorial factor levels: `swa` → `checkpoint_avg`
   - MLflow tags: `post_training_method: swa` → `post_training_method: checkpoint_avg`
   - Test files: `test_swa_plugin.py` → `test_checkpoint_averaging_plugin.py`
   - ~18 files affected (see full list in audit)

2. **Update CLAUDE.md** with explicit ban on terminology invention

3. **Update knowledge graph** — `domains/post_training.yaml` must distinguish
   between real SWA and checkpoint averaging

### Before GCP production run

4. **Implement REAL SWA** as an additional post-training method using
   `torch.optim.swa_utils`:
   - Add `swa_start_epoch`, `swa_lr` to training config
   - Modify training loop to collect snapshots and use SWALR
   - Add `update_bn()` after averaging
   - Test that it actually produces different results from checkpoint averaging

5. **Factorial design update**: The factor `post_training_method` should have levels:
   `[none, checkpoint_avg, real_swa]` — testing both approaches

## How Claude Code Should Have Caught This

1. **Read the paper first** (Rule #13: Read Context Before Implementing)
2. **Web search** for "pytorch swa implementation" → finds `torch.optim.swa_utils`
3. **Verify against library** (Rule #3: Library-First) → PyTorch already provides SWA
4. **Never confabulate** (Rule #12) → Don't invent your own definition of a well-known algorithm

The failure was: Claude Code was asked to "add SWA support" and instead of looking up
what SWA actually is, confidently implemented a simpler algorithm and gave it the wrong name.
This is confabulation applied to algorithm implementation.
