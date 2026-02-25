# Metalearning Failure Report #001: Training Launched Without Multi-Metric Config

**Date**: 2026-02-25
**Severity**: HIGH — wasted ~40 minutes of GPU training
**Category**: Agent execution error — config bypass

## What Happened

After implementing the multi-metric tracking system across 3 phases (57 tests, 3 commits),
the agent launched convergence training using `train_monitored.py` directly with CLI args:

```bash
uv run python scripts/train_monitored.py --compute gpu_low --loss dice_ce ...
```

This **bypassed the experiment YAML config** (`configs/experiments/dynunet_losses.yaml`)
which contains the checkpoint section with 3 tracked metrics (val_loss, val_dice, val_f1_foreground).

As a result, `TrainingConfig()` was created with the **default CheckpointConfig** which
only tracks `val_loss`. The training saved `best_val_loss.pth` but NOT `best_val_dice.pth`
or `best_val_f1_foreground.pth`.

## The Whole Point We Missed

The multi-metric tracking system was built specifically so that:
1. Each tracked metric gets its own best checkpoint with model weights
2. Post-training: you can load any checkpoint for inference, ensemble, or comparison
3. You can test hypotheses like "best by val_loss" vs "best by DSC" for ensemble selection

Without the per-metric checkpoints, we have metric_history.json (all metrics recorded)
but **no model weights** for the best-by-val_dice or best-by-val_f1 epochs. The model
weights at those epochs are lost forever.

## Root Cause

The `train_monitored.py` script was designed to receive checkpoint config from
`run_experiment.py` via `args.checkpoint_config`. When launched directly from CLI,
this attribute is never set, and the default single-metric config is used silently.

**Two failures compounded**:
1. Agent chose the wrong entry point (train_monitored.py instead of run_experiment.py)
2. No warning or validation that multi-metric config was missing when training with
   metrics that match known tracked metrics

## Corrective Actions

### Immediate
- [x] Kill the incorrectly launched training
- [x] Restart training through `run_experiment.py` which reads the YAML config
- [x] Verify multi-metric checkpoints are being saved before leaving training unattended

### Preventive
- [ ] Add a CLI flag to `train_monitored.py` for `--experiment-config` that reads the YAML
- [x] Add a startup log warning: "Using default checkpoint config (single val_loss only)"
  when no multi-metric config is provided
- [x] Add to LEARNINGS.md: always use run_experiment.py as the entry point for YAML-driven runs

## Impact

- ~40 minutes of GPU time wasted on fold 0 of dice_ce
- Training was progressing well (val_loss: 0.679 → 0.140, val_dice: 0.424 → 0.906)
- The metric_history.json is salvaged but model weights for best-by-dice epoch are lost
- Training must be restarted from scratch

## Lesson

**Never bypass the config entry point.** If the user implemented a YAML-driven config
system, always use the YAML-driven entry point. The CLI script is a lower-level tool
that doesn't pick up experiment-level configuration.
