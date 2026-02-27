# DynUNet Loss Variation v2 — Experiment Report

> **Experiment**: `dynunet_loss_variation_v2`
> **Date**: 2026-02-26 14:42 → 2026-02-27 16:14 (25 h 32 min)
> **Branch**: `feat/experiment-evaluation`
> **Hardware**: NVIDIA RTX 2070 Super (8 GB), 63 GB RAM, Intel i9-9900K
> **Dataset**: MiniVess — 70 volumes, 512x512xZ (Z: 5–110 slices), native resolution

## 1. Experiment Design

This experiment compares four loss functions for 3D microvessel segmentation using
DynUNet, with 3-fold cross-validation and 100 epochs per fold. The goal is to
understand the **topology-accuracy tradeoff** between standard overlap losses and
topology-preserving losses.

All folds use the same deterministic splits (`configs/splits/3fold_seed42.json`,
seed=42). Each fold trains on ~46–47 volumes and validates on ~23–24 volumes, with
100% MONAI CacheDataset caching.

Six best-model checkpoints are tracked per fold, each saved by its own metric.
MetricsReloaded evaluation (clDice, MASD with 95% bootstrap CIs) runs on the
best `val_compound_masd_cldice` checkpoint using sliding window inference on
full-resolution validation volumes.

```json
{
  "experiment": {
    "name": "dynunet_loss_variation_v2",
    "model": "dynunet",
    "losses": ["dice_ce", "cbdice", "dice_ce_cldice", "cbdice_cldice"],
    "num_folds": 3,
    "max_epochs": 100,
    "seed": 42,
    "total_fold_runs": 12,
    "compute_profile": "gpu_low",
    "extended_metric_frequency": 5,
    "primary_metric": "val_compound_masd_cldice"
  },
  "tracked_metrics": [
    {"name": "val_loss", "direction": "minimize", "patience": 30},
    {"name": "val_dice", "direction": "maximize", "patience": 30},
    {"name": "val_f1_foreground", "direction": "maximize", "patience": 30},
    {"name": "val_cldice", "direction": "maximize", "patience": 30},
    {"name": "val_masd", "direction": "minimize", "patience": 30},
    {"name": "val_compound_masd_cldice", "direction": "maximize", "patience": 30}
  ],
  "hardware": {
    "gpu": "NVIDIA RTX 2070 Super",
    "gpu_vram_mb": 8192,
    "ram_gb": 62.7,
    "peak_gpu_mb": 7399,
    "peak_ram_gb": 42.0,
    "num_workers": 2,
    "cache_rate": 1.0
  },
  "runtime": {
    "start": "2026-02-26T14:42:52",
    "end": "2026-02-27T16:14:00",
    "total_hours": 25.52,
    "monitor_snapshots": 2280,
    "warnings": 0,
    "crashes": 0
  }
}
```

## 2. Loss Function Descriptions

**`dice_ce`** — The nnU-Net standard (Isensee et al., 2021). Equal-weight combination
of Dice loss and cross-entropy. Serves as the baseline since it is the most widely
validated loss for medical image segmentation. Does not incorporate any topology or
boundary awareness.

**`cbdice`** — Centerline Boundary Dice Loss (Shi et al., 2024). A diameter-aware
loss that decomposes vessel supervision into three components: standard Dice (40%),
centerline Dice (30%), and boundary Dice (30%). The centerline and boundary components
use skeleton and surface extraction to provide geometry-aware gradients. Does not use
MONAI's SoftclDice.

**`dice_ce_cldice`** — Custom compound: 50% DiceCE + 50% SoftclDice (Shit et al.,
2021). Combines the region-based accuracy of DiceCE with the topology-preserving
property of soft centreline Dice. The clDice component computes soft skeletonization
via iterative min-pooling/max-pooling (3 iterations) and penalizes skeleton
discontinuities.

**`cbdice_cldice`** — Custom compound: 50% cbDice + 50% VesselCompoundLoss
(dice_ce_cldice). The most topology-heavy loss in the experiment, combining
diameter-aware centerline+boundary supervision with skeleton-following topology
preservation. This loss replaced the `warp` loss which completely failed in v1
(DSC ~0.015).

```json
{
  "losses": {
    "dice_ce": {
      "class": "monai.losses.DiceCELoss",
      "components": {"dice": 0.5, "cross_entropy": 0.5},
      "topology_aware": false
    },
    "cbdice": {
      "class": "CenterlineBoundaryDiceLoss (vendored)",
      "components": {"dice": 0.4, "centerline_dice": 0.3, "boundary_dice": 0.3},
      "topology_aware": true,
      "source": "Shi et al. (2024), MICCAI"
    },
    "dice_ce_cldice": {
      "class": "VesselCompoundLoss (custom)",
      "components": {"dice_ce": 0.5, "soft_cldice": 0.5},
      "topology_aware": true,
      "source": "Shit et al. (2021), CVPR"
    },
    "cbdice_cldice": {
      "class": "CbDiceClDiceLoss (custom)",
      "components": {"cbdice": 0.5, "vessel_compound": 0.5},
      "topology_aware": true,
      "replaces": "warp (failed in v1)"
    }
  }
}
```

## 3. Per-Fold Results

The evaluation was performed on the best `val_compound_masd_cldice` checkpoint for each
fold using sliding window inference on full-resolution validation volumes. MetricsReloaded
computes centreline DSC (clDice) and measured MASD with 95% bootstrap confidence intervals.

```json
{
  "per_fold_results": {
    "dice_ce": {
      "fold_0": {
        "dsc": 0.8164, "dsc_ci": [0.7934, 0.8382],
        "cldice": 0.8130, "cldice_ci": [0.7627, 0.8518],
        "masd": 2.2360, "masd_ci": [1.0760, 4.2173],
        "best_val_loss": 0.1853,
        "final_train_loss": 0.1663,
        "train_volumes": 46, "val_volumes": 24
      },
      "fold_1": {
        "dsc": 0.8433, "dsc_ci": [0.8143, 0.8691],
        "cldice": 0.8574, "cldice_ci": [0.8181, 0.8930],
        "masd": 1.2552, "masd_ci": [0.6019, 2.3492],
        "best_val_loss": 0.1622,
        "final_train_loss": 0.1685,
        "train_volumes": 47, "val_volumes": 23
      },
      "fold_2": {
        "dsc": 0.8130, "dsc_ci": [0.7833, 0.8425],
        "cldice": 0.8247, "cldice_ci": [0.7683, 0.8700],
        "masd": 1.5390, "masd_ci": [0.8353, 2.4304],
        "best_val_loss": 0.1757,
        "final_train_loss": 0.1620,
        "train_volumes": 47, "val_volumes": 23
      }
    },
    "cbdice": {
      "fold_0": {
        "dsc": 0.7399, "dsc_ci": [0.7037, 0.7711],
        "cldice": 0.7801, "cldice_ci": [0.7295, 0.8240],
        "masd": 3.0374, "masd_ci": [1.7832, 4.9562],
        "best_val_loss": 0.1598,
        "final_train_loss": 0.1532,
        "train_volumes": 46, "val_volumes": 24
      },
      "fold_1": {
        "dsc": 0.7966, "dsc_ci": [0.7518, 0.8367],
        "cldice": 0.8271, "cldice_ci": [0.7782, 0.8714],
        "masd": 1.5182, "masd_ci": [0.8455, 2.5715],
        "best_val_loss": 0.1460,
        "final_train_loss": 0.1480,
        "train_volumes": 47, "val_volumes": 23
      },
      "fold_2": {
        "dsc": 0.7634, "dsc_ci": [0.7271, 0.7969],
        "cldice": 0.7905, "cldice_ci": [0.7315, 0.8454],
        "masd": 1.8192, "masd_ci": [1.0850, 2.7134],
        "best_val_loss": 0.1542,
        "final_train_loss": 0.1421,
        "train_volumes": 47, "val_volumes": 23
      }
    },
    "dice_ce_cldice": {
      "fold_0": {
        "dsc": 0.7230, "dsc_ci": [0.6948, 0.7479],
        "cldice": 0.9074, "cldice_ci": [0.8846, 0.9267],
        "masd": 2.6976, "masd_ci": [1.4885, 4.7532],
        "best_val_loss": 0.2340,
        "final_train_loss": 0.1719,
        "train_volumes": 46, "val_volumes": 24
      },
      "fold_1": {
        "dsc": 0.7589, "dsc_ci": [0.7190, 0.7931],
        "cldice": 0.9060, "cldice_ci": [0.8741, 0.9316],
        "masd": 1.4627, "masd_ci": [0.9502, 2.2955],
        "best_val_loss": 0.1998,
        "final_train_loss": 0.1799,
        "train_volumes": 47, "val_volumes": 23
      },
      "fold_2": {
        "dsc": 0.7266, "dsc_ci": [0.6940, 0.7581],
        "cldice": 0.9003, "cldice_ci": [0.8723, 0.9238],
        "masd": 1.7197, "masd_ci": [1.1388, 2.4755],
        "best_val_loss": 0.2190,
        "final_train_loss": 0.1697,
        "train_volumes": 47, "val_volumes": 23
      }
    },
    "cbdice_cldice": {
      "fold_0": {
        "dsc": 0.7553, "dsc_ci": [0.7282, 0.7786],
        "cldice": 0.9077, "cldice_ci": [0.8860, 0.9260],
        "masd": 2.4623, "masd_ci": [1.3035, 4.4112],
        "best_val_loss": 0.1935,
        "final_train_loss": 0.1639,
        "train_volumes": 46, "val_volumes": 24
      },
      "fold_1": {
        "dsc": 0.7936, "dsc_ci": [0.7557, 0.8269],
        "cldice": 0.9142, "cldice_ci": [0.8832, 0.9402],
        "masd": 1.2724, "masd_ci": [0.7709, 2.0853],
        "best_val_loss": 0.1704,
        "final_train_loss": 0.1685,
        "train_volumes": 47, "val_volumes": 23
      },
      "fold_2": {
        "dsc": 0.7659, "dsc_ci": [0.7341, 0.7946],
        "cldice": 0.8960, "cldice_ci": [0.8656, 0.9227],
        "masd": 1.4775, "masd_ci": [0.9228, 2.1530],
        "best_val_loss": 0.1847,
        "final_train_loss": 0.1608,
        "train_volumes": 47, "val_volumes": 23
      }
    }
  }
}
```

## 4. Cross-Loss Summary

```json
{
  "cross_loss_means": {
    "dice_ce": {
      "mean_dsc": 0.8242, "std_dsc": 0.0172,
      "mean_cldice": 0.8317, "std_cldice": 0.0225,
      "mean_masd": 1.6767, "std_masd": 0.4919
    },
    "cbdice": {
      "mean_dsc": 0.7666, "std_dsc": 0.0284,
      "mean_cldice": 0.7992, "std_cldice": 0.0245,
      "mean_masd": 2.1249, "std_masd": 0.7762
    },
    "dice_ce_cldice": {
      "mean_dsc": 0.7362, "std_dsc": 0.0197,
      "mean_cldice": 0.9046, "std_cldice": 0.0037,
      "mean_masd": 1.9600, "std_masd": 0.6397
    },
    "cbdice_cldice": {
      "mean_dsc": 0.7716, "std_dsc": 0.0196,
      "mean_cldice": 0.9060, "std_cldice": 0.0091,
      "mean_masd": 1.7374, "std_masd": 0.6103
    }
  },
  "rankings": {
    "by_dsc":    ["dice_ce", "cbdice_cldice", "cbdice", "dice_ce_cldice"],
    "by_cldice": ["cbdice_cldice", "dice_ce_cldice", "dice_ce", "cbdice"],
    "by_masd":   ["dice_ce", "cbdice_cldice", "dice_ce_cldice", "cbdice"]
  }
}
```

### Interpretation

**`dice_ce` is the best overlap loss.** With a mean DSC of 0.824, it outperforms all
topology-aware losses by 5–9 percentage points. This is expected — DiceCE directly
optimizes volumetric overlap without any competing objective. Its MASD (1.68) is also
the best, meaning predictions are closest to the ground truth surface. However, its
clDice (0.832) is the second-lowest, indicating that topological continuity of the
vessel tree is not preserved as well.

**`cbdice` underperforms on all metrics.** Mean DSC (0.767), clDice (0.799), and MASD
(2.12) are all worse than the baseline. The centerline+boundary decomposition from
Shi et al. (2024) appears to fragment the optimization signal too much for our small
dataset and 8 GB VRAM constraint. The fold-0 MASD of 3.04 is particularly poor. This
loss may benefit from longer training or larger batch sizes.

**`dice_ce_cldice` dominates topology preservation.** The clDice of 0.904 is
dramatically higher than `dice_ce`'s 0.832 — an 8.7% improvement in centreline overlap.
The standard deviation across folds is remarkably low (0.004), showing that the
topology benefit is consistent and not fold-dependent. However, this comes at a steep
cost: DSC drops to 0.736 (−8.8% vs baseline) and MASD worsens to 1.96 (+17% vs
baseline). The model preserves vessel connectivity but with less precise boundaries.

**`cbdice_cldice` is the best balanced loss.** It matches `dice_ce_cldice`'s topology
(clDice 0.906 vs 0.905) while recovering significant overlap and surface accuracy:
DSC 0.772 (+3.5% vs `dice_ce_cldice`) and MASD 1.74 (−11% vs `dice_ce_cldice`). By
combining cbDice's diameter-aware boundary supervision with clDice's skeleton
preservation, it achieves the best tradeoff point. Its MASD (1.74) is closer to the
baseline (1.68) than any other topology-aware loss.

### The Topology-Accuracy Tradeoff

The results reveal a clear and consistent pattern: adding clDice supervision shifts
~9% of DSC into ~7% of clDice improvement. This is the fundamental tradeoff documented
in Shit et al. (2021) — skeleton-aware losses improve topological continuity at the
expense of volumetric precision because they redirect gradient signal from boundary
refinement toward centreline alignment.

The compound primary metric (`val_compound_masd_cldice = 0.5*(1 - masd/50) + 0.5*clDice`)
is designed to navigate this tradeoff. Under this metric, `cbdice_cldice` should rank
highest because it achieves near-optimal clDice with the least MASD penalty.

## 5. Fold Variance

```json
{
  "fold_variance_analysis": {
    "observation": "Fold 0 consistently shows worse MASD across all losses",
    "fold_0_masd_range": [2.236, 3.037],
    "fold_1_masd_range": [1.255, 1.518],
    "fold_2_masd_range": [1.478, 1.819],
    "likely_cause": "Fold 0 validation set (24 volumes) contains harder cases or outlier volumes (e.g., mv02 with atypical voxel spacing 4.97 um)",
    "fold_split_sizes": {
      "fold_0": {"train": 46, "val": 24},
      "fold_1": {"train": 47, "val": 23},
      "fold_2": {"train": 47, "val": 23}
    }
  }
}
```

Fold 0 has the highest MASD across all four losses (2.24–3.04), while folds 1 and 2
are consistently lower (1.26–1.82). This suggests the fold 0 validation set contains
volumes that are harder to segment precisely — possibly including outlier volumes with
atypical voxel spacing or particularly thin vessel structures. Fold 1 is consistently
the easiest, producing the best metrics for every loss.

The fold variance in clDice is much smaller for the topology-aware losses (std 0.004
for `dice_ce_cldice`) than for the baseline (std 0.023 for `dice_ce`). This indicates
that **topology-aware losses generalize more consistently across data splits**, even if
their absolute overlap scores are lower.

## 6. Training Dynamics

```json
{
  "training_dynamics": {
    "dice_ce": {
      "epoch1_val_loss": [1.1148, 1.0829, 1.1067],
      "epoch100_val_loss": [0.1855, 0.1622, 0.1758],
      "convergence": "smooth, monotonic decrease",
      "train_val_gap": "minimal (0.01-0.02), no overfitting"
    },
    "cbdice": {
      "epoch1_val_loss": [0.5720, 0.5615, 0.5700],
      "epoch100_val_loss": [0.1598, 0.1460, 0.1542],
      "convergence": "smooth, lower starting loss (different scale)",
      "train_val_gap": "minimal, well-regularized"
    },
    "dice_ce_cldice": {
      "epoch1_val_loss": [0.9980, 0.9789, 0.9973],
      "epoch100_val_loss": [0.2341, 0.1997, 0.2191],
      "convergence": "smooth but higher final loss (topology component adds irreducible term)",
      "train_val_gap": "moderate (0.04-0.06), slight underfitting from clDice regularization"
    },
    "cbdice_cldice": {
      "epoch1_val_loss": [0.7855, 0.7707, 0.7841],
      "epoch100_val_loss": [0.1936, 0.1704, 0.1848],
      "convergence": "smooth, intermediate between cbdice and dice_ce_cldice",
      "train_val_gap": "moderate, well-behaved"
    }
  }
}
```

All four losses converge smoothly with no NaN values, training instabilities, or
sudden divergences across 1,200 total epochs. The cosine annealing learning rate
schedule (peak 1e-4, warmup 10 epochs) drives learning rate to zero by epoch 100,
ensuring clean convergence.

The `dice_ce_cldice` and `cbdice_cldice` losses have higher final val_loss values
(0.19–0.23) compared to `dice_ce` (0.16–0.19) and `cbdice` (0.15–0.16). This is not
a sign of worse training — it reflects the topology components adding an irreducible
loss term. The clDice component penalizes imperfect skeleton overlap that cannot be
eliminated without perfectly matching all centreline voxels, so the loss floor is
naturally higher.

## 7. Resource Utilization

```json
{
  "resource_utilization": {
    "peak_gpu_mb": 7399,
    "peak_gpu_utilization_pct": 90.3,
    "peak_ram_gb": 42.0,
    "peak_ram_utilization_pct": 67.0,
    "peak_process_rss_gb": 17.7,
    "swap_usage_gb": 13.5,
    "avg_epoch_time_sec": {
      "normal_epoch": 32,
      "extended_epoch_with_metricsreloaded": 240,
      "effective_average": 74
    },
    "avg_fold_time_hours": 2.13,
    "total_time_hours": 25.52,
    "monitor_warnings": 0,
    "oom_events": 0
  }
}
```

The experiment ran within the 8 GB VRAM budget with no OOM events. Peak GPU usage
(7,399 MB) was reached during `cbdice` training, which requires additional memory for
skeleton and boundary extraction. The MetricsReloaded extended evaluation (every 5th
epoch) adds ~3.5 minutes per evaluation due to skeleton computation on 23–24
full-resolution volumes, but the frequency-5 schedule keeps the average epoch time
at ~74 seconds.

## 8. Conclusions and Next Steps

### Key Findings

1. **The topology-accuracy tradeoff is real and consistent.** Adding clDice supervision
   improves centreline DSC by +7–9% but costs −5–9% volumetric DSC.

2. **`cbdice_cldice` offers the best tradeoff.** It achieves 0.906 clDice (matching the
   best topology loss) with only −5.3% DSC penalty (vs −8.8% for `dice_ce_cldice`).

3. **`cbdice` alone is not competitive.** Without the clDice component, the
   centerline+boundary decomposition underperforms the baseline on all metrics.

4. **Fold variance in topology metrics is lower for topology-aware losses.** clDice
   standard deviation is 5x smaller for `dice_ce_cldice` (0.004) vs `dice_ce` (0.023),
   suggesting more robust topological generalization.

5. **Zero instabilities across 1,200 epochs.** No NaN losses, no OOM, no crashes.
   The training pipeline is production-ready.

### Recommended Next Steps

- **Cross-loss statistical comparison** with paired bootstrap testing (Issue #79 pattern)
- **Compound metric ranking** to determine the primary-metric winner
- **Ensemble construction** from best checkpoints across losses (Issue #80)
- **Boundary Loss** (Issue #100) and **Generalized Surface Loss** (Issue #101) as v3
  additions to bridge the topology-surface gap
- **Pure clDice ablation** to isolate the topology contribution vs. the compound

### References

- Isensee et al. (2021) — nnU-Net (Nature Methods)
- Shit et al. (2021) — clDice (CVPR)
- Shi et al. (2024) — cbDice (MICCAI)
- Kervadec et al. (2019) — Boundary Loss (MIDL)
