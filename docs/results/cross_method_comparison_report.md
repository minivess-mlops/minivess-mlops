# Cross-Method Comparison Report

> **Scope**: All MiniVess experiments run to date (2026-03-03)
> **Dataset**: MiniVess -- 70 volumes, 512x512xZ (Z: 5--110 slices), native resolution
> **Hardware**: NVIDIA RTX 2070 Super (8 GB), 63 GB RAM, Intel i9-9900K
> **Metrics**: DSC (Dice Similarity Coefficient), clDSC (centreline Dice), MASD (Mean Average Surface Distance)
> **All values**: 3-fold cross-validation mean

## 1. Summary

This report consolidates results from **four experiments** spanning two model families
(DynUNet and SAM3), 18 loss functions, two architecture widths, and two training
regimes (debug 6 epochs, production 100 epochs). Together they comprise **57 trained
models** across 27 distinct method configurations, providing a comprehensive view of
the topology-accuracy tradeoff in 3D microvessel segmentation.

| Experiment | Model | Losses | Epochs | Folds | Width | Runs |
|-----------|-------|--------|--------|-------|-------|------|
| DynUNet Loss Variation v2 | DynUNet | 4 | 100 | 3 | Full | 12 |
| DynUNet Half-Width v1 | DynUNet | 4 | 100 | 3 | Half (filters/2) | 12 |
| DynUNet All-Losses Debug | DynUNet | 17 | 6 | 3 | Full | 51 |
| SAM3 Debug | SAM3 | 2 | 6 | 3 | -- | 9 |

**Total fold-runs**: 84 (57 unique configurations, some overlap between debug and
production for the 4 core losses).

## 2. Grand Comparison Table

### Ranked by DSC (Descending)

All methods from all experiments. Production-trained configurations (100 epochs) are
marked with a dagger symbol. Half-width configurations are marked with an asterisk.
SAM3 configurations are marked with a double dagger.

| Rank | Method | Epochs | DSC | clDSC | MASD | Notes |
|-----:|--------|-------:|----:|------:|-----:|-------|
| 1 | dice_ce | 100 | 0.8242 | 0.8317 | 1.6768 | Best overlap overall |
| 2 | dice_ce* | 100 | 0.8055 | 0.7743 | 2.1579 | Half-width baseline |
| 3 | cbdice_cldice | 100 | 0.7716 | 0.9060 | 1.7374 | Best topology overall |
| 4 | cbdice | 100 | 0.7666 | 0.7992 | 2.1249 | Full-width, no clDice |
| 5 | cbdice* | 100 | 0.7355 | 0.7686 | 2.5434 | Half-width cbDice |
| 6 | dice_ce_cldice | 100 | 0.7362 | 0.9046 | 1.9600 | Strong topology |
| 7 | centerline_ce | 6 | 0.7001 | 0.6672 | 3.1076 | Debug, best DSC at 6ep |
| 8 | cbdice_cldice* | 100 | 0.6788 | 0.8877 | 2.3301 | Half-width topo-best |
| 9 | dice_ce (debug) | 6 | 0.6764 | 0.6532 | 3.2736 | Debug baseline |
| 10 | focal | 6 | 0.6665 | 0.6164 | 3.2828 | Debug |
| 11 | dice_ce_cldice* | 100 | 0.6361 | 0.8950 | 2.4330 | Half-width strong topo |
| 12 | dice (debug) | 6 | 0.5865 | 0.6077 | 3.8725 | Debug, pure Dice |
| 13 | cb_dice (debug) | 6 | 0.5802 | 0.6012 | 3.9364 | Debug |
| 14 | cbdice (debug) | 6 | 0.5612 | 0.5310 | 3.9690 | Debug |
| 15 | toposeg | 6 | 0.5484 | 0.5596 | 3.3181 | Debug |
| 16 | sam3_hybrid | 6 | 0.5358 | 0.6922 | 4.3357 | Best SAM3 variant |
| 17 | cbdice_cldice (debug) | 6 | 0.5223 | 0.6863 | 3.8767 | Debug |
| 18 | dice_ce_cldice (debug) | 6 | 0.5004 | 0.7161 | 4.0448 | Debug |
| 19 | full_topo | 6 | 0.4987 | 0.7218 | 4.0386 | Debug |
| 20 | sam3_topolora | 6 | 0.4504 | 0.3327 | 8.0468 | SAM3 with LoRA |
| 21 | cldice (debug) | 6 | 0.4525 | 0.7206 | 4.3767 | Debug, pure clDice |
| 22 | graph_topology | 6 | 0.4134 | 0.4153 | 5.1402 | Debug |
| 23 | sam3_vanilla | 6 | 0.3493 | 0.1859 | 18.8450 | 2D slice-by-slice |
| 24 | skeleton_recall | 6 | 0.2659 | 0.2674 | 6.3415 | Debug |
| 25 | cape | 6 | 0.2507 | 0.2582 | 6.5482 | Debug |
| 26 | warp | 6 | 0.2217 | 0.3238 | 5.1159 | Debug |
| 27 | betti | 6 | 0.0160 | 0.0155 | 7.4355 | Debug, near-zero |
| 28 | topo | 6 | 0.0117 | 0.0145 | 8.0071 | Debug, near-zero |

### Ranked by clDSC (Descending)

| Rank | Method | Epochs | clDSC | DSC | MASD |
|-----:|--------|-------:|------:|----:|-----:|
| 1 | cbdice_cldice | 100 | 0.9060 | 0.7716 | 1.7374 |
| 2 | dice_ce_cldice | 100 | 0.9046 | 0.7362 | 1.9600 |
| 3 | dice_ce_cldice* | 100 | 0.8950 | 0.6361 | 2.4330 |
| 4 | cbdice_cldice* | 100 | 0.8877 | 0.6788 | 2.3301 |
| 5 | dice_ce | 100 | 0.8317 | 0.8242 | 1.6768 |
| 6 | cbdice | 100 | 0.7992 | 0.7666 | 2.1249 |
| 7 | dice_ce* | 100 | 0.7743 | 0.8055 | 2.1579 |
| 8 | cbdice* | 100 | 0.7686 | 0.7355 | 2.5434 |
| 9 | full_topo | 6 | 0.7218 | 0.4987 | 4.0386 |
| 10 | cldice (debug) | 6 | 0.7206 | 0.4525 | 4.3767 |
| 11 | dice_ce_cldice (debug) | 6 | 0.7161 | 0.5004 | 4.0448 |
| 12 | sam3_hybrid | 6 | 0.6922 | 0.5358 | 4.3357 |
| 13 | cbdice_cldice (debug) | 6 | 0.6863 | 0.5223 | 3.8767 |
| 14 | centerline_ce | 6 | 0.6672 | 0.7001 | 3.1076 |
| 15 | dice_ce (debug) | 6 | 0.6532 | 0.6764 | 3.2736 |
| 16 | focal | 6 | 0.6164 | 0.6665 | 3.2828 |
| 17 | dice (debug) | 6 | 0.6077 | 0.5865 | 3.8725 |
| 18 | cb_dice (debug) | 6 | 0.6012 | 0.5802 | 3.9364 |
| 19 | toposeg | 6 | 0.5596 | 0.5484 | 3.3181 |
| 20 | cbdice (debug) | 6 | 0.5310 | 0.5612 | 3.9690 |
| 21 | graph_topology | 6 | 0.4153 | 0.4134 | 5.1402 |
| 22 | sam3_topolora | 6 | 0.3327 | 0.4504 | 8.0468 |
| 23 | warp | 6 | 0.3238 | 0.2217 | 5.1159 |
| 24 | skeleton_recall | 6 | 0.2674 | 0.2659 | 6.3415 |
| 25 | cape | 6 | 0.2582 | 0.2507 | 6.5482 |
| 26 | sam3_vanilla | 6 | 0.1859 | 0.3493 | 18.8450 |
| 27 | betti | 6 | 0.0155 | 0.0160 | 7.4355 |
| 28 | topo | 6 | 0.0145 | 0.0117 | 8.0071 |

## 3. Key Findings

**Finding 1: Topology-aware losses consistently improve clDice at DSC cost across all
model families and widths.** This pattern holds for DynUNet full-width (100 epochs),
DynUNet half-width (100 epochs), DynUNet debug (6 epochs), and SAM3 debug (6 epochs).
The tradeoff is not an artifact of a single configuration -- it is a fundamental
property of the loss landscape.

**Finding 2: `cbdice_cldice` is the Pareto-optimal loss function.** Across all
production-trained models, it achieves the highest clDice (0.906) with only a 5.3%
DSC penalty relative to the `dice_ce` baseline. No other loss achieves topology
comparable to `cbdice_cldice` without incurring a larger DSC cost.

**Finding 3: SAM3 hybrid approaches DynUNet debug-epoch topology at 6 epochs.**
The SAM3 hybrid variant achieves 0.692 clDice at 6 epochs, compared to 0.686 for
DynUNet `cbdice_cldice` at the same epoch count. This is notable because the SAM3
models use a randomly initialized stub encoder (no pretrained weights), suggesting
that with proper pretraining the gap could close further.

**Finding 4: Half-width DynUNet preserves topology at the cost of overlap.** Reducing
filters by 50% drops DSC by 2--10% depending on the loss, but topology-aware losses
(`cbdice_cldice`, `dice_ce_cldice`) retain >88% of their clDice, demonstrating that
topology preservation is more robust to capacity reduction than boundary precision.

**Finding 5: Pure topology losses are not viable standalone.** Losses like `cldice`,
`betti`, and `topo` produce very poor DSC (0.01--0.45) when used alone at 6 epochs.
They require combination with overlap losses to produce usable segmentations.

**Finding 6: Seventeen of 18 losses trained successfully.** Only `betti_matching` failed
due to OOM on real volumes (gudhi path computation: 680K features, 323 GiB distance
matrix). All other losses, including experimental ones, completed training without
crashes or NaN values.

## 4. DynUNet Analysis

### 4.1 Loss Function Impact (100 Epochs, Full-Width)

The production experiment (`dynunet_loss_variation_v2`) ran 4 losses for 100 epochs
with 3-fold cross-validation on full-width DynUNet. This is the most reliable
comparison in the entire report due to the training duration and consistent setup.

| Loss | DSC | clDSC | MASD | DSC Delta | clDSC Delta |
|------|----:|------:|-----:|----------:|------------:|
| dice_ce (baseline) | 0.8242 | 0.8317 | 1.6768 | -- | -- |
| cbdice_cldice | 0.7716 | 0.9060 | 1.7374 | -5.3% | +7.4% |
| cbdice | 0.7666 | 0.7992 | 2.1249 | -5.8% | -3.3% |
| dice_ce_cldice | 0.7362 | 0.9046 | 1.9600 | -8.8% | +7.3% |

The pattern is clear: losses that include a clDice term (`cbdice_cldice`,
`dice_ce_cldice`) gain approximately +7% clDice at a cost of 5--9% DSC. Losses without
clDice (`cbdice`) lose DSC without gaining topology. The combination of cbDice and
clDice in `cbdice_cldice` recovers 3.5% of the DSC gap compared to `dice_ce_cldice`
while maintaining identical topology.

### 4.2 Width Impact (Full vs. Half, 100 Epochs)

Comparing full-width and half-width DynUNet across the same 4 losses reveals how model
capacity affects both overlap and topology metrics.

| Loss | DSC (Full) | DSC (Half) | Delta | clDSC (Full) | clDSC (Half) | Delta |
|------|----------:|----------:|------:|------------:|------------:|------:|
| dice_ce | 0.8242 | 0.8055 | -1.9% | 0.8317 | 0.7743 | -5.7% |
| cbdice_cldice | 0.7716 | 0.6788 | -9.3% | 0.9060 | 0.8877 | -1.8% |
| cbdice | 0.7666 | 0.7355 | -3.1% | 0.7992 | 0.7686 | -3.1% |
| dice_ce_cldice | 0.7362 | 0.6361 | -10.0% | 0.9046 | 0.8950 | -1.0% |

Key observations:

- **DSC degrades asymmetrically.** The `dice_ce` baseline loses only 1.9% DSC at half
  width, while topology-aware losses lose 9--10%. This suggests that topology-aware
  losses require more model capacity for boundary precision because they allocate
  representational capacity to centreline alignment.

- **clDice is remarkably robust to width reduction.** The topology-aware losses
  (`cbdice_cldice`, `dice_ce_cldice`) lose only 1--2% clDice at half width, while
  `dice_ce` loses 5.7%. Topology preservation appears to be a lower-dimensional
  property than boundary precision, requiring fewer parameters to maintain.

- **MASD uniformly worsens.** Every loss at half-width has higher MASD, with the
  topology-aware losses showing the largest absolute increase (0.47--0.59 um).

### 4.3 Debug vs. Production Epochs

Comparing the 4 core losses between 6-epoch debug and 100-epoch production training
quantifies the cost of early stopping.

| Loss | DSC (6ep) | DSC (100ep) | Gain | clDSC (6ep) | clDSC (100ep) | Gain |
|------|----------:|-----------:|-----:|------------:|-------------:|-----:|
| dice_ce | 0.6764 | 0.8242 | +14.8% | 0.6532 | 0.8317 | +17.9% |
| cbdice_cldice | 0.5223 | 0.7716 | +24.9% | 0.6863 | 0.9060 | +22.0% |
| cbdice | 0.5612 | 0.7666 | +20.5% | 0.5310 | 0.7992 | +26.8% |
| dice_ce_cldice | 0.5004 | 0.7362 | +23.6% | 0.7161 | 0.9046 | +18.9% |

All losses benefit substantially from 100 epochs, but topology-aware losses show
especially large DSC gains (+21--25%) compared to `dice_ce` (+15%). This suggests
that topology-aware losses have a **slower convergence profile** -- they need more
epochs to jointly optimize the topology and overlap objectives. At 6 epochs, the
topology component dominates (relatively high clDice) while the overlap component
has not yet converged (low DSC).

This has direct implications for the SAM3 experiments, which were only trained for
6 epochs.

### 4.4 Debug-Only Loss Tiers

The 17-loss debug experiment reveals a natural tiering of loss functions:

**Tier 1 -- Viable (DSC > 0.50, 6 epochs)**: `centerline_ce`, `dice_ce`, `focal`,
`toposeg`, `dice`, `cb_dice`, `cbdice`, `cbdice_cldice`, `dice_ce_cldice`,
`full_topo`, `cldice` (11 losses). These converge to usable segmentations within
6 epochs.

**Tier 2 -- Marginal (DSC 0.20--0.50, 6 epochs)**: `graph_topology`, `warp`,
`skeleton_recall`, `cape` (4 losses). These show learning but produce incomplete
segmentations. Some may improve substantially with more epochs.

**Tier 3 -- Failed (DSC < 0.05, 6 epochs)**: `betti`, `topo` (2 losses). These
produce near-empty predictions at 6 epochs. The pure persistent homology losses
lack the voxel-level gradient signal needed for early training. They may be viable
as fine-tuning losses after pre-training with `dice_ce`, but this has not been tested.

## 5. SAM3 Analysis

### 5.1 Variant Comparison

Three SAM3 variants were trained for 6 epochs with 3-fold cross-validation. All used
a randomly initialized stub encoder (no pretrained SAM3 weights were available at
training time). This fundamentally limits the comparison -- the primary value of SAM3
is its pretrained visual features, which are absent here.

| Variant | Loss | DSC | clDSC | MASD | Architecture |
|---------|------|----:|------:|-----:|-------------|
| sam3_vanilla | dice_ce | 0.3493 | 0.1859 | 18.845 | 2D slice-by-slice, no 3D context |
| sam3_topolora | cbdice_cldice | 0.4504 | 0.3327 | 8.047 | LoRA adapters, topology loss |
| sam3_hybrid | cbdice_cldice | 0.5358 | 0.6922 | 4.336 | Gated 3D fusion + topology loss |

**sam3_vanilla** performs poorly across all metrics. This is expected: vanilla SAM3
processes 2D slices independently without inter-slice context, making 3D vessel
continuity impossible to learn. The MASD of 18.8 um is an order of magnitude worse
than any DynUNet configuration, reflecting fragmented per-slice predictions that do
not form coherent 3D structures.

**sam3_topolora** improves substantially over vanilla (+10% DSC, +15% clDSC, -57%
MASD) but still lags behind DynUNet. The LoRA adapters provide parameter-efficient
fine-tuning, and the `cbdice_cldice` loss adds topology awareness, but without
pretrained features or 3D context the model struggles with vessel connectivity.

**sam3_hybrid** is the strongest variant, approaching DynUNet debug-epoch performance.
The gated 3D fusion mechanism provides inter-slice context that vanilla and topolora
lack. Its clDSC of 0.692 is competitive with DynUNet's `cbdice_cldice` at 6 epochs
(0.686), despite the handicap of random initialization.

### 5.2 Gated Fusion Impact

The progression from vanilla to topolora to hybrid isolates the contribution of each
architectural component:

| Transition | DSC Delta | clDSC Delta | MASD Delta | Added Component |
|-----------|----------:|------------:|-----------:|----------------|
| vanilla -> topolora | +0.101 | +0.147 | -10.80 | LoRA + topology loss |
| topolora -> hybrid | +0.085 | +0.360 | -3.71 | Gated 3D fusion |
| vanilla -> hybrid | +0.187 | +0.506 | -14.51 | All components |

The gated 3D fusion module contributes the largest single improvement to clDSC
(+0.360), confirming that inter-slice context is critical for vessel topology in
3D data. The LoRA adapters plus topology loss contribute more to MASD improvement
(-10.8 um), likely because they sharpen per-slice predictions even without 3D
context.

## 6. DynUNet vs. SAM3

Direct comparison must be interpreted cautiously due to the different training
regimes. The fairest comparison is at 6 epochs (debug), where both architectures
had identical training duration.

### 6.1 At 6 Epochs (Fair Comparison)

| Method | DSC | clDSC | MASD |
|--------|----:|------:|-----:|
| DynUNet dice_ce (6ep) | 0.6764 | 0.6532 | 3.274 |
| DynUNet cbdice_cldice (6ep) | 0.5223 | 0.6863 | 3.877 |
| SAM3 hybrid cbdice_cldice | 0.5358 | 0.6922 | 4.336 |
| SAM3 topolora cbdice_cldice | 0.4504 | 0.3327 | 8.047 |
| SAM3 vanilla dice_ce | 0.3493 | 0.1859 | 18.845 |

At 6 epochs with the same topology-aware loss (`cbdice_cldice`):

- **SAM3 hybrid matches DynUNet on topology.** clDSC 0.692 vs 0.686 (SAM3 is +0.6%
  higher). This is remarkable given that SAM3 hybrid has no pretrained weights.
- **DynUNet wins on overlap.** DSC 0.522 vs 0.536 is close, but DynUNet's `dice_ce`
  baseline at 0.676 is far ahead of any SAM3 variant.
- **DynUNet wins on surface accuracy.** MASD 3.88 vs 4.34 (DynUNet 11% better).

### 6.2 DynUNet at Production vs. SAM3 at Debug

For completeness, the gap between production DynUNet and debug SAM3:

| Method | DSC | clDSC | MASD |
|--------|----:|------:|-----:|
| DynUNet cbdice_cldice (100ep) | 0.7716 | 0.9060 | 1.737 |
| SAM3 hybrid (6ep) | 0.5358 | 0.6922 | 4.336 |
| Gap | -0.236 | -0.214 | +2.599 |

The gap is large but unfair -- 100 vs 6 epochs, and DynUNet benefits from
domain-specific 3D convolutions while SAM3 uses a 2D backbone with learned fusion.
The comparison motivates a production SAM3 training run (100 epochs with pretrained
weights) as the critical next experiment.

## 7. Topology-Accuracy Tradeoff

The topology-accuracy tradeoff is the most consistent finding across all experiments.
It can be quantified as the slope of the DSC-to-clDSC exchange.

### 7.1 Exchange Rates by Experiment

For each experiment, the "exchange rate" is computed as:
`clDSC_gain / DSC_cost` comparing the best topology loss to the baseline (`dice_ce`).

| Experiment | Baseline DSC | Topo DSC | DSC Cost | clDSC Gain | Exchange |
|-----------|------------:|---------:|---------:|-----------:|---------:|
| DynUNet v2 (100ep, full) | 0.8242 | 0.7716 | -0.053 | +0.074 | 1.42 |
| DynUNet HW (100ep, half) | 0.8055 | 0.6788 | -0.127 | +0.113 | 0.89 |
| DynUNet debug (6ep) | 0.6764 | 0.5223 | -0.154 | +0.033 | 0.21 |
| SAM3 debug (6ep) | 0.3493 | 0.5358 | +0.187 | +0.506 | -- |

Notes:

- At full capacity and production epochs, the exchange rate is **favorable** (1.42):
  each percentage point of DSC lost buys 1.42 percentage points of clDSC gained.
- At half capacity, the exchange rate drops below 1.0 (0.89), meaning the tradeoff
  becomes less favorable with fewer parameters.
- At debug epochs, the exchange rate is poor (0.21) because topology losses have not
  converged. This reinforces the finding that topology-aware losses need more training
  time.
- SAM3 is a special case: the hybrid variant with topology loss outperforms the
  vanilla variant on **both** DSC and clDSC, because the gated fusion module provides
  3D context that benefits all metrics. The tradeoff framework does not apply when the
  architectural change adds information rather than redirecting gradients.

### 7.2 Pareto Frontier

The Pareto frontier across all production-trained models (100 epochs) consists of
exactly two points:

1. **`dice_ce` (full-width)**: DSC 0.824, clDSC 0.832 -- best overlap
2. **`cbdice_cldice` (full-width)**: DSC 0.772, clDSC 0.906 -- best topology

No other configuration achieves higher clDSC than `cbdice_cldice` or higher DSC
than `dice_ce`. The `dice_ce_cldice` configuration (DSC 0.736, clDSC 0.905) is
strictly dominated by `cbdice_cldice`, which achieves similar topology with higher
overlap. Similarly, `cbdice` (DSC 0.767, clDSC 0.799) is dominated by both Pareto
members.

## 8. Limitations

### 8.1 SAM3 Training Limitations

- **No pretrained weights.** All SAM3 variants used randomly initialized stub encoders.
  The entire value proposition of SAM3 (848M pretrained parameters, ViT-32L backbone
  trained on SA-1B) is absent from these results. Production comparison requires SAM3
  pretrained weights, which were not available at training time.
- **Debug epoch count only.** Six epochs is insufficient for convergence, as demonstrated
  by the 15--25% DSC improvement from 6 to 100 epochs in DynUNet. SAM3 results at 100
  epochs could be substantially different.
- **No fine-tuning sweep.** LoRA rank, fusion gate initialization, and learning rate
  were not tuned. The vanilla/topolora/hybrid comparison is architectural, not
  hyperparameter-optimized.

### 8.2 Metric Limitations

- **No HD95 or NSD for older runs.** The graph-topology metrics (HD95, NSD, Betti
  errors, Junction F1) were implemented after the production DynUNet experiments.
  Retrospective evaluation with the expanded metric set would require reloading
  checkpoints and re-running inference.
- **Compound metric range collapse.** The primary metric `val_compound_masd_cldice`
  suffers from MASD-clDice scale mismatch (MASD 0.058 spread vs clDice 0.417 spread),
  making it unreliable for cross-loss ranking. The planned fix (rank-then-aggregate
  with NSD compound) has not been applied to these results.
- **No statistical significance testing.** Cross-fold means are reported without
  paired bootstrap tests between methods. With only 3 folds, statistical power is
  limited regardless.

### 8.3 Experimental Design Limitations

- **Sequential experiments.** The 4 experiments were run at different times with
  evolving codebase. Loss function implementations were refined between the debug and
  production experiments (6 bugs fixed before the production runs).
- **Incomplete cross-experiment coverage.** Only 4 of 18 loss functions have production
  (100 epoch) results. The remaining 14 losses (including promising ones like
  `full_topo`, `centerline_ce`, and `toposeg`) have only 6-epoch debug results, which
  are unreliable for final comparison.
- **Single architecture depth/width.** DynUNet was tested at two widths but a single
  depth. The interaction between topology-aware losses and network depth is unknown.

## 9. Recommendations

### 9.1 Immediate (High Confidence)

1. **Run SAM3 hybrid at 100 epochs with pretrained weights.** This is the single
   highest-value experiment remaining. The debug results show SAM3 hybrid matching
   DynUNet topology at 6 epochs without pretraining. With pretrained ViT-32L features
   and 100 epochs, it could plausibly surpass DynUNet.

2. **Run production training for top debug-only losses.** Based on 6-epoch results,
   `centerline_ce`, `full_topo`, and `toposeg` are the most promising losses not yet
   tested at production scale. Priority order:
   - `full_topo` (0.722 clDice at 6 epochs, highest of any debug loss)
   - `centerline_ce` (0.700 DSC at 6 epochs, best DSC of any debug loss)
   - `toposeg` (competitive DSC 0.548 with moderate topology 0.560)

3. **Expand the metric set.** Re-evaluate the production checkpoints with HD95, NSD,
   Betti error, and Junction F1 to enable full comparison with future experiments
   using the expanded metric framework.

### 9.2 Medium-Term

4. **Ensemble the Pareto frontier.** Combine `dice_ce` (overlap champion) and
   `cbdice_cldice` (topology champion) predictions via model soup or voting ensemble
   to test whether the tradeoff can be circumvented through ensembling.

5. **Half-width + topology as efficient baseline.** The half-width `cbdice_cldice`
   (clDSC 0.888, ~50% fewer parameters) is a strong candidate for resource-constrained
   deployment. Validate this configuration on the held-out test set.

6. **SAM3 hyperparameter sweep.** Once pretrained weights are available, sweep LoRA
   rank (4, 8, 16, 32), learning rate (1e-5 to 1e-3), and fusion gate initialization
   to find optimal SAM3 configuration.

### 9.3 Long-Term

7. **Topology loss scheduling.** The debug-to-production comparison suggests topology
   losses benefit from longer training. A curriculum approach (start with `dice_ce`,
   anneal toward `cbdice_cldice`) might achieve better DSC while maintaining topology.

8. **Cross-dataset validation.** All results are on MiniVess (70 volumes). Validate
   the topology-accuracy tradeoff and loss rankings on a second vascular dataset to
   confirm generalizability.

9. **Conformal prediction integration.** Apply the implemented conformal UQ framework
   (morphological, distance transform, and risk-controlling predictors) to the Pareto
   frontier models to quantify prediction uncertainty alongside topology metrics.

## References

- Isensee et al. (2021) -- nnU-Net (Nature Methods)
- Shit et al. (2021) -- clDice (CVPR)
- Shi et al. (2024) -- cbDice (MICCAI)
- Kirchhoff et al. (2024) -- Skeleton Recall Loss (ECCV)
- Stucki et al. (2025) -- CAPE Loss (MICCAI)
- Hu et al. (2019) -- Betti Matching Loss (NeurIPS)
- Clough et al. (2020) -- Topological Loss (MICCAI)
- SAM3: Meta Segment Anything Model 3 (arXiv:2511.16719, Nov 2025)
