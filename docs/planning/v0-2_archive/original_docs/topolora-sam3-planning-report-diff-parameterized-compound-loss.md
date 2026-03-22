# TopoLoRA Paper Loss — Differently-Parameterized Compound Loss

**Date**: 2026-03-22
**Branch**: `test/debug-factorial-4th-pass`
**Status**: IMPLEMENTING

---

## Motivation

The TopoLoRA paper (Khazem et al. 2025) uses a compound loss with **different weight
balance** than our existing `dice_ce_cldice`. Both use the same 3 components (region +
overlap + topology), but the paper treats clDice as **supplementary** (20% relative
weight) while ours treats it as **dominant** (50% relative weight).

Testing both parameterizations across ALL 4 models is a clean factorial question:
does the weight balance between region accuracy and topology preservation matter?

## Loss Comparison

| Property | Our `dice_ce_cldice` | Paper's `bce_dice_05cldice` |
|----------|---------------------|----------------------------|
| **Components** | DiceCE + clDice | BCE + Dice + clDice |
| **Region term** | 0.5 * DiceCE (= 0.25 Dice + 0.25 CE) | 1.0 BCE + 1.0 Dice |
| **Topology term** | 0.5 * clDice | 0.5 * clDice |
| **Relative clDice weight** | **50%** (dominant) | **20%** (supplementary) |
| **Philosophy** | Topology-balanced | Region-first, topology regularizer |
| **Source** | Our default compound | Khazem et al. 2025, SegLab |
| **clDice iterations** | 10 (matching paper) | 10 |

## Implementation Plan

### Step 1: New Loss Class

Add `BceDiceClDiceLoss` to `src/minivess/pipeline/loss_functions.py`:
- Uses `torch.nn.BCEWithLogitsLoss` (not MONAI DiceCELoss)
- Uses `monai.losses.DiceLoss` (standard)
- Uses existing `_WrappedSoftclDiceLoss` (iter_=10, matching paper)
- Default weights: `lambda_bce=1.0, lambda_dice=1.0, lambda_cldice=0.5`

### Step 2: Register in Factory

Add `"bce_dice_05cldice"` to:
- `_LIBRARY_COMPOUND_LOSSES` frozenset
- `build_loss_function()` dispatch

### Step 3: Add to Factorial Configs

Update all 4 factorial configs:
- `configs/factorial/paper_full.yaml`: 4 losses (was 3)
- `configs/factorial/debug.yaml`: 4 losses (identical to production, Rule 27)
- `configs/factorial/smoke_local.yaml`: add if >1 loss
- `configs/factorial/smoke_test.yaml`: keep at 1 loss (pipeline validation)

Update deprecated configs:
- `configs/experiment/debug_factorial.yaml`: add to losses list
- `configs/hpo/paper_factorial.yaml`: add to losses list

### Step 4: Update Tests

- Add test for new loss in `tests/v2/unit/test_loss_functions.py`
- Update factorial config tests: 480 → 640 conditions (4×4×2×2×2×5)
- Update training grid tests: 24 → 32 cells

### Step 5: Update Visualization

- Add color + label in `src/minivess/pipeline/viz/plot_config.py`

## Factorial Impact

| Metric | Before (3 losses) | After (4 losses) | Change |
|--------|-------------------|-------------------|--------|
| Training cells | 4×3×2 = 24 | 4×4×2 = 32 | +33% |
| Total conditions | 480 | 640 | +33% |
| Debug GPU jobs | 24 | 32 | +$2-3 |
| Production GPU jobs | 72 | 96 | +$22 |

## Rename of Existing Loss

The existing `dice_ce_cldice` name doesn't communicate its weight balance.
However, renaming touches **175+ files** (source, tests, configs, docs, KG).

**Decision**: Defer rename to separate PR. Document the weight difference in
class docstrings and the planning report. The factorial config comments make
the distinction clear.

## Cross-References

- Paper: [Khazem et al. (2025)](https://arxiv.org/html/2601.02273v1)
- Code: https://github.com/salimkhazem/Seglab
- Loss registry: `src/minivess/pipeline/loss_functions.py`
- Prior analysis: `docs/planning/topolora-sam3-planning-report.md`
