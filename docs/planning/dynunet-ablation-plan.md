# DynUNet Width Ablation Study — Implementation Plan (Issue #33)

## Current State
- No DynUNet adapter
- Loss factory supports: dice_ce, dice, focal
- No topology-aware losses (SoftclDiceLoss)
- No ablation config framework

## Architecture

### New Module 1: `src/minivess/adapters/dynunet.py`
- **DynUNetAdapter** — ModelAdapter wrapping MONAI DynUNet
- Configurable filters, kernel_size, strides
- Deep supervision support

### New Module 2: `src/minivess/pipeline/ablation.py`
- **AblationConfig** — Dataclass defining width × loss ablation grid
- **build_ablation_grid()** — Generate experiment configs from grid
- Width presets: FULL [32,64,128,256], HALF [16,32,64,128], QUARTER [8,16,32,64]

### Modified Module: `src/minivess/pipeline/loss_functions.py`
- Add "dice_ce_cldice" loss → compound DiceCE + SoftclDiceLoss
- Add "cldice" loss → standalone SoftclDiceLoss

### Modified Files
- `src/minivess/config/models.py` — Add MONAI_DYNUNET to ModelFamily
- `src/minivess/adapters/__init__.py` — Export DynUNetAdapter
- `src/minivess/pipeline/__init__.py` — Export ablation functions

## Test Plan
- `tests/v2/unit/test_dynunet_ablation.py` (~18 tests)
  - TestDynUNetAdapter: instanceof, forward, config, checkpoint
  - TestCompoundLoss: clDice loss functions
  - TestAblationConfig: grid generation, width presets
