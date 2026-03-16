---
name: monai-losses
description: "MONAI loss functions used in MinIVess — compound losses, topology-aware losses, and custom implementations"
metadata:
  languages: "python"
  versions: "1.4.0"
  revision: 1
  updated-on: "2026-03-16"
  source: maintainer
  tags: "monai,loss,segmentation,topology,dice"
---

# MONAI Loss Functions for Vascular Segmentation

## Default Loss: CbDiceClDiceLoss (cbdice_cldice)

The default single-model loss for this project. Combines centerline-aware Dice
with class-balanced Dice for topology preservation.

```python
from monai.losses import DiceCELoss
from minivess.pipeline.losses import CbDiceClDiceLoss

# Default — best topology (0.906 clDice), −5.3% DSC penalty
loss = CbDiceClDiceLoss(include_background=False, to_onehot_y=True, softmax=True)

# Alternative — higher DSC (0.824) but worse topology (0.832 clDice)
loss = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
```

## Available Losses

| Loss Name | Class | Source | Default For |
|-----------|-------|--------|-------------|
| `cbdice_cldice` | `CbDiceClDiceLoss` | Custom (MONAI components) | All models except SAM3 Vanilla |
| `dice_ce` | `DiceCELoss` | MONAI | SAM3 Vanilla |
| `dice_ce_cldice` | `DiceCEClDiceLoss` | Custom | DynUNet (extra) |
| `skeleton_recall` | `SkeletonRecallLoss` | Custom | DynUNet (extra) |

## Key Constraints

- `include_background=False` — always exclude background from loss computation
- `to_onehot_y=True` — labels are integer-encoded, convert to one-hot
- `softmax=True` — model outputs logits, not probabilities
- Loss config in `configs/method_capabilities.yaml` under `model_default_loss`
- NEVER hardcode loss names — use config-driven dispatch via `build_loss_for_test()`

## MONAI Loss API Notes

```python
# MONAI DiceLoss — standard interface
from monai.losses import DiceLoss, DiceCELoss, GeneralizedDiceLoss

# All MONAI losses expect:
#   input: (B, C, H, W, D) — model predictions (logits or softmax)
#   target: (B, 1, H, W, D) — integer labels (to_onehot_y converts)
# Output: scalar loss value

# clDice requires skeletonization — uses soft_skeleton from monai.losses
from monai.losses import SoftclDiceLoss
```

## AMP Policy (Critical — MONAI #4243)

Mixed precision during validation causes NaN with 3D selective scan operations.
Policy D04: train=ON, validation=OFF.

```python
# In trainer.py — AMP only during training
with torch.amp.autocast("cuda", enabled=self.amp_enabled and is_training):
    output = model(batch_data)
```
