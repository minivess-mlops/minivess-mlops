# Model Adapters Design

## Implementation Files

| File | Purpose |
|------|---------|
| `src/minivess/adapters/base.py` | ModelAdapter ABC |
| `src/minivess/adapters/dynunet_adapter.py` | DynUNet (MONAI-native) |
| `src/minivess/adapters/segresnet_adapter.py` | SegResNet (MONAI-native) |
| `src/minivess/adapters/sam3_adapter.py` | SAM3 (external, adapted to MONAI) |
| `src/minivess/adapters/sam3_backbone.py` | SAM3 ViT-32L backbone |
| `src/minivess/adapters/sam3_decoder.py` | SAM3 lightweight Conv decoder |
| `src/minivess/adapters/sam3_vram_check.py` | VRAM verification at build time |

## VRAM Budget (8 GB GPU)

| Model | VRAM | Patch Size | Status |
|-------|------|-----------|--------|
| DynUNet | 3.5 GB | (128,128,32) | Production |
| SAM3 Vanilla | 2.9 GB | (64,64,3) | Verified |
| SAM3 Hybrid | 7.5 GB | (32,32,3) | Marginal |

## SAM3 Key Facts

- SAM3 = Meta's Segment Anything Model 3 (Nov 2025, github.com/facebookresearch/sam3)
- NOT SAM2 — different architecture (ViT-32L vs Hiera), different weights
- 848M params, 1008×1008 input, SDPA mandatory
- HF decoder: lightweight Conv (66K params), not HF Sam3MaskDecoder
