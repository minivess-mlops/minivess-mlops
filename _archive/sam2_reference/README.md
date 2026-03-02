# SAM2 Reference Code (ARCHIVED — WRONG MODEL)

**DO NOT USE THIS CODE FOR SAM3 IMPLEMENTATION.**

## What This Is

This directory contains SAM2-based (Segment Anything Model 2) adapter code that was
implemented by mistake when SAM3 (Segment Anything Model 3) was intended.

## Why It Exists

On 2026-03-02, Claude implemented the entire SAM3 variants plan (~1500 lines, 91 tests)
using **SAM2 Hiera-Tiny** as the backbone, when the user explicitly requested **SAM3**
(Meta's official successor released November 2025). The user had even warned:
"don't confuse SAMv3 with SAMv2."

Full incident report: `.claude/metalearning/2026-03-02-sam3-implementation-fuckup.md`

## SAM2 ≠ SAM3

| Aspect | SAM2 (this code) | SAM3 (what should be used) |
|--------|------------------|---------------------------|
| Release | July 2024 | November 2025 |
| Params | Hiera-Tiny: 38.9M | 848M (single variant) |
| Backbone | Hiera (hierarchical ViT) | ViT-32L (1024-dim, 648M) |
| Input size | 1024×1024 | 1008×1008 |
| Prompts | Points, boxes, masks | Text, points, boxes, masks, exemplars |
| Package | `pip install sam2` | `pip install -e .` (git clone) |
| Repository | github.com/facebookresearch/sam2 | github.com/facebookresearch/sam3 |

## What's Reusable (~60%)

The following patterns transfer to SAM3 with minor modifications:
- `slice_inference.py` — slice-by-slice pattern (change resize from 1024→1008)
- `model_builder.py` — factory dispatch pattern
- `sam3_gates.py` — go/no-go gate evaluation logic
- Experiment config YAMLs (update model references)
- Gated fusion architecture (adjust dims for SAM3's 1024/256 features)

## What's NOT Reusable (~40%)

- `sam2_backbone.py` — completely different model loading API
- `sam2_decoder.py` — SAM3 has its own decoder
- LoRA targets — SAM2 targets attention q_proj/v_proj, SAM3 targets FFN mlp.lin1/mlp.lin2
- VRAM budget — SAM2 Hiera-Tiny fits 8GB, SAM3 needs 16GB+

## Correct SAM3 Implementation Plan

See: `docs/planning/sam3-implementation-plan.xml`
