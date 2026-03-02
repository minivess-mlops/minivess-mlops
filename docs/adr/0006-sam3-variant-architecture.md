# ADR-0006: SAM3 Variant Architecture

## Status

Accepted

## Date

2026-03-02

## Context

The MinIVess MLOps platform needs to demonstrate SAM3 (Segment Anything Model 3, Ravi et al. 2025, arXiv:2511.16719) integration as a foundation model comparison point against the established DynUNet baseline (0.824 DSC / 0.906 clDice with cbdice_cldice). SAM3 is a 2D image segmentation model with a ViT-32L perception encoder (848M params, 1008x1008 input), trained on natural images. Its application to 3D microvessel segmentation requires architectural decisions around:

1. **Backbone**: SAM3 ViT-32L (648M backbone, 1024-dim embeddings, 256-dim FPN).
2. **2D-to-3D strategy**: SAM3 processes 2D images, but MiniVess data is 3D (512x512xZ).
3. **Adaptation strategy**: Frozen vs. LoRA fine-tuned vs. hybrid encoder usage.
4. **Prompt mode**: SAM3 supports point/box/mask/concept prompts, but we need fully automatic segmentation.
5. **VRAM budget**: All variants must fit within 8GB (RTX 2070 Super).

## Decision

We implement three SAM3-based adapters as a controlled experiment:

### V1: Sam3VanillaAdapter — Frozen baseline
- Frozen SAM3 ViT-32L perception encoder + trainable lightweight mask decoder
- Slice-by-slice 2D inference (iterate over Z dimension)
- Null prompt embeddings (fully automatic, no interactive prompts)
- Loss: `dice_ce` (standard, no topology supervision)
- VRAM: ~3.0 GB

### V2: Sam3TopoLoraAdapter — Topology-guided LoRA
- Same as V1 + LoRA adapters on encoder FFN layers (mlp.lin1, mlp.lin2)
- LoRA rank=16, alpha=32 (default), wrapping `nn.Linear` and `nn.Conv2d` layers
- Loss: `cbdice_cldice` (topology-aware, matching DynUNet champion)
- V1→V2 is a strict controlled experiment: ONLY difference is LoRA + loss
- Backbone created with `freeze=False` (gradient flow for LoRA), FPN neck frozen separately
- VRAM: ~3.5 GB

### V3: Sam3HybridAdapter — 3D decoder fusion
- Frozen SAM3 ViT-32L as feature extractor (256-dim FPN output per slice)
- DynUNet encoder/decoder for 3D processing
- `GatedFeatureFusion`: `f_3d + sigmoid(alpha) * proj_conv(f_sam.detach())`
  - `gate_alpha` initialized to 0.0 (pure DynUNet at training start)
  - `f_sam.detach()` blocks gradient flow to frozen SAM encoder
- Axial projection: Conv3d kernel=(3,1,1) for inter-slice smoothing of stacked 2D features
- VRAM: ~7.5 GB (requires batch_size=1 + AMP)

### Key choices

| Decision | Choice | Alternatives Considered |
|----------|--------|----------------------|
| Backbone | SAM3 ViT-32L (648M) | Smaller ViT variants — SAM3 has single backbone size |
| Input size | 1008x1008 (native) | 1024x1024 — would require padding, SAM3 uses patch_size=14 |
| 2D→3D | Slice-by-slice | SAM3 video mode — designed for temporal, not spatial Z |
| Prompts | Null embeddings | Points/boxes/concepts — requires interactive annotation |
| Fusion | Gated residual at output | Cross-attention at bottleneck — higher VRAM, complexity |
| LoRA targets | mlp.lin1, mlp.lin2 (FFN) | q_proj, v_proj (attention) — FFN has more parameters |
| SAM3 dependency | Optional (stub for testing) | Required — would break installs without SAM3 |
| Weight loading | Native sam3 pkg or HuggingFace | Single path — dual reduces install friction |

## Consequences

**Positive:**

- Three variants form a controlled ablation study: frozen → LoRA → hybrid.
- Each variant implements the `ModelAdapter` ABC, integrating with existing training, evaluation, and serving pipelines with zero model-specific code.
- Go/no-go gates (G1, G2, G3) provide clear decision criteria.
- VRAM budgets verified to fit 8GB, enabling reproducibility on commodity hardware.
- Stub encoder (`_StubSam3Encoder`) enables CI testing without SAM3 installed.
- Conditional `torch.no_grad()` in backbone methods: frozen mode uses no_grad, unfrozen (LoRA) allows gradient flow.
- Feature caching infrastructure (`sam3_feature_cache.py`) reduces VRAM for hybrid variant.

**Negative:**

- Slice-by-slice inference loses inter-slice context (V1/V2). The hybrid (V3) partially addresses this via axial projection.
- SAM3 on microvasculature is expected to significantly underperform DynUNet. This is the intended scientific finding — demonstrating the domain gap.
- ViT-32L is large (648M backbone). Feature caching or AMP is needed to fit 8GB.

**Neutral:**

- The `sam3_*` naming convention refers to Meta's SAM3 (Segment Anything Model 3, Nov 2025). "Vanilla/TopoLoRA/Hybrid" distinguish our three adaptation strategies.
- 2.5D input (3 adjacent slices as RGB) is deferred to a future iteration as a stretch goal.
