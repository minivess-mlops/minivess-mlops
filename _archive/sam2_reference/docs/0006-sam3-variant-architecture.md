# ADR-0006: SAM3 Variant Architecture

## Status

Accepted

## Date

2026-03-02

## Context

The MinIVess MLOps platform needs to demonstrate SAM2 (Segment Anything Model 2) integration as a foundation model comparison point against the established DynUNet baseline. SAM2 is a 2D image segmentation model trained on natural images (SA-V dataset), and its application to 3D microvessel segmentation requires architectural decisions around:

1. **Backbone selection**: SAM2 offers Hiera-Tiny (38.9M), Hiera-Small (64.8M), Hiera-B+ (119M), and Hiera-Large (224M) variants.
2. **2D-to-3D strategy**: SAM2 processes 2D images, but MiniVess data is 3D (512x512xZ).
3. **Adaptation strategy**: Frozen vs. fine-tuned vs. hybrid encoder usage.
4. **Prompt mode**: SAM2 expects point/box/mask prompts, but we need fully automatic segmentation.
5. **VRAM budget**: All variants must fit within 8GB (RTX 2070 Super).

## Decision

We implement three SAM2-based adapters (collectively called "SAM3") as a controlled experiment:

### V1: Sam3VanillaAdapter — Frozen baseline
- Frozen SAM2 Hiera-Tiny encoder + trainable lightweight mask decoder
- Slice-by-slice 2D inference (iterate over Z dimension)
- Null prompt embeddings (learned, no interactive prompts)
- Loss: `dice_ce` (standard, no topology supervision)
- VRAM: ~3.0 GB

### V2: Sam3TopoLoraAdapter — Topology-guided LoRA
- Same as V1 + PEFT LoRA (r=16, alpha=32) on encoder attention layers
- LoRA targets `nn.Linear` layers in Hiera encoder (q_proj, v_proj style)
- Loss: `cbdice_cldice` (topology-aware, matching DynUNet champion)
- V1→V2 is a strict controlled experiment: ONLY difference is LoRA + loss
- VRAM: ~3.5 GB

### V3: Sam3HybridAdapter — 3D decoder fusion
- Frozen SAM2 encoder as feature extractor
- DynUNet encoder/decoder for 3D processing
- `GatedFeatureFusion`: `f_3d + sigmoid(alpha) * proj_conv(f_sam.detach())`
  - `gate_alpha` initialized to 0.0 (pure DynUNet at training start)
  - `f_sam.detach()` blocks gradient flow to frozen SAM encoder
- `AxialProjection`: 1D Conv3d along Z-axis for inter-slice context
- VRAM: ~7.5 GB (requires batch_size=1 + AMP)

### Key choices

| Decision | Choice | Alternatives Considered |
|----------|--------|----------------------|
| Backbone | Hiera-Tiny (38.9M) | Hiera-B+ (119M) — exceeds 8GB VRAM |
| 2D→3D | Slice-by-slice | SAM2 video mode — designed for temporal, not spatial Z |
| Prompts | Null embeddings | Points/boxes — requires interactive annotation |
| Fusion | Gated concat at output | Cross-attention at bottleneck — higher VRAM, complexity |
| LoRA targets | nn.Linear in Hiera | Conv3d — Hiera uses Linear for attention, not Conv |
| SAM2 dependency | Optional `[sam]` extras | Required — would break installs without SAM2 |

## Consequences

**Positive:**

- Three variants form a controlled ablation study: frozen → LoRA → hybrid.
- Each variant implements the `ModelAdapter` ABC, integrating with existing training, evaluation, and serving pipelines with zero model-specific code.
- Go/no-go gates (G1, G2, G3) provide clear decision criteria.
- VRAM budgets verified to fit 8GB, enabling reproducibility on commodity hardware.
- Stub encoder (`_StubHieraEncoder`) enables testing without the SAM2 package installed.

**Negative:**

- Slice-by-slice inference loses inter-slice context (V1/V2). The hybrid (V3) partially addresses this via AxialProjection.
- SAM2 on microvasculature is expected to significantly underperform DynUNet. This is the intended scientific finding — demonstrating the domain gap.
- Hiera-Tiny is the weakest SAM2 variant. Better results are possible with Hiera-B+ but would require >8GB VRAM.

**Neutral:**

- The `sam3_*` naming convention distinguishes our three adaptation strategies from SAM2 (the Meta model) and MedSAM3D (a different project). "SAM3" = our SAM variant study.
- 2.5D input (3 adjacent slices as RGB) is deferred to a future iteration as a stretch goal.
