---
title: "Mamba Model Capacity Matching for MiniVess Segmentation"
status: reference
created: "2026-03-05"
---

# Mamba Model Capacity Matching for MiniVess Segmentation

**Date:** 2026-03-05 (revised — earlier estimates were wrong, see §1)
**Context:** Comparing `comma_mamba` and `ulike_mamba` against `dynunet` (full-width and
half-width) on the MiniVess 70-volume microvessel segmentation benchmark.

 Capacity matching report completely rewritten with measured (not estimated) numbers:

  ┌────────────────────────────┬───────────────────────┐
  │           Model            │   Params (measured)   │
  ├────────────────────────────┼───────────────────────┤
  │ DynUNet [32,64,128,256]    │ 5.641M (not "15-20M") │
  ├────────────────────────────┼───────────────────────┤
  │ DynUNet [16,32,64,128]     │ 1.411M (not "4-5M")   │
  ├────────────────────────────┼───────────────────────┤
  │ UlikeMamba init_filters=32 │ 4.116M                │
  ├────────────────────────────┼───────────────────────┤
  │ CommaMamba init_filters=32 │ 5.174M                │
  └────────────────────────────┴───────────────────────┘

  Key finding that changes the narrative: The Mamba models at default init_filters=32 are
  already iso-param with DynUNet full-width (UlikeMamba is 73%, CommaMamba is 92%). The
  paper's "efficiency comparison" framing was wrong — this is already an architectural
  comparison at comparable scale.

  Revised tiers now include a Tier 3 DynUNet multi-width ablation (eighth/quarter/half/full
  width) to establish a CNN capacity scaling curve on MiniVess, which grounds all Mamba
  comparisons scientifically. Tier 0 requires no config changes — current defaults are
  correct.

---

## 1. Correction: Estimated Params Were Wrong

The initial version of this report used theoretical estimates for parameter counts.
**All numbers below are measured directly** from the models via
`sum(p.numel() for p in model.parameters())`.

Two additional discoveries made during measurement:

**Bug fixed**: `CommaAdapter.__init__` ignored `config.architecture_params['init_filters']`
and always used the constructor default (32). The adapter always had 5.174M params regardless
of the YAML config. Fixed in `src/minivess/adapters/comma.py` — it now reads from
`architecture_params` like `MambaAdapter` does.

**MLflow confirmed**: `arch_filters` logged as `[32, 64, 128, 256]` for `dynunet_loss_variation_v2`
and `[16, 32, 64, 128]` for `dynunet_half_width_v1`.

---

## 2. How the Literature Handles Capacity Matching

The short answer: **the field does not use strict iso-parameter comparisons.** Three
patterns dominate:

### 2.1 Scaffold inheritance (U-Mamba approach)
Build directly on the nnUNet framework, replace encoder blocks with Mamba layers, keep
everything else. U-Mamba (Ma et al., 2024) ends up at 58.47M parameters — 88% larger
than nnUNet's 31.18M — without any explicit parameter control.

**Critical result** (Isensee et al., 2024): The "No-Mamba Base" ablation — same residual
U-Net backbone with Mamba layers disabled — achieves identical performance to U-Mamba Bot
and U-Mamba Enc across six datasets (BTCV, ACDC, LiTS, BraTS, KiTS, AMOS). **The claimed
gains over nnUNet are attributable to the residual U-Net backbone, not the SSM layers.**
This is arguably the most important result for our experimental design.

### 2.2 Efficiency-focused design (nnMamba, LightUMamba approach)
Intentionally target fewer parameters than nnUNet and report efficiency wins.
nnMamba (2024) achieves 73.98% mDice on AMOS MRI-Test vs nnUNet's 67.63% — with
*half* the parameters (15.55M vs 31.18M). LightUMamba (Kazaj et al., 2025) at 5.70M
shows competitive accuracy with no statistically significant difference from nnUNet
while requiring significantly longer training time.

### 2.3 Standardized protocols, natural sizes (Claims-to-Evidence approach)
Fix training hyperparameters (batch size, patch, loss, augmentation) identically across
all models, compare at natural parameter counts. Kazaj et al. (2025) concluded:
"Mamba-based architectures achieved competitive accuracy with no statistically significant
difference from nnUNet and U2Net, while using fewer parameters. However, they required
significantly longer training time." CNN models remain the best speed-accuracy tradeoff.

---

## 3. Measured Parameter Inventory

### 3.1 DynUNet — measured counts

| Filter config | Params (measured) | Experiment |
|--------------|-------------------|------------|
| `[32, 64, 128, 256]` | **5.641M** | `dynunet_loss_variation_v2` |
| `[16, 32, 64, 128]` | **1.411M** | `dynunet_half_width_v1` |
| `[8, 16, 32, 64]` | **0.353M** | (proposed Tier 2) |
| `[4, 8, 16, 32]` | **0.088M** | (proposed Tier 3) |

### 3.2 UlikeMamba — measured counts (multiples of 8 required by GroupNorm)

| init_filters | Params (measured) | Nearest DynUNet tier |
|-------------|-------------------|---------------------|
| 8 | 0.262M | quarter-width (0.353M, −26%) |
| 16 | 1.036M | half-width (1.411M, −27%) |
| 24 | 2.320M | (between tiers) |
| **32** | **4.116M** | **full-width (5.641M, −27%) ← current default** |
| 40 | 6.422M | full-width (5.641M, +14%) |
| 48 | 9.240M | — |
| 64 | 16.409M | — |
| 80 | 25.623M | — |

### 3.3 CommaMamba — measured counts (no GroupNorm constraint)

| init_filters | Params (measured) | Nearest DynUNet tier |
|-------------|-------------------|---------------------|
| 12 | 0.731M | (between tiers) |
| 14 | 0.994M | (between tiers) |
| 16 | 1.297M | half-width (1.411M, −8%) |
| 18 | 1.640M | (between tiers) |
| 20 | 2.024M | — |
| 24 | 2.913M | — |
| 28 | 3.963M | — |
| **32** | **5.174M** | **full-width (5.641M, −8%) ← current default** |
| 34 | 5.840M | full-width (5.641M, +3%) |
| 36 | 6.546M | — |
| 40 | 8.080M | — |
| 60 | 18.168M | — |

### 3.4 Literature reference points

| Model | Params | vs nnUNet | Venue |
|-------|--------|-----------|-------|
| nnUNet PlainConvUNet 3D | 31.18M | baseline | MICCAI benchmark |
| SegMamba (Xing et al., 2024) | 22.86M | −27% | MICCAI 2024 |
| nnMamba (2024) | 15.55M | −50% | arXiv:2402.03526 |
| EM-Net (2024) | 39.41M | +26% | MICCAI 2024 |
| U-Mamba (Ma et al., 2024) | 58.47M | +88% | arXiv:2401.04722 |
| VM-UNet (2024) | 27.43M | −12% | ACM TOMM 2024 |
| UlikeMamba_3dMT (Wang et al., 2025) | ~30–32M | ≈ parity | arXiv:2503.19308 |
| LightUMamba (Kazaj et al., 2025) | 5.70M | −82% | arXiv:2503.01306 |
| COMMA (Shi et al., 2025) | not stated | FLOPs ≈ ½ SegMamba | arXiv:2503.02332 |

Note: Our DynUNet implementation (MONAI DynUNet) is a *different codebase* from nnUNet's
PlainConvUNet 3D. The 5.641M vs 31.18M difference reflects distinct filter choices, not a
comparison of MONAI vs nnU-Net at the same config.

---

## 4. Key Finding: Current Defaults Are Already Iso-Param

The critical insight from measuring actual parameters:

> **At `init_filters=32`, both Mamba variants have parameter counts within 8–27% of
> DynUNet full-width `[32,64,128,256]`.**
>
> - UlikeMamba (4.116M) vs DynUNet full-width (5.641M) → **73% of DynUNet**
> - CommaMamba (5.174M) vs DynUNet full-width (5.641M) → **92% of DynUNet**

This fundamentally changes the experimental framing. The current defaults are **not** an
extreme efficiency comparison — they are a direct architectural comparison at comparable scale.
Training at current defaults answers the question:

> *"Does the SSM mechanism help compared to a DynUNet CNN at equivalent parameter count?"*

This is exactly the question that Isensee et al. (2024) asked (and answered "No" for nnUNet
6 datasets). Our MiniVess context may be different due to tubular topology requirements.

---

## 5. Revised Capacity Tiers

Four tiers covering the full efficiency vs capacity spectrum:

### Tier 0 — Full-width iso-param (CURRENT defaults)
**Question:** At equivalent capacity (~5M), does the SSM mechanism outperform a well-tuned DynUNet?
This is the scientifically cleanest comparison answering the Isensee et al. question on microvessel data.

| Model | Config | Params (measured) | vs DynUNet full-width |
|-------|--------|-------------------|----------------------|
| DynUNet full-width | filters=[32,64,128,256] | **5.641M** | baseline |
| UlikeMamba | init_filters=32 | **4.116M** | −27% |
| CommaMamba | init_filters=32, d_state=16 | **5.174M** | −8% |

### Tier 1 — Half-width iso-param (~1.4M)
**Question:** At small model size, does architecture matter more?

| Model | Config | Params (measured) | vs DynUNet half-width |
|-------|--------|-------------------|----------------------|
| DynUNet half-width | filters=[16,32,64,128] | **1.411M** | baseline |
| UlikeMamba | init_filters=16 | **1.036M** | −27% |
| CommaMamba | init_filters=16, d_state=16 | **1.297M** | −8% |

### Tier 2 — Quarter-width (~0.35M)
**Question:** Extreme parameter efficiency — can any architecture work at 88K–353K params?

| Model | Config | Params (measured) | vs DynUNet quarter-width |
|-------|--------|-------------------|--------------------------|
| DynUNet quarter-width | filters=[8,16,32,64] | **0.353M** | baseline |
| UlikeMamba | init_filters=8 | **0.262M** | −26% |
| CommaMamba | — | no close match | CommaMamba minimum ~0.73M |

### Tier 3 — Multi-width DynUNet ablation (new idea)
**Question:** How does DynUNet performance scale with capacity on MiniVess?
Run DynUNet at all four widths to establish a capacity scaling curve.

| Model | Config | Params (measured) |
|-------|--------|-------------------|
| DynUNet eighth-width | filters=[4,8,16,32] | 0.088M |
| DynUNet quarter-width | filters=[8,16,32,64] | 0.353M |
| DynUNet half-width | filters=[16,32,64,128] | 1.411M (done) |
| DynUNet full-width | filters=[32,64,128,256] | 5.641M (done) |

This generates a capacity scaling curve for the CNN baseline, making the Mamba comparisons
at each tier scientifically grounded.

---

## 6. Architecture Capacity Controls

### 6.1 DynUNet
```python
# src/minivess/adapters/dynunet.py
# Capacity controlled exclusively via filters list:
filters: list[int] = [32, 64, 128, 256]   # full-width: 5.641M
#                    [16, 32,  64, 128]   # half-width: 1.411M (tested)
#                    [ 8, 16,  32,  64]   # quarter-width: 0.353M
#                    [ 4,  8,  16,  32]   # eighth-width: 0.088M
```

### 6.2 UlikeMamba (`src/minivess/adapters/mamba.py`)
```python
# init_filters MUST be a multiple of 8 (GroupNorm constraint)
# Configurable via YAML architecture_params.init_filters
```

### 6.3 CommaMamba (`src/minivess/adapters/comma.py`)
```python
# init_filters can be any positive integer
# d_state universally 16 across all 2024-2025 Mamba medical imaging papers
# Bug fixed 2026-03-05: now reads from config.architecture_params
```

---

## 7. Parameter Count Verification Script

```bash
uv run python -c "
from minivess.config.models import ModelConfig
from minivess.adapters.model_builder import build_adapter

configs = [
    # DynUNet
    ('dynunet', 'filters=[32,64,128,256]', {'filters': [32,64,128,256]}),
    ('dynunet', 'filters=[16,32,64,128]',  {'filters': [16,32,64,128]}),
    ('dynunet', 'filters=[8,16,32,64]',    {'filters': [8,16,32,64]}),
    # Mamba Tier 0
    ('ulike_mamba', 'init_filters=32',     {'init_filters': 32}),
    ('comma_mamba', 'init_filters=32',     {'init_filters': 32, 'd_state': 16}),
    # Mamba Tier 1
    ('ulike_mamba', 'init_filters=16',     {'init_filters': 16}),
    ('comma_mamba', 'init_filters=16',     {'init_filters': 16, 'd_state': 16}),
    # Mamba Tier 2
    ('ulike_mamba', 'init_filters=8',      {'init_filters': 8}),
]
for family, label, ap in configs:
    cfg = ModelConfig(family=family, name='verify', in_channels=1, out_channels=2,
                     architecture_params=ap)
    m = build_adapter(cfg)
    n = sum(p.numel() for p in m.parameters())
    print(f'{family:15s} {label:30s}  {n/1e6:.3f}M')
"
```

---

## 8. Experimental Design Recommendation

**Primary recommendation:** Run Tier 0 first (current defaults — Mamba vs DynUNet full-width,
already iso-param), then Tier 1 (half-width matching) if Tier 0 shows no advantage.

**Rationale:**
1. Tier 0 with current defaults is already the scientifically correct comparison — no config
   changes needed. CommaMamba (5.2M) vs DynUNet (5.6M) is within 8%.
2. If Mamba shows topology advantages at Tier 0 (clDice improvement), Tier 1 tests whether
   the advantage persists at smaller scale.
3. Tier 3 (DynUNet multi-width ablation) provides a capacity scaling curve that gives
   context for all tier comparisons.

**Loss function:** Use `cbdice_cldice` (project default) for all variants to enable
direct metric comparison with existing DynUNet runs.

**Epochs:** 100 (same as DynUNet experiments). Kazaj et al. (2025) notes that Mamba
architectures converge more slowly than CNN baselines — 100 epochs may underestimate
Mamba performance relative to CNNs.

**Mamba training time caveat:** Mamba's sequential state-space scanning is not as
amenable to GPU parallelism as convolution. Expect longer wall-clock times per epoch
even at matched parameter counts.

---

## 9. Critical Caveat: The Isensee et al. (2024) Result

Before investing significant GPU time, the Isensee et al. (2024) "nnU-Net Revisited"
finding is essential context:

> *"We show that many recently proposed architectures achieve no improvement over a
> properly configured nnU-Net ResEnc baseline, and that the reported gains in prior
> work are often due to training differences rather than the proposed architectural
> innovations."*

For **U-Mamba specifically**, the No-Mamba Base achieves the same performance as U-Mamba
across six datasets. This does not mean Mamba is useless — it means that:

1. The SSM mechanism's theoretical O(n) advantage over O(n log n) attention may not
   manifest on the ~96³ patch sizes typical in medical segmentation.
2. Microvessel segmentation (MiniVess) may be different: long-range tubular connectivity
   is exactly the use case where SSM linear-complexity should help.
3. The topology-aware losses (`cbdice_cldice`) we use may interact differently with
   Mamba's sequential processing vs CNN spatial filtering.

This makes the MiniVess comparison genuinely scientifically interesting.

---

## 10. Summary Table: Planned Experimental Configurations

| Tier | Model | Config | Params (measured) | VRAM est. | Scientific Question |
|------|-------|--------|-------------------|-----------|---------------------|
| **T0** | DynUNet full | [32,64,128,256] | **5.641M** | ~7.4 GB | Baseline (done) |
| **T0** | UlikeMamba | init_filters=32 | **4.116M** | ~3 GB | CNN vs SSM at 5M params |
| **T0** | CommaMamba | init_filters=32, d_state=16 | **5.174M** | ~4 GB | CNN vs SSM at 5M params |
| **T1** | DynUNet half | [16,32,64,128] | **1.411M** | ~4 GB | Baseline (done) |
| **T1** | UlikeMamba | init_filters=16 | **1.036M** | ~2 GB | CNN vs SSM at 1.4M params |
| **T1** | CommaMamba | init_filters=16, d_state=16 | **1.297M** | ~3 GB | CNN vs SSM at 1.4M params |
| **T2** | DynUNet quarter | [8,16,32,64] | **0.353M** | ~2 GB | CNN extreme efficiency |
| **T2** | UlikeMamba | init_filters=8 | **0.262M** | ~2 GB | SSM extreme efficiency |
| **T3** | DynUNet eighth | [4,8,16,32] | **0.088M** | ~1 GB | CNN capacity scaling |

All Tier 0 Mamba runs need no YAML config changes — `train_alternative_variants.sh` uses
current defaults.

---

## 11. Next Steps

1. **Run Tier 0** with `./scripts/train_alternative_variants.sh --skip-heavy` (current defaults)
2. **Create DynUNet quarter-width YAML** (`configs/dynunet_quarter_width_v1.yaml`) for Tier 2
3. **Create Mamba Tier 1 YAMLs** (`configs/mamba_half_width_v1.yaml`) with `init_filters=16`
4. **Compare metrics** using `uv run python scripts/compare_models.py`
5. If Mamba shows topology advantages (clDice improvement), run Tier 1 and Tier 3

VesselFM (~30M params) requires RunPod (GitHub Issue #366) — not included in local tiers above.

---

## References

- Isensee, F., Wald, T., Ulrich, C., Baumgartner, M., Roy, S., Maier-Hein, K., & Jäger, P. (2024).
  nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation. *MICCAI 2024*.
  arXiv:2404.09556

- Kazaj, P. M., et al. (2025). From Claims to Evidence: A Unified Framework and Critical Analysis
  of CNN vs. Transformer vs. Mamba in Medical Image Segmentation. arXiv:2503.01306

- Ma, J., Li, F., & Wang, B. (2024). U-Mamba: Enhancing Long-range Dependency for Biomedical
  Image Segmentation. arXiv:2401.04722

- Shi, G., Zhang, H., & Tian, J. (2025). COMMA: Coordinate-aware Modulated Mamba Network for 3D
  Dispersed Vessel Segmentation. arXiv:2503.02332

- Wang, C., Xie, Y., Chen, Q., Zhou, Y., & Wu, Q. (2025). A Comprehensive Analysis of Mamba for
  3D Volumetric Medical Image Segmentation. arXiv:2503.19308

- Xing, Z., Ye, T., Yang, Y., Liu, G., & Zhu, L. (2024). SegMamba: Long-range Sequential Modeling
  Mamba For 3D Medical Image Segmentation. *MICCAI 2024*. arXiv:2401.13560

- nnMamba (2024). nnMamba: 3D Biomedical Image Segmentation, Classification and Landmark Detection
  with State Space Model. arXiv:2402.03526

- EM-Net (2024). EM-Net: Efficient Channel and Frequency Learning with Mamba for 3D Medical Image
  Segmentation. *MICCAI 2024*. arXiv:2409.17675

- VM-UNet (2024). VM-UNet: Vision Mamba UNet for Medical Image Segmentation. *ACM TOMM 2024*.
  arXiv:2402.02491

- Wittmann, B., et al. (2024). vesselFM: A Foundation Model for Universal 3D Blood Vessel
  Segmentation. *CVPR 2025*. arXiv:2411.17386
