# Mamba Model Capacity Matching for MiniVess Segmentation

**Date:** 2026-03-05
**Context:** Comparing `comma_mamba` and `ulike_mamba` against `dynunet` (full-width and
half-width) on the MiniVess 70-volume microvessel segmentation benchmark.

---

## 1. The Problem

Our two Mamba variants currently use `init_filters=32` (default), yielding ~2M parameters.
Our DynUNet full-width `[32, 64, 128, 256]` has ~15–20M trainable parameters, and the
established half-width variant `[16, 32, 64, 128]` has ~4–5M. A 10× parameter gap makes
any performance comparison an *efficiency* comparison, not an *iso-parameter architectural*
comparison. These are two different questions and the choice must be deliberate.

---

## 2. How the Literature Handles This

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

### 2.2 Efficiency-focused design (nnMamba, EM-Net approach)
Intentionally target fewer parameters than nnUNet and report efficiency wins.
nnMamba (2024) achieves 73.98% mDice on AMOS MRI-Test vs nnUNet's 67.63% — with
*half* the parameters (15.55M vs 31.18M). EM-Net (2024) at 39.41M outperforms
Swin UNETR (62.19M) and 3D UX-Net (53.01M).

### 2.3 Standardized protocols, natural sizes (Claims-to-Evidence approach)
Fix training hyperparameters (batch size, patch, loss, augmentation) identically across
all models, compare at natural parameter counts. Kazaj et al. (2025) concluded:
"Mamba-based architectures achieved competitive accuracy with no statistically significant
difference from nnUNet and U2Net, while using fewer parameters. However, they required
significantly longer training time." CNN models remain the best speed-accuracy tradeoff.

---

## 3. Current Parameter Inventory

### 3.1 MiniVess implementations

| Model | Family key | Default config | Estimated params | VRAM (gpu_low) | Profile |
|-------|-----------|---------------|-----------------|---------------|---------|
| DynUNet full-width | `dynunet` | filters=[32,64,128,256] | ~15–20M | ~7.4 GB | `dynunet.yaml` |
| DynUNet half-width | `dynunet` | filters=[16,32,64,128] | ~4–5M | ~4 GB | `dynunet_half_width_v1.yaml` |
| UlikeMamba | `ulike_mamba` | init_filters=32 | **~2M** | ~3 GB | `mamba.yaml` |
| CommaMamba | `comma_mamba` | init_filters=32, d_state=16 | **~2–3M** | ~4 GB | *(none)* |
| VesselFM | `vesselfm` | pre-trained DynUNet | ~30M | ~10 GB | `vesselfm.yaml` |

### 3.2 Literature reference points

| Model | Params | vs nnUNet | FLOPs | Venue |
|-------|--------|-----------|-------|-------|
| nnUNet PlainConvUNet 3D | 31.18M | baseline | 680 G | MICCAI benchmark |
| SegMamba (Xing et al., 2024) | 22.86M | −27% | 13,000 G | MICCAI 2024 |
| nnMamba (2024) | 15.55M | −50% | 141 G | arXiv:2402.03526 |
| EM-Net (2024) | 39.41M | +26% | 898 G | MICCAI 2024 |
| U-Mamba (Ma et al., 2024) | 58.47M | +88% | — | arXiv:2401.04722 |
| VM-UNet (2024) | 27.43M | −12% | 4 G | ACM TOMM 2024 |
| UlikeMamba_3dMT (Wang et al., 2025) | ~30–32M | ≈ parity | 93 G | arXiv:2503.19308 |
| LightUMamba (Kazaj et al., 2025) | 5.70M | −82% | — | arXiv:2503.01306 |
| COMMA (Shi et al., 2025) | not stated | FLOPs ≈ ½ SegMamba | — | arXiv:2503.02332 |

The **nnUNet-class range (20–35M)** is where competitive Mamba 3D segmentation models
cluster. Our current 2M default sits an order of magnitude below this.

---

## 4. Architecture Capacity Controls

### 4.1 DynUNet

```python
# src/minivess/adapters/dynunet.py
# Capacity controlled exclusively via filters list:
filters: list[int] = [32, 64, 128, 256]   # default full-width
#                    [16, 32,  64, 128]   # half-width (tested)
#                    [48, 96, 192, 384]   # 1.5× width
```

Parameter scaling: approximately quadratic with filter size. Halving filters ≈ ÷4 params.

### 4.2 UlikeMamba (`src/minivess/adapters/mamba.py`)

```python
def __init__(self, config: ModelConfig, *, init_filters: int = 32) -> None:
    f = params.get("init_filters", init_filters)
    # 4-level encoder: f → 2f → 4f → 8f (bottleneck)
    # Mamba block: _MambaBlock3D(channels) with d_state=16 hardcoded
```

The `_MambaBlock3D` uses 3D depthwise conv + 1×1 projections. No `expand` factor;
parameter growth is approximately linear in `f²` per level.

Configurable via YAML `architecture_params.init_filters`.

### 4.3 CommaMamba (`src/minivess/adapters/comma.py`)

```python
def __init__(self, config: ModelConfig, init_filters: int = 32, d_state: int = 16) -> None:
    # 4-level encoder: f → 2f → 4f → 8f
    # MambaBlock per level with expand=2 (d_inner = 2 × d_model)
    # Per-block params ≈ 3 × expand × d_model² (dominant projection term)
```

Both `init_filters` and `d_state` are configurable. `d_state` is
**universally set to 16** across all 2024–2025 Mamba medical imaging papers reviewed.
No paper ablates `d_state` for segmentation — it is treated as a fixed design constant.
`expand=2` is likewise never ablated in the medical domain.

---

## 5. Proposed Capacity Tiers

Three experimental tiers are proposed, each answering a different scientific question:

### Tier 0 — Efficiency baseline (current default)
**Question:** Can a 2M Mamba network compete with a 15–20M CNN on microvessel segmentation?
**Motivation:** nnMamba achieved better mDice than nnUNet at 50% params; is the efficiency
advantage even more extreme at 2M? Establishes the minimum viable Mamba.

| Model | Config | Params (est.) |
|-------|--------|--------------|
| DynUNet half-width | filters=[16,32,64,128] | ~4–5M |
| UlikeMamba | init_filters=32 | ~2M |
| CommaMamba | init_filters=32, d_state=16 | ~2–3M |

### Tier 1 — DynUNet half-width matching
**Question:** At matched parameter count (~4–5M), does Mamba architecture outperform CNN?
**Configuration adjustments:**

```yaml
# UlikeMamba matching DynUNet half-width
model: ulike_mamba
architecture_params:
  init_filters: 48     # estimated ~4–5M params

# CommaMamba matching DynUNet half-width
model: comma_mamba
architecture_params:
  init_filters: 40     # estimated ~4–5M params
  d_state: 16
```

### Tier 2 — DynUNet full-width matching (nnUNet-class)
**Question:** At equivalent capacity (~15–20M), does the SSM mechanism provide any benefit
over a well-tuned CNN baseline? (The Isensee et al. 2024 "No-Mamba Base" ablation made
this question concrete.)

```yaml
# UlikeMamba matching DynUNet full-width
model: ulike_mamba
architecture_params:
  init_filters: 80     # estimated ~15–18M params

# CommaMamba matching DynUNet full-width
model: comma_mamba
architecture_params:
  init_filters: 72     # estimated ~15–18M params
  d_state: 16
```

> **Note:** These estimates are based on quadratic scaling of conv layers and need to be
> empirically verified with `sum(p.numel() for p in model.parameters())` before committing
> to training. See Section 6.

---

## 6. Parameter Count Verification (Required Before Training)

The estimates above are theoretical. Before running 100-epoch training runs, verify
actual parameter counts with:

```bash
uv run python -c "
from minivess.config.models import ModelConfig, ModelFamily
from minivess.adapters.model_builder import build_adapter

for family, filters_or_init in [
    ('ulike_mamba', {'init_filters': 32}),
    ('ulike_mamba', {'init_filters': 48}),
    ('ulike_mamba', {'init_filters': 80}),
    ('comma_mamba', {'init_filters': 32, 'd_state': 16}),
    ('comma_mamba', {'init_filters': 40, 'd_state': 16}),
    ('comma_mamba', {'init_filters': 72, 'd_state': 16}),
]:
    cfg = ModelConfig(
        family=family,
        in_channels=1,
        out_channels=2,
        architecture_params=filters_or_init,
    )
    m = build_adapter(cfg, use_stub=True)
    n = sum(p.numel() for p in m.parameters())
    print(f'{family:15s} {str(filters_or_init):40s}  {n/1e6:.2f}M')
"
```

Adjust `init_filters` until each tier's Mamba count is within ±10% of the DynUNet target.

---

## 7. Experimental Design Recommendation

**Primary recommendation:** Run Tier 0 first (current defaults) as a smoke test, then
Tier 1 (half-width matching) as the scientifically cleanest comparison.

**Rationale:**
1. The Isensee et al. (2024) result makes Tier 2 (full-width matching) the most rigorous
   test of whether Mamba's SSM mechanism adds value. But it requires 80+ init_filters,
   which may strain gpu_low VRAM.
2. Tier 1 at ~4–5M is more VRAM-friendly and directly comparable to the half-width
   DynUNet run we already have results for.
3. Tier 0 (current defaults) is a legitimate efficiency experiment in the spirit of
   nnMamba — but should be labeled clearly as such, not as a fair architectural comparison.

**Loss function:** Use `cbdice_cldice` (project default) for all variants to enable
direct metric comparison with existing DynUNet runs.

**Epochs:** 100 (same as DynUNet experiments). Kazaj et al. (2025) notes that Mamba
architectures converge more slowly than CNN baselines — 100 epochs may underestimate
Mamba performance relative to CNNs. Consider monitoring clDice at epoch 50 and 100.

---

## 8. Critical Caveat: The Isensee et al. (2024) Result

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
   is exactly the use case where SSM linear-complexity should help, as it can model
   the full sequence of Z-slices without the window size restrictions of convolution.
3. The topology-aware losses (`cbdice_cldice`) we use may interact differently with
   Mamba's sequential processing vs CNN spatial filtering.

This makes the MiniVess comparison genuinely scientifically interesting, not just
a replication of published nnUNet/Mamba experiments.

---

## 9. Summary Table: Planned Experimental Configurations

| Tier | Model | init_filters | d_state | Est. Params | VRAM | Scientific Question |
|------|-------|-------------|---------|------------|------|---------------------|
| **T0** | DynUNet half | [16,32,64,128] | — | ~4–5M | ~4 GB | Baseline (existing results) |
| **T0** | UlikeMamba | 32 | — | ~2M | ~3 GB | Extreme efficiency |
| **T0** | CommaMamba | 32 | 16 | ~2–3M | ~4 GB | Extreme efficiency |
| **T1** | UlikeMamba | 48 | — | ~4–5M† | ~4 GB | Iso-param vs DynUNet half-width |
| **T1** | CommaMamba | 40 | 16 | ~4–5M† | ~4 GB | Iso-param vs DynUNet half-width |
| **T2** | DynUNet full | [32,64,128,256] | — | ~15–20M | ~7.4 GB | Baseline (existing results) |
| **T2** | UlikeMamba | 80 | — | ~15–18M† | ~6–7 GB | Iso-param vs DynUNet full-width |
| **T2** | CommaMamba | 72 | 16 | ~15–18M† | ~6–7 GB | Iso-param vs DynUNet full-width |

† Estimates — verify with parameter count script (Section 6) before training.

---

## 10. Next Steps

1. **Verify parameter counts** using the script in Section 6
2. **Adjust YAML configs** for Tier 1 and Tier 2 based on actual counts
3. **Run Tier 0** with `./scripts/train_alternative_variants.sh --skip-heavy` (current defaults)
4. **Run Tier 1** with adjusted init_filters configs
5. **Compare metrics** using `uv run python scripts/compare_models.py`
6. If Mamba shows topology advantages (clDice improvement), proceed to Tier 2

GitHub Issues:
- #366 — VesselFM fine-tuning on RunPod (external compute)

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
