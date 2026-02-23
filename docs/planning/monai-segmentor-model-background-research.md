# MONAI Segmentation Model Selection for Tubular Vasculature

> **Phase 12 Research Report** | MinIVess MLOps
> **Decision**: Which single MONAI-compatible segmentation architecture to use
> **Task**: 3D tubular vascular segmentation (thin, expansive, branching structures)
> **Analogous domains**: Road extraction (remote sensing), neurite tracing (connectomics), airway segmentation

---

## 1. Executive Summary

This report evaluates segmentation architectures for 3D tubular vascular structures, where
topology preservation (connected centerlines, no false breaks) matters as much as volumetric
overlap (Dice). The analysis draws on challenge results (SMILE-UHURA, TopCoW, KiTS, Medical
Segmentation Decathlon), recent benchmarking studies (Isensee et al., MICCAI 2024), and
practical MONAI integration considerations.

### Recommendation

**DynUNet** (MONAI's nnU-Net architecture) with topology-aware compound loss (Dice + CE + clDice)
is the recommended single model for MinIVess MLOps, replacing SegResNet + SwinUNETR.

| Criterion | DynUNet | SegResNet | SwinUNETR |
|-----------|---------|-----------|-----------|
| Vascular benchmark Dice | **Best** (0.84 SMILE-UHURA†) | Good | Weakest (see §3.3) |
| Topology (clDice) | **Best** (w/ topology loss) | Good (w/ loss) | Weakest |
| Hyperparameter tuning | **Semi-auto** (via nnU-Net planning) | Manual (or Auto3DSeg) | Manual + pre-train |
| GPU memory (24GB, AMP) | 160^3 patches | **192^3 patches** | 96-128^3 patches |
| Parameters | 31.2M (5-level)‡ | **4.7M** (`init_filters=32`)‡ | 62.2M (`feature_size=48`)‡ |
| Training speed | Medium | **Fastest** | Slowest |
| ONNX export | Clean | Clean | **Problematic** (issue #5125) |
| Multi-task extensibility | Good (deep supervision) | **Best** (SegResNetDS2) | Complex |
| Anisotropic data handling | **Best** (auto kernels/strides) | Manual | Manual |
| Challenge wins | **Most** (MSD, TopCoW, BraTS) | BraTS 2018, Auto3DSeg default | Few |

> † SMILE-UHURA top entry (LSGroup) used an enhanced nnU-Net with multi-scale dilated
> convolution aggregation, not vanilla DynUNet. See §3.1 for details.
>
> ‡ Parameter counts are configuration-dependent. DynUNet ranges ~10-40M depending on
> depth; SegResNet ~4.7-19M depending on `init_filters`; SwinUNETR ~15-62M depending on
> `feature_size`. Values shown are common defaults.

**Key insight**: The architecture matters less than the loss function for vessel topology.
Pairing *any* CNN backbone with clDice loss + deep supervision outperforms a better
architecture with standard Dice+CE loss. Therefore:

1. **Quick win (immediate)**: Add clDice loss to existing SegResNet — expected +5-15%
   clDice improvement with zero architecture change
2. **Architecture migration**: Switch to DynUNet for its data-adaptive configuration
   (anisotropic spacing, optimal patch/kernel sizing via nnU-Net planning)

---

## 2. Why Tubular Structures Are Hard

Vascular segmentation differs from organ segmentation in fundamental ways:

1. **Extreme class imbalance**: Vessels occupy <5% of most volumes. Standard Dice loss
   under-weights thin branches.
2. **Topology criticality**: A single broken voxel disconnects an entire vessel branch.
   Standard overlap metrics (Dice) cannot detect this — a model scoring 0.95 Dice may
   have catastrophically broken connectivity.
3. **Multi-scale structure**: Vessels range from 1-voxel-wide capillaries to 20+ voxel
   arteries within the same volume. Diameter imbalance biases training toward large vessels.
4. **Elongated geometry**: Vessels are orders of magnitude longer than wide. Receptive
   field and patch size must capture sufficient longitudinal context.
5. **Branching topology**: Tree/graph structure with bifurcations, loops (Circle of Willis),
   and anastomoses requires preservation of specific topological features (Betti numbers).

### What Actually Helps (Ranked by Impact)

Based on comprehensive survey of 2021-2025 literature:

| Rank | Intervention | Impact | Reference |
|------|-------------|--------|-----------|
| 1 | **Topology-aware loss** (clDice, cbDice, Skeleton Recall) | +5-15% clDice | Shit et al. (2021), Kirchhoff et al. (2024) |
| 2 | **Multi-task learning** (mask + centerline/SDF) | +3-8% clDice | Rouge et al. (2024), Ma et al. (2025) |
| 3 | **Deep supervision** | +1-3% Dice | Standard in nnU-Net |
| 4 | **Architecture choice** (CNN vs Transformer) | 2-5% Dice | Isensee et al. (2024) |
| 5 | **Foreground oversampling** (33%+ patches with vessels) | +1-2% Dice | nnU-Net default |
| 6 | **Foundation model pre-training** | +5-25% in few-shot | vesselFM (Wittmann et al., 2025) |

---

## 3. Architecture Comparison

### 3.1 DynUNet (nnU-Net Architecture in MONAI)

**Origin**: Reimplementation of nnU-Net (Isensee et al., 2021) within MONAI. Based on:
- "Automated Design of Deep Learning Methods for Biomedical Image Segmentation" (Zhu et al., 2019)
- "nnU-Net: Self-adapting Framework" (Isensee et al., 2021)

**Self-adapting mechanism**: Three-tier configuration automatically derived from dataset properties:

| Tier | What's Determined | How |
|------|-------------------|-----|
| **Blueprint** (fixed) | Loss (Dice+CE), augmentation, LR schedule (poly), optimizer (SGD), 1000 epochs | Data-independent defaults |
| **Rule-based** (automatic) | Patch size, kernel sizes, strides, network depth, batch size | Dataset fingerprint (spacing, size, intensity) |
| **Empirical** (cross-val) | Best config (2D/3D), post-processing, ensemble | 5-fold cross-validation |

The `get_kernels_strides()` algorithm (from MONAI tutorial):
```python
# For each dimension:
spacing_ratio = spacing[dim] / min(spacings)
if spacing_ratio <= 2 and spatial_size[dim] >= 8:
    stride[dim] = 2; kernel[dim] = 3
else:
    stride[dim] = 1; kernel[dim] = 1
# Iterate until all strides = 1 (bottleneck ~4x4x4)
```

**Why it matters for tubular structures**: Anisotropic vascular data (e.g., 0.3×0.3×0.6mm)
gets anisotropic kernels/strides automatically — no manual tuning to avoid down-sampling
thin vessels in the coarse dimension.

**Vascular benchmark results**:

| Benchmark | nnU-Net Variant | Dice | clDice | Notes |
|-----------|----------------|------|--------|-------|
| SMILE-UHURA (7T brain vessels) | LSGroup (nnU-Net + multi-scale dilated conv) | 0.838 | — | Top methods all nnU-Net variants (Chatterjee et al., 2024) |
| SMILE-UHURA (7T brain vessels) | PBI (vanilla nnU-Net) | 0.787 | — | Baseline without enhancement |
| TopCoW (cerebral vessels) | nnU-Net baseline | 93.55 | — | Binary segmentation, 13-class: 85.36 |
| TopCoW + Skeleton Recall | nnU-Net + topology loss | 93.72 | — | +1.23 on multi-class (Kirchhoff et al., 2024) |
| Kidney vessels (HiP-CT) | nnU-Net | 0.9523 | 0.8631 | Relatively large vessels |
| MSD Task08 (hepatic vessels) | nnU-Net | Top | — | Won Medical Segmentation Decathlon |

> Note: On SMILE-UHURA, the best nnU-Net entry (LSGroup) used architectural enhancements
> (multi-scale dilated convolution aggregation block), not vanilla DynUNet. The vanilla
> nnU-Net entry (PBI) scored 0.787 combined — still competitive but 2 points lower. No
> SwinUNETR or SegResNet entries were found in the challenge for direct comparison.

**MONAI API**: `monai.networks.nets.DynUNet` — stable since MONAI v0.3, ONNX EXPORT_PASSED.

```python
from monai.networks.nets import DynUNet

model = DynUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    kernel_size=[[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]],
    strides=[[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2]],
    upsample_kernel_size=[[2,2,2], [2,2,2], [2,2,2], [2,2,2]],
    norm_name="instance",
    deep_supervision=True,
    deep_supr_num=3,
    res_block=True,  # Use residual blocks (nnU-Net ResEnc)
)
```

### 3.2 SegResNet (Current Implementation)

**Origin**: Myronenko (2019), BraTS 2018 winner. Encoder-decoder with residual blocks + VAE regularization.

**Strengths**:
- Smallest parameter count (4.7M with `init_filters=32`)
- Largest patch sizes on 24GB GPU (192^3)
- Fastest training (13.8h on A100 for TotalSegmentator)
- Clean ONNX export
- SegResNetDS/DS2 variants for deep supervision and multi-task

**Weaknesses for tubular structures**:
- No automated configuration — manual selection of `init_filters`, `blocks_down`, patch size
- Standard 3×3×3 kernels — limited receptive field for long vessel continuity
- No built-in anisotropic kernel handling
- Lower Dice than DynUNet on challenging datasets (2-5 points, per Isensee et al., 2024)

**When SegResNet wins**: Memory-constrained environments, fastest prototyping, when used
with Auto3DSeg (which handles configuration automatically).

### 3.3 SwinUNETR (Current Implementation)

**Origin**: Hatamizadeh et al. (2022). Swin Transformer encoder + CNN decoder.

**Finding from Isensee et al. (MICCAI 2024) — "nnU-Net Revisited"**:

> **Caveat**: These benchmarks are on organ segmentation (BTCV, KiTS, AMOS, LiTS), not
> vascular datasets. The gap on vascular tasks is expected to be similar or larger due to
> the locality bias argument below, but has not been directly measured on the same vascular
> benchmark with controlled settings.

| Dataset | nnU-Net | SwinUNETR | Gap |
|---------|---------|-----------|-----|
| BTCV (abdominal) | 83.08 | 78.89 | -4.2 |
| KiTS (kidney) | 86.04 | 81.27 | -4.8 |
| AMOS (abdominal) | 88.64 | 83.81 | -4.8 |
| LiTS (liver) | 80.09 | 76.50 | -3.6 |

> "Transformer-based architectures (SwinUNETR, nnFormer, CoTr) fail to match the
> performance of CNNs, including the original nnU-Net." — Isensee et al. (2024)

Note: MedNeXt (Roy et al., MICCAI 2023), a ConvNeXt-based architecture, outperformed both
standard nnU-Net and SwinUNETR on several of these benchmarks (84.70 on BTCV, 92.65 on ACDC).
MedNeXt is available in MONAI but is not yet as widely adopted.

**Why transformers are expected to underperform for thin structures**:
1. Shifted-window attention is *local* within each window (~7^3 voxels), not truly global
2. Tokenization at coarse resolution loses sub-voxel vessel detail
3. CNN inductive biases (locality, translation equivariance) better match vascular morphology
4. Small medical datasets insufficient for transformers to learn what CNNs encode structurally
5. Retinal vessel studies confirm: SwinUNETR "encounters challenges when segmenting blood
   vessel images characterized by numerous small regions and dense boundaries" (TD Swin-UNet, 2024)

**Additional practical issues**:
- ONNX export broken (MONAI issue #5125 — "Failed to export ONNX attribute")
- Heaviest memory (96-128^3 patches max on 24GB)
- Slowest training convergence (300-2000 epochs)
- Needs pre-training for competitive results

**Verdict**: SwinUNETR should be **deprioritized** in MinIVess MLOps. Evidence from organ
segmentation benchmarks (Isensee et al., 2024) and theoretical analysis of thin structures
both suggest CNN architectures are better suited to vascular segmentation. ONNX export
issues add practical friction. However, keeping SwinUNETR as a comparison baseline validates
the ModelAdapter abstraction across paradigms (CNN vs Transformer) and may provide ensemble
diversity (CNN+Transformer ensembles can outperform CNN-only due to complementary errors).

### 3.4 Other Considered Architectures

#### vesselFM (Wittmann et al., CVPR 2025)

First 3D vessel-specific foundation model. **Uses nnU-Net backbone**.

| Dataset | vesselFM Zero-Shot | vesselFM 5-Shot | Best Other 5-Shot |
|---------|-------------------|-----------------|-------------------|
| SMILE-UHURA | **74.66** | **78.77** | 61.17 (VISTA3D) |
| BvEM | **67.49** | **78.11** | 57.86 (SAM-Med3D) |
| OCTA | **46.94** | **75.70** | 54.25 (VISTA3D) |

vesselFM's zero-shot exceeds other foundation models' few-shot results. Key: it validates
the nnU-Net architecture family for vessels. Available on HuggingFace.

#### COMMA (Shi et al., 2025) — Mamba-Based

Coordinate-aware Modulated Mamba: local CNN branch + global Mamba branch.

| Dataset | Dice | clDice | NSD |
|---------|------|--------|-----|
| KiPA | 86.36 | 84.31 | 92.87 |
| ASOCA | 84.20 | 82.00 | 88.58 |
| PARSE | 83.84 | 76.96 | 83.54 |

Impressive results but: requires custom CUDA kernels, not in MONAI, experimental maturity.
**Monitor but don't adopt yet.**

#### HarmonySeg (Huang et al., ICCV 2025)

State-of-the-art tubular segmentation with vesselness-guided attention and growth-suppression
balanced loss. Best results on hepatic vessels (Dice 66.79, clDice 72.04) and airways.
Not yet available as a MONAI network.

---

## 4. Topology-Aware Losses for Vessels

### 4.1 Loss Function Landscape

| Loss | Topology Strength | Computation | Multi-class | In MONAI | Reference |
|------|-------------------|-------------|-------------|----------|-----------|
| **Dice + CE** | None | Lowest | Yes | Yes | Standard |
| **clDice** | Good (homotopy equiv.) | Low | **No** (binary) | **Yes** | Shit et al. (2021) |
| **cbDice** | Better (+ geometry) | Low | Yes | No | Shi et al. (2024) |
| **clCE** | Good (+ noise robust) | Low | No | No | Acebes et al. (2024) |
| **Skeleton Recall** | Good (connectivity) | **Very low** | **Yes** | No | Kirchhoff et al. (2024) |
| **Betti Matching** | Strong (exact) | High | Yes | No | Stucki et al. (2024) |
| **Topograph** | Strongest (strict) | Medium | Yes | No | ICLR 2025 |
| **Warping** | Strong (critical pts) | High | No | No | Hu (2022) |

### 4.2 Recommended Loss: Dice + CE + clDice

MONAI has native clDice at `monai.losses.cldice.SoftclDiceLoss`. The compound loss:

> **Important**: `SoftclDiceLoss` is inherently binary — it operates on single-channel
> probability maps. For multi-class output (e.g., 2-channel softmax), extract the
> vessel foreground channel before passing to clDice. For multi-class vessel
> segmentation (artery vs vein), apply clDice per-class.

```python
import torch.nn as nn
from monai.losses import DiceCELoss
from monai.losses.cldice import SoftclDiceLoss

class VesselCompoundLoss(nn.Module):
    def __init__(self, lambda_dicece=0.5, lambda_cldice=0.5):
        super().__init__()
        self.dice_ce = DiceCELoss(softmax=True, to_onehot_y=True)
        self.cldice = SoftclDiceLoss(smooth=1e-5, iter_=50)
        self.lambda_dicece = lambda_dicece
        self.lambda_cldice = lambda_cldice

    def forward(self, pred, target):
        # DiceCE handles multi-class natively
        loss_dicece = self.dice_ce(pred, target)
        # clDice needs binary: extract vessel foreground channel
        pred_vessel = pred[:, 1:2, ...]  # (B, 1, D, H, W) vessel probability
        target_binary = (target == 1).float()  # (B, 1, D, H, W) binary mask
        loss_cldice = self.cldice(pred_vessel, target_binary)
        return self.lambda_dicece * loss_dicece + self.lambda_cldice * loss_cldice
```

> **Numerical stability note**: `SoftclDiceLoss` with `iter_=50` can be numerically
> unstable on very thin structures (<2 voxels wide). Monitor for NaN gradients during
> early training. Consider starting with `iter_=10` and increasing.

**Impact on TopCoW challenge** (nnU-Net + Skeleton Recall Loss):
- Binary Dice: 93.55 → **93.72** (+0.17)
- Multi-class Dice: 85.36 → **86.59** (+1.23)
- Kirchhoff et al. (2024)

### 4.3 Multi-Task Extensions

For maximum topology preservation, add auxiliary regression heads:

1. **Centerline distance transform**: Regress distance-to-centerline at each voxel.
   Provides gradient signal in vessel interior where binary loss is zero.
2. **Signed Distance Function (SDF)**: VesselSDF (Esposito et al., MICCAI 2025) shows
   SDF regression eliminates floating segments and improves boundary precision.
3. **SDF pre-training** (SDF-TopoNet, Ma et al., 2025): Pre-train with SDF regression,
   then fine-tune with segmentation loss. Only ~4 minutes extra training, substantial
   topology improvement.

DynUNet's deep supervision provides natural multi-scale output points for attaching
auxiliary task heads.

---

## 5. Practical Considerations

### 5.1 GPU Memory (24GB RTX 3090/4090)

| Model | Max Patch Size | Batch Size | Params |
|-------|---------------|------------|--------|
| SegResNet | 192^3 | 1 | 4.7M |
| DynUNet | 160^3 | 1 | 31.2M |
| DynUNet (ResEnc M) | 128^3 | 2 | ~15M |
| SwinUNETR | 96-128^3 | 1 | 62.2M |

For vascular data: larger patches capture more vessel continuity. DynUNet at 160^3 is
sufficient for most vascular datasets. SegResNet's 192^3 advantage matters less when
topology-aware losses handle connectivity.

### 5.2 Training Time

From NVIDIA Clara benchmarks (TotalSegmentator, 104 classes, A100 80GB — **not directly
transferable** to 2-class vascular segmentation on consumer GPUs, shown for relative comparison):
- SegResNet: **13.8 hours**
- SwinUNETR: **15.6 hours**
- DynUNet: ~15-20 hours (estimated, not in Auto3DSeg benchmarks)

Actual training times are dataset and hardware dependent. For 2-class binary vascular
segmentation on 24GB GPUs, expect roughly 2-4x shorter than these 104-class benchmarks.

### 5.3 Hybrid Approach: nnU-Net Planning + MONAI DynUNet

The optimal workflow eliminates manual hyperparameter tuning:

1. Install nnU-Net v2: `pip install nnunetv2`
2. Run fingerprinting: `nnUNetv2_plan_and_preprocess -d DATASET_ID`
3. Read `nnUNetPlans.json` (human-readable):
   - `conv_kernel_sizes` → DynUNet `kernel_size`
   - `pool_op_kernel_sizes` → DynUNet `strides`
   - `patch_size` → training patch dimensions
   - `batch_size` → training batch size
4. Configure MONAI DynUNet with extracted parameters
5. Add clDice loss (not in nnU-Net defaults)
6. Train with MONAI pipeline (transforms, data loading, MLflow tracking)

**What you gain over pure nnU-Net**: MONAI ecosystem (transforms, Model Zoo, MONAI Deploy
MAP packaging for clinical deployment), custom loss integration, ONNX export.

**What you lose vs. full nnU-Net**: No automatic 2D/3D config selection, no automated
post-processing optimization, no ensemble management. These can be added manually.

### 5.4 ONNX Export

| Model | ONNX Export | TensorRT | Notes |
|-------|-----------|----------|-------|
| DynUNet | **PASS** | Works | Standard ops only |
| SegResNet | **PASS** | Works | Clean export |
| SwinUNETR | **FAIL** | Partial | MONAI issue #5125, requires workarounds |

### 5.5 What About Reducing to SegResNet Only?

SegResNet is the user's existing implementation and has compelling practical advantages.
However, for **tubular vascular segmentation specifically**:

1. DynUNet handles anisotropic data automatically; SegResNet requires manual kernel tuning
2. DynUNet has deeper architecture options (ResEnc L/XL); SegResNet is limited by
   `blocks_down` configuration
3. nnU-Net variants are top performers in all vascular challenges examined
4. vesselFM (CVPR 2025) validates nnU-Net/DynUNet as the vessel segmentation backbone
5. The hyperparameter tuning advantage is eliminated by nnU-Net planning

**SegResNet remains the recommended fallback** if GPU memory is severely constrained (<16GB)
or if the team decides the benchmark gaps are acceptable for faster iteration.

---

## 6. Evaluation Metrics for Tubular Structures

### 6.1 Required Metrics

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Dice** | Volumetric overlap | Standard, but topology-blind |
| **clDice** | Centerline Dice (topology) | Captures vessel connectivity preservation |
| **Betti errors** (β₀, β₁) | Connected components, loops | Topological correctness |
| **NSD** (Normalized Surface Distance) | Boundary accuracy | Vessel wall delineation |

### 6.2 Evaluation Pitfalls (Berger et al., IPMI 2025)

Critical warnings from "Pitfalls of Topology-Aware Image Segmentation":

1. **Connectivity definition matters**: Switching connectivity settings causes *negative
   correlations* (ρ = -0.37 to -0.85) in method rankings
2. **Label artifacts**: Up to 43% of measured Betti matching errors come from annotation noise
3. **Report β₀ and β₁ separately** — aggregated Betti numbers mask dimensional distinctions

---

## 7. Foundation Models as Future Option

### vesselFM (Wittmann et al., CVPR 2025)

- **Architecture**: nnU-Net backbone (validates our choice)
- **Training**: 17 vessel datasets + domain randomization + flow matching generative model
- **Zero-shot**: Dice 74.66 on SMILE-UHURA (other FMs: 48.32 max)
- **Few-shot (5 volumes)**: Dice 78.77 on SMILE-UHURA (other FMs: 61.17 max)
- **Availability**: Checkpoints on HuggingFace, code on GitHub
- **Implication**: When MinIVess data is sufficient, vesselFM fine-tuning may outperform
  training from scratch. DynUNet compatibility ensures smooth transition.

---

## 8. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **MONAI DynUNet diverges from nnU-Net v2**: MONAI's DynUNet is a reimplementation, not a wrapper. New nnU-Net features (ResEncL/XL, cascaded models) may not appear in MONAI. | Medium | Monitor MONAI releases; fallback to full nnU-Net pipeline if gap widens. DynUNet covers the core architecture. |
| **nnU-Net planning dependency**: The workflow depends on nnU-Net v2 for fingerprinting. If output format changes, integration breaks. | Low | Pin nnU-Net version; the `nnUNetPlans.json` format is stable and human-readable. |
| **clDice numerical instability**: `SoftclDiceLoss` with iterative skeletonization can produce NaN gradients on very thin structures. | Medium | Start with `iter_=10`, use gradient clipping, monitor for NaN. Skeleton Recall Loss (Kirchhoff et al., 2024) is a computationally cheaper alternative. |
| **Single-model risk**: If MinIVess data has properties poorly suited to DynUNet, no validated fallback. | Medium | Keep SegResNet adapter as tested fallback. The ModelAdapter ABC makes switching trivial. |
| **Community trajectory**: MONAI's roadmap favors SegResNet (Auto3DSeg default) and foundation models (VISTA-3D) over DynUNet. | Low | DynUNet is stable and widely used; SegResNet as Auto3DSeg default doesn't deprecate DynUNet. vesselFM (nnU-Net backbone) validates the architecture family long-term. |

### Data Volume Decision Gate

If annotated training volumes are limited:
- **<10 volumes**: Consider vesselFM fine-tuning as primary approach (few-shot Dice 78.77
  on SMILE-UHURA with only 5 volumes)
- **10-50 volumes**: DynUNet from scratch with clDice loss + aggressive augmentation
- **>50 volumes**: DynUNet from scratch will likely match or exceed vesselFM fine-tuning

vesselFM uses nnU-Net backbone, so the DynUNet adapter can load vesselFM checkpoints
with minimal modification — the approaches are complementary, not exclusive.

---

## 9. Final Recommendation

### Primary: DynUNet with Topology-Aware Training

```
Architecture:  MONAI DynUNet (res_block=True, deep_supervision=True)
Configuration: nnU-Net v2 fingerprinting → auto kernel/stride/patch
Loss:          Dice+CE (0.5) + clDice (0.5) compound
Metrics:       Dice, clDice, β₀ error, β₁ error, NSD
Future:        Add SDF/centerline auxiliary heads; vesselFM fine-tuning
```

### PRD Update Recommendation

| Model | New Probability | Status | Rationale |
|-------|----------------|--------|-----------|
| **DynUNet** | **0.35** | **viable → primary** | Challenge winner, self-adapting, vessel-validated |
| SegResNet | 0.20 → **0.15** | resolved → **fallback** | Lighter but manual tuning, lower benchmarks |
| vesselFM | 0.15 → **0.20** | viable | Best vessel FM, nnU-Net backbone validates arch |
| nnU-Net v2 (external) | 0.15 → **0.10** | viable | Full pipeline, but DynUNet covers the architecture |
| SwinUNETR | 0.15 → **0.05** | resolved → **deprioritized** | Broken ONNX, 4-7pt gap, no vessel advantage |
| COMMA Mamba | 0.10 → **0.10** | experimental | Monitor, not in MONAI |
| VISTA-3D | 0.15 → **0.05** | viable → **deprioritized** | Degrades 88→27% on small structures |

### Deprioritize SwinUNETR

SwinUNETR underperforms CNN architectures on organ segmentation benchmarks (4-7 Dice
points, Isensee et al., 2024) and is expected to perform similarly or worse on thin
vascular structures. It has ONNX export issues and the highest memory footprint.
However, maintaining the SwinUNETR adapter at low priority serves the project's
learning-first philosophy: it validates the ModelAdapter abstraction across paradigms
and may contribute ensemble diversity in future experiments.

### Implementation Roadmap

**Phase A — Quick win (on existing SegResNet, no architecture change):**
1. **Implement clDice loss** (MONAI native `SoftclDiceLoss`) — expected +5-15% clDice
2. **Add topology metrics** (clDice, Betti errors via `gudhi` or custom)
3. Baseline SegResNet + clDice results → establishes topology-aware benchmark

**Phase B — DynUNet width-ablation study (primary contribution):**
4. **Implement DynUNet adapter** (ModelAdapter ABC pattern)
5. **Add nnU-Net planning integration** (fingerprint → DynUNet config)
6. **Width ablation**: DynUNet `filters` parameter controls channel progression:
   - **Full** (default): `[32, 64, 128, 256]` (~31M params)
   - **Half** (÷2): `[16, 32, 64, 128]` (~8M params)
   - **Quarter** (÷4): `[8, 16, 32, 64]` (~2M params)
7. **Loss ablation**: Cross with Dice+CE vs Dice+CE+clDice vs Dice+CE+Skeleton Recall
8. **Systematic comparison**: 3 widths × 2-3 losses = 6-9 configs via Hydra-zen sweeps,
   tracked in MLflow, compared via DuckDB analytics

This width × loss ablation is novel: nobody has systematically evaluated DynUNet
capacity scaling for tubular vascular segmentation with topology-aware losses. The tiny
MinIVess dataset (Tahir et al., 2023) motivates smaller models to avoid overfitting —
quarter-width DynUNet at ~2M params is comparable to SegResNet's 4.7M.

**The MLOps pipeline is the product, not SOTA.** The ablation demonstrates reproducible
experiment management (Hydra-zen), data versioning (DVC), metric tracking (MLflow),
topology-aware evaluation (clDice), and model comparison — a template for how researchers
can systematically find SOTA for their own tasks.

**Phase C — Advanced (optional):**
9. SDF/centerline auxiliary heads on DynUNet decoder
10. vesselFM checkpoint loading for fine-tuning (if data is limited)

---

## References

- Acebes, C. et al. (2024). "Centerline-Cross Entropy Loss for Vessel-Like Structure Segmentation." *MICCAI 2024*.
- Berger, A. H. et al. (2025). "Pitfalls of Topology-Aware Image Segmentation." *IPMI 2025*. arXiv:2412.14619.
- Chatterjee, S. et al. (2024). "SMILE-UHURA Challenge." arXiv:2411.09593.
- Esposito, M. et al. (2025). "VesselSDF." *MICCAI 2025*. arXiv:2506.16556.
- Hatamizadeh, A. et al. (2022). "Swin UNETR: Transformer-Based Semantic Segmentation." *MICCAI 2022*.
- Huang, Y. et al. (2025). "HarmonySeg: Harmonizing Diverse Tubular Structure Segmentation." *ICCV 2025*. arXiv:2504.07827.
- Isensee, F. et al. (2021). "nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation." *Nature Methods* 18: 203-211.
- Isensee, F. et al. (2024). "nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation." *MICCAI 2024*. arXiv:2404.09556.
- Kirchhoff, Y. et al. (2024). "Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures." *ECCV 2024*. arXiv:2404.03010.
- Ma, J. et al. (2025). "SDF-TopoNet." arXiv:2503.14523.
- Mou, L. et al. (2021). "CS²-Net: Deep Learning Segmentation of Curvilinear Structures." *Medical Image Analysis* 67: 101874.
- Myronenko, A. (2019). "3D MRI Brain Tumor Segmentation Using Autoencoder Regularization." *BraTS 2018*.
- Rouge, P. et al. (2024). "Topology-Aware Cascaded Multitask U-Net for Cerebrovascular Segmentation." *PLOS ONE*. 10.1371/journal.pone.0311439.
- Shi, P. et al. (2024). "cbDice: Centerline Boundary Dice Loss for Vascular Segmentation." *MICCAI 2024*.
- Shi, P. et al. (2025). "COMMA: Coordinate-Aware Modulated Mamba Network for 3D Dispersed Vessel Segmentation." arXiv:2503.02332.
- Shit, S. et al. (2021). "clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation." *CVPR 2021*. arXiv:2003.07311.
- Stucki, N. et al. (2024). "Efficient Betti Matching for 3D Topology-Aware Segmentation." arXiv:2407.04683.
- Wittmann, B. et al. (2025). "vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation." *CVPR 2025*. arXiv:2411.17386.
- Roy, S. et al. (2023). "MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation." *MICCAI 2023*. arXiv:2303.09975.
- Zhu, W. et al. (2019). "Automated Design of Deep Learning Methods for Biomedical Image Segmentation." arXiv:1904.08128.
