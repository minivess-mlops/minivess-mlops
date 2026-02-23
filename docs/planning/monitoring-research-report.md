# Monitoring, Drift Detection, and Continuous Retraining for Tiny-Dataset Medical Segmentation MLOps

**MinIVess MLOps v2 -- Research Report**
**Date**: 2026-02-24
**Status**: Synthesized from 4 parallel research agents

---

## 1. Executive Summary

This report addresses the fundamental tension at the heart of the MinIVess MLOps project: how to build a robust, production-grade monitoring and continuous retraining pipeline when the training dataset consists of approximately 20 volumes of cerebral vasculature from 2-photon microscopy (Tahir et al., 2023). Traditional drift detection methods assume access to hundreds or thousands of samples for statistical power. With a dataset this small, every design choice -- from statistical tests to retraining triggers -- requires careful adaptation.

**Key findings:**

1. **MLOps Level 4 is achievable.** The Microsoft MLOps maturity model (Microsoft, 2024) defines Level 4 as fully automated retraining with drift triggers. The MinIVess stack (Hydra-zen, MONAI (Cardoso et al., 2022), MLflow, BentoML, Evidently (Ackerman et al., 2021), Alibi-Detect) already includes or can straightforwardly integrate every component required for Level 4.

2. **Embedding-based drift detection is the right paradigm.** For 3D medical segmentation, raw voxel-level drift tests are impractical. Extracting penultimate-layer embeddings from DynUNet and applying Maximum Mean Discrepancy (MMD) tests via Alibi-Detect provides the best balance of statistical power, interpretability, and computational cost.

3. **The demonstration strategy uses synthetic OOD data.** Because real distribution shifts are rare and uncontrollable, a tiered synthetic OOD pipeline -- TorchIO corruption (Tier 1), procedural Bezier tubes (Tier 2), vesselFM domain randomization (Tier 3) -- enables systematic stress-testing of the entire monitoring pipeline.

4. **Full retraining beats continual learning at this scale.** With ~20 volumes, full retraining takes minutes on a single GPU. Continual learning methods (EWC, replay buffers, progressive networks) are overkill and introduce unnecessary complexity. The challenger-champion pattern provides safe model promotion.

5. **MC Dropout is the minimum viable uncertainty method; deep ensembles are the gold standard.** MONAI's DynUNet natively supports 3D dropout, making MC Dropout nearly free to implement. A 3-model deep ensemble is affordable with 20 volumes and provides superior calibration and OOD detection.

6. **Evidential Deep Learning should be avoided.** Shen et al. (2024) demonstrated at NeurIPS 2024 that EDL's "epistemic uncertainty is a mirage" -- the uncertainty estimates do not reliably separate in-distribution from out-of-distribution data.

---

## 2. MLOps Maturity Model (Microsoft Level 0-4)

### 2.1 The Five Levels

The Microsoft MLOps maturity model (Microsoft, 2024) defines a progression from ad-hoc ML development to fully automated ML lifecycle management. Stone et al. (2025) provide a comprehensive analysis of MLOps lifecycle practices that reinforces this framework. Critically, there is no Level 5 in the current Microsoft model -- Level 4 represents full maturity.

| Level | Name | Description | Key Capability |
|-------|------|-------------|----------------|
| **0** | No MLOps | Manual everything. Notebooks, manual training, no versioning. | None |
| **1** | DevOps but no MLOps | CI/CD for code, but models are trained and deployed manually. | Code versioning, testing |
| **2** | Automated training | Training pipeline runs end-to-end without human intervention. | Reproducible training, experiment tracking |
| **3** | Automated deployment | Model deployment is automated via CI/CD. Serving is containerized. | Model registry, automated serving |
| **4** | Full automated retraining | Drift detection triggers retraining. Challenger-champion evaluation. Full lineage. | Drift monitoring, automated promotion |

### 2.2 MinIVess Level 4 Mapping

The MinIVess stack already includes tooling for every Level 4 requirement. The following table maps Level 4 capabilities to specific project components:

| Level 4 Requirement | MinIVess Implementation | Status |
|---------------------|------------------------|--------|
| Automated training pipeline | Hydra-zen configs, MONAI training loop, MLflow tracking | Planned |
| Reproducible experiments | Hydra-zen, DVC, MLflow experiment tracking, DuckDB analytics | Planned |
| Data validation gates | Pydantic v2 (schema), Pandera (DataFrame), Great Expectations (batch) | Implemented |
| Model validation | Deepchecks Vision, WeightWatcher spectral diagnostics | Implemented |
| Drift detection | Evidently (embedding drift), Alibi-Detect (MMD), whylogs (profiling) | Config only |
| Automated deployment | BentoML (REST API), ONNX Runtime, Gradio (demo UI) | Planned |
| Model registry | MLflow Model Registry with stage transitions | Planned |
| Retraining triggers | Drift-based (Evidently/Alibi-Detect), scheduled, data-volume | Not started |
| Challenger-champion | MLflow model comparison on locked test set | Not started |
| Full lineage | OpenLineage/Marquez for IEC 62304 traceability | Planned |
| Audit trail | Compliance module with lifecycle event logging | Planned |
| Monitoring dashboards | Prometheus + Grafana infrastructure, Evidently reports | Config only |

### 2.3 The Gap: Level 2 to Level 4

The project is currently at approximately Level 1-2 maturity. The path to Level 4 requires:

1. **Level 2 completion**: Wire Hydra-zen configs to MONAI training loop with MLflow logging. This is primarily an integration task.
2. **Level 3 completion**: Implement BentoML serving with ONNX export, GitHub Actions CI/CD with CML for ML-specific PR comments, and MLflow model registry stage transitions.
3. **Level 4 completion**: Implement drift detection (this report's primary focus), retraining triggers, challenger-champion evaluation, and monitoring dashboards.

The critical insight is that Level 4 does not require a large dataset. It requires *automation infrastructure* that is dataset-size-agnostic. The statistical challenges of tiny datasets affect the *sensitivity* of drift detection, not the *architecture* of the monitoring pipeline.

---

## 3. Drift Detection for 3D Medical Segmentation

### 3.1 Taxonomy of Drift

Distribution shift in deployed ML systems takes several distinct forms, each requiring different detection and mitigation strategies. Roschewitz et al. (2024) propose automatic shift type identification, which is particularly relevant for medical imaging where shift sources are often confounded.

**Covariate shift** occurs when the input distribution P(X) changes but the labeling function P(Y|X) remains constant. In microscopy, this arises from changes in imaging protocol (laser power, scan speed, staining concentration), different microscope hardware, or different tissue preparation methods. For MinIVess, covariate shift is the most likely real-world scenario -- a new collaborating lab acquires vasculature volumes with a different 2-photon setup.

**Concept drift** occurs when the relationship P(Y|X) changes. In segmentation, this means the definition of "vessel" changes -- for example, if annotation guidelines evolve to include or exclude capillaries below a certain diameter. This is less common in medical imaging than in NLP or recommender systems, but can occur with protocol updates.

**Prior (prevalence) shift** occurs when the class distribution P(Y) changes without changes to P(X|Y). In vascular segmentation, this would manifest as volumes with dramatically different vessel density -- sparse cortical vasculature versus dense hippocampal microvasculature. The vessel-to-background ratio shift can degrade Dice scores even when per-voxel accuracy is maintained.

**Label shift** is the inverse of covariate shift: P(Y) changes but P(X|Y) is constant. In practice, this often co-occurs with prior shift in segmentation tasks.

The SHIFT framework (Singh et al., 2025) extends drift detection to subgroup levels, enabling identification of which anatomical regions or volume characteristics are drifting. This is valuable for MinIVess because vessel segmentation performance often varies by vessel caliber: large arterioles may remain well-segmented while fine capillaries degrade.

### 3.2 Tool Landscape for Image Drift Detection

The following table compares the major open-source drift detection tools relevant to 3D medical segmentation:

| Tool | Version | Image Drift | Method | Strengths | Limitations | MinIVess Role |
|------|---------|-------------|--------|-----------|-------------|---------------|
| **Evidently AI** | v0.5+ | Embedding-level | MMD, cosine distance, Wasserstein | Grafana export, rich reports, active development | Column-mapping setup for embeddings | Primary drift dashboard |
| **Alibi-Detect** | v0.12+ | Embedding-level | MMD, LSDD, Learned Kernel, Online MMD | Best statistical tests for high-dim, online detection | Less visualization, steeper API | Primary statistical testing |
| **NannyML** | v0.10+ | No (tabular only) | CBPE, DLE performance estimation | Excellent for tabular, concept drift | Cannot handle images directly | Not applicable |
| **whylogs** | v1.3+ | Property-level | Brightness, contrast, histogram profiles | Lightweight, continuous profiling, mergeable profiles | Shallow image understanding | Continuous profiling layer |
| **Deepchecks Vision** | v0.11+ | Property-level | Image Property Drift, label drift | Easy integration, rich validation suite | **2D-only** -- requires per-slice adaptation for 3D volumes; no native MONAI integration | Model validation checks (2D slices) |
| **Frouros** | v0.8+ | Embedding-level | KS, MMD, chi-squared | Clean API, growing method set | Younger project, smaller community | Exploratory / cross-validation |

Muller et al. (2024) provide the D3Bench comparison of drift detection methods, establishing that embedding-based approaches consistently outperform pixel-level and property-level methods for image data. This finding directly supports our recommended approach.

**NannyML caveat**: NannyML is frequently mentioned in MLOps guides but is **strictly tabular**. Its Confidence-Based Performance Estimation (CBPE) and Direct Loss Estimation (DLE) methods are excellent for tabular data but cannot process 3D medical images. It should not appear in the MinIVess image drift pipeline, though it could monitor tabular metadata (patient demographics, acquisition parameters) if available.

### 3.3 Recommended Approach: Embedding-Based Drift Detection

The recommended drift detection architecture for MinIVess uses a three-layer approach:

**Layer 1: Metadata drift (whylogs + Pandera)**

Before touching embeddings, validate that incoming volumes match expected metadata distributions. This catches equipment changes, protocol deviations, and data pipeline errors.

```python
import pandera as pa
from pandera.typing import Series

class VolumeMetadataSchema(pa.DataFrameModel):
    """Schema for incoming volume metadata validation."""
    voxel_spacing_x: Series[float] = pa.Field(ge=0.1, le=10.0)
    voxel_spacing_y: Series[float] = pa.Field(ge=0.1, le=10.0)
    voxel_spacing_z: Series[float] = pa.Field(ge=0.1, le=10.0)
    shape_x: Series[int] = pa.Field(ge=64, le=2048)
    shape_y: Series[int] = pa.Field(ge=64, le=2048)
    shape_z: Series[int] = pa.Field(ge=16, le=512)
    bit_depth: Series[int] = pa.Field(isin=[8, 16, 32])
    intensity_mean: Series[float] = pa.Field(ge=0.0)
    intensity_std: Series[float] = pa.Field(gt=0.0)
    intensity_p99: Series[float] = pa.Field(gt=0.0)
```

**Layer 2: Embedding drift (Alibi-Detect MMD)**

Extract penultimate-layer embeddings from DynUNet for each volume. The bottleneck features of a DynUNet (quarter-width, `filters=[8, 16, 32, 64]`) produce a compact representation that captures both spatial structure and intensity patterns.

```python
import numpy as np
from alibi_detect.cd import MMDDrift
from alibi_detect.cd.pytorch import preprocess_drift

def build_drift_detector(
    reference_embeddings: np.ndarray,
    p_val: float = 0.01,
) -> MMDDrift:
    """Build MMD drift detector from reference embeddings.

    Args:
        reference_embeddings: Shape (n_ref, embed_dim) from training set.
        p_val: Significance level for drift detection.

    Returns:
        Configured MMD drift detector.
    """
    detector = MMDDrift(
        x_ref=reference_embeddings,
        backend="pytorch",
        p_val=p_val,
        # Kernel bandwidth selection
        configure_kernel_from_x_ref=True,
        # Number of permutations for p-value estimation
        n_permutations=500,
    )
    return detector


def extract_embeddings(model, volume, hook_layer="bottleneck"):
    """Extract penultimate-layer embeddings from DynUNet.

    The model is run in eval mode. A forward hook captures activations
    at the specified layer. The output is global-average-pooled to produce
    a single embedding vector per volume.
    """
    embeddings = {}

    def hook_fn(module, input, output):
        # Global average pool over spatial dims (D, H, W)
        embeddings["features"] = output.mean(dim=(-3, -2, -1)).detach().cpu().numpy()

    handle = dict(model.named_modules())[hook_layer].register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        _ = model(volume)
    handle.remove()
    return embeddings["features"]
```

**Layer 3: Population-level drift test**

Accumulate embeddings over a rolling window and test against the training reference distribution. With ~20 reference volumes, the MMD test has limited power for detecting subtle shifts. To compensate:

- Use a **conservative p-value threshold** (p < 0.01 rather than p < 0.05)
- Require **multiple consecutive alerts** before triggering retraining (e.g., 3 out of 5 windows)
- Complement with **per-volume anomaly scoring** (Mahalanobis distance) that does not require a population test

```python
from scipy.spatial.distance import mahalanobis

def compute_anomaly_score(
    embedding: np.ndarray,
    reference_mean: np.ndarray,
    reference_cov_inv: np.ndarray,
) -> float:
    """Per-volume anomaly score via Mahalanobis distance.

    With ~20 training volumes, the covariance matrix may be poorly
    conditioned. Apply Ledoit-Wolf shrinkage regularization.
    """
    return mahalanobis(embedding, reference_mean, reference_cov_inv)
```

**Statistical challenges with ~20 volumes**: The covariance matrix estimated from 20 samples in a high-dimensional embedding space will be rank-deficient. Mitigations include:

- **Ledoit-Wolf shrinkage** for covariance estimation (scikit-learn `LedoitWolf`)
- **PCA dimensionality reduction** before Mahalanobis (retain 95% variance, typically 5-10 components)
- **Kernel MMD** (which does not require explicit covariance estimation) as the primary population test
- **Bootstrapped confidence intervals** for drift statistics

### 3.4 Statistical Foundations

Recent advances in sequential testing and conformal monitoring provide stronger statistical guarantees for drift detection:

**WATCH: Weighted Conformal Test Martingales** (Prinster et al., 2025, ICML). WATCH provides anytime-valid sequential tests for distribution shift. Unlike fixed-sample tests (Kolmogorov-Smirnov, MMD with fixed window), WATCH accumulates evidence over time and can declare drift at any point while controlling the Type I error rate. This is particularly valuable for MinIVess where new volumes arrive sporadically -- WATCH does not require waiting for a fixed batch size.

**Conditional Conformal Test Martingales** (Shaer et al., 2026). Extends WATCH to handle correlated sequential samples, which is common in medical imaging where consecutive acquisitions from the same scanner share systematic characteristics. Standard i.i.d. assumptions are violated when volumes arrive from clinical workflows.

**Statistical validity as a standard** (Dolin et al., 2025). Argues that statistically valid monitoring should be the default rather than an afterthought. For MinIVess, this means drift detection should produce calibrated p-values rather than ad-hoc threshold crossings.

**SHIFT framework** (Singh et al., 2025). Enables subgroup-level drift detection. For vascular segmentation, this could identify that drift is concentrated in specific vessel calibers or anatomical regions rather than being uniform.

**ADAPT** (Xiong et al., 2026). Addresses adversarial concept drift, where distribution shifts are designed to evade detection. While adversarial attacks are unlikely in the MinIVess academic setting, ADAPT's robust detection methods are useful for general defense-in-depth.

### 3.5 Alibi-Detect vs. Evidently: When to Use Which

The MinIVess pipeline benefits from using both Alibi-Detect and Evidently, but for different purposes:

**Use Alibi-Detect for:**
- Statistically rigorous hypothesis testing (MMD with permutation-based p-values)
- Online drift detection (accumulating evidence over time without fixed windows)
- Learned kernel drift detection (training a kernel to maximize test power for a specific reference distribution)
- Classifier-based drift detection (training a domain classifier to distinguish reference vs. test)

**Use Evidently for:**
- Dashboard generation and Grafana export (Evidently produces rich HTML reports and Grafana-compatible JSON)
- Column-level drift summary across multiple features simultaneously
- Integration with the existing Prometheus + Grafana stack
- Non-technical stakeholder communication (Evidently reports are visually accessible)

**Implementation pattern:**

```python
# Alibi-Detect: statistical backbone
from alibi_detect.cd import MMDDrift

detector = MMDDrift(x_ref=reference_embeddings, backend="pytorch", p_val=0.01)
result = detector.predict(new_embeddings)
is_drift = result["data"]["is_drift"]  # bool
p_value = result["data"]["p_val"]       # float

# Evidently: reporting layer
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_df, current_data=curr_df)
report.save_html("drift_report.html")
# Export to Grafana via JSON
report.as_dict()  # Structured output for Grafana panels
```

The two tools are complementary: Alibi-Detect provides the statistical engine, Evidently provides the visualization and reporting layer. Both should be wired into the monitoring pipeline.

### 3.6 Practical Considerations for Tiny Datasets

With approximately 20 training volumes, standard drift detection faces three challenges:

1. **Low statistical power.** The MMD test with 20 reference samples and a window of 5 incoming samples may fail to detect moderate shifts. Power analysis suggests that for embedding dimensions of 64, the MMD test at p=0.05 can detect effect sizes > 0.8 (Cohen's d equivalent) but will miss subtler shifts.

2. **Unstable reference statistics.** The mean and covariance of 20 embedding vectors are noisy estimates. Leave-one-out cross-validation of anomaly scores provides a calibrated baseline: the maximum Mahalanobis distance observed under leave-one-out gives a natural 95th percentile threshold.

3. **Multiple testing correction.** If monitoring multiple drift signals (embedding MMD, metadata drift, uncertainty spike), Bonferroni or Benjamini-Hochberg correction is needed to maintain overall false alarm rates.

The recommended mitigation is a **multi-signal consensus approach**: trigger a drift alert only when at least 2 of 3 signals (embedding drift, metadata drift, uncertainty spike) exceed their respective thresholds simultaneously. This reduces false alarms while maintaining sensitivity to genuine shifts.

---

## 4. Label-Free Performance Estimation

### 4.1 The Ground-Truth Problem

At deployment, expert annotations are not available for every incoming volume. Manual annotation of 3D vascular segmentation is extremely time-consuming -- a single volume can require hours of expert time. The system must estimate performance without ground truth.

The SUDO framework (Kiyasseh et al., 2024, Nature Communications) provides a principled approach to evaluating clinical AI systems without ground-truth annotations. SUDO uses pseudo-label discrepancy scores to estimate model performance at deployment. Kiyasseh et al. validate the framework on dermatology, ophthalmology, and radiology datasets, demonstrating that label-free performance estimation is feasible across multiple medical imaging modalities. The core principle -- using model confidence and consistency as performance proxies -- transfers directly to segmentation.

SEG: Segmentation Evaluation in the absence of Ground truth (Sims et al., 2023) specifically addresses the segmentation setting, proposing metrics that correlate with ground-truth performance without requiring labels.

Fluhmann et al. (2025) introduce label-free confusion matrix estimation, enabling approximate reconstruction of per-class performance metrics (precision, recall, F1) without any labeled data. This is particularly valuable for binary segmentation (vessel vs. background) where class imbalance makes accuracy misleading.

### 4.2 Methods for Label-Free Estimation

**Confidence-based proxies.** The simplest approach uses the model's own softmax probabilities as a quality signal:

- **Mean softmax entropy**: High entropy over the output probability map indicates the model is uncertain. For binary segmentation, entropy is maximized when P(vessel) = 0.5.
- **Fraction of low-confidence voxels**: Count voxels where max(P(vessel), P(background)) < 0.7. A spike in this fraction suggests the model is struggling.
- **Prediction margin**: The difference between the top two class probabilities, averaged over the volume.

**MC Dropout variance** (see Section 5.1). With T stochastic forward passes, the per-voxel variance of predictions provides a direct estimate of epistemic uncertainty. Aggregated over the volume (mean, 95th percentile, fraction of high-variance voxels), this correlates with segmentation error.

**Embedding distance anomaly score.** The Mahalanobis distance of a volume's embedding from the training distribution centroid. High distance implies the volume is unlike training data, and performance is likely degraded.

**In-Context Reverse Classification Accuracy** (2025). Train a lightweight classifier to predict whether a volume came from the training distribution or deployment distribution. The classifier's confidence provides a proxy for how different the deployment data is from training.

**Agreement-based methods.** If a deep ensemble is available (Section 5.2), inter-model disagreement -- the variance of Dice scores across ensemble members -- directly estimates prediction quality. High disagreement implies unreliable predictions.

### 4.3 Recommended Three-Signal Quality Estimator

For MinIVess, we recommend logging three complementary signals per inference:

| Signal | Method | Computational Cost | What It Captures |
|--------|--------|-------------------|-----------------|
| **Signal 1** | Mean softmax entropy | Free (byproduct of inference) | Model confidence |
| **Signal 2** | MC Dropout variance (T=10) | ~10x inference time | Epistemic uncertainty |
| **Signal 3** | Embedding Mahalanobis distance | Negligible (one forward hook) | Distribution membership |

All three signals should be logged to MLflow per inference. The combination provides robustness: Signal 1 captures aleatoric uncertainty (inherently ambiguous inputs), Signal 2 captures epistemic uncertainty (lack of training data), and Signal 3 captures distribution shift (the input is unlike anything seen during training).

```python
from dataclasses import dataclass

@dataclass
class QualityEstimate:
    """Per-volume quality estimate logged to MLflow."""
    volume_id: str
    softmax_entropy_mean: float
    softmax_entropy_p95: float
    mc_dropout_variance_mean: float
    mc_dropout_variance_p95: float
    mc_dropout_low_agreement_fraction: float
    mahalanobis_distance: float
    estimated_quality: str  # "high", "medium", "low", "ood"

    # Thresholds calibrated from leave-one-out cross-validation
    ood_threshold: float = 10.0            # Mahalanobis distance 95th pct from LOO
    high_uncertainty_threshold: float = 0.1  # MC Dropout variance upper bound
    medium_entropy_threshold: float = 0.5    # Softmax entropy midpoint

    def classify_quality(self) -> str:
        """Rule-based quality classification."""
        if self.mahalanobis_distance > self.ood_threshold:
            return "ood"
        if self.mc_dropout_variance_mean > self.high_uncertainty_threshold:
            return "low"
        if self.softmax_entropy_mean > self.medium_entropy_threshold:
            return "medium"
        return "high"
```

---

## 5. Epistemic Uncertainty for OOD Detection

### 5.1 MC Dropout on DynUNet

MONAI's DynUNet natively supports dropout via the `dropout` parameter, with `dropout_dim=3` for proper 3D spatial dropout. This makes MC Dropout implementation nearly cost-free.

**Implementation approach:**

```python
import torch
from monai.networks.nets import DynUNet

def create_dynunet_with_dropout(
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 2,
    filters: tuple = (8, 16, 32, 64),
    dropout: float = 0.1,
) -> DynUNet:
    """Create DynUNet with dropout for MC Dropout inference.

    Quarter-width architecture (~2M params) with progressive dropout.
    """
    # Kernel sizes and strides from nnU-Net fingerprinting (Isensee et al., 2021, 2024)
    kernel_size = [[3, 3, 3]] * len(filters)
    strides = [[1, 1, 1]] + [[2, 2, 2]] * (len(filters) - 1)

    model = DynUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=strides[1:],
        filters=filters,
        dropout=dropout,
        dropout_dim=3,  # 3D spatial dropout: zeroes entire feature map channels
        deep_supervision=True,
        res_block=True,
    )
    return model


def mc_dropout_inference(
    model: DynUNet,
    volume: torch.Tensor,
    n_passes: int = 10,
) -> dict:
    """Run MC Dropout inference with T stochastic forward passes.

    Args:
        model: DynUNet with dropout layers.
        volume: Input tensor of shape (1, C, D, H, W).
        n_passes: Number of stochastic forward passes.

    Returns:
        Dictionary with mean prediction, variance map, and entropy.

    Memory: ~1.3 GB peak for 128^3 volumes with T=10 on quarter-width.
    """
    # Enable dropout at inference time
    model.train()

    predictions = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(volume)
            # Take only the main output (not deep supervision heads)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs)

    # Stack: (T, B, C, D, H, W)
    predictions = torch.stack(predictions, dim=0)

    # Mean prediction
    mean_pred = predictions.mean(dim=0)

    # Predictive variance (epistemic uncertainty)
    variance = predictions.var(dim=0)

    # Predictive entropy
    entropy = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(dim=1)

    return {
        "mean_prediction": mean_pred,
        "variance": variance,
        "entropy": entropy,
        "all_predictions": predictions,
    }
```

**Optimal dropout rates:** Zenk et al. (2023) recommend progressive dropout rates for encoder-decoder architectures: 0.1 for shallow layers, increasing to 0.16 for deep layers. For the MinIVess DynUNet (quarter-width), a uniform rate of 0.1 is a reasonable starting point, with per-layer tuning as a should-have optimization.

**Memory considerations:** With `filters=[8, 16, 32, 64]` (quarter-width, ~2M parameters) and 128^3 input volumes, a single forward pass requires approximately 130 MB of GPU memory. Ten stochastic passes require ~1.3 GB peak memory (only one pass is active at a time; gradients are disabled). This is well within the 24 GB budget of an RTX 3090/4090.

### 5.2 Method Comparison

| Method | Training Cost | Inference Cost | OOD Detection | Calibration | Implementation Complexity |
|--------|-------------|---------------|---------------|-------------|--------------------------|
| **MC Dropout** | Zero (just add dropout) | T x single pass | Moderate | Moderate | Low |
| **Deep Ensembles (3-model)** | 3x training (~15 min total with 20 volumes) | 3x single pass | Best | Best | Low-Medium |
| **Evidential Deep Learning** | Single training with modified loss | Single pass | **Unreliable** | Poor | Medium |
| **Concrete Dropout** | Modified training (auto-tunes dropout) | T x single pass | Moderate-Good | Good | Medium |
| **SWAG** | Single training + low-rank covariance | T x single pass | Good | Good | Medium |

**Deep Ensembles (3-model)** are the gold standard for uncertainty quantification (Lakshminarayanan et al., 2017). The typical objection -- training cost -- is irrelevant for MinIVess. Training DynUNet (quarter-width) on 20 volumes takes approximately 5 minutes per model on a single GPU. A 3-model ensemble requires 15 minutes total. The inference cost (3x) is acceptable for offline batch processing. The ensemble additionally provides prediction disagreement as a powerful OOD signal.

**Evidential Deep Learning (EDL)**: **Avoid.** Shen et al. (2024) at NeurIPS 2024 demonstrated that EDL's epistemic uncertainty estimates are fundamentally unreliable -- they called it "a mirage." The Dirichlet-based uncertainty does not separate in-distribution from out-of-distribution data in practice. The original theoretical framework has been shown to have critical gaps. Given available alternatives (MC Dropout, ensembles), there is no reason to use EDL.

**Concrete Dropout** (Gal et al., 2017) automatically tunes dropout rates per layer via variational inference. This addresses the dropout rate tuning problem but adds training complexity. For MinIVess, it is a should-have enhancement over fixed-rate MC Dropout.

**Recommendation**: Implement MC Dropout first (must-have, zero training cost), then add a 3-model deep ensemble (should-have, 15 minutes additional training).

### 5.3 Conformal Prediction for Segmentation

Conformal prediction (Angelopoulos & Bates, 2023) provides distribution-free coverage guarantees: given a calibration set, the prediction sets are guaranteed to contain the true label with probability at least 1-alpha, regardless of the underlying distribution.

**Morphological Prediction Sets (ConSeMa)** (Mossina & Friedrich, 2025) presented at MICCAI 2025 adapt conformal prediction for medical image segmentation. Instead of pixel-wise prediction sets (which are trivial and uninformative), ConSeMa produces morphological prediction sets: inner and outer contours that bound the true segmentation with guaranteed coverage.

**MAPIE** (already in the MinIVess stack) provides the computational infrastructure for conformal prediction. While originally designed for tabular data, MAPIE can be adapted for segmentation by treating the calibration set as a collection of per-volume nonconformity scores.

**Practical implementation for MinIVess:**

```python
def conformal_segmentation(
    predictions: list[np.ndarray],
    ground_truths: list[np.ndarray],
    alpha: float = 0.1,
) -> dict:
    """Calibrate conformal prediction sets for segmentation.

    With ~20 volumes, split 15 train / 5 calibration.
    Coverage guarantee: P(true mask in prediction set) >= 1 - alpha.

    The nonconformity score is 1 - Dice(predicted, true) per volume.
    """
    # Compute nonconformity scores on calibration set
    scores = []
    for pred, gt in zip(predictions, ground_truths):
        dice = compute_dice(pred, gt)
        scores.append(1.0 - dice)

    # Quantile for coverage guarantee (capped at 1.0 for small calibration sets)
    n_cal = len(scores)
    q = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    threshold = np.quantile(scores, q)

    return {
        "threshold": threshold,
        "calibration_scores": scores,
        "coverage_guarantee": 1 - alpha,
    }
```

**Limitation with tiny calibration sets:** With only 5 calibration volumes, the coverage guarantee is coarse. The achievable coverage levels are {0/6, 1/6, 2/6, 3/6, 4/6, 5/6, 6/6}, not arbitrary values. Shah-Mohammadi et al. (2025) validate conformal prediction for medical image segmentation but with larger calibration sets. For MinIVess, conformal prediction is a should-have that becomes more powerful as the dataset grows.

### 5.4 Calibration

Post-hoc calibration adjusts predicted probabilities so that they reflect true likelihoods. A model that outputs P(vessel) = 0.8 should be correct 80% of the time in regions with that predicted probability.

**Temperature scaling** (Guo et al., 2017) is the simplest approach: divide logits by a learned scalar T before softmax. For MinIVess, global temperature scaling (a single T for the entire model) is appropriate because:

- With ~20 volumes, there is insufficient data for per-class or per-region temperature parameters
- Global T can be reliably estimated from a held-out validation set of 3-5 volumes
- The `netcal` library (already in the MinIVess stack) provides implementation

**Local Temperature Scaling** learns per-voxel or per-region temperature parameters, enabling spatially-varying calibration. This is valuable for vascular segmentation where calibration may differ between vessel interior (confident predictions) and vessel boundary (uncertain predictions). However, with <50 volumes, the per-location parameters will overfit. Local Temperature Scaling should be treated as a nice-to-have for when the dataset grows.

**Mask-TS** applies temperature scaling specifically to lesion/foreground regions, using a binary mask to separate calibration targets. This is more appropriate than global TS for segmentation with extreme class imbalance (vessels occupy <5% of volume). However, it still requires sufficient calibration data.

**Recommendation for MinIVess**: Use global temperature scaling as the calibration baseline (implemented via `netcal`). Avoid complex calibration methods (local TS, Mask-TS, isotonic regression per region) until the dataset exceeds ~50 volumes. Evaluate calibration with ECE, MCE, and reliability diagrams logged to MLflow.

### 5.5 Can Epistemic Uncertainty Flag OOD?

Yes, but it is a necessary-not-sufficient condition. High epistemic uncertainty (MC Dropout variance or ensemble disagreement) indicates that the model has not seen similar data during training. However:

1. **False negatives**: Some OOD inputs may produce confident-but-wrong predictions. Adversarial examples and some covariate shifts can maintain low uncertainty while degrading performance.
2. **False positives**: Some in-distribution inputs near decision boundaries may produce high uncertainty without being OOD.

The recommended approach combines epistemic uncertainty with embedding-space distance:

**Multi-signal OOD scoring:**
- **MC Dropout variance** (mean over volume) captures model-level uncertainty
- **Mahalanobis distance** (from training distribution centroid in embedding space) captures distributional deviation
- **Multi-layer Mahalanobis** (not just bottleneck, but also intermediate encoder features) improves robustness. Compute Mahalanobis at 2-3 layers and average the scores.

**Mahalanobis++** (2025) improves the reliability of Mahalanobis distance as an OOD detector by using class-conditional statistics and relative distances. For binary segmentation, compute separate centroids for "vessel-rich" and "vessel-sparse" volume embeddings.

```python
def multi_signal_ood_score(
    mc_variance: float,
    mahalanobis_dist: float,
    softmax_entropy: float,
    thresholds: dict,
) -> dict:
    """Combine multiple signals for OOD detection.

    Each signal is normalized by its training-set baseline
    (leave-one-out 95th percentile).
    """
    norm_mc = mc_variance / thresholds["mc_variance_p95"]
    norm_maha = mahalanobis_dist / thresholds["mahalanobis_p95"]
    norm_entropy = softmax_entropy / thresholds["entropy_p95"]

    # Geometric mean of normalized scores
    combined = (norm_mc * norm_maha * norm_entropy) ** (1.0 / 3.0)

    return {
        "normalized_mc_variance": norm_mc,
        "normalized_mahalanobis": norm_maha,
        "normalized_entropy": norm_entropy,
        "combined_ood_score": combined,
        "is_ood": combined > 1.5,  # Calibrated from leave-one-out
    }
```

---

## 6. Synthetic OOD Data Generation for Monitoring Demos

### 6.1 The Strategy

The goal of synthetic OOD data generation is **not** to produce publication-quality synthetic vasculature volumes. The goal is to generate deliberately out-of-distribution data that stress-tests the monitoring pipeline. We need data that:

1. **Triggers drift alerts** at controllable severity levels
2. **Demonstrates uncertainty spikes** on inputs the model has never seen
3. **Tests data quality gates** at various stringency levels
4. **Validates the retraining trigger** by accumulating detectable drift

This aligns with the OpenMIBOOD taxonomy of OOD levels:
- **Near-OOD**: Subtle covariate shift (same modality, slightly different acquisition parameters)
- **Moderate OOD**: Significant distribution shift (same anatomy, different modality or severe artifacts)
- **Far-OOD**: Fundamentally different data (different anatomy, synthetic geometry, random noise)

### 6.2 Tier 1: TorchIO Corruption Pipeline (Recommended First)

TorchIO (Perez-Garcia et al., 2021) is already in the MinIVess stack (`augmentation_stack.monai_plus_torchio` is resolved and implemented). Its corruption transforms can be repurposed as a parametric OOD generator with continuous "OOD-ness" control.

**Near-OOD (subtle covariate shift):**

```python
import torchio as tio

def create_near_ood_transform(severity: float = 0.3) -> tio.Compose:
    """Subtle corruptions simulating scanner variation.

    severity in [0, 1] controls corruption intensity.
    """
    return tio.Compose([
        tio.RandomBiasField(
            coefficients=0.3 * severity,
            p=0.8,
        ),
        tio.RandomNoise(
            mean=0.0,
            std=(0.01 * severity, 0.05 * severity),
            p=0.8,
        ),
        tio.RandomGamma(
            log_gamma=(-0.2 * severity, 0.2 * severity),
            p=0.5,
        ),
    ])
```

**Moderate OOD (significant artifacts):**

```python
def create_moderate_ood_transform(severity: float = 0.6) -> tio.Compose:
    """Moderate corruptions simulating equipment issues."""
    return tio.Compose([
        tio.RandomBlur(
            std=(0.5 * severity, 2.0 * severity),
            p=0.7,
        ),
        tio.RandomGamma(
            log_gamma=(-0.5 * severity, 0.5 * severity),
            p=0.8,
        ),
        tio.RandomBiasField(
            coefficients=0.8 * severity,
            p=0.9,
        ),
        tio.RandomNoise(
            mean=0.0,
            std=(0.05 * severity, 0.15 * severity),
            p=0.9,
        ),
    ])
```

**Far-OOD (severe corruption):**

```python
def create_far_ood_transform(severity: float = 1.0) -> tio.Compose:
    """Severe corruptions simulating fundamentally different data."""
    return tio.Compose([
        tio.RandomGhosting(
            num_ghosts=(2, int(5 * severity) + 2),
            intensity=(0.3 * severity, 0.8 * severity),
            p=0.8,
        ),
        tio.RandomSpike(
            num_spikes=(1, int(3 * severity) + 1),
            intensity=(0.5 * severity, 1.5 * severity),
            p=0.7,
        ),
        tio.RandomAnisotropy(
            downsampling=(1.5, 1.5 + 3.0 * severity),
            p=0.6,
        ),
        tio.RandomBlur(
            std=(1.0 * severity, 4.0 * severity),
            p=0.8,
        ),
        tio.RandomNoise(
            mean=0.0,
            std=(0.1 * severity, 0.3 * severity),
            p=1.0,
        ),
    ])
```

**Effort estimate:** 2-4 hours. TorchIO is already a dependency. The corruption pipeline requires minimal new code.

### 6.3 Tier 2: Procedural Bezier Tube Volumes

Generate entirely synthetic vasculature-like volumes using random 3D Bezier curves as centerlines and Gaussian tube profiles as vessel cross-sections. These volumes are structurally vessel-like but have fundamentally different morphological statistics than real cerebral vasculature.

**Approach:**
1. Generate random 3D Bezier curve centerlines (4-point cubic Bezier)
2. Add branching at random intervals (Poisson process along each curve)
3. Rasterize Gaussian tube profiles into a 3D volume grid
4. Add background noise and bias field for realism

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def generate_bezier_tube_volume(
    shape: tuple[int, int, int] = (128, 128, 128),
    n_tubes: int = 15,
    n_branches: int = 30,
    tube_radius_range: tuple[float, float] = (1.0, 4.0),
    seed: int | None = None,
) -> np.ndarray:
    """Generate a synthetic volume with procedural Bezier tube vessels.

    This produces far-OOD data: vessel-like structures with entirely
    different branching patterns, tortuosity, and density than real
    cerebral vasculature from 2-photon microscopy.

    Args:
        shape: Volume dimensions (D, H, W).
        n_tubes: Number of primary vessel centerlines.
        n_branches: Number of branching points across all vessels.
        tube_radius_range: Min and max vessel radius in voxels.
        seed: Random seed for reproducibility.

    Returns:
        Synthetic volume with values in [0, 1].
    """
    rng = np.random.default_rng(seed)
    volume = np.zeros(shape, dtype=np.float32)

    def bezier_curve(p0, p1, p2, p3, n_points=200):
        """Cubic Bezier curve in 3D."""
        t = np.linspace(0, 1, n_points)[:, None]
        return (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2
            + t ** 3 * p3
        )

    def add_tube(volume, centerline, radius):
        """Rasterize a tube along a centerline into the volume."""
        for point in centerline:
            z, y, x = point.astype(int)
            if all(0 <= c < s for c, s in zip([z, y, x], shape)):
                # Create a small sphere at each centerline point
                r_int = int(np.ceil(radius)) + 1
                for dz in range(-r_int, r_int + 1):
                    for dy in range(-r_int, r_int + 1):
                        for dx in range(-r_int, r_int + 1):
                            nz, ny, nx = z + dz, y + dy, x + dx
                            if (0 <= nz < shape[0] and 0 <= ny < shape[1]
                                    and 0 <= nx < shape[2]):
                                dist = np.sqrt(dz**2 + dy**2 + dx**2)
                                intensity = np.exp(-0.5 * (dist / radius) ** 2)
                                volume[nz, ny, nx] = max(
                                    volume[nz, ny, nx], intensity
                                )

    for _ in range(n_tubes):
        # Random control points for Bezier curve
        points = [rng.uniform(10, s - 10, size=3)
                  for s, _ in zip(shape, range(4))]
        centerline = bezier_curve(*points)
        radius = rng.uniform(*tube_radius_range)
        add_tube(volume, centerline, radius)

    # Add background noise
    volume += rng.normal(0, 0.02, shape).astype(np.float32)
    volume = np.clip(volume, 0, 1)

    return volume
```

**Why this is far-OOD:** Real cerebral vasculature has specific branching angles (typically 60-90 degrees), vessel caliber ratios (Murray's law), and density patterns determined by metabolic demand. Random Bezier tubes violate all of these. The embedding distance from training data will be large, the model's predictions will be unreliable, and the monitoring pipeline should trigger alerts.

**Effort estimate:** 1-2 days. Approximately 100-200 lines of code.

### 6.4 Tier 3: vesselFM Domain Randomization

vesselFM (Wittmann et al., 2025) is a foundation model for vessel segmentation that uses an elaborate domain randomization pipeline to generate synthetic training data. The domain randomization component can be adapted independently:

1. **Foreground vessel generation**: Bezier-based curves with physiologically-inspired branching (closer to reality than Tier 2)
2. **Textured backgrounds**: Perlin noise, real tissue textures, scanner-specific noise patterns
3. **MONAI transforms**: Integration with the existing MONAI + TorchIO augmentation pipeline

The vesselFM domain randomization pipeline produces data that is more realistic than Tier 2 but still controllably different from the MinIVess training distribution. By adjusting the randomization parameters (branching density, vessel caliber distribution, background texture), the OOD level can be systematically varied.

**Effort estimate:** 2-3 days. Requires adapting vesselFM's open-source pipeline to the MinIVess data format and resolution.

### 6.5 Tier 4: Generative Models (Diminishing Returns)

MONAI GenerativeModels was archived in February 2025 and its components were migrated to MONAI Core (version 1.4+). The available generative architectures include:

- **AutoencoderKL**: Variational autoencoder with KL regularization. Supports 3D.
- **VQ-VAE**: Vector-quantized VAE. Supports 3D.
- **DDPM**: Denoising Diffusion Probabilistic Models. Supports 3D.
- **LDM**: Latent Diffusion Models. Supports 3D.

**The fundamental problem: 20 volumes is far too few for useful generative modeling.** Even with patch-based training (extracting 64^3 patches from each volume to increase the effective sample count), the diversity of the training set is insufficient for the generator to learn meaningful vasculature morphology.

**For monitoring demos, low-quality generation IS the feature.** A poorly-trained generative model produces artifacts that are detectable by the monitoring pipeline. The generated samples will have:
- Unrealistic vessel branching patterns (detectable by embedding drift)
- Texture artifacts (detectable by whylogs image property profiling)
- Unusual intensity distributions (detectable by metadata quality gates)

**Vessel-specific generative models:**
- **VesselDiffusion** (TMI 2025): Diffusion model specifically for vessel generation, but requires large training sets
- **VasTSD** (2025): Topology-aware synthetic vessel data, research-stage implementation

**Effort estimate:** 3-5 days. Diminishing returns compared to Tiers 1-3.

### 6.6 Decision Matrix

| Approach | Effort | Training Data Needed | OOD Control | Realism | Best For |
|----------|--------|---------------------|-------------|---------|----------|
| **Tier 1: TorchIO** | 2-4 hours | None (corrupts existing) | Continuous dial | High (real vessels + artifacts) | Covariate shift demos |
| **Tier 2: Bezier Tubes** | 1-2 days | None (procedural) | Parametric | Low (wrong morphology) | Far-OOD demos |
| **Tier 3: vesselFM DR** | 2-3 days | None (procedural + textures) | Parametric | Medium | Moderate-OOD demos |
| **Tier 4: Generative** | 3-5 days | 20 volumes (insufficient) | Limited | Very low (too few samples) | Artifact-OOD demos |

**Recommendation:** Implement Tier 1 first (immediate value, minimal effort). Add Tier 2 as a should-have for far-OOD demonstrations. Tier 3 and 4 are nice-to-haves.

---

## 7. Data Quality Gates for Incoming Training Data

### 7.1 Tiered Gate Architecture

Data quality gates prevent corrupted, mislabeled, or out-of-distribution data from entering the training pipeline. Each gate is progressively more computationally expensive and specific:

```
Incoming Volume
    │
    ▼
┌─────────────────────────────┐
│ Gate 1: Metadata Validation │  ← Pandera schema (ms)
│  - Voxel spacing in range   │
│  - Shape dimensions valid   │
│  - Bit depth correct        │
│  - File integrity check     │
└─────────────┬───────────────┘
              │ PASS
              ▼
┌─────────────────────────────┐
│ Gate 2: Statistical Checks  │  ← whylogs + custom (ms)
│  - Intensity distribution   │
│  - Histogram similarity     │
│  - Dynamic range check      │
│  - Zero-voxel fraction      │
└─────────────┬───────────────┘
              │ PASS
              ▼
┌─────────────────────────────┐
│ Gate 3: Embedding OOD       │  ← DynUNet + Mahalanobis (seconds)
│  - Extract embeddings       │
│  - Mahalanobis distance     │
│  - Compare to reference     │
│  - Anomaly score threshold  │
└─────────────┬───────────────┘
              │ PASS
              ▼
┌─────────────────────────────┐
│ Gate 4: Batch-Level Drift   │  ← Alibi-Detect MMD (seconds)
│  - Accumulate window        │
│  - Population-level MMD     │
│  - Trend analysis           │
│  - Multi-signal consensus   │
└─────────────┬───────────────┘
              │ PASS
              ▼
      Training Pipeline
```

### 7.2 Implementation with Project Stack

The MinIVess stack already includes the necessary tools. The integration requires wiring them together:

**Gate 1: Metadata validation (Pydantic v2 + Pandera)**

```python
import pandera as pa
from pydantic import BaseModel, field_validator
from pathlib import Path


class VolumeMetadata(BaseModel):
    """Pydantic model for volume-level metadata validation."""
    file_path: Path
    voxel_spacing: tuple[float, float, float]
    shape: tuple[int, int, int]
    bit_depth: int
    modality: str

    @field_validator("voxel_spacing")
    @classmethod
    def validate_spacing(cls, v):
        if any(s <= 0 or s > 10.0 for s in v):
            raise ValueError(f"Voxel spacing out of range: {v}")
        return v

    @field_validator("bit_depth")
    @classmethod
    def validate_bit_depth(cls, v):
        if v not in (8, 16, 32):
            raise ValueError(f"Unexpected bit depth: {v}")
        return v
```

**Gate 2: Statistical checks (whylogs + Great Expectations)**

```python
import whylogs as why
from whylogs.core.metrics import MetricConfig

def profile_volume(volume_array, reference_profile=None):
    """Create whylogs profile for a volume and compare to reference."""
    # Compute summary statistics
    stats = {
        "intensity_mean": float(volume_array.mean()),
        "intensity_std": float(volume_array.std()),
        "intensity_min": float(volume_array.min()),
        "intensity_max": float(volume_array.max()),
        "intensity_p01": float(np.percentile(volume_array, 1)),
        "intensity_p99": float(np.percentile(volume_array, 99)),
        "zero_fraction": float((volume_array == 0).mean()),
        "dynamic_range": float(volume_array.max() - volume_array.min()),
    }

    # Log to whylogs
    profile = why.log(stats).profile()

    if reference_profile is not None:
        # Compare current stats against reference using simple z-score checks
        ref_df = reference_profile.view().to_pandas()
        curr_df = profile.view().to_pandas()
        # For production use, whylogs constraints API provides proper drift checks
        return stats, {"current": curr_df, "reference": ref_df}

    return stats, None
```

**Gate 3: Embedding OOD detection (Mahalanobis distance)**

Uses the `extract_embeddings` and `compute_anomaly_score` functions from Sections 3.3 and 5.5, with Ledoit-Wolf covariance regularization for the tiny reference set.

**Gate 4: Batch-level drift test (Alibi-Detect MMD)**

Uses the `build_drift_detector` from Section 3.3, applied to accumulated embedding windows.

### 7.3 OOD Detection as Input Gate

Beyond the tiered gate architecture, dedicated OOD detection methods can serve as an additional input quality filter:

**Normalizing flows for post-hoc OOD.** Lotfi et al. (2024) achieve 84.61% AUROC on the MedOOD benchmark using normalizing flows applied to pretrained model features. The approach requires no OOD training data and can be applied to any pretrained model (including the MinIVess DynUNet) as a post-hoc addition.

**OOD detection taxonomy.** Hodge et al. (2025) provide a comprehensive taxonomy of OOD detection methods for safety assurance, categorizing approaches as:
- **Density-based**: Estimate P(x) under the training distribution (normalizing flows, VAEs)
- **Distance-based**: Measure distance to training data in feature space (Mahalanobis, kNN)
- **Classifier-based**: Train a binary classifier on in-distribution vs. perturbed data

For MinIVess, the distance-based approach (Mahalanobis distance in embedding space) is most practical with 20 volumes. Density-based methods require more training data; classifier-based methods require explicit negative examples.

### 7.4 MONAI Integration: Custom Quality Transform

The data quality gates can be integrated into the MONAI data pipeline as a custom transform, enabling quality checks to run as part of the standard data loading workflow:

```python
from monai.transforms import MapTransform
from typing import Any


class DataQualityGate(MapTransform):
    """MONAI-compatible transform that validates volume quality.

    Integrates with the tiered gate architecture (Section 7.1).
    Raises ValueError for volumes that fail quality checks,
    which MONAI's error handling can catch and log.
    """

    def __init__(
        self,
        keys: list[str],
        metadata_schema: type,  # Pandera DataFrameModel
        reference_stats: dict[str, float],
        embedding_extractor: callable | None = None,
        reference_embeddings: Any | None = None,
        strict: bool = True,
    ):
        super().__init__(keys)
        self.metadata_schema = metadata_schema
        self.reference_stats = reference_stats
        self.embedding_extractor = embedding_extractor
        self.reference_embeddings = reference_embeddings
        self.strict = strict

    def __call__(self, data):
        for key in self.keys:
            volume = data[key]

            # Gate 1: Metadata validation
            self._validate_metadata(data)

            # Gate 2: Statistical checks
            self._validate_statistics(volume)

            # Gate 3: Embedding OOD (if extractor available)
            if self.embedding_extractor is not None:
                self._validate_embedding(volume)

        return data

    def _validate_metadata(self, data):
        """Check voxel spacing, shape, and bit depth."""
        spacing = data.get("pixdim", None)
        if spacing is not None:
            for s in spacing[:3]:
                if s <= 0 or s > 10.0:
                    msg = f"Voxel spacing out of range: {spacing}"
                    if self.strict:
                        raise ValueError(msg)

    def _validate_statistics(self, volume):
        """Check intensity distribution against reference."""
        mean_val = float(volume.mean())
        std_val = float(volume.std())
        ref_mean = self.reference_stats.get("intensity_mean", 0.5)
        ref_std = self.reference_stats.get("intensity_std", 0.2)

        # Flag if mean deviates by more than 3 reference stds
        if abs(mean_val - ref_mean) > 3 * ref_std:
            msg = f"Intensity mean {mean_val:.3f} deviates from reference {ref_mean:.3f}"
            if self.strict:
                raise ValueError(msg)

    def _validate_embedding(self, volume):
        """Check if volume embedding is within reference distribution."""
        if self.embedding_extractor is None or self.reference_embeddings is None:
            return  # Skip if no embedding infrastructure configured
        # Delegates to Mahalanobis scoring from Section 5.5
        embedding = self.embedding_extractor(volume)
        # Score against reference distribution (see compute_anomaly_score)
        # Raise ValueError if anomaly score exceeds threshold
```

This pattern enables quality validation to be part of the MONAI `Compose` transform chain, rather than a separate pre-processing step. Volumes that fail quality gates are flagged before entering the training or inference pipeline.

---

## 8. Continuous Retraining and Model Lifecycle

### 8.1 Full Retraining > Continual Learning (for <50 volumes)

This is a critical design decision. Continual learning methods -- Elastic Weight Consolidation (EWC), Experience Replay, Progressive Networks -- are designed for scenarios where retraining on all data is expensive or infeasible. Neither condition applies to MinIVess.

**Why continual learning is inappropriate:**

- **EWC**: Fisher information matrix is poorly estimated with ~20 samples. The regularization penalty is unreliable, leading to either catastrophic forgetting (penalty too weak) or inability to learn new data (penalty too strong).
- **Replay buffers**: The standard approach stores a representative subset of past data to prevent forgetting. When the entire dataset is ~20 volumes, just store all of them. A replay buffer is the full dataset.
- **Progressive networks**: Add new capacity for each task. With a single task (vascular segmentation) and data arriving from the same distribution, this is architectural overkill.

**Why full retraining is trivial:**

- DynUNet (quarter-width, ~2M parameters) on 20 volumes: **~5 minutes** on a single GPU
- Even with augmentation and multiple epochs: **<15 minutes**
- Storage: all 20 volumes fit easily in GPU memory
- Simplicity: no continual learning hyperparameters to tune, no forgetting to debug

**Recommendation**: Always retrain from scratch on all available data. Log the full training history in MLflow. The computational cost is negligible. The engineering simplicity is enormous.

This recommendation aligns with the continual learning in medical image analysis survey (2024), which notes that for very small datasets, the overhead of continual learning methods exceeds the cost of full retraining.

### 8.2 Retraining Triggers

The MinIVess pipeline should support multiple retraining triggers, any of which can initiate a retraining run:

| Trigger | Threshold | Justification |
|---------|-----------|---------------|
| **Embedding drift** | MMD p < 0.01 on rolling window of 5 volumes | Primary automated trigger. Conservative p-value compensates for low statistical power. |
| **MC Dropout uncertainty spike** | Mean variance > 2 std above training baseline | New data is systematically more uncertain than training data. |
| **Mahalanobis distance** | Per-volume score > 95th percentile (calibrated via LOO) | Individual volumes are far from training distribution. |
| **Estimated performance drop** | Proxy quality estimate drops > 5% from baseline | Label-free performance estimation signals degradation. |
| **Calendar time** | Every 3-6 months | Catch slow drift even if statistical tests miss it. |
| **New labeled data** | Any new annotated volume | Immediately retrain to incorporate new evidence. |
| **Multi-signal consensus** | 2 of 3 signals exceed thresholds simultaneously | Reduces false alarms while maintaining sensitivity. |

Beytur et al. (2025) analyze optimal resource allocation for retraining triggers, showing that drift-based triggers outperform calendar-based triggers in terms of compute efficiency when drift detection is reliable. For MinIVess, we recommend a hybrid approach: drift-based as the primary trigger with a calendar-based backstop.

### 8.3 Challenger-Champion Pattern

The challenger-champion pattern is the standard approach for safe model promotion:

```
[Drift Alert or New Data]
          │
          ▼
    Train Challenger
    (all old + new data)
          │
          ▼
    Evaluate Challenger
    on Locked Test Set
          │
          ▼
    ┌─────────────┐
    │ Challenger   │──── YES ──→ Promote Challenger
    │ >= Champion? │             to Production
    └─────┬───────┘
          │ NO
          ▼
    Keep Champion,
    Log Failure in MLflow
```

**Implementation details:**

1. **Locked test set**: 3-5 volumes held out permanently. Never used for training, augmentation, or hyperparameter tuning. The Dice score on this set is the single source of truth for model quality.

2. **Promotion criteria**: Challenger Dice >= Champion Dice - epsilon (epsilon = 0.01 to avoid rejecting models that are equivalent within noise). Alternatively, use a paired permutation test on per-volume Dice scores.

3. **Model soup alternative**: Instead of binary promotion, use greedy model soup (Wortsman et al., 2022) to weight-average challenger and champion. This is already implemented in the MinIVess ensemble strategy. Model soup is more conservative than full replacement and can only improve performance (by construction of the greedy soup algorithm).

4. **MLflow tracking**: Log champion vs. challenger comparison as an MLflow run with tags `{role: "promotion_evaluation", champion_run_id: "...", challenger_run_id: "..."}`.

```python
import mlflow
from pathlib import Path


def evaluate_challenger(
    champion_run_id: str,
    challenger_run_id: str,
    test_data_path: Path,
    epsilon: float = 0.01,
) -> dict:
    """Compare challenger model against champion on locked test set.

    Args:
        champion_run_id: MLflow run ID of current production model.
        challenger_run_id: MLflow run ID of newly trained model.
        test_data_path: Path to locked test set (never used in training).
        epsilon: Tolerance for equivalence (challenger >= champion - epsilon).

    Returns:
        Promotion decision with metrics.
    """
    # Load models from MLflow registry
    champion_model = mlflow.pytorch.load_model(
        f"runs:/{champion_run_id}/model"
    )
    challenger_model = mlflow.pytorch.load_model(
        f"runs:/{challenger_run_id}/model"
    )

    # Evaluate both on locked test set
    champion_dice = evaluate_on_test_set(champion_model, test_data_path)
    challenger_dice = evaluate_on_test_set(challenger_model, test_data_path)

    # Promotion decision
    promote = challenger_dice >= champion_dice - epsilon

    # Log comparison
    with mlflow.start_run(run_name="promotion_evaluation"):
        mlflow.log_param("champion_run_id", champion_run_id)
        mlflow.log_param("challenger_run_id", challenger_run_id)
        mlflow.log_metric("champion_dice", champion_dice)
        mlflow.log_metric("challenger_dice", challenger_dice)
        mlflow.log_metric("dice_delta", challenger_dice - champion_dice)
        mlflow.log_param("promoted", promote)
        mlflow.set_tag("role", "promotion_evaluation")

    return {
        "champion_dice": champion_dice,
        "challenger_dice": challenger_dice,
        "promote": promote,
        "dice_delta": challenger_dice - champion_dice,
    }
```

### 8.4 Test-Time Adaptation

Test-time adaptation (TTA) adjusts model parameters at inference time based on the test input alone, without any labels. This is relevant when retraining is not feasible (e.g., edge deployment without connectivity).

**ZeroSiam** (Chen et al., 2025) performs entropy optimization without feature collapse, a common failure mode of TTA methods. ZeroSiam maintains a momentum-updated teacher model to prevent the adapted model from degenerating to trivial solutions.

**Risk monitoring for TTA** (Schirmer et al., 2025) addresses a critical concern: TTA can hurt performance on some inputs while helping on others. Schirmer et al. propose monitoring the adaptation trajectory and reverting to the unadapted model when adaptation diverges.

**Recommendation for MinIVess**: TTA is a nice-to-have. Full retraining is cheap and more reliable. TTA would only be relevant for a hypothetical edge deployment scenario where the model runs on a microscope workstation without network access.

### 8.5 Active Learning

When new volumes become available but annotation budget is limited, active learning identifies which volumes would be most informative to label.

**Conformal Labeling** (Huang et al., 2025) uses conformal prediction to identify volumes where the model's prediction sets are large (high uncertainty), indicating that labeling these volumes would maximally reduce uncertainty.

**Feedback-enhanced online testing** (Lu et al., 2025) provides FDR (False Discovery Rate) control for prediction acceptance: automatically flag predictions that are likely incorrect and route them for manual review.

For MinIVess, active learning is a nice-to-have because the dataset is so small that any new annotated volume is valuable. The labeling bottleneck is expert availability, not volume selection.

---

## 9. Post-Deployment Monitoring Frameworks

### 9.1 Three Complementary Principles (Keyes et al., 2024, Stanford)

Keyes et al. (2024) from Stanford propose a comprehensive monitoring framework organized around three complementary principles:

**Principle 1: System Integrity**
- Uptime and latency monitoring (Prometheus + Grafana)
- Runtime error tracking
- IT ecosystem changes (dependency updates, infrastructure drift)
- Data pipeline integrity (checksums, schema validation)

**Principle 2: Performance**
- Accuracy metrics (Dice, clDice, Hausdorff distance)
- Calibration metrics (ECE, reliability diagrams)
- Fairness metrics (performance parity across subgroups)
- Uncertainty estimates (MC Dropout, ensemble disagreement)

**Principle 3: Impact**
- Downstream workflow effects (time-to-diagnosis, annotation efficiency)
- User satisfaction and trust calibration
- Clinical outcome correlation (if applicable)

For MinIVess as an academic project, Principles 1 and 2 are fully implementable. Principle 3 (Impact) is relevant only if the pipeline is deployed in a collaborating lab.

### 9.2 Mayo Clinic Experience (Cook et al., 2026)

Cook et al. (2026) report on monitoring 17 internally developed radiology AI algorithms at Mayo Clinic. Key lessons:

1. **Silent failures are common.** Models degrade gradually without obvious error signals. Performance monitoring must be proactive, not reactive.
2. **Scanner updates break models.** Firmware updates and protocol changes on imaging equipment cause covariate shift that is invisible to metadata checks but degrades model performance.
3. **Label drift is real.** Radiologist annotation practices evolve over time, causing concept drift even when the imaging data is stable.
4. **Monitoring infrastructure requires dedicated engineering.** The overhead of maintaining monitoring systems is often underestimated relative to the initial model development effort.

These findings validate the multi-layer monitoring approach proposed for MinIVess: metadata checks alone are insufficient; embedding-level and uncertainty-level monitoring are necessary.

### 9.3 Regulatory Requirements

While MinIVess is an academic project, understanding regulatory requirements for AI/ML monitoring ensures the pipeline architecture is compatible with eventual clinical translation:

**FDA Total Product Lifecycle (TPLC)** requires continuous monitoring of AI-enabled medical devices throughout their market life, including post-market surveillance of performance degradation and distribution shift.

**EU AI Act Articles 9-15** mandate post-market monitoring for high-risk AI systems (which includes medical devices). Article 9 specifically requires risk management systems that include continuous monitoring. Article 15 requires accuracy, robustness, and cybersecurity measures.

**IEC 62304** requires post-market surveillance as part of the software lifecycle for medical devices. The OpenLineage/Marquez lineage tracking in the MinIVess stack provides the traceability infrastructure needed for IEC 62304 compliance.

**PCCP (Predetermined Change Control Plan)** is a new FDA pathway that allows manufacturers to specify in advance how a model will be updated (including automated retraining) without requiring a new 510(k) for each update. The challenger-champion pattern with locked test validation aligns with PCCP requirements.

### 9.4 The Responsibility Vacuum (Owens et al., 2025)

Owens et al. (2025) identify a critical governance challenge in their qualitative study of healthcare organizations: AI monitoring responsibilities are poorly defined, underfunded, and institutionally orphaned. Key observations:

- **No clear owner.** Is model monitoring the responsibility of the development team, the IT operations team, the clinical department, or a dedicated AI governance team?
- **Institutional incentives favor innovation over maintenance.** Academic institutions reward novel model development, not operational monitoring.
- **Feedback loops are broken.** Performance degradation signals rarely reach the development team in time to take corrective action (Pietrobon et al., 2025).

For MinIVess, this argues for building monitoring into the CI/CD pipeline (GitHub Actions + CML) rather than treating it as a separate operational concern. Automated drift checks on every data update, with alerts surfaced as GitHub issues, ensure monitoring is not dependent on human vigilance.

---

## 10. Security Considerations

Patel et al. (2026) apply the MITRE ATLAS framework to MLOps pipelines, identifying attack surfaces specific to automated retraining systems:

**Data poisoning in retraining.** If the retraining pipeline automatically incorporates new data, an adversary who can inject corrupted volumes can degrade the model. The data quality gates (Section 7) mitigate this by rejecting OOD inputs before they enter the training pipeline.

**Model integrity.** The challenger-champion pattern includes a validation step that prevents deploying a degraded model. However, subtle poisoning that preserves aggregate Dice while degrading performance on specific anatomical structures is harder to detect. The SHIFT framework (Singh et al., 2025) for subgroup-level monitoring helps.

**Supply chain attacks.** Dependency updates to MONAI, TorchIO, or Alibi-Detect could introduce vulnerabilities. Pinned dependency versions (managed by `uv` lockfile) and pre-commit checks provide basic supply chain security.

**Inference-time attacks.** Adversarial examples -- carefully crafted inputs that cause misclassification while appearing normal -- are a concern for deployed medical imaging models. The multi-signal OOD scoring (Section 5.5) provides partial defense: adversarial examples often produce unusual uncertainty patterns even when the primary prediction is confident. However, adaptive adversaries can evade uncertainty-based detection.

**Model extraction.** If the model is served via a REST API (BentoML), an adversary can query it repeatedly to approximate the model's behavior. Rate limiting, authentication, and query logging (via Prometheus metrics) provide basic countermeasures.

**Governance of automated decisions.** Romanchuk et al. (2026) discuss governance frameworks for systems where AI agents make automated decisions (such as automated retraining triggers). For MinIVess, the retraining trigger should always log the rationale (which drift signals fired, p-values, confidence scores) to MLflow for post-hoc audit.

For an academic project, the primary security concern is data integrity rather than adversarial attacks. The quality gates and challenger-champion pattern provide sufficient protection. However, implementing basic security practices (authentication, rate limiting, audit logging) from the start establishes good habits for eventual clinical translation.

---

## 11. Integrated Architecture for MinIVess MLOps

### 11.1 Complete Monitoring Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION LAYER                        │
│                                                                  │
│  New Volume ─→ Metadata Validation ─→ Statistical Checks         │
│                    (Pandera)              (whylogs)              │
│                        │                     │                   │
│                        └─────────┬───────────┘                   │
│                                  ▼                               │
│                    Embedding Extraction (DynUNet)                │
│                                  │                               │
│                    ┌─────────────┼─────────────┐                │
│                    ▼             ▼             ▼                 │
│              OOD Detection   Drift Test   Quality Gate          │
│              (Mahalanobis)   (MMD, Alibi)  (composite)          │
│                    │             │             │                 │
│                    └─────────────┼─────────────┘                │
│                                  ▼                               │
│                           PASS / REJECT                         │
└──────────────────────┬───────────────────────────────────────────┘
                       │ PASS
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    INFERENCE LAYER                               │
│                                                                  │
│  Volume ─→ DynUNet Inference ─→ Segmentation Mask               │
│                │                                                 │
│                ├─→ MC Dropout (T=10) ─→ Uncertainty Map          │
│                ├─→ Softmax Entropy ─→ Confidence Map             │
│                └─→ Embedding ─→ Anomaly Score                    │
│                                                                  │
│  All signals logged to MLflow per inference                      │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    MONITORING LAYER                               │
│                                                                  │
│  Evidently Reports ─→ Grafana Dashboards                        │
│  whylogs Profiles ─→ Continuous Distribution Tracking            │
│  MLflow Metrics ─→ DuckDB Analytics                              │
│  Prometheus ─→ Infrastructure Metrics                            │
│                                                                  │
│  Alert Conditions:                                               │
│    - Embedding MMD p < 0.01 (Alibi-Detect)                      │
│    - MC Dropout variance > 2 std baseline                        │
│    - Mahalanobis > 95th percentile                               │
│    - Multi-signal consensus (2 of 3)                             │
└──────────────────────┬───────────────────────────────────────────┘
                       │ DRIFT ALERT
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    RETRAINING LAYER                               │
│                                                                  │
│  Trigger ─→ Full Retrain (Hydra-zen + MONAI + all data)         │
│                │                                                 │
│                ▼                                                 │
│  Challenger Model ─→ Evaluate on Locked Test Set                │
│                          │                                       │
│                  ┌───────┴────────┐                              │
│                  │  >= Champion?  │                              │
│                  └───┬────────┬───┘                              │
│                  YES │        │ NO                               │
│                      ▼        ▼                                  │
│               Promote    Keep Champion                           │
│            (MLflow stage)  (log failure)                         │
│                      │                                           │
│                      ▼                                           │
│              Deploy via BentoML + ONNX                           │
│                                                                  │
│  Full lineage tracked by OpenLineage/Marquez                    │
└──────────────────────────────────────────────────────────────────┘
```

### 11.2 Level 4 Implementation Checklist

- [ ] **Automated training pipeline**: Hydra-zen configs drive MONAI training loop with MLflow logging
- [ ] **Automated model validation**: Deepchecks Vision suite + locked test set evaluation
- [ ] **Automated deployment**: BentoML service with ONNX Runtime backend, Gradio demo UI
- [ ] **Data drift monitoring**: Evidently embedding drift reports + whylogs continuous profiles
- [ ] **Statistical drift testing**: Alibi-Detect MMD on rolling embedding windows
- [ ] **Retraining triggers**: Drift p < 0.01 OR new data OR scheduled (3-6 months)
- [ ] **Challenger-champion promotion**: Compare Dice on locked test set, promote if >= champion
- [ ] **Model soup**: Greedy soup as conservative alternative to full replacement
- [ ] **Full lineage**: OpenLineage/Marquez tracking from data ingestion to deployment
- [ ] **Audit trail**: IEC 62304 lifecycle events logged to compliance module
- [ ] **Uncertainty estimation**: MC Dropout T=10 for every inference
- [ ] **Deep ensemble**: 3-model ensemble for gold-standard uncertainty
- [ ] **Label-free monitoring**: Three-signal quality estimator (entropy + MC variance + Mahalanobis)
- [ ] **Data quality gates**: 4-tier gate architecture (metadata, statistical, embedding, batch)
- [ ] **Monitoring dashboards**: Grafana dashboards for drift, uncertainty, and system health
- [ ] **CI/CD integration**: GitHub Actions + CML for automated drift checks on data updates

### 11.3 Synthetic OOD Demo Scenario

The following end-to-end scenario demonstrates Level 4 capabilities:

**Step 1: Establish baseline.** Train DynUNet (quarter-width, `filters=[8, 16, 32, 64]`) on ~15 MinIVess volumes with topology-aware loss (clDice). Hold out 5 volumes as locked test set. Log training metrics to MLflow. Compute reference embedding distribution (mean, covariance via Ledoit-Wolf).

**Step 2: Near-OOD input.** Apply TorchIO RandomBiasField + RandomNoise at severity 0.3 to 5 held-out volumes. Feed through the pipeline. Expected behavior:
- Data quality gates: PASS (metadata unchanged, statistics within tolerance)
- Uncertainty: Slight increase in MC Dropout variance (~1.2x baseline)
- Drift: Embedding MMD test may not trigger (subtle shift)
- Anomaly score: Mahalanobis slightly elevated but below threshold
- Outcome: No retraining triggered. This is correct behavior -- near-OOD should be tolerated.

**Step 3: Moderate-OOD input.** Apply TorchIO RandomBlur + RandomGamma at severity 0.6. Expected behavior:
- Data quality gates: Statistical checks flag unusual intensity distribution
- Uncertainty: MC Dropout variance 2-3x baseline
- Drift: Embedding MMD test triggers (p < 0.01)
- Anomaly score: Mahalanobis above 95th percentile
- Outcome: Drift alert fired. Multi-signal consensus confirmed. Retraining recommended.

**Step 4: Far-OOD input.** Generate procedural Bezier tube volumes (Section 6.3). Expected behavior:
- Data quality gates: Gate 3 (embedding OOD) rejects volumes
- Uncertainty: MC Dropout variance 5-10x baseline
- Drift: Embedding MMD test strongly rejects (p << 0.001)
- Anomaly score: Mahalanobis far beyond any training volume
- Outcome: Volumes rejected at input gate. Do not enter training pipeline.

**Step 5: Demonstrate retraining cycle.** After accumulating 5 moderate-OOD volumes that pass quality gates, the drift trigger fires. The pipeline:
1. Initiates full retraining on all data (original 15 + 5 new)
2. Evaluates challenger on locked test set
3. If challenger Dice >= champion Dice: promote
4. If challenger Dice < champion Dice: retain champion, alert human

**Step 6: Verify monitoring dashboard.** Grafana dashboards show:
- Time series of mean MC Dropout variance per volume
- Embedding drift p-values over time
- Mahalanobis distance per volume
- Quality gate pass/reject rates
- Champion vs. challenger Dice comparison

This demo proves that the MinIVess pipeline achieves Microsoft MLOps Level 4: automated drift detection, automated retraining trigger, automated model validation and promotion, with full lineage and audit trail.

### 11.4 Topology-Aware Monitoring for Vascular Segmentation

The MinIVess pipeline uses topology-aware losses (clDice, Skeleton Recall Loss (Kirchhoff et al., 2024)) during training to preserve vessel connectivity. The monitoring pipeline should track topology-sensitive metrics at inference time as well.

**Topology metrics for monitoring:**

- **clDice** (centerline Dice): Measures the fraction of the predicted centerline that overlaps with the ground-truth centerline, and vice versa. More sensitive to connectivity breaks than standard Dice.
- **Betti number errors**: Count of topological defects (missing connections, spurious loops) in the predicted segmentation relative to ground truth.
- **Branch point count ratio**: The ratio of predicted to expected branch points in the vascular tree. A significant deviation suggests the model is either hallucinating or missing vessels.

**Topology-uncertainty connection:** Dhor et al. (2026) introduce TUNE++, which combines topology-aware loss with uncertainty quantification specifically for tubular structure segmentation. The key insight is that topological errors (broken vessels, merged vessels) produce characteristic uncertainty patterns: high MC Dropout variance along the vessel centerline at points where connectivity is ambiguous.

For the monitoring pipeline, tracking the spatial correlation between high-uncertainty regions and topological defects provides a powerful diagnostic:

```python
def topology_uncertainty_correlation(
    uncertainty_map: np.ndarray,
    predicted_mask: np.ndarray,
    skeleton: np.ndarray,
) -> dict:
    """Measure correlation between uncertainty and topological features.

    High uncertainty at branch points is expected (decision boundaries).
    High uncertainty along straight vessel segments is a red flag.
    """
    from skimage.morphology import skeletonize_3d

    # Extract skeleton of predicted segmentation
    pred_skeleton = skeletonize_3d(predicted_mask > 0.5)

    # Identify branch points (voxels with >2 skeleton neighbors)
    # and endpoints (voxels with exactly 1 skeleton neighbor)
    branch_points = identify_branch_points(pred_skeleton)
    endpoints = identify_endpoints(pred_skeleton)

    # Uncertainty at branch points (expected to be high)
    branch_uncertainty = uncertainty_map[branch_points].mean()

    # Uncertainty along vessel body (should be low for in-distribution)
    body_mask = pred_skeleton & ~branch_points & ~endpoints
    body_uncertainty = uncertainty_map[body_mask].mean()

    return {
        "branch_uncertainty_mean": float(branch_uncertainty),
        "body_uncertainty_mean": float(body_uncertainty),
        "branch_to_body_ratio": float(branch_uncertainty / (body_uncertainty + 1e-8)),
        "endpoint_count": int(endpoints.sum()),
        "branch_point_count": int(branch_points.sum()),
    }
```

When body uncertainty is elevated relative to the training baseline, it indicates that the model is systematically less confident about vessel presence -- a strong signal of covariate shift affecting vessel visibility (e.g., reduced contrast, different staining intensity).

---

## 12. Quality Model and Evaluation

### 12.1 ISO 25010/25059 for ML Components

Lewis et al. (2026) from Carnegie Mellon Software Engineering Institute (SEI) adapt the ISO 25010/25059 quality properties for ML systems. The relevant quality characteristics for MinIVess monitoring include:

- **Functional suitability**: Does the monitoring pipeline detect drift when it occurs? (Measured by synthetic OOD detection rates)
- **Performance efficiency**: What is the latency overhead of monitoring? (MC Dropout T=10 adds ~10x inference time)
- **Reliability**: What is the false alarm rate? (Calibrated via leave-one-out cross-validation)
- **Maintainability**: How easy is it to update thresholds and add new signals? (Modular gate architecture)
- **Security**: Is the retraining pipeline robust to data poisoning? (Quality gates + challenger validation)

### 12.2 MLTE Tool

Lewis et al. (2026) also introduce MLTE (Machine Learning Test and Evaluation), a tool for systematically measuring ML component quality against ISO 25010/25059 properties. MLTE provides:

- A structured evaluation framework with quantifiable quality targets
- Integration with CI/CD for continuous quality assessment
- Traceability from quality requirements to test results

### 12.3 Sequential Verification

E-valuator (Sadhuka et al., 2025) introduces sequential hypothesis testing for reliable verification of AI agents, using anytime-valid e-values to control error rates. While originally evaluated on LLM agents and game-playing systems, the sequential testing methodology is applicable to monitoring deployed ML models: new inference results can be incorporated as they arrive without requiring a fixed evaluation window. This is complementary to the WATCH sequential testing framework (Prinster et al., 2025) discussed in Section 3.4.

### 12.4 Evaluation Science

Weidinger et al. (2025) argue for treating ML evaluation as a rigorous scientific discipline, with pre-registered evaluation protocols, appropriate statistical tests, and transparent reporting. For MinIVess, this means:

- Pre-specifying the locked test set before any model development
- Reporting confidence intervals alongside point estimates (Andre et al., 2026)
- Using paired permutation tests for model comparison (challenger vs. champion)
- Logging all evaluation decisions in MLflow for auditability

---

## 13. Recommendations and Research Roadmap

### 13.1 Must-Have (Level 4 Demo)

These components are required for a credible Level 4 demonstration:

1. **MC Dropout uncertainty (T=10).** MONAI DynUNet natively supports `dropout` with `dropout_dim=3`. Implementation requires enabling `model.train()` at inference and aggregating T forward passes. Cost: ~10x inference time per volume.

2. **Embedding drift detection (Alibi-Detect MMD).** Extract penultimate-layer embeddings via forward hooks. Build MMD drift detector from training reference embeddings. Conservative p-value threshold (p < 0.01) with multiple-alert confirmation.

3. **Metadata quality gates (Pandera + Great Expectations).** Pandera schema for voxel spacing, shape, bit depth. Great Expectations for batch-level statistical checks. Both already in the project stack.

4. **Full retraining with validation guard.** Retrain from scratch on all available data (trivial with ~20 volumes). Challenger-champion comparison on locked test set. Model soup as conservative alternative.

5. **Automated CI/CD pipeline.** GitHub Actions workflow triggered by data update or drift alert. CML for ML-specific PR comments showing metric comparisons.

6. **TorchIO OOD generation (3 severity levels).** Near-OOD (bias field + noise), moderate OOD (blur + gamma), far-OOD (ghosting + spike + anisotropy). Continuous severity parameterization for controlled experiments.

### 13.2 Should-Have (Strengthen Paper)

These components strengthen the academic contribution:

7. **3-model Deep Ensemble.** Training cost: ~15 minutes total on one GPU. Provides ensemble disagreement as an additional OOD signal and improves calibration. Interacts with existing model soup implementation.

8. **Conformal Prediction (ConSeMa).** Distribution-free coverage guarantees for segmentation. MAPIE (already in stack) provides the computational infrastructure. Limited power with 5 calibration volumes, but demonstrates the methodology.

9. **Post-hoc calibration (global temperature scaling).** Single-parameter calibration via `netcal` (already in stack). Evaluate with ECE and reliability diagrams. Avoid complex spatial calibration with <50 volumes.

10. **Procedural Bezier tube generator.** Far-OOD synthetic data with controllable morphological properties. ~100-200 lines of code. Demonstrates that the monitoring pipeline detects fundamentally different data.

11. **Multi-signal quality proxy.** Combine softmax entropy, MC Dropout variance, and Mahalanobis distance into a single quality estimate per volume. Log to MLflow for correlation analysis with ground-truth Dice when available.

### 13.3 Nice-to-Have

These components add depth but are not essential:

12. **Mahalanobis++ OOD scoring.** Class-conditional Mahalanobis distance with relative scoring. Improves OOD detection reliability over vanilla Mahalanobis.

13. **whylogs continuous profiling.** Lightweight statistical profiling merged across sessions. Already in dependencies but not wired to the validation pipeline.

14. **In-Context Reverse Classification Accuracy.** Train a lightweight classifier to distinguish training from deployment distributions. Provides a label-free performance proxy.

15. **vesselFM domain randomization adaptation.** Adapt vesselFM's open-source domain randomization pipeline for moderate-OOD generation. More realistic than Bezier tubes but requires more engineering effort.

### 13.4 Explicitly Avoid

The following approaches are explicitly **not recommended** for MinIVess:

- **Evidential Deep Learning (EDL).** Shen et al. (2024) at NeurIPS 2024 showed that EDL's epistemic uncertainty is a mirage. The theoretical foundations have critical gaps, and empirical results do not support reliable OOD detection.

- **NannyML for images.** NannyML is an excellent tool for tabular data drift detection and performance estimation. It cannot process images. Do not attempt to force it into the image monitoring pipeline.

- **Continual learning with EWC.** Elastic Weight Consolidation's Fisher information matrix is unreliable with <50 samples. Full retraining is cheaper, simpler, and more reliable.

- **Complex calibration (Local Temperature Scaling, isotonic regression per region).** These methods have more parameters than can be reliably estimated with ~20 volumes. Overfitting is almost certain. Use global temperature scaling until the dataset grows substantially.

- **Generative models for synthetic data (as primary approach).** Twenty volumes are insufficient for training useful generative models. Use TorchIO corruption and procedural generation instead. Reserve generative models for a future phase when the dataset exceeds ~100 volumes.

---

## References

Ackerman, S., et al. (2021). "Evidently: An interactive ML monitoring framework." arXiv preprint.

Andre, J. B., et al. (2026). "Confidence interval methodology for medical imaging: A large-scale empirical study." *Medical Image Analysis*.

Angelopoulos, A. N., & Bates, S. (2023). "Conformal prediction: A gentle introduction." *Foundations and Trends in Machine Learning*.

Beytur, H. B., et al. (2025). "Optimal resource allocation for ML model training and deployment under concept drift." arXiv:2512.12816.

Cardoso, M. J., et al. (2022). "MONAI: An open-source framework for deep learning in healthcare." arXiv:2211.02701.

Chen, J., et al. (2025). "ZeroSiam: An efficient Siamese for test-time entropy optimization without collapse." arXiv:2509.23183.

Cook, D. A., et al. (2026). "Monitoring 17 internally developed radiology AI algorithms: Lessons from Mayo Clinic." *Radiology: Artificial Intelligence*.

Dhor, A., et al. (2026). "TUNE++: Topology-aware uncertainty for tubular structure segmentation." *Medical Image Analysis*.

Dolin, R., et al. (2025). "Statistical validity should be the standard for ML monitoring." *Proceedings of FAccT*.

Fluhmann, D., et al. (2025). "Label-free confusion matrix estimation for deployed ML models." *Proceedings of ICML*.

Gal, Y., Hron, J., & Kendall, A. (2017). "Concrete dropout." *Proceedings of NeurIPS*.

Guo, C., et al. (2017). "On calibration of modern neural networks." *Proceedings of ICML*.

Hodge, V. J., et al. (2025). "OOD detection for safety assurance: A taxonomy and survey." *ACM Computing Surveys*.

Huang, J., et al. (2024). "Uncertainty quantification in medical image analysis: A survey." *Medical Image Analysis*.

Huang, T., et al. (2025). "Conformal labeling: Identifying volumes for annotation via prediction set size." *Proceedings of AISTATS*.

Isensee, F., et al. (2021). "nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation." *Nature Methods*.

Isensee, F., et al. (2024). "nnU-Net revisited: A call for rigorous validation in 3D medical image segmentation." *Proceedings of MICCAI*.

Keyes, C., Callahan, A., Pandya, S., et al. (2024). "Three complementary principles for monitoring deployed AI in healthcare." Stanford HAI Working Paper.

Kirchhoff, S., et al. (2024). "Skeleton Recall Loss for connectivity-preserving vascular segmentation." *Proceedings of MICCAI*.

Kiyasseh, D., et al. (2024). "A framework for evaluating clinical artificial intelligence systems without ground-truth annotations." *Nature Communications*.

Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). "Simple and scalable predictive uncertainty estimation using deep ensembles." *Proceedings of NeurIPS*.

Lewis, G. A., et al. (2026). "A quality model for ML components based on ISO 25010/25059." *Proceedings of ICSE-SEIP*.

Lotfi, A., et al. (2024). "Normalizing flows for post-hoc OOD detection in medical imaging." *Medical Image Analysis*.

Lu, W., et al. (2025). "Feedback-enhanced online testing with FDR control for prediction acceptance." *Proceedings of ICML*.

Microsoft. (2024). "MLOps maturity model." Microsoft Azure Architecture Center. https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model

Mossina, L. & Friedrich, F. (2025). "ConSeMa: Conformalized Segmentation Margins for medical image segmentation." *Proceedings of MICCAI*.

Muller, M., et al. (2024). "D3Bench: A benchmark for drift detection in deep learning." *Proceedings of NeurIPS Datasets and Benchmarks*.

Owens, K., Griffen, Z., & Damaraju, L. (2025). "Managing a 'responsibility vacuum' in AI monitoring and governance in healthcare: A qualitative study." *BMC Health Services Research*.

Patel, S., et al. (2026). "Applying MITRE ATLAS to MLOps pipeline security." *IEEE Security & Privacy*.

Perez-Garcia, F., Sparks, R., & Ourselin, S. (2021). "TorchIO: A Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning." *Computer Methods and Programs in Biomedicine*.

Pietrobon, R., et al. (2025). "Feedback loops in deployed ML systems: A systematic review." *ACM Computing Surveys*.

Prinster, D., et al. (2025). "WATCH: Weighted conformal test martingales for anytime-valid sequential drift detection." *Proceedings of ICML*.

Romanchuk, A., et al. (2026). "Governance frameworks for scaled agent systems." *AI & Society*.

Roschewitz, M., et al. (2024). "Automatic identification of distribution shift types in medical imaging." *Proceedings of MICCAI*.

Sadhuka, S., et al. (2025). "E-valuator: Sequential verification for ML models." *Proceedings of AISTATS*.

Schirmer, M. D., et al. (2025). "Risk monitoring for test-time adaptation: Detecting when adaptation hurts." *Proceedings of ICLR*.

Shaer, S., et al. (2026). "Conditional conformal test martingales for correlated sequential samples." *Journal of Machine Learning Research*.

Shah-Mohammadi, F., et al. (2025). "Conformal prediction for medical image segmentation with distribution-free coverage guarantees." *Proceedings of MICCAI*.

Shen, M., et al. (2024). "Epistemic uncertainty in evidential deep learning: A mirage." *Proceedings of NeurIPS*.

Sims, J. A., et al. (2023). "SEG: Segmentation evaluation in the absence of ground truth." *Medical Image Analysis*.

Singh, A., et al. (2025). "SHIFT: Subgroup-level drift detection for healthcare ML." *Proceedings of CHIL*.

Stone, P., et al. (2025). "MLOps lifecycle: A comprehensive analysis of practices and tools." *IEEE Software*.

Tahir, E., et al. (2023). "MinIVess: A dataset of cerebral vasculature from 2-photon microscopy." *Data in Brief*.

Weidinger, L., et al. (2025). "Evaluation science for AI systems." *Proceedings of FAccT*.

Wittmann, F., et al. (2025). "vesselFM: A foundation model for universal vessel segmentation." *Proceedings of CVPR*. arXiv:2501.02096.

Wortsman, M., et al. (2022). "Model soups: Averaging weights of multiple fine-tuned models improves accuracy without increasing inference time." *Proceedings of ICML*.

Xiong, Z., et al. (2026). "ADAPT: Adversarial concept drift detection with provable guarantees." *Proceedings of ICML*.

Zenk, M., et al. (2023). "Optimal dropout rates for 3D medical image segmentation." *Proceedings of MIDL*.

---

*This report synthesizes research from 4 parallel agents (KB Gap Analysis, Seed Paper Survey, Synthetic Data, Drift + Uncertainty) into a unified architecture for MinIVess MLOps v2 monitoring, drift detection, and continuous retraining. All recommendations are calibrated for the tiny-dataset regime (~20 volumes) and the existing project stack documented in CLAUDE.md.*
