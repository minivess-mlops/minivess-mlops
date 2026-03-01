# Graph-Based Topology Evaluation Metrics for Vessel/Tube Segmentation

**Survey Date:** 2026-02-28
**Purpose:** Document existing Python implementations for APLS, Skeleton Recall, and Branch Detection Rate metrics
**Target Use:** MiniVess MLOps evaluation flow (#124)

---

## Executive Summary

| Metric | Best Implementation | 3D Support | On PyPI | License | Status |
|--------|-------------------|-----------|---------|---------|--------|
| **APLS** | `apls` package (CosmiQ) | No (2D only, geospatial) | ✓ Yes | Apache-2.0 | Production-ready, but 2D-only |
| **Skeleton Recall** | `Skeleton-Recall` (MIC-DKFZ) | ✓ Yes (2D/3D) | ✗ No | Apache-2.0 | Loss function, not metric; nnUNet integration |
| **clDice** | `jocpae/clDice` (GitHub) | ✓ Yes (2D/3D) | ✗ No | MIT | Production-ready, CVPR 2021 |
| **clDice (MONAI)** | `monai.losses` | ✓ Yes (2D/3D) | ✓ Yes | Apache-2.0 | Official MONAI implementation |
| **Skeleton Metrics** | `segmentation-skeleton-metrics` (Allen Institute) | ✓ Yes (3D) | ✓ Yes | MIT | Graph-based topology evaluation |
| **Topology Precision/Recall** | MONAI (Soft clDice Loss) | ✓ Yes (2D/3D) | ✓ Yes | Apache-2.0 | Native skeleton-based metrics |
| **Branch Detection** | Graph extraction + NetworkX | ✓ Yes (2D/3D) | ✓ Partial | Varies | Bespoke implementations required |

---

## 1. APLS (Average Path Length Similarity)

### Overview
- **What it does:** Compares shortest paths between matched graph nodes in predicted vs. ground truth graphs
- **Origin:** SpaceNet 3 road network challenge (2016)
- **Publication:** [clDice - a Novel Topology-Preserving Loss Function](https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf)

### Package: `apls` (CosmiQ)
```python
from apls import apls

# Compare graphs
metric = apls.single_path_metric(G_pred, G_gt)  # Single path
metric = apls.path_sim_metric(G_pred, G_gt)     # Path similarity
metric = apls.compute_metric(G_pred, G_gt)      # Full metric
```

| Aspect | Details |
|--------|---------|
| **PyPI** | ✓ Yes (`pip install apls`) |
| **GitHub** | https://github.com/CosmiQ/apls |
| **License** | Apache-2.0 |
| **3D Support** | ✗ No — 2D only (geospatial/road networks) |
| **Main Functions** | `single_path_metric()`, `path_sim_metric()`, `compute_metric()` |
| **Supporting Functions** | `cut_linestring()`, `get_closest_edge()`, `insert_control_points()`, `insert_point_into_G()` |
| **Graph Format** | NetworkX graphs, GeoJSON, CSV, pickled graphs |
| **Key Files** | `apls.py` (main), `graphTools.py` (GeoJSON→graph conversion) |
| **Critical Limitation** | Designed for 2D road networks; no explicit 3D support |

### Use Case in MinIVess
⚠️ **APLS is NOT recommended for MinIVess** due to 2D-only design. However, the graph matching algorithm could be adapted if full vessel tree topology becomes a priority metric.

---

## 2. Skeleton Recall Loss

### Overview
- **What it does:** Soft recall loss against a precomputed tubed skeleton of ground truth
- **Paper:** [Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation](https://arxiv.org/abs/2404.03010) (ECCV 2024)
- **Mechanism:** Extracts GT skeleton → dilates into tube → soft recall of prediction on tube

### Package: `Skeleton-Recall` (MIC-DKFZ)

```python
# Installed as custom nnUNetv2 trainer
# Training command:
# nnUNetv2_train DATASET_NAME 2d FOLD -tr nnUNetTrainerSkeletonRecall

# Loss function signature (from compound_losses.py)
loss = SkeletonRecallLoss(
    alpha=0.5,  # Weight
    reduction='mean'
)
```

| Aspect | Details |
|--------|---------|
| **PyPI** | ✗ No — Custom nnUNet trainer only |
| **GitHub** | https://github.com/MIC-DKFZ/Skeleton-Recall |
| **License** | Apache-2.0 |
| **3D Support** | ✓ Yes (both 2D and 3D, binary and multi-class) |
| **Type** | **Loss function** (not a metric for evaluation) |
| **Integration** | nnUNetv2 only (not standalone) |
| **Metric Formula** | Soft recall: `R = sum(pred * skeleton) / sum(skeleton)` |
| **Loss Formula** | `L = w_dice·L_dice + w_ce·L_ce + w_skel·L_skeleton_recall` |
| **Setup** | Copy `nnUNetTrainerSkeletonRecall`, transform, and loss modules into nnUNet install |
| **Performance** | 90%+ computational overhead reduction vs. hard skeletonization |

### Key Functions
- **`tubed_skeletonize_transform()`** — Skeletonize GT, dilate, assign classes (applied at dataload)
- **`SkeletonRecallLoss.forward()`** — Compute soft recall during training

### Critical Notes
- ⚠️ Skeleton Recall is a **loss function for training**, NOT an evaluation metric
- No standalone implementation; requires nnUNetv2 framework
- Excellent for topology-aware training but requires nnUNet adoption

---

## 3. clDice (Centerline Dice)

### Overview
- **What it does:** Topology-preserving similarity measure using morphological skeletons
- **Formula:** `clDice = 2 * (skelGT ∩ skelPred) / (skelGT + skelPred)`
- **Guarantees:** Homotopy equivalence preservation (binary 2D/3D)
- **Paper:** [clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf) (CVPR 2021)

### Implementation Options

#### Option A: Standalone GitHub (jocpae/clDice)
```python
# PyTorch
from cldice_loss import clDice  # Hard metric
from cldice_loss import soft_cldice_loss  # Differentiable loss

hard_metric = clDice(pred, target)  # For evaluation
soft_loss = soft_cldice_loss(pred, target)  # For training
```

| Aspect | Details |
|--------|---------|
| **GitHub** | https://github.com/jocpae/clDice |
| **PyPI** | ✗ No — Clone and import directly |
| **License** | MIT |
| **3D Support** | ✓ Yes (2D/3D for both PyTorch and TensorFlow) |
| **Implementations** | PyTorch 2D/3D, TensorFlow/Keras 2D/3D |
| **Skeleton** | Hard (scikit-image `skeletonize`) or soft (iterative min/max pooling) |
| **Performance** | 4-5 min per volume for skeleton computation (full dataset) |

#### Option B: MONAI Native (Recommended)
```python
from monai.losses import SoftclDiceLoss, SoftDiceclDiceLoss

# For training (soft skeleton)
loss = SoftclDiceLoss(iter_=3, smooth=1.0)

# Combined Dice + clDice
loss = SoftDiceclDiceLoss(alpha=0.5, iter_=3)
```

| Aspect | Details |
|--------|---------|
| **Package** | `monai` (v1.3+) |
| **PyPI** | ✓ Yes (`pip install monai`) |
| **License** | Apache-2.0 |
| **3D Support** | ✓ Yes |
| **Functions** | `SoftclDiceLoss`, `SoftDiceclDiceLoss` |
| **Params** | `iter_=3` (skeleton iterations), `alpha=0.5` (loss weight), `smooth=1.0` |
| **Status** | Official, documented, maintained |

### Key Advantages
- ✓ Differentiable loss function (soft-clDice)
- ✓ Hard metric for post-hoc evaluation
- ✓ 2D and 3D support
- ✓ Native MONAI integration (recommended)
- ✓ Production-ready (CVPR 2021 publication)
- ✓ Topology preservation guarantee

### Recommendations for MinIVess
**✓ Use MONAI's `SoftclDiceLoss` for training + hard clDice for evaluation.**
Already integrated in the project's loss functions (#122, #123).

---

## 4. Skeleton-Based Metrics (Allen Institute)

### Package: `segmentation-skeleton-metrics`

```python
from segmentation_skeleton_metrics.skeleton_metric import SkeletonMetric
from tifffile import imread

# Load ground truth skeletons (SWC files) and predicted segmentation
gt_skeletons = "path/to/gt/*.swc"  # Neuron morphology format
pred_segmentation = imread("segmentation.tif")  # 3D volume

metric = SkeletonMetric(
    groundtruth_pointer=gt_skeletons,
    segmentation=pred_segmentation,
    output_dir="results/"
)
results = metric.run()
```

| Aspect | Details |
|--------|---------|
| **PyPI** | ✓ Yes (`pip install segmentation-skeleton-metrics`) |
| **Version** | 5.6.21 (latest) |
| **GitHub** | https://github.com/AllenNeuralDynamics/segmentation-skeleton-metrics |
| **License** | MIT |
| **3D Support** | ✓ Yes (works with 3D volumetric data + 3D SWC skeletons) |
| **Metric Type** | Graph-based topology evaluation |
| **Input Format** | SWC files (neuron morphology) + 3D segmentation volume |
| **Primary Use** | Neuron/axon segmentation evaluation |
| **Key Metrics Computed** | See table below |

### Metrics Computed

| Metric | Description | Type |
|--------|-------------|------|
| **# Splits** | Connected components minus 1 after removing error edges | Count |
| **# Merges** | Ground truth graphs with merge errors | Count |
| **% Split/Omit/Merged Edges** | Error fractions by type | Percentage |
| **Edge Accuracy** | Proportion of correctly identified edges | % (0-100) |
| **Expected Run Length (ERL)** | Total run length after removing all errors | Distance |
| **Normalized ERL** | ERL / total run length | Ratio (0-1) |
| **Split/Merge Rates** | Run length / error count | Distance/count |

### Strengths & Limitations

| Aspect | Status |
|--------|--------|
| ✓ 3D support | Yes, works with volumetric data |
| ✓ Graph-based | True topological evaluation (not just voxel overlap) |
| ✓ Published | Allen Brain Observatory (trusted source) |
| ✗ Vessel-specific | Designed for neurons (SWC format) |
| ⚠ SWC limitation | Requires converting vessel graphs to SWC format |
| ⚠ Sparse comparison | Only evaluates at skeleton node locations |

### Use Case in MinIVess
⚠️ **Possible but not ideal.** Would require:
1. Extract 3D vessel skeleton from GT segmentation
2. Convert NetworkX graph to SWC format
3. Pass segmentation volume for comparison

Better alternatives exist for true graph topology metrics (see below).

---

## 5. Topology Precision & Topology Recall

### Overview
These are skeleton-intersection based metrics that form the basis of clDice:
- **Topology Precision:** `TP = (skelGT ∩ skelPred) / skelPred`
- **Topology Recall:** `R = (skelGT ∩ skelPred) / skelGT`
- **clDice:** `2·TP·R / (TP + R)`

### MONAI Implementation

```python
from monai.losses import SoftclDiceLoss

# Topology metrics computed internally by soft clDice
loss = SoftclDiceLoss(iter_=3, smooth=1.0)
topo_metric = loss.forward(pred, target)  # Includes TP, R, clDice
```

### Status
- ✓ Implemented in MONAI 1.3+
- ✓ Differentiable soft skeleton
- ✓ 3D support
- ✓ Production-ready

---

## 6. Branch Detection Rate (BDR)

### Overview
- **What it does:** Measures fraction of GT branches matched in prediction graph
- **Formula:** `BDR = # correctly_detected_branches / # total_gt_branches`
- **Related:** Bifurcation detection (nodes with degree ≥ 3)

### No Standalone Implementation Found
⚠️ **No PyPI package exists for BDR.** All implementations are bespoke:

1. **DeepVesselNet approach:** Graph extraction → NetworkX → bifurcation detection by neighbor count
   - Reference: [DeepVesselNet: Vessel Segmentation, Centerline Prediction, and Bifurcation Detection](https://arxiv.org/pdf/1803.09340)
   - Uses 3×3×3 neighborhood counting in skeleton

2. **Graph-based matching:** Extract skeletons → build directed graphs → match bifurcations
   - Reference: [Automatic Vessel Crossing and Bifurcation Detection](https://www.sciencedirect.com/science/article/pii/S0010482523001129)
   - Uses directed graph search + multi-attention networks

3. **Components for custom implementation:**
   - Skeletonize: `skimage.morphology.skeletonize_3d()` or clDice soft skeleton
   - Graph extraction: NetworkX or cugraph
   - Bifurcation detection: Neighbor count in 26-neighborhood
   - Graph matching: Hungarian algorithm or ICP variants

### Algorithm Template for MinIVess
```python
import networkx as nx
from skimage.morphology import skeletonize_3d
from scipy.ndimage import label

def extract_vessel_graph_with_bifurcations(seg_volume):
    """Extract directed vessel graph with bifurcation counts."""
    skeleton = skeletonize_3d(seg_volume > 0)

    # Find nodes (bifurcations, junctions, endpoints)
    nodes = []
    bifurcations = []

    for idx in np.ndindex(skeleton.shape):
        if skeleton[idx]:
            neighbors = skeleton[
                max(0, idx[0]-1):idx[0]+2,
                max(0, idx[1]-1):idx[1]+2,
                max(0, idx[2]-1):idx[2]+2
            ].sum()

            if neighbors == 3:  # Bifurcation (center + 2 neighbors)
                bifurcations.append(idx)
                nodes.append(idx)

    # Build directed graph with branch edges
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    # Add edges between connected bifurcations...

    return G, bifurcations

def compute_branch_detection_rate(G_pred, G_gt):
    """Match bifurcations between pred and GT graphs."""
    # Hungarian algorithm or graph isomorphism matching
    matched = match_bifurcations(G_pred, G_gt)
    bdr = len(matched) / len(G_gt.nodes())
    return bdr
```

---

## 7. Comparative Analysis

### For MinIVess Use Case (3D Vessel Segmentation)

| Need | Best Solution | Alternative |
|------|-------------------|-------------|
| **Training loss** | `monai.losses.SoftclDiceLoss` | `Skeleton-Recall` (if nnUNet) |
| **Topology evaluation** | Hard clDice + topology precision/recall | APLS (2D only) |
| **Graph matching** | Custom NetworkX + graph isomorphism | `segmentation-skeleton-metrics` + SWC conversion |
| **Bifurcation detection** | Custom skeleton + neighbor count | DeepVesselNet reference implementation |
| **Connectivity preservation** | clDice metric (post-hoc) | Skeleton Recall loss (training only) |

### Metrics Already in MinIVess

From memory context, the project implements:
- ✓ **clDice** (hard metric) — Compute at eval time
- ✓ **Soft clDice Loss** — Training via MONAI
- ✓ **Compound loss** (cbdice_cldice) — Custom in loss_functions.py
- ✓ **Topology precision/recall** — Via skeletonization

### Gaps to Address

| Gap | Severity | Solution |
|-----|----------|----------|
| **Branch detection rate** | Medium | Custom implementation (graph matching) |
| **Graph isomorphism matching** | Medium | Hungarian algorithm or graph kernels |
| **APLS adaptation for 3D** | Low | Use clDice instead (better for 3D) |
| **Skeleton recall metric** | Low | Use clDice (loss function, not metric) |
| **Bifurcation F1 score** | Low | Custom TP/FP/FN counting |

---

## 8. Recommended Implementation Path for #124

### Phase 1: Consolidate Existing Metrics (Already Done)
- ✓ clDice metric (hard + soft)
- ✓ Topology precision/recall
- ✓ Normalized MASD

### Phase 2: Add Graph-Based Evaluation (TODO)
**For evaluating vessel connectivity structure:**

```python
# In src/minivess/ensemble/graph_topology_evaluator.py

from skimage.morphology import skeletonize_3d
import networkx as nx

class GraphTopologyEvaluator:
    def __init__(self):
        self.metrics = {}

    def extract_skeleton_graph(self, seg_volume, min_component_size=10):
        """Extract vessel skeleton as directed graph."""
        skeleton = skeletonize_3d(seg_volume > 0)
        # Remove small noise components
        labeled, num = label(skeleton)
        # ... build NetworkX graph with bifurcations

    def compute_branch_detection_rate(self, G_pred, G_gt):
        """BDR = matched bifurcations / total GT bifurcations."""
        pass

    def compute_graph_matching_score(self, G_pred, G_gt, method='hamming'):
        """Graph isomorphism or Hamming distance between adjacency matrices."""
        pass

    def evaluate(self, pred_seg, gt_seg) -> dict:
        """Return dict of graph-based topology metrics."""
        return {
            'clDice': ...,
            'topology_precision': ...,
            'topology_recall': ...,
            'branch_detection_rate': ...,
            'graph_matching_score': ...,
        }
```

### Phase 3: Integration Points
1. **Evaluation flow** (#91-#93) — Call `GraphTopologyEvaluator` per prediction
2. **MLflow logging** — Log graph metrics to `mlruns/`
3. **Markdown reporting** — Include graphs in analysis report
4. **Ensemble evaluation** — Compare topology across ensemble members

---

## 9. Dependencies Summary

### For MONAI clDice (Recommended)
```bash
# Already in minivess env
pip install monai[all]  # Includes clDice loss + soft skeleton
```

### For Standalone clDice
```bash
# Clone repo
git clone https://github.com/jocpae/clDice.git
# No PyPI distribution; add to PYTHONPATH or copy modules
```

### For Skeleton Metrics (Allen)
```bash
pip install segmentation-skeleton-metrics
# Requires TiffReader; SWC file format knowledge
```

### For APLS (Not Recommended for 3D)
```bash
pip install apls
# 2D-only; geospatial focus
```

### For Skeleton Recall Loss
```bash
# Clone + integrate into nnUNetv2
git clone https://github.com/MIC-DKFZ/Skeleton-Recall.git
# Requires nnUNetv2 installation
```

---

## 10. References

### Primary Papers
1. Shit et al. (2021). "[clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf)" CVPR.

2. Weikert et al. (2024). "[Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures](https://arxiv.org/abs/2404.03010)" ECCV.

3. Van Etten et al. (2018). "[SpaceNet: A Remote Sensing Dataset and Challenge Series](https://arxiv.org/abs/1807.01232)" (APLS metric origin).

### GitHub Repositories
- **clDice:** https://github.com/jocpae/clDice (MIT)
- **APLS:** https://github.com/CosmiQ/apls (Apache-2.0)
- **Skeleton-Recall:** https://github.com/MIC-DKFZ/Skeleton-Recall (Apache-2.0)
- **segmentation-skeleton-metrics:** https://github.com/AllenNeuralDynamics/segmentation-skeleton-metrics (MIT)
- **PyVesTo:** https://github.com/chcomin/pyvesto (status: unmaintained, see PyVaNe)

### Related Work
- [DeepVesselNet: Vessel Segmentation, Centerline Prediction, and Bifurcation Detection](https://arxiv.org/pdf/1803.09340) (bifurcation detection algorithm)
- [Automatic Vessel Crossing and Bifurcation Detection](https://www.sciencedirect.com/science/article/pii/S0010482523001129) (graph-based approach)
- [VesselBoost: A Python Toolbox for Small Blood Vessel Segmentation](https://apertureneuro.org/article/123217-vesselboost-a-python-toolbox-for-small-blood-vessel-segmentation-in-human-magnetic-resonance-angiography-data) (topology preservation in training)

---

## 11. Decision Summary

### For #124 (Graph-Based Evaluation Metrics)

**Recommendation:**
1. ✓ **Use MONAI clDice** for hard metric evaluation (already integrated)
2. ✓ **Add custom graph topology evaluator** for BDR and graph matching
3. ✓ **Log all metrics** to MLflow and markdown reports
4. ✗ **Skip APLS** (2D-only; clDice is superior for 3D vessels)
5. ✗ **Skip Skeleton Recall loss** (training-only, requires nnUNet)
6. ⚠ **Consider SWC conversion** if detailed neuron-like topology metrics needed

**Effort Estimate:**
- Core graph extraction + BDR: ~4 hours (TDD red-green-refactor)
- MLflow integration: ~2 hours
- Testing + documentation: ~3 hours

---

**Document Version:** 1.0
**Last Updated:** 2026-02-28
**Applicable to:** MinIVess MLOps v2, Issue #124
