# Graph-Constrained Models Execution Plan

> **Branch**: `feat/graph-constrained-models`
> **Source**: `docs/planning/graph-connectivity-analysis.md` (80+ paper survey)
> **Created**: 2026-02-28

## Overview

This plan organises 22 GitHub Issues that bring graph-level topology awareness into
MinIVess. The codebase already has topology-aware losses (cbdice_cldice, clDice) and
conformal UQ, but lacks graph-level evaluation metrics, centreline/graph extraction,
GNN architecture modules, advanced topology losses, and topological UQ.

### Issue Number Mapping

| Plan # | GitHub # | Title |
|--------|----------|-------|
| 1 | #112 | ccDice evaluation metric |
| 2 | #113 | Betti number error + persistence distance |
| 3 | #114 | Skeleton recall loss |
| 4 | #115 | CAPE loss |
| 5 | #116 | Centreline extraction utility |
| 6 | #117 | Junction F1 score |
| 7 | #118 | Topology-aware experiment config |
| 8 | #119 | TFFM (GAT feature fusion) |
| 9 | #120 | Centreline prediction head |
| 10 | #121 | Vessel graph extraction pipeline |
| 11 | #122 | Betti matching loss |
| 12 | #123 | Compound graph topology loss |
| 13 | #124 | Graph-based evaluation metrics |
| 14 | #125 | Topology-aware experiment sweep |
| 15 | #126 | Topograph post-processing |
| 16 | #127 | Learned reconnection |
| 17 | #128 | FlowAxis parameterization |
| 18 | #129 | VesselFormer |
| 19 | #130 | Topological UQ |
| 20 | #131 | Murray's law compliance |
| 21 | #132 | GraphSeg deformable priors |
| 22 | #133 | TopoSegNet critical points |

## Issue Inventory

### P0 — Critical Foundation (7 issues)

These provide the metric infrastructure and loss functions needed by everything else.

#### Issue 1: Add connected-component Dice (ccDice) evaluation metric
- **Size**: S | **Dependencies**: None
- **Labels**: `graph-topology`, `P0`, `metrics`
- **Files**: `src/minivess/pipeline/topology_metrics.py` (new),
  `configs/metric_registry.yaml`
- **Description**: Implement ccDice (Shi et al., 2024) — decomposes a segmentation into
  connected components, matches them via IoU, and computes per-component Dice. Critical
  because standard Dice hides fragmentation: a mask with 50 disconnected fragments can
  score 0.90 Dice but 0.3 ccDice. The metric must handle 3D volumes efficiently using
  `scipy.ndimage.label` or `cc3d`. Register in `build_metrics()` factory.
- **Acceptance criteria**:
  - `compute_ccdice(pred, target)` returns float in [0, 1]
  - Handles empty predictions (return 0.0) and perfect match (return 1.0)
  - 3D test with known component count
  - Unit tests with synthetic masks (connected, fragmented, empty)

#### Issue 2: Add Betti number error and persistence diagram distance metrics
- **Size**: M | **Dependencies**: Issue 1
- **Labels**: `graph-topology`, `P0`, `metrics`
- **Files**: `src/minivess/pipeline/topology_metrics.py`
- **Description**: Implement Betti number error (|beta_0_pred - beta_0_gt|,
  |beta_1_pred - beta_1_gt|) for connected components (beta_0) and loops (beta_1).
  Optionally compute Wasserstein distance between persistence diagrams if `gudhi` is
  available (graceful ImportError → NaN fallback). Beta_0 is critical for vascular
  segmentation — it directly measures fragmentation. Beta_1 measures spurious loops.
- **Acceptance criteria**:
  - `compute_betti_error(pred, target)` returns dict with `beta0_error`, `beta1_error`
  - `compute_persistence_distance(pred, target)` returns float or NaN if gudhi missing
  - Tests with known topologies (sphere=β₁=0, torus=β₁=1)

#### Issue 3: Add skeleton recall loss (Kirchhoff et al., ECCV 2024)
- **Size**: M | **Dependencies**: None
- **Labels**: `graph-topology`, `P0`, `loss`
- **Files**: `src/minivess/pipeline/vendored_losses/skeleton_recall.py` (new),
  `src/minivess/pipeline/loss_functions.py`
- **Description**: Implement skeleton recall loss from "Skeleton Recall Loss for
  Connectedness-Aware Segmentation" (Kirchhoff et al., ECCV 2024). The loss penalises
  missed skeleton voxels — voxels that lie on the morphological skeleton of the ground
  truth but are not covered by the prediction. This is differentiable via soft
  skeletonization. Integrate into `build_loss_function()` factory as `skeleton_recall`.
- **Acceptance criteria**:
  - Loss is differentiable (backward pass succeeds)
  - Gradient is non-zero for missed skeleton voxels
  - Perfect prediction → loss ≈ 0
  - Registered in loss factory, selectable via config
  - VRAM test: forward+backward on (1,1,16,16,8) patch < 500 MB

#### Issue 4: Add CAPE loss (Connectivity-Aware Path Enforcement, MICCAI 2025)
- **Size**: L | **Dependencies**: None
- **Labels**: `graph-topology`, `P0`, `loss`
- **Files**: `src/minivess/pipeline/vendored_losses/cape.py` (new),
  `src/minivess/pipeline/loss_functions.py`
- **Description**: Implement CAPE loss (Luo et al., MICCAI 2025) — samples random
  endpoint pairs on the ground truth skeleton, checks if they remain connected in the
  prediction via differentiable path queries. Path connectivity is approximated via
  soft geodesic distance (heat kernel) or learnable message passing. Penalises
  predictions that disconnect paths that should be connected.
- **Acceptance criteria**:
  - Loss is differentiable
  - Connected prediction → low loss; severed prediction → high loss
  - Configurable number of sampled endpoint pairs (default 64)
  - Registered in loss factory as `cape`
  - VRAM test: forward+backward on (1,1,16,16,8) patch < 1 GB

#### Issue 5: Add centreline extraction utility (skeletonization + graph)
- **Size**: M | **Dependencies**: None
- **Labels**: `graph-topology`, `P0`, `infrastructure`
- **Files**: `src/minivess/pipeline/centreline_extraction.py` (new)
- **Description**: Implement centreline extraction from binary 3D masks. Pipeline:
  (1) morphological thinning via `skimage.morphology.skeletonize` (Lee94 algorithm),
  (2) skeleton-to-graph conversion using `skan.Skeleton` or custom 26-connectivity
  tracing, (3) return `CentrelineGraph` dataclass with nodes (xyz + radius), edges
  (length, mean radius), and junction/endpoint labels. This is the foundation for
  Junction F1, APLS, vessel graph pipeline, and graph-based UQ.
- **Acceptance criteria**:
  - `extract_centreline(mask_3d)` returns `CentrelineGraph`
  - Graph has correct node count for simple tube (2 endpoints, 0 junctions)
  - Graph has correct node count for Y-bifurcation (3 endpoints, 1 junction)
  - Edge lengths are within 10% of ground truth for synthetic test volumes
  - Works on real MiniVess patch (16x16x8)

#### Issue 6: Add Junction F1 score evaluation metric
- **Size**: M | **Dependencies**: Issue 5
- **Labels**: `graph-topology`, `P0`, `metrics`
- **Files**: `src/minivess/pipeline/topology_metrics.py`
- **Description**: Implement Junction F1 score — extract junction points from predicted
  and ground truth centreline graphs (via Issue 5), match junctions within a distance
  tolerance (default 3 voxels), compute precision/recall/F1. This is critical for
  evaluating bifurcation preservation, the key clinical requirement for vascular
  segmentation. Junction detection uses degree-3+ nodes from `CentrelineGraph`.
- **Acceptance criteria**:
  - `compute_junction_f1(pred_mask, gt_mask, tolerance=3)` returns dict with P/R/F1
  - Perfect bifurcation match → F1 = 1.0
  - Missed bifurcation → recall < 1.0
  - Spurious bifurcation → precision < 1.0
  - Tests with synthetic bifurcation volumes

#### Issue 7: Topology-aware experiment config and metric integration
- **Size**: M | **Dependencies**: Issues 1, 2, 6
- **Labels**: `graph-topology`, `P0`, `integration`
- **Files**: `configs/experiments/dynunet_topology.yaml`,
  `src/minivess/pipeline/trainer.py`, `configs/metric_registry.yaml`
- **Description**: Wire all P0 topology metrics (ccDice, Betti error, Junction F1)
  into the training/evaluation loop. Create experiment config that runs the existing
  losses (dice_ce, dice_ce_cldice, cbdice_cldice) but tracks topology metrics at
  evaluation time. Add topology metrics to the MetricsReloaded-frequency epoch
  callback (every 5 epochs, since skeleton extraction is expensive). Update
  `metric_registry.yaml` with new metric definitions.
- **Acceptance criteria**:
  - `dynunet_topology.yaml` is valid and loadable
  - Topology metrics appear in MLflow run logs
  - Metrics computed every 5 epochs alongside existing MetricsReloaded metrics
  - No VRAM regression (topology metrics are computed on CPU after prediction)

---

### P1 — Core Research Contributions (7 issues)

These build on P0 infrastructure to deliver novel research components.

#### Issue 8: Implement TFFM (GAT feature fusion) as pluggable adapter wrapper
- **Size**: L | **Dependencies**: None
- **Labels**: `graph-topology`, `P1`, `architecture`
- **Files**: `src/minivess/adapters/tffm_wrapper.py` (new),
  `src/minivess/adapters/graph_modules.py` (new)
- **Description**: Implement Topology Feature Fusion Module (TFFM) from DS2Net
  (Wang et al., 2025) as a wrapper around existing `ModelAdapter`. TFFM uses a Graph
  Attention Network (GAT) to fuse multi-scale features from the encoder, modelling
  inter-scale topology relationships. Implementation pattern:
  `TFFMWrapper(DynUNetAdapter(config))` — wraps any adapter's encoder output.
  Requires `torch_geometric` (optional dependency, graceful ImportError).
- **Acceptance criteria**:
  - `TFFMWrapper` wraps any `ModelAdapter` and produces same output shape
  - GAT attention weights are logged as artifacts
  - Forward pass works without `torch_geometric` (fallback to linear fusion)
  - VRAM test: forward+backward on (1,1,16,16,8) < 2 GB

#### Issue 9: Implement centreline prediction head (multi-task seg + centreline)
- **Size**: L | **Dependencies**: Issue 5
- **Labels**: `graph-topology`, `P1`, `architecture`
- **Files**: `src/minivess/adapters/centreline_head.py` (new),
  `src/minivess/pipeline/loss_functions.py`
- **Description**: Add auxiliary centreline prediction head for multi-task learning.
  The model predicts both the segmentation mask and a centreline distance map
  (soft skeleton). Inspired by Kirchhoff et al. (2024) and Shit et al. (2021).
  The centreline head is a 1x1x1 conv on the decoder output. Loss is weighted
  sum: `total = α*seg_loss + β*centreline_mse`. Centreline GT generated via Issue 5.
  Output stored in `SegmentationOutput.metadata['centreline_map']`.
- **Acceptance criteria**:
  - Multi-task model produces both mask and centreline map
  - Centreline loss is differentiable and non-zero
  - α, β configurable via YAML
  - Does not break single-task training when disabled

#### Issue 10: Implement vessel graph extraction pipeline (mask → graph → export)
- **Size**: XL | **Dependencies**: Issue 5
- **Labels**: `graph-topology`, `P1`, `pipeline`
- **Files**: `src/minivess/pipeline/vessel_graph.py` (new),
  `src/minivess/pipeline/graph_export.py` (new)
- **Description**: Full vessel graph extraction pipeline: binary mask →
  skeletonization → graph extraction → pruning → radius estimation → export.
  Export formats: NetworkX graph (pickle), GraphML (for Gephi/Cytoscape),
  SWC (neuroscience standard), VTP (VTK polydata for ParaView). Pruning removes
  spurious branches shorter than `min_branch_length` (default 3 voxels). Radius
  estimation via distance transform at skeleton voxels.
- **Acceptance criteria**:
  - `extract_vessel_graph(mask)` returns annotated NetworkX graph
  - `export_graph(graph, format='graphml')` produces valid file
  - Pruning removes short branches correctly
  - Radius values match distance transform
  - Works on real MiniVess volume

#### Issue 11: Implement Betti matching loss (Stucki et al., ICML 2023)
- **Size**: L | **Dependencies**: Issue 2
- **Labels**: `graph-topology`, `P1`, `loss`
- **Files**: `src/minivess/pipeline/vendored_losses/betti_matching.py` (new),
  `src/minivess/pipeline/loss_functions.py`
- **Description**: Implement Betti matching loss from "Topologically Faithful
  Image Segmentation via Topology Matching" (Stucki et al., ICML 2023).
  Matches persistence diagram points between prediction and ground truth,
  penalises unmatched topological features (extra/missing components or loops).
  Requires computing filtration on soft predictions — use cubical persistence
  via `gudhi` or the differentiable approach from the paper. Registered as
  `betti_matching` in loss factory.
- **Acceptance criteria**:
  - Loss is differentiable via soft filtration
  - Penalises topology differences (extra components, missing loops)
  - Graceful fallback if `gudhi` not installed (warning + skip)
  - VRAM test on (1,1,16,16,8) patch

#### Issue 12: Implement compound graph topology loss
- **Size**: M | **Dependencies**: Issues 3, 4
- **Labels**: `graph-topology`, `P1`, `loss`
- **Files**: `src/minivess/pipeline/loss_functions.py`
- **Description**: Implement compound loss combining existing cbdice_cldice with
  new topology losses: `graph_topology = w1*cbdice_cldice + w2*skeleton_recall +
  w3*cape`. Default weights: w1=0.5, w2=0.3, w3=0.2. Configurable via YAML.
  This is the primary research contribution — a loss that optimises for both
  voxel overlap AND graph connectivity. Registered as `graph_topology` in factory.
- **Acceptance criteria**:
  - Loss is differentiable
  - Weights configurable via `loss_weights` config
  - Default weights produce balanced gradients (no single term dominates)
  - VRAM test on (1,1,16,16,8) patch
  - Backward pass non-zero for all three components

#### Issue 13: Graph-based evaluation metrics (APLS, skeleton recall metric, BDR)
- **Size**: L | **Dependencies**: Issues 10, 5
- **Labels**: `graph-topology`, `P1`, `metrics`
- **Files**: `src/minivess/pipeline/topology_metrics.py`
- **Description**: Implement graph-level evaluation metrics that operate on extracted
  vessel graphs: (1) APLS (Average Path Length Similarity) — compare shortest paths
  between matched node pairs in predicted vs GT graph, (2) skeleton recall metric —
  fraction of GT skeleton voxels covered by prediction, (3) branch detection rate —
  fraction of GT branches (edges) that have a matching predicted branch within
  tolerance. These require Issue 10's graph extraction pipeline.
- **Acceptance criteria**:
  - `compute_apls(pred_graph, gt_graph)` returns float in [0, 1]
  - `compute_skeleton_recall(pred_mask, gt_mask)` returns float in [0, 1]
  - `compute_branch_detection_rate(pred_graph, gt_graph)` returns float in [0, 1]
  - Tests with synthetic graphs (perfect match, partial match, no match)

#### Issue 14: Topology-aware experiment sweep (loss ablation with graph metrics)
- **Size**: L | **Dependencies**: All P0 + P1
- **Labels**: `graph-topology`, `P1`, `experiment`
- **Files**: `configs/experiments/dynunet_graph_topology.yaml`
- **Description**: Design and run experiment sweep comparing topology-aware losses
  using graph-level metrics. Sweep: {dice_ce, cbdice_cldice, graph_topology,
  skeleton_recall, betti_matching} × 3 folds × 100 epochs. Evaluation with full
  graph metric suite (ccDice, Betti error, Junction F1, APLS, BDR). This
  experiment produces the main results table for the paper.
- **Acceptance criteria**:
  - Valid experiment YAML with all loss/metric combinations
  - Estimated VRAM < 8 GB for all configurations
  - Results logged to MLflow `dynunet_graph_topology` experiment
  - Comparison report generated via existing comparison.py

---

### P2 — Exploratory / Nice-to-Have (8 issues)

These are research extensions that depend on P0/P1 infrastructure.

#### Issue 15: Implement Topograph post-processing (Lux et al., ICLR 2025)
- **Size**: XL | **Dependencies**: Issue 2
- **Labels**: `graph-topology`, `P2`, `post-processing`
- **Files**: `src/minivess/pipeline/topology_postprocessing.py` (new)
- **Description**: Implement Topograph — a post-processing method that modifies
  segmentation predictions to match target Betti numbers using persistence-guided
  voxel flipping. Given a prediction with wrong topology, identifies critical
  voxels (births/deaths in persistence diagram) and flips them to correct the
  topology. Requires persistence computation via `gudhi`.

#### Issue 16: Implement learned reconnection post-processing (Greco et al., 2024)
- **Size**: XL | **Dependencies**: Issue 5
- **Labels**: `graph-topology`, `P2`, `post-processing`
- **Files**: `src/minivess/pipeline/learned_reconnection.py` (new)
- **Description**: Implement learned gap reconnection: detect disconnected endpoints
  in the skeleton graph, train a small MLP/GNN to predict whether two nearby
  endpoints should be connected, reconnect via geodesic path in the distance
  transform. Inspired by Greco et al. (2024) agent-based reconnection.

#### Issue 17: Implement FlowAxis continuous vessel parameterization (Wu et al., 2026)
- **Size**: XL | **Dependencies**: None
- **Labels**: `graph-topology`, `P2`, `architecture`
- **Files**: `src/minivess/adapters/flowaxis.py` (new)
- **Description**: Implement FlowAxis — predicts a continuous 1D parameterization
  along vessel centrelines (analogous to s-coordinate in fluid dynamics). The model
  outputs a scalar field that increases monotonically along each vessel branch.
  This enables automatic centreline extraction without morphological thinning.

#### Issue 18: Implement VesselFormer joint segmentation + graph extraction
- **Size**: XL | **Dependencies**: Issues 5, 10
- **Labels**: `graph-topology`, `P2`, `architecture`
- **Files**: `src/minivess/adapters/vesselformer.py` (new)
- **Description**: Implement VesselFormer (Luo et al., 2024) — a transformer-based
  architecture that jointly predicts segmentation masks and vessel graphs. Uses
  deformable attention to detect vessel nodes and predicts edges between them.
  End-to-end differentiable graph extraction without post-processing.

#### Issue 19: Topological UQ — connectivity confidence via conformal prediction
- **Size**: XL | **Dependencies**: Issues 5, 10
- **Labels**: `graph-topology`, `P2`, `uq`
- **Files**: `src/minivess/ensemble/topological_uq.py` (new)
- **Description**: Extend conformal prediction framework to provide topology-aware
  uncertainty. Instead of per-voxel prediction sets, compute connectivity confidence:
  "probability that edge (u,v) exists in the true vascular graph". Uses ensemble
  disagreement on graph structure (edge presence across ensemble members) to
  calibrate connectivity confidence via conformal quantiles.

#### Issue 20: Murray's law compliance metric for vessel graph validation
- **Size**: S | **Dependencies**: Issue 10
- **Labels**: `graph-topology`, `P2`, `metrics`
- **Files**: `src/minivess/pipeline/topology_metrics.py`
- **Description**: Implement Murray's law compliance metric: at each bifurcation
  in the extracted vessel graph, measure r_parent^3 vs r_daughter1^3 + r_daughter2^3
  (Murray, 1926). Report mean absolute deviation from Murray's law across all
  bifurcations. This validates biological plausibility of the extracted graph.
  Kirstetter et al. (2024) showed deviations correlate with pathology.

#### Issue 21: GraphSeg deformable graph priors (Liu et al., NeurIPS 2025)
- **Size**: XL | **Dependencies**: Issue 8
- **Labels**: `graph-topology`, `P2`, `architecture`
- **Files**: `src/minivess/adapters/graphseg.py` (new)
- **Description**: Implement GraphSeg — uses deformable graph priors to guide
  segmentation. A graph template (atlas) deforms to fit the observed image,
  providing structural priors that prevent topological errors. Extends the
  TFFM wrapper pattern from Issue 8.

#### Issue 22: TopoSegNet scalable topology loss via critical points (IJCV 2025)
- **Size**: L | **Dependencies**: None
- **Labels**: `graph-topology`, `P2`, `loss`
- **Files**: `src/minivess/pipeline/vendored_losses/toposeg.py` (new)
- **Description**: Implement TopoSegNet topology loss (Gupta & Essa, IJCV 2025) —
  identifies topological critical points (where topology changes) via discrete
  Morse theory and penalises misclassification at those points. More scalable
  than full persistence computation (Betti matching) because it only requires
  local critical point detection, not global filtration.

---

## Dependency Graph

```
                    ┌─────────────────────────┐
                    │   Phase 1 (parallel)     │
                    │                          │
                    │  #1 ccDice   #3 SkelLoss │
                    │  #5 Centreline           │
                    └─────┬───────┬───────┬────┘
                          │       │       │
                    ┌─────▼───────▼───────▼────┐
                    │   Phase 2 (dependent)     │
                    │                          │
                    │  #2 Betti    #4 CAPE     │
                    │  #6 Junc F1 (← #5)      │
                    └─────┬───────┬───────┬────┘
                          │       │       │
                    ┌─────▼───────▼───────▼────┐
                    │   Phase 3 (integration)   │
                    │                          │
                    │  #7 Config  #8 TFFM      │
                    │  #9 Cent. Head (← #5)    │
                    └─────┬───────┬───────┬────┘
                          │       │       │
                    ┌─────▼───────▼───────▼────┐
                    │   Phase 4 (P1 core)       │
                    │                          │
                    │  #10 Graph Pipeline       │
                    │  #11 Betti Matching       │
                    │  #12 Compound Loss        │
                    └─────┬───────┬───────┬────┘
                          │       │       │
                    ┌─────▼───────▼───────▼────┐
                    │   Phase 5 (evaluation)    │
                    │                          │
                    │  #13 Graph Metrics        │
                    │  #14 Topology Experiment  │
                    └──────────────────────────┘
                              │
                    ┌─────────▼────────────────┐
                    │   Phase 6 (P2)            │
                    │  #15-#22 in any order     │
                    └──────────────────────────┘
```

## Key Architecture Decisions

1. **All new losses** → `build_loss_function()` factory in `loss_functions.py`
2. **All new metrics** → `topology_metrics.py` (new module) + `metric_registry.yaml`
3. **Architecture modules** → wrapper pattern: `TFFMWrapper(DynUNetAdapter(config))`
4. **SegmentationOutput.metadata** → carries auxiliary outputs (centreline map, graph)
5. **Optional deps** (`torch_geometric`, `gudhi`) → graceful ImportError with NaN fallback
6. **VRAM constraint** → all implementations verified on 8 GB with (1,1,16,16,8) patches

## Key References

- Shi et al. (2024) — ccDice metric
- Stucki et al. (2023) — Betti matching loss (ICML)
- Kirchhoff et al. (2024) — Skeleton recall loss (ECCV)
- Luo et al. (2025) — CAPE loss (MICCAI)
- Shit et al. (2021) — clDice (CVPR)
- Wang et al. (2025) — DS2Net / TFFM
- Lux et al. (2025) — Topograph post-processing (ICLR)
- Gupta & Essa (2025) — TopoSegNet (IJCV)
- Kirstetter et al. (2024) — Murray's law in microvascular trees
- Luo et al. (2024) — VesselFormer
- Wu et al. (2026) — FlowAxis
- Liu et al. (2025) — GraphSeg (NeurIPS)
