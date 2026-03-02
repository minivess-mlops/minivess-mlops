# Topology-Aware Segmentation: Discussion Notes

## Implemented Approaches

Three topology-aware approaches were implemented and evaluated for 3D vessel
segmentation, all operating natively in 3D (no 2D slice processing):

### Approach A: Disconnect-to-Connect (D2C) 3D Augmentation
- **Source**: Luo et al. (2023), adapted from 2D retinal to 3D volumetric
- **Mechanism**: Randomly disconnects vessel branches at junctions during training,
  forcing the model to learn reconnection patterns
- **Implementation**: `DisconnectToConnectd` MONAI-compatible transform
- **Prediction P1**: D2C augmentation improves topology preservation (clDice)
  by > 3 percentage points over baseline

### Approach B: Generic Multi-Task Framework
- **Mechanism**: Config-driven auxiliary heads (SDF regression, centerline distance)
  provide implicit topological supervision via loss gradient
- **Implementation**: `MultiTaskAdapter` wrapping any base model, `MultiTaskLoss`
  with per-head loss weighting
- **Prediction P3**: Multi-task learning improves clDice by > 1 percentage point
- **Design principle**: Completely generic — researchers add new auxiliary tasks
  via YAML config only, no code changes required

### Approach C: TFFMBlock3D (Graph Attention Feature Fusion)
- **Source**: Ahmed et al. (2026), WACV 2026, ported from 2D to 3D
- **Mechanism**: Graph attention network applied to bottleneck features via
  forward hook composition
- **Implementation**: `TFFMWrapper` wrapping any ModelAdapter
- **Prediction P4**: TFFM improves topology metrics (informational, no threshold gate)
- **Known limitation**: Isotropic grid (8x8x8) may be suboptimal for MiniVess
  anisotropic data (Z=5-110 vs XY=512)

## Deferred Approaches (Future Work)

The following approaches were surveyed in the literature review but deferred:

### Deferred — Require Extensive Architecture Changes
- **VesselFormer** (Nadeem et al., 2025): Transformer-based, requires custom
  attention architecture incompatible with MONAI DynUNet
- **ViG3D-UNet**: Vision GNN backbone, would require complete model replacement
- **GraphSeg**: Graph neural network for segmentation, fundamentally different
  architecture

### Deferred — 2D-Only Methods Not Trivially Lifted to 3D
- **DeformCL** (Wang et al., 2025): Deformable centerline loss, 2D-specific
  curvature computation. Lifting to 3D requires 3D Frenet-Serret frames
- **TopographNet**: Topological post-processing, 2D persistence diagram-based
- **FlowAxis**: Blood flow-based axis extraction, 2D simulation-dependent

### Deferred — Computational Constraints
- **Topograph post-processing**: O(n^3) persistent homology on full volumes
- **Learned reconnection**: Requires separate reconnection network training
- **Generative conformal prediction**: Training overhead exceeds budget

## 3D-Only Constraint Rationale

All implemented approaches operate natively in 3D. The 2D slice processing
approach was explicitly excluded because:

1. **Vessel disconnection**: 2D slices through tortuous 3D vessels create
   artificial disconnections that don't exist in the volume
2. **Topology inconsistency**: A vessel that appears disconnected in adjacent
   2D slices may be perfectly connected in 3D
3. **Information loss**: Axial, coronal, and sagittal slices capture different
   topology; no single orientation preserves 3D connectivity
4. **MiniVess specifics**: Z-range of 5-110 slices means some volumes have
   very few slices — 2D approaches would miss inter-slice connectivity

## SAM3 Parallel Branch

SAM3 (Segment Anything Model 3) is explored as a **separate parallel branch**
(`feat/sam3-integration`) and is **not part of this topology evaluation**.
The topology approaches (D2C, multi-task, TFFM) are model-agnostic and can
be applied to SAM3 in the future via the same wrapper composition pattern.

## Lifted-to-3D Feasibility Table

| Method | 2D Source | 3D Feasibility | Blocking Issue |
|--------|-----------|----------------|----------------|
| D2C | Luo 2023 | Done (this work) | None |
| TFFM | Ahmed 2026 | Done (this work) | Anisotropic grid |
| DeformCL | Wang 2025 | Medium | 3D Frenet-Serret frames |
| TopographNet | — | Low | 3D persistence diagrams O(n^3) |
| FlowAxis | — | Low | 3D CFD simulation required |
| VesselFormer | Nadeem 2025 | Medium | Custom architecture |

## Remaining Literature Gaps

1. **No 3D vessel-specific benchmark**: MiniVess is small (70 volumes);
   larger 3D vessel datasets needed for generalization claims
2. **Topology metric standardization**: clDice is established but Betti error
   thresholds are dataset-dependent
3. **Multi-scale topology**: Current approaches operate at single resolution;
   hierarchical topology (capillary → arteriole → artery) unexplored
4. **Computational cost**: TFFM + multi-task + D2C combined overhead needs
   careful profiling on production hardware
