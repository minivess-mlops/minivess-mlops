# Graph-Based Topology Metrics for Vessel Segmentation - Research Index

**Project:** MinIVess MLOps v2
**Issue:** #124 (Graph-based evaluation metrics)
**Research Date:** 2026-02-28
**Status:** Complete, ready for implementation

---

## Research Summary

Comprehensive survey of existing Python implementations for graph-based evaluation metrics in vessel/tube segmentation, focusing on three specific metrics:

1. **APLS** (Average Path Length Similarity) — SpaceNet road network metric
2. **Skeleton Recall** — Connectivity-preserving loss for thin tubes
3. **Branch Detection Rate (BDR)** — Fraction of matched branches

**Key Finding:** No single package implements all three. Recommendations prioritize MONAI's clDice (best for 3D vessels) + custom BDR evaluator.

---

## Documentation Files

### 1. graph-topology-metrics-survey.md (MAIN REFERENCE)
**Length:** 491 lines
**Purpose:** Comprehensive technical analysis of all metrics

**Contents:**
- Executive summary table (all 6 metrics)
- Detailed breakdown of APLS, Skeleton Recall, clDice, clDice (MONAI), Topology Precision/Recall, Skeleton Metrics
- PyPI availability & GitHub locations
- 3D support analysis
- Installation instructions
- Comparative analysis by use case
- Decision matrix for MinIVess
- Implementation path for #124 (3 phases)
- Full references to 15+ papers

**Best for:** Understanding the landscape of available metrics and making implementation decisions

**Quick Access:**
- Section 1: APLS (page 2-4) — **Finding: 2D-only, skip for MinIVess**
- Section 2: Skeleton Recall (page 4-6) — **Finding: Loss function, not metric; nnUNet-specific**
- Section 3: clDice (page 6-9) — **Finding: Best choice, 3D support, both PyTorch & TensorFlow**
- Section 4: MONAI clDice (page 9-10) — **Finding: Recommended, production-ready**
- Section 6: Branch Detection Rate (page 13-15) — **Finding: No package, custom implementation needed**

---

### 2. graph-topology-metrics-quick-reference.yaml (LOOKUP TABLE)
**Length:** 257 lines
**Purpose:** Machine-readable reference for quick lookups

**Contents:**
- YAML-structured metrics database with:
  * Package name, short name, purpose
  * PyPI availability, GitHub links, licenses
  * 3D support status
  * Key functions and parameters
  * Implementation recommendations
  * Notes and caveats
- Implementation status (already in MinIVess)
- Decision matrix by use case
- Comprehensive references section

**Best for:**
- Quick package lookups
- Automated tooling (CI/CD checks)
- Decision workflows
- Migration planning

**Top-level keys:** `metrics`, `implementation_status`, `decision_matrix`, `references`

**Example queries:**
```yaml
metrics.cldice_monai:
  on_pypi: true
  license: "Apache-2.0"
  support_3d: true
  use_for_minivess: "✓ YES - RECOMMENDED for training"
```

---

### 3. graph-topology-metrics-code-examples.md (IMPLEMENTATION)
**Length:** 632 lines
**Purpose:** Working code samples ready for implementation

**Contents:**

#### Section 1: Using MONAI clDice (Recommended)
- Installation via `uv add monai>=1.3.0`
- Training with `SoftclDiceLoss` and `SoftDiceclDiceLoss`
- Post-hoc evaluation with hard clDice metric
- Includes `HardClDiceEvaluator` class with topology metrics

#### Section 2: Using Standalone clDice (GitHub Version)
- Installation from GitHub repo
- PyTorch implementation with `SoftclDiceLoss` and `clDice`
- Training loop integration

#### Section 3: Custom Graph Topology Evaluator (Branch Detection Rate)
- **Complete, production-ready implementation** (300+ lines)
- `GraphTopologyEvaluator` class with:
  * `compute_hard_skeleton()` — Scikit-image skeletonization
  * `extract_bifurcations()` — 26-connectivity neighbor counting
  * `build_vessel_graph()` — NetworkX directed graph construction
  * `match_graphs()` — Hungarian algorithm for bifurcation matching
  * `compute_branch_detection_rate()` — BDR computation
  * `evaluate()` — Full metric evaluation
- `GraphTopologyMetrics` dataclass for results
- Effort estimate: ~4 hours to integrate (TDD)

#### Section 4: Integration into MLflow
- Logging graph metrics to MLflow runs
- Markdown report generation with interpretation guide

#### Section 5: Unit Tests (TDD)
- pytest fixtures and test cases
- Tests for skeleton computation, perfect matches, bifurcation detection
- Graph matching and BDR tests

#### Section 6: MONAI Integration in Training Loop
- `TopoAwareTrainer` class with:
  * Training epoch with topology-aware loss
  * Validation with hard clDice metric
  * Per-sample evaluation and aggregation

**Best for:** Copy-paste implementations, TDD red-green-refactor cycles

---

## Quick Decision Guide

### For Issue #124 Implementation:

**MUST DO:**
```
✓ Use MONAI clDice (SoftclDiceLoss) for topology-aware training
✓ Implement custom GraphTopologyEvaluator for BDR + bifurcation detection
✓ Log metrics to MLflow + generate markdown reports
```
See: `graph-topology-metrics-code-examples.md` Section 3 + 4

**SHOULD DO:**
```
✓ Add unit tests for custom evaluator
✓ Integrate into evaluation flow (#91-#93)
✓ Document topology thresholds (clDice > 0.85 = excellent)
```

**SKIP:**
```
✗ APLS — 2D-only, not suitable for 3D vessels
✗ Skeleton Recall loss — Skip unless adopting nnUNetv2
✗ segmentation-skeleton-metrics — Neuron-focused, requires SWC conversion
```

---

## Package Recommendation Matrix

| Use Case | Package | Status | Effort | Why |
|----------|---------|--------|--------|-----|
| **Training** | `monai.losses.SoftclDiceLoss` | ✓ Ready | Low | Official, maintained, 3D |
| **Evaluation** | clDice + custom BDR | ⚠ Partial | Medium | MONAI for clDice, custom for BDR |
| **Connectivity** | MONAI topology metrics | ✓ Ready | Low | Embedded in SoftclDiceLoss |
| **Topology** | Custom `GraphTopologyEvaluator` | ⚠ TODO | Medium | Bifurcations + graph matching |
| **Logging** | MLflow + ExperimentTracker | ✓ Ready | Low | Already integrated |
| **Reporting** | Custom markdown generator | ⚠ TODO | Low | Template provided |

---

## Implementation Timeline (Recommended)

### Phase 1: Foundation (Week 1)
- **Red:** Write failing tests for `GraphTopologyEvaluator`
- **Green:** Implement skeleton → bifurcation extraction
- **Verify:** Test on synthetic T-shaped vessel structures
- **Time:** 4-6 hours (TDD cycles)

### Phase 2: Graph Matching (Week 1)
- **Red:** Write tests for bifurcation matching
- **Green:** Implement Hungarian algorithm wrapper
- **Verify:** Test on simple NetworkX graphs
- **Time:** 2-3 hours

### Phase 3: MLflow Integration (Week 2)
- **Red:** Write tests for metric logging
- **Green:** Implement tracking calls
- **Verify:** Check MLflow UI
- **Time:** 1-2 hours

### Phase 4: Reporting & Documentation (Week 2)
- **Red:** Write tests for markdown generation
- **Green:** Implement report templates
- **Verify:** Generate sample reports
- **Time:** 1-2 hours

**Total Effort:** ~10-13 hours (2 weeks, following TDD)

---

## Key Findings Summary

### APLS (Average Path Length Similarity)
- **Package:** `apls` (pip install apls)
- **3D Support:** NO — 2D-only (geospatial roads)
- **Status:** Production-ready but unsuitable
- **Functions:** `apls.single_path_metric()`, `apls.compute_metric()`
- **License:** Apache-2.0
- **Recommendation:** SKIP for MinIVess

### Skeleton Recall Loss
- **Package:** Custom nnUNetv2 trainer (GitHub)
- **3D Support:** YES
- **Type:** Loss function (not evaluation metric)
- **Status:** ECCV 2024, production-ready
- **License:** Apache-2.0
- **Recommendation:** Use IF adopting nnUNetv2, otherwise skip

### Branch Detection Rate (BDR)
- **Package:** None (no standalone implementation)
- **3D Support:** YES
- **Algorithm:** Skeleton → bifurcations → graph matching
- **Status:** Requires custom implementation
- **Recommendation:** Implement custom evaluator (~4 hours)

### clDice (MONAI) — PRIMARY CHOICE
- **Package:** `monai` (pip install monai>=1.3)
- **3D Support:** YES
- **Type:** Both loss AND metric
- **Status:** Production-ready, official MONAI implementation
- **License:** Apache-2.0
- **Recommendation:** PRIMARY choice for topology evaluation

### Topology Precision & Recall
- **Package:** Part of MONAI SoftclDiceLoss
- **3D Support:** YES
- **Formula:** TP = (skelGT ∩ skelPred) / skelPred; R = (skelGT ∩ skelPred) / skelGT
- **Status:** Embedded in clDice computation
- **Recommendation:** Use as components of clDice metric

### Skeleton Metrics (Allen Institute)
- **Package:** `segmentation-skeleton-metrics` (pip install)
- **3D Support:** YES
- **Input:** SWC files (neuron morphology) + 3D volumes
- **Status:** Production-ready, neuron-focused
- **License:** MIT
- **Recommendation:** Conditional (requires SWC conversion for vessels)

---

## References

### Papers
- [clDice - a Novel Topology-Preserving Loss Function](https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf) (Shit et al., CVPR 2021)
- [Skeleton Recall Loss for Connectivity Conserving](https://arxiv.org/abs/2404.03010) (Weikert et al., ECCV 2024)
- [DeepVesselNet: Vessel Segmentation, Centerline Prediction, and Bifurcation Detection](https://arxiv.org/pdf/1803.09340) (2020)
- [Automatic Vessel Crossing and Bifurcation Detection](https://www.sciencedirect.com/science/article/pii/S0010482523001129) (2023)

### GitHub Repositories
- [clDice](https://github.com/jocpae/clDice) (MIT license)
- [APLS](https://github.com/CosmiQ/apls) (Apache-2.0)
- [Skeleton-Recall](https://github.com/MIC-DKFZ/Skeleton-Recall) (Apache-2.0)
- [segmentation-skeleton-metrics](https://github.com/AllenNeuralDynamics/segmentation-skeleton-metrics) (MIT)
- [PyVesTo](https://github.com/chcomin/pyvesto) (Unmaintained)

### MONAI Documentation
- [MONAI Loss Functions](https://docs.monai.io/en/stable/losses.html)
- [SoftclDiceLoss](https://docs.monai.io/en/stable/api/monai.losses.html#softcldicelosssoftcldicelosssoftdicelosssoftcldicelosssoftdiceloss)

---

## Integration Checklist for #124

- [ ] Read `graph-topology-metrics-survey.md` for full context
- [ ] Review MONAI clDice implementation (Section 3 of survey)
- [ ] Copy `GraphTopologyEvaluator` code from `graph-topology-metrics-code-examples.md`
- [ ] Write failing tests for bifurcation detection (RED phase)
- [ ] Implement skeleton extraction + bifurcation finding (GREEN phase)
- [ ] Implement graph matching with Hungarian algorithm (GREEN phase)
- [ ] Write MLflow logging integration (Section 4 of examples)
- [ ] Generate markdown reports (Section 4 of examples)
- [ ] Run full test suite: `uv run pytest tests/ -x -q`
- [ ] Check types: `uv run mypy src/`
- [ ] Check lint: `uv run ruff check src/`
- [ ] Commit with proper TDD history
- [ ] Update Issue #124 with completion status

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-28 | 1.0 | Initial comprehensive survey + 3 documentation files |

---

## Document Navigation

**If you need...**
- Comprehensive technical analysis → `graph-topology-metrics-survey.md`
- Quick package lookup → `graph-topology-metrics-quick-reference.yaml`
- Working code examples → `graph-topology-metrics-code-examples.md`
- High-level overview → This file (GRAPH-TOPOLOGY-METRICS-INDEX.md)

---

**Created for Issue #124**
**Last Updated:** 2026-02-28
**Status:** Research complete, ready for implementation (Phase 2 onwards)
