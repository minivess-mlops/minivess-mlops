# Topology-Aware Loss Functions — Implementation Plan (Issue #5)

## Current State
- clDice implemented (SoftclDiceLoss via MONAI, Issue #33)
- VesselCompoundLoss: DiceCE + clDice
- No cbDice, no Betti loss

## Architecture

### Extended Module: `src/minivess/pipeline/loss_functions.py`
- **ClassBalancedDiceLoss** — DiceLoss with inverse-frequency class weighting
- **BettiLoss** — Penalizes Betti-0 (connected component count) differences
- **TopologyCompoundLoss** — DiceCE + clDice + Betti (configurable weights)
- Extend `build_loss_function()` with: "cb_dice", "betti", "full_topo"

## Test Plan
- `tests/v2/unit/test_topology_loss.py` (~14 tests)
  - TestClassBalancedDice: basic, class weighting effect, gradient flow
  - TestBettiLoss: basic, perfect prediction=0, different topologies
  - TestTopologyCompoundLoss: basic, components, build_loss_function integration
