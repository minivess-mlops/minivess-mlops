# Debug Training All 18 Losses — Pre-PR Validation

## Context

Branch `feat/graph-constrained-models` (25 closed issues, 2123 tests, 23 commits ahead of main)
implements topology-aware losses, metrics, and architecture modules. Before creating the PR,
we need to validate ALL 18 loss functions through a short debug training run to discover and
TDD-fix implementation bugs. The purpose is **mechanical stability validation**, not performance
evaluation — that comes later with the real 100-epoch sweep.

## Verified Bug Inventory

| # | Bug | Severity | File | Root Cause |
|---|-----|----------|------|------------|
| 1 | Standalone `cldice` missing preprocessing | MEDIUM | `loss_functions.py` | Bare `SoftclDiceLoss()` receives raw logits + 1-ch labels; needs softmax + one-hot |
| 2 | `cbdice` avg_pool3d kernel_size=5 crash | HIGH | `cbdice.py` | `F.avg_pool3d(kernel_size=5)` crashes when any spatial dim < 5 |
| 3 | TopoLoss NaN at Z<=7 | MEDIUM | `coletra.py` | `avg_pool3d(kernel_size=4)` on Z=5 produces Z=1, then `torch.diff(dim=2)` yields empty tensor |
| 4 | TopoSegLoss gradient break | LOW | `toposeg.py` | Returns `torch.tensor(0.0)` without `requires_grad` when no critical points found |
| 5 | `betti_matching` 100x scale mismatch | MEDIUM | `betti_matching.py` | Returns ~84 while all other losses return 0.3-0.8 |

## Exit Criterion

**"All 18 losses complete 6 epochs x 3 folds without crash (no RuntimeError, no NaN, no OOM)."**
