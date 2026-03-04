# Boundary Loss & Generalized Surface Loss Implementation Plan

**Issues:** #100 (Boundary Loss), #101 (Generalized Surface Loss)
**Date:** 2026-03-04
**Branch:** `feat/distance-losses` (from `main`)
**Classification:** HYBRID (library EDT for GT processing, custom differentiable loss computation)

## References

- Kervadec, H., Bouchtiba, J., Desrosiers, C., Granger, E., de Guise, J., & Bhatt Dolz, J. (2019). Boundary loss for highly unbalanced segmentation. *MIDL 2019*. https://github.com/LIVIAETS/boundary-loss
- Celaya, A., Actor, J.A., Muthusivarajan, R., Gates, E., Chung, C., Schellingerhout, D., Riviere, B., & Fuentes, D. (2024). A Generalized Surface Loss for Reducing the Hausdorff Distance in Medical Imaging Segmentation. *arXiv:2302.03868*.

---

## 1. Current Architecture Summary

### 1.1 Loss Factory (`src/minivess/pipeline/loss_functions.py`)

All losses currently follow the same calling convention:

```python
criterion = build_loss_function(loss_name)
loss = criterion(logits, labels)  # logits=(B,C,D,H,W), labels=(B,1,D,H,W)
```

The factory returns `nn.Module` instances. Classification tiers (`_LIBRARY_LOSSES`, `_HYBRID_LOSSES`, `_EXPERIMENTAL_LOSSES`) drive a one-time warning system via `_emit_loss_warning()`.

### 1.2 Training Loop Dispatch (`src/minivess/pipeline/trainer.py`)

The `SegmentationTrainer._compute_loss()` method already handles two calling conventions via isinstance dispatch:

```python
def _compute_loss(self, output, batch, labels):
    from minivess.pipeline.multitask_loss import MultiTaskLoss
    if isinstance(self.criterion, MultiTaskLoss):
        return self.criterion(output, batch)       # full batch dict
    return self.criterion(output.logits, labels)    # simple (logits, labels)
```

This existing pattern is the extension point for distance-based losses.

### 1.3 SDF Generation (`src/minivess/pipeline/sdf_generation.py`)

```python
def compute_sdf_from_mask(mask: np.ndarray) -> np.ndarray:
    dist_outside = distance_transform_edt(1 - binary)
    dist_inside = distance_transform_edt(binary)
    sdf = dist_outside - dist_inside
    return sdf
```

**Critical gap:** No `sampling=` parameter is passed to `distance_transform_edt()`. Distances are in voxel units, not physical units. For MiniVess (spacing 0.31-4.97 um/voxel, anisotropic Z), this produces incorrect distance fields.

### 1.4 Multi-Task Target Framework (`src/minivess/data/multitask_targets.py`)

```python
@dataclasses.dataclass
class AuxTargetConfig:
    name: str                                     # key in MONAI data dict
    suffix: str                                   # file suffix for precomputed NIfTI
    compute_fn: Callable[[np.ndarray], np.ndarray] # on-the-fly computation
```

`LoadAuxiliaryTargetsd` loads/computes auxiliary targets and injects them into the MONAI data dict. Spatial transforms (crop, flip, rotate) propagate to all keys including auxiliary targets.

### 1.5 Transform Pipeline (`src/minivess/data/transforms.py`)

Auxiliary targets are injected after `LoadImaged` + `EnsureChannelFirstd` + `NormalizeIntensityd`, then included in all spatial augmentations (`RandRotate90d`, `RandFlipd`, `RandCropByPosNegLabeld`).

---

## 2. Design Decisions

### 2.1 Loss Interface Extension: Batch-Dict Pattern (Recommended)

**Decision:** Extend `SegmentationTrainer._compute_loss()` to support a third calling convention for losses that need auxiliary data from the batch dict, without requiring MultiTaskLoss.

Three calling conventions (after this change):

| Convention | When | Signature |
|-----------|------|-----------|
| Standard | `criterion(logits, labels)` | All existing region/overlap losses |
| MultiTask | `criterion(output, batch)` | `MultiTaskLoss` with aux heads |
| BatchAware | `criterion(logits, labels, batch)` | Distance-based losses needing precomputed maps |

Implementation approach: Add a `BatchAwareLoss` ABC (or protocol) that `BoundaryLoss` and `GeneralizedSurfaceLoss` inherit from. The trainer dispatches on isinstance:

```python
# src/minivess/pipeline/loss_protocols.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import torch
from torch import nn


class BatchAwareLoss(nn.Module, ABC):
    """Base class for losses that need auxiliary data from the batch dict.

    Subclasses declare their required batch keys via `required_batch_keys`.
    The trainer calls `forward(logits, labels, batch)` instead of `forward(logits, labels)`.
    """

    @property
    @abstractmethod
    def required_batch_keys(self) -> frozenset[str]:
        """Batch dict keys this loss requires (e.g., {"distance_map"})."""
        ...

    @abstractmethod
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        ...
```

Updated trainer dispatch:

```python
def _compute_loss(self, output, batch, labels):
    from minivess.pipeline.multitask_loss import MultiTaskLoss
    from minivess.pipeline.loss_protocols import BatchAwareLoss

    if isinstance(self.criterion, MultiTaskLoss):
        return self.criterion(output, batch)
    if isinstance(self.criterion, BatchAwareLoss):
        return self.criterion(output.logits, labels, batch)
    return self.criterion(output.logits, labels)
```

**Rationale:** This avoids conflating "needs distance map" with "is multi-task." BatchAwareLoss is a clean orthogonal concept. Losses like BoundaryLoss are still single-task segmentation losses -- they just need precomputed auxiliary data alongside the label.

### 2.2 Distance Map Key Name

Use the generic key `"distance_map"` in the MONAI data dict, NOT `"boundary_loss_sdf"` or `"gsl_dist"`. Both Boundary Loss and GSL consume the same signed distance field. The key is task-agnostic per CLAUDE.md Principle #8.

### 2.3 Spacing-Aware EDT

Modify `compute_sdf_from_mask()` to accept an optional `sampling` parameter:

```python
def compute_sdf_from_mask(
    mask: np.ndarray,
    sampling: tuple[float, ...] | None = None,
) -> np.ndarray:
```

When `sampling` is provided, distances are in physical units (micrometers for MiniVess). When `None`, distances are in voxel units (backward-compatible).

### 2.4 Loss Prerequisite Registry

Add a `_LOSS_PREREQUISITES` dict to `loss_functions.py` that declares what each loss requires beyond `(logits, labels)`:

```python
_LOSS_PREREQUISITES: dict[str, frozenset[str]] = {
    "boundary": frozenset({"distance_map"}),
    "dice_ce_boundary": frozenset({"distance_map"}),
    "gsl": frozenset({"distance_map"}),
    "dice_ce_gsl": frozenset({"distance_map"}),
}
```

A pre-training validator checks that the data pipeline provides these keys.

---

## 3. Shared Infrastructure (Phase 1)

### 3.1 Spacing-Aware EDT

**File:** `src/minivess/pipeline/sdf_generation.py`

```python
def compute_sdf_from_mask(
    mask: np.ndarray,
    sampling: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Compute signed distance field from a binary 3D mask.

    Args:
        mask: Binary 3D array (0 = background, >0 = foreground).
        sampling: Physical voxel spacing per axis (e.g., (0.5, 0.5, 2.0) um).
            When provided, distances are in physical units.
            When None, distances are in voxel units (backward-compatible).

    Returns:
        SDF array (float32). Negative inside foreground, positive outside.
    """
    binary = (mask > 0).astype(np.uint8)
    dist_outside = distance_transform_edt(1 - binary, sampling=sampling).astype(np.float32)
    dist_inside = distance_transform_edt(binary, sampling=sampling).astype(np.float32)
    sdf = dist_outside - dist_inside
    return np.asarray(sdf)
```

### 3.2 Spacing Extraction Utility

**File:** `src/minivess/data/spacing_utils.py` (new)

```python
"""Utilities for extracting and validating voxel spacing from NIfTI headers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Anisotropy ratio threshold for emitting error with distance-based losses
ANISOTROPY_ERROR_THRESHOLD = 5.0


def extract_spacing_from_meta(meta_dict: dict[str, Any]) -> tuple[float, ...]:
    """Extract voxel spacing from MONAI metadata dict.

    MONAI's LoadImaged stores spacing in meta_dict["pixdim"] or
    meta_dict["affine"]. This function extracts physical spacing
    regardless of which is available.

    Returns:
        Tuple of physical spacing per spatial axis (e.g., (0.5, 0.5, 2.0)).
    """
    # MONAI stores spacing in pixdim[1:4] (NIfTI convention)
    if "pixdim" in meta_dict:
        pixdim = np.asarray(meta_dict["pixdim"])
        return tuple(float(x) for x in pixdim[1:4])

    # Fallback: extract from affine matrix diagonal
    if "affine" in meta_dict:
        affine = np.asarray(meta_dict["affine"])
        spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        return tuple(float(x) for x in spacing[:3])

    logger.warning("No spacing info found in metadata; defaulting to (1.0, 1.0, 1.0)")
    return (1.0, 1.0, 1.0)


def check_anisotropy(
    spacing: tuple[float, ...],
    volume_id: str | None = None,
) -> float:
    """Compute max anisotropy ratio and log error if above threshold.

    Returns:
        Max ratio between any two spacing axes.
    """
    spacing_arr = np.array(spacing)
    if spacing_arr.min() <= 0:
        logger.error("Invalid spacing %s (non-positive values)", spacing)
        return float("inf")

    ratio = float(spacing_arr.max() / spacing_arr.min())
    if ratio > ANISOTROPY_ERROR_THRESHOLD:
        logger.error(
            "Extreme anisotropy detected%s: spacing=%s, ratio=%.1fx. "
            "Distance-based losses (boundary, GSL) may produce unreliable gradients. "
            "Consider resampling or excluding this volume.",
            f" in {volume_id}" if volume_id else "",
            spacing,
            ratio,
        )
    return ratio
```

### 3.3 Spacing-Aware AuxTargetConfig for Distance Maps

**File:** `src/minivess/data/multitask_targets.py` (modify)

The current `AuxTargetConfig.compute_fn` signature is `Callable[[np.ndarray], np.ndarray]`. Distance maps need spacing. Two options:

**Option A (recommended):** Add an optional `compute_fn_with_meta` field:

```python
@dataclasses.dataclass
class AuxTargetConfig:
    name: str
    suffix: str
    compute_fn: Callable[[np.ndarray], np.ndarray]
    # Optional: compute function that also receives metadata (e.g., spacing)
    compute_fn_with_meta: Callable[[np.ndarray, dict[str, Any]], np.ndarray] | None = None
```

In `LoadAuxiliaryTargetsd._load_or_compute()`, prefer `compute_fn_with_meta` when available:

```python
def _load_or_compute(self, label, volume_id, config, meta_dict=None):
    # ... try precomputed file first ...

    if config.compute_fn_with_meta is not None and meta_dict is not None:
        return config.compute_fn_with_meta(label, meta_dict).astype(np.float32)

    return config.compute_fn(label).astype(np.float32)
```

**Option B:** Change `compute_fn` signature to always accept `(mask, meta_dict)` -- but this breaks all existing callers.

Option A is backward-compatible. The `compute_fn_with_meta` for distance maps would be:

```python
def compute_distance_map_with_spacing(mask: np.ndarray, meta_dict: dict) -> np.ndarray:
    """Compute spacing-aware signed distance field for distance-based losses."""
    from minivess.data.spacing_utils import extract_spacing_from_meta, check_anisotropy
    from minivess.pipeline.sdf_generation import compute_sdf_from_mask

    spacing = extract_spacing_from_meta(meta_dict)
    check_anisotropy(spacing, volume_id=meta_dict.get("filename_or_obj"))
    return compute_sdf_from_mask(mask, sampling=spacing)
```

### 3.4 Loss Prerequisite Registry

**File:** `src/minivess/pipeline/loss_functions.py` (modify)

```python
# Losses that require precomputed auxiliary data in the batch dict.
# Key = loss name, Value = frozenset of required batch keys.
_LOSS_PREREQUISITES: dict[str, frozenset[str]] = {
    "boundary": frozenset({"distance_map"}),
    "dice_ce_boundary": frozenset({"distance_map"}),
    "gsl": frozenset({"distance_map"}),
    "dice_ce_gsl": frozenset({"distance_map"}),
}


def get_loss_prerequisites(loss_name: str) -> frozenset[str]:
    """Return the set of batch-dict keys required by a loss function.

    Returns empty frozenset for losses that only need (logits, labels).
    """
    return _LOSS_PREREQUISITES.get(loss_name, frozenset())
```

### 3.5 Pre-Training Prerequisite Validator

**File:** `src/minivess/data/validation.py` (modify)

```python
class LossPrerequisiteError(ValueError):
    """Raised when a loss function's data prerequisites are not met."""


def validate_loss_prerequisites(
    loss_name: str,
    aux_config_names: frozenset[str],
) -> None:
    """Validate that the data pipeline provides all keys required by the loss.

    Parameters
    ----------
    loss_name:
        Loss function identifier.
    aux_config_names:
        Set of auxiliary target names configured in the data pipeline.

    Raises
    ------
    LossPrerequisiteError
        If any required key is missing from the data pipeline configuration.
    """
    from minivess.pipeline.loss_functions import get_loss_prerequisites

    required = get_loss_prerequisites(loss_name)
    if not required:
        return

    missing = required - aux_config_names
    if missing:
        msg = (
            f"Loss '{loss_name}' requires batch keys {sorted(required)} "
            f"but the data pipeline only provides {sorted(aux_config_names)}. "
            f"Missing: {sorted(missing)}. "
            f"Add AuxTargetConfig(name='{next(iter(missing))}', ...) to your "
            f"data pipeline configuration."
        )
        raise LossPrerequisiteError(msg)
```

---

## 4. Loss Interface Extension (Phase 2)

### 4.1 BatchAwareLoss Protocol

**File:** `src/minivess/pipeline/loss_protocols.py` (new)

```python
"""Loss function protocols for extended calling conventions.

Standard losses:    criterion(logits, labels) -> Tensor
BatchAware losses:  criterion(logits, labels, batch) -> Tensor
MultiTask losses:   criterion(output, batch) -> Tensor
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class BatchAwareLoss(nn.Module, ABC):
    """Abstract base for losses needing precomputed auxiliary batch data.

    Subclasses declare required batch keys and receive the full batch dict
    in forward(). The SegmentationTrainer dispatches on isinstance.
    """

    @property
    @abstractmethod
    def required_batch_keys(self) -> frozenset[str]:
        """Batch dict keys this loss requires beyond 'image' and 'label'."""
        ...

    @abstractmethod
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Compute loss with access to auxiliary batch data.

        Parameters
        ----------
        logits:
            Model output (B, C, D, H, W).
        labels:
            Ground truth labels (B, 1, D, H, W).
        batch:
            Full batch dict containing auxiliary keys.
        """
        ...
```

### 4.2 Trainer Dispatch Update

**File:** `src/minivess/pipeline/trainer.py` (modify `_compute_loss`)

```python
def _compute_loss(self, output, batch, labels):
    from minivess.pipeline.multitask_loss import MultiTaskLoss
    from minivess.pipeline.loss_protocols import BatchAwareLoss

    if isinstance(self.criterion, MultiTaskLoss):
        return self.criterion(output, batch)
    if isinstance(self.criterion, BatchAwareLoss):
        return self.criterion(output.logits, labels, batch)
    return self.criterion(output.logits, labels)
```

Also update the validation path in `_validate_epoch()` (lines ~328-333) to handle `BatchAwareLoss`:

```python
if isinstance(self.criterion, MultiTaskLoss):
    loss = self.criterion.seg_criterion(logits, labels)
elif isinstance(self.criterion, BatchAwareLoss):
    loss = self.criterion(logits, labels, batch)
else:
    loss = self.criterion(logits, labels)
```

---

## 5. Boundary Loss Implementation (Phase 3 -- Issue #100)

### 5.1 Core Loss Module

**File:** `src/minivess/pipeline/vendored_losses/boundary_loss.py` (new)

```python
"""Boundary Loss (Kervadec et al., MIDL 2019).

Differentiable surface distance proxy via dot product of softmax predictions
with precomputed signed distance maps.

L_boundary = (1/N) * sum(softmax_pred_fg * dist_map)

where dist_map is negative inside the foreground and positive outside.
Minimizing this loss pushes predicted foreground toward the GT boundary.

Reference: https://github.com/LIVIAETS/boundary-loss
Classification: HYBRID — library EDT for GT, custom loss forward path.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from minivess.pipeline.loss_protocols import BatchAwareLoss


class BoundaryLoss(BatchAwareLoss):
    """Boundary Loss using precomputed signed distance maps.

    Parameters
    ----------
    dist_map_key:
        Key in the batch dict containing the signed distance map (B, 1, D, H, W).
    softmax:
        Whether to apply softmax to logits before computing the loss.
    """

    def __init__(
        self,
        dist_map_key: str = "distance_map",
        *,
        softmax: bool = True,
    ) -> None:
        super().__init__()
        self.dist_map_key = dist_map_key
        self.softmax = softmax

    @property
    def required_batch_keys(self) -> frozenset[str]:
        return frozenset({self.dist_map_key})

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Compute boundary loss.

        L = (1/N) * sum(pred_fg * dist_map)

        Where pred_fg is the foreground probability and dist_map is negative
        inside GT foreground, positive outside. Minimizing pushes predictions
        toward GT boundaries.

        Parameters
        ----------
        logits:
            Model output (B, C, D, H, W) with C >= 2 channels.
        labels:
            Ground truth (B, 1, D, H, W) -- unused by boundary loss itself
            but accepted for interface compatibility.
        batch:
            Must contain self.dist_map_key with shape (B, 1, D, H, W).
        """
        dist_map = batch[self.dist_map_key]  # (B, 1, D, H, W)

        if self.softmax:
            probs = torch.softmax(logits, dim=1)
        else:
            probs = logits

        # Foreground probability (classes 1+)
        pred_fg = probs[:, 1:, ...]  # (B, C-1, D, H, W)

        # For binary segmentation, dist_map is (B, 1, D, H, W) matching pred_fg
        # For multi-class, broadcast or expand dist_map as needed
        loss: torch.Tensor = (pred_fg * dist_map).mean()
        return loss
```

### 5.2 Compound: DiceCE + Boundary with Scheduling

**File:** `src/minivess/pipeline/vendored_losses/boundary_loss.py` (continued)

```python
class DiceCEBoundaryLoss(BatchAwareLoss):
    """Compound loss: alpha * DiceCE + (1 - alpha) * BoundaryLoss.

    Alpha scheduling: starts at 1.0 (pure DiceCE) and linearly decays to
    alpha_min over schedule_epochs. This is essential because BoundaryLoss
    alone cannot learn from scratch (it has no region-based gradient).

    Parameters
    ----------
    alpha_init:
        Initial weight for DiceCE (default 1.0 = pure DiceCE at start).
    alpha_min:
        Minimum alpha after scheduling (default 0.01).
    schedule_epochs:
        Number of epochs over which alpha decays (default 100).
    dist_map_key:
        Key for distance map in batch dict.
    """

    def __init__(
        self,
        alpha_init: float = 1.0,
        alpha_min: float = 0.01,
        schedule_epochs: int = 100,
        dist_map_key: str = "distance_map",
        *,
        softmax: bool = True,
        to_onehot_y: bool = True,
    ) -> None:
        super().__init__()
        self.alpha_init = alpha_init
        self.alpha_min = alpha_min
        self.schedule_epochs = schedule_epochs
        self._current_epoch = 0

        from monai.losses import DiceCELoss

        self.dice_ce = DiceCELoss(
            softmax=softmax,
            to_onehot_y=to_onehot_y,
            lambda_ce=0.5,
            lambda_dice=0.5,
        )
        self.boundary = BoundaryLoss(
            dist_map_key=dist_map_key,
            softmax=softmax,
        )

    @property
    def required_batch_keys(self) -> frozenset[str]:
        return self.boundary.required_batch_keys

    @property
    def alpha(self) -> float:
        """Current scheduling alpha (linearly decays from alpha_init to alpha_min)."""
        if self.schedule_epochs <= 0:
            return self.alpha_min
        progress = min(self._current_epoch / self.schedule_epochs, 1.0)
        return self.alpha_init - progress * (self.alpha_init - self.alpha_min)

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for alpha scheduling."""
        self._current_epoch = epoch

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Compute scheduled compound loss."""
        a = self.alpha
        dice_ce_loss = self.dice_ce(logits, labels)
        boundary_loss = self.boundary(logits, labels, batch)
        result: torch.Tensor = a * dice_ce_loss + (1.0 - a) * boundary_loss
        return result
```

### 5.3 Epoch Scheduling Hook

The `SegmentationTrainer.train_epoch()` must call `criterion.set_epoch(epoch)` if the criterion supports it. Add at the top of `train_epoch()`:

```python
if hasattr(self.criterion, "set_epoch"):
    self.criterion.set_epoch(self._current_epoch)
```

This is a duck-typing approach (no isinstance check needed), keeping it generic for any loss that wants epoch-based scheduling.

### 5.4 Factory Registration

In `build_loss_function()`:

```python
elif loss_name == "boundary":
    from minivess.pipeline.vendored_losses.boundary_loss import BoundaryLoss
    loss_fn = BoundaryLoss(softmax=softmax)
elif loss_name == "dice_ce_boundary":
    from minivess.pipeline.vendored_losses.boundary_loss import DiceCEBoundaryLoss
    loss_fn = DiceCEBoundaryLoss(softmax=softmax, to_onehot_y=to_onehot_y)
```

Add to classification dicts:

```python
_HYBRID_LOSSES["boundary"] = (
    "scipy EDT (GT distance map) + differentiable dot-product loss "
    "(Kervadec et al. MIDL 2019)"
)
_HYBRID_LOSSES["dice_ce_boundary"] = (
    "DiceCE + BoundaryLoss with linear alpha scheduling "
    "(Kervadec et al. MIDL 2019)"
)
```

Add to prerequisites:

```python
_LOSS_PREREQUISITES["boundary"] = frozenset({"distance_map"})
_LOSS_PREREQUISITES["dice_ce_boundary"] = frozenset({"distance_map"})
```

---

## 6. Generalized Surface Loss Implementation (Phase 4 -- Issue #101)

### 6.1 Core Loss Module

**File:** `src/minivess/pipeline/vendored_losses/gsl_loss.py` (new)

```python
"""Generalized Surface Loss (Celaya et al., 2024).

Bounded alternative to Boundary Loss. Output is in [0, 1] range.

L_GSL = 1 - [sum_k w_k * sum_i (D_i^k * (1 - (T_i^k + P_i^k)))^2]
            / [sum_k w_k * sum_i (D_i^k)^2]

where:
  D_i^k = signed distance transform value at voxel i for class k
  T_i^k = ground truth indicator for class k at voxel i
  P_i^k = predicted probability for class k at voxel i
  w_k   = class weight (precomputed across entire dataset)

Key properties:
  - Bounded to [0, 1] (unlike BoundaryLoss which is unbounded)
  - No epoch-based scheduling needed (can be combined directly)
  - Class weights computed ONCE across entire dataset, not per-batch

Reference: arXiv:2302.03868
Classification: HYBRID -- library EDT for GT, custom loss forward path.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from minivess.pipeline.loss_protocols import BatchAwareLoss


class GeneralizedSurfaceLoss(BatchAwareLoss):
    """Generalized Surface Loss (GSL).

    Parameters
    ----------
    dist_map_key:
        Key in batch dict for signed distance map (B, 1, D, H, W).
    class_weights:
        Per-class weights. If None, uniform weights are used.
        For binary segmentation, pass (w_bg, w_fg).
        Precomputed across the entire dataset (not per-batch).
    softmax:
        Whether to apply softmax to logits.
    eps:
        Epsilon for numerical stability in denominator.
    """

    def __init__(
        self,
        dist_map_key: str = "distance_map",
        class_weights: tuple[float, ...] | None = None,
        *,
        softmax: bool = True,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.dist_map_key = dist_map_key
        self.softmax = softmax
        self.eps = eps

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None  # type: ignore[assignment]

    @property
    def required_batch_keys(self) -> frozenset[str]:
        return frozenset({self.dist_map_key})

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Compute Generalized Surface Loss.

        Parameters
        ----------
        logits:
            (B, C, D, H, W) model output.
        labels:
            (B, 1, D, H, W) integer ground truth.
        batch:
            Must contain self.dist_map_key (B, 1, D, H, W) signed distance map.
        """
        dist_map = batch[self.dist_map_key]  # (B, 1, D, H, W)

        if self.softmax:
            probs = torch.softmax(logits, dim=1)  # (B, C, D, H, W)
        else:
            probs = logits

        n_classes = probs.shape[1]

        # One-hot encode labels: (B, C, D, H, W)
        labels_squeeze = labels.long()
        if labels_squeeze.ndim == logits.ndim:
            labels_squeeze = labels_squeeze[:, 0]  # (B, D, H, W)
        labels_onehot = torch.zeros_like(probs)
        for c in range(n_classes):
            labels_onehot[:, c] = (labels_squeeze == c).float()

        # Expand distance map to all classes: (B, C, D, H, W)
        # The same distance map applies to all classes for binary segmentation.
        # For multi-class, each class would have its own distance map;
        # for now, broadcast the single-channel map.
        dist_expanded = dist_map.expand_as(probs)

        # GSL formula: 1 - [sum(w_k * sum((D * (1 - (T + P)))^2))] / [sum(w_k * sum(D^2))]
        residual = 1.0 - (labels_onehot + probs)  # (B, C, D, H, W)
        numerator_per_class = (dist_expanded * residual).pow(2)  # (B, C, D, H, W)
        denominator_per_class = dist_expanded.pow(2)  # (B, C, D, H, W)

        spatial_dims = tuple(range(2, logits.ndim))  # (2, 3, 4) for 3D

        # Sum over spatial dims: (B, C)
        num_summed = numerator_per_class.sum(dim=spatial_dims)
        den_summed = denominator_per_class.sum(dim=spatial_dims)

        # Apply class weights
        if self.class_weights is not None:
            weights = self.class_weights.to(probs.device)
            # Reshape for broadcasting: (1, C)
            weights = weights.unsqueeze(0)
        else:
            weights = torch.ones(1, n_classes, device=probs.device)

        weighted_num = (weights * num_summed).sum(dim=1)  # (B,)
        weighted_den = (weights * den_summed).sum(dim=1)  # (B,)

        gsl_per_sample = 1.0 - weighted_num / (weighted_den + self.eps)  # (B,)
        result: torch.Tensor = gsl_per_sample.mean()
        return result
```

### 6.2 Dataset-Level Class Weight Computation

**File:** `src/minivess/data/class_weights.py` (new)

```python
"""Dataset-level class weight computation for losses like GSL.

Weights are computed ONCE across the entire dataset, not per-batch.
This follows Celaya et al. (2024) prescription.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def compute_class_weights_from_dataset(
    label_paths: list[Path],
    num_classes: int = 2,
) -> tuple[float, ...]:
    """Compute inverse-frequency class weights across entire dataset.

    Parameters
    ----------
    label_paths:
        Paths to all label NIfTI files in the dataset.
    num_classes:
        Number of segmentation classes (including background).

    Returns
    -------
    Tuple of per-class weights, normalized to sum to 1.
    """
    import nibabel as nib

    class_counts = np.zeros(num_classes, dtype=np.float64)

    for path in label_paths:
        label = np.asarray(
            nib.load(str(path)).dataobj,  # type: ignore[attr-defined]
            dtype=np.int64,
        )
        for c in range(num_classes):
            class_counts[c] += (label == c).sum()

    total = class_counts.sum()
    if total == 0:
        logger.warning("All labels are empty; using uniform class weights")
        return tuple(1.0 / num_classes for _ in range(num_classes))

    # Inverse frequency
    inv_freq = total / (num_classes * class_counts + 1e-10)
    # Normalize to sum=1
    weights = inv_freq / inv_freq.sum()

    logger.info(
        "Dataset class weights (n=%d labels): %s",
        len(label_paths),
        {f"class_{c}": f"{w:.4f}" for c, w in enumerate(weights)},
    )
    return tuple(float(w) for w in weights)
```

### 6.3 Compound: DiceCE + GSL

```python
class DiceCEGSLLoss(BatchAwareLoss):
    """Compound loss: lambda_dice_ce * DiceCE + lambda_gsl * GSL.

    Unlike DiceCE + BoundaryLoss, no scheduling is needed because GSL
    is bounded to [0, 1] and can be combined directly from epoch 0.

    Parameters
    ----------
    lambda_dice_ce:
        Weight for DiceCE component.
    lambda_gsl:
        Weight for GSL component.
    dist_map_key:
        Key for distance map in batch dict.
    class_weights:
        Per-class weights for GSL (precomputed across dataset).
    """

    def __init__(
        self,
        lambda_dice_ce: float = 0.5,
        lambda_gsl: float = 0.5,
        dist_map_key: str = "distance_map",
        class_weights: tuple[float, ...] | None = None,
        *,
        softmax: bool = True,
        to_onehot_y: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_dice_ce = lambda_dice_ce
        self.lambda_gsl = lambda_gsl

        from monai.losses import DiceCELoss

        self.dice_ce = DiceCELoss(
            softmax=softmax,
            to_onehot_y=to_onehot_y,
            lambda_ce=0.5,
            lambda_dice=0.5,
        )
        self.gsl = GeneralizedSurfaceLoss(
            dist_map_key=dist_map_key,
            class_weights=class_weights,
            softmax=softmax,
        )

    @property
    def required_batch_keys(self) -> frozenset[str]:
        return self.gsl.required_batch_keys

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        dice_ce_loss = self.dice_ce(logits, labels)
        gsl_loss = self.gsl(logits, labels, batch)
        result: torch.Tensor = (
            self.lambda_dice_ce * dice_ce_loss + self.lambda_gsl * gsl_loss
        )
        return result
```

### 6.4 Factory Registration

```python
elif loss_name == "gsl":
    from minivess.pipeline.vendored_losses.gsl_loss import GeneralizedSurfaceLoss
    loss_fn = GeneralizedSurfaceLoss(softmax=softmax)
elif loss_name == "dice_ce_gsl":
    from minivess.pipeline.vendored_losses.gsl_loss import DiceCEGSLLoss
    loss_fn = DiceCEGSLLoss(softmax=softmax, to_onehot_y=to_onehot_y)
```

Classification:

```python
_HYBRID_LOSSES["gsl"] = (
    "scipy EDT (GT distance map) + bounded surface distance loss "
    "(Celaya et al. 2024)"
)
_HYBRID_LOSSES["dice_ce_gsl"] = (
    "DiceCE + GeneralizedSurfaceLoss compound "
    "(Celaya et al. 2024)"
)
```

---

## 7. Data Flow Diagram

```
NIfTI Volume Loading
        |
        v
LoadImaged(keys=["image", "label"])
        |
        v
EnsureChannelFirstd
        |
        v
NormalizeIntensityd(keys="image")
        |
        v
LoadAuxiliaryTargetsd  ------>  compute_fn_with_meta(label, meta_dict)
  |                                     |
  |                                     v
  |                             extract_spacing_from_meta(meta_dict)
  |                                     |
  |                                     v
  |                             check_anisotropy(spacing)
  |                                     |
  |                                     v
  |                             compute_sdf_from_mask(mask, sampling=spacing)
  |                                     |
  |                                     v
  |                             batch["distance_map"] = SDF  (B, 1, D, H, W)
  |
  v
EnsureChannelFirstd(keys=["distance_map"])
        |
        v
SpatialPadd / RandCropByPosNegLabeld  (distance_map included in spatial_keys)
        |
        v
  Training Loop
        |
        v
SegmentationTrainer._compute_loss(output, batch, labels)
        |
        v
  isinstance(criterion, BatchAwareLoss)?
        |           |
       yes          no
        |           |
        v           v
  criterion(        criterion(
    logits,           logits,
    labels,           labels
    batch           )
  )
        |
        v
  BoundaryLoss:  loss = mean(pred_fg * dist_map)
  -- or --
  GSL:           loss = 1 - weighted_sum(D*(1-(T+P)))^2 / weighted_sum(D^2)
```

---

## 8. Validation & Guards (Phase 5)

### 8.1 Pre-Training Validation

Called during training setup (in `train_monitored.py` or equivalent):

```python
# After building loss and data config, before training starts:
from minivess.data.validation import validate_loss_prerequisites

aux_names = frozenset(c.name for c in aux_configs) if aux_configs else frozenset()
validate_loss_prerequisites(loss_name, aux_names)
```

### 8.2 Anisotropy Guard

Already handled in `check_anisotropy()` (Section 3.2). Emits `logger.error()` when ratio exceeds 5x. Does NOT raise an exception (the researcher may have reasons to proceed).

### 8.3 Test Plan

#### Unit Tests

**File:** `tests/unit/test_boundary_loss.py`

| Test | Assertion |
|------|-----------|
| `test_boundary_loss_forward_shape` | Output is scalar tensor |
| `test_boundary_loss_gradient_flow` | `loss.backward()` produces non-None gradients on logits |
| `test_boundary_loss_perfect_prediction` | Loss is minimized (negative) when pred matches GT |
| `test_boundary_loss_worst_prediction` | Loss is maximized (positive) when pred is inverted GT |
| `test_boundary_loss_unbounded_range` | Loss can be negative (this is expected behavior) |
| `test_boundary_loss_requires_distance_map` | Raises `KeyError` if `distance_map` not in batch |
| `test_boundary_loss_required_batch_keys` | `.required_batch_keys` returns `{"distance_map"}` |

**File:** `tests/unit/test_gsl_loss.py`

| Test | Assertion |
|------|-----------|
| `test_gsl_forward_shape` | Output is scalar tensor |
| `test_gsl_gradient_flow` | `loss.backward()` produces non-None gradients |
| `test_gsl_bounded_output` | Loss is in `[0, 1]` range for all tested inputs |
| `test_gsl_perfect_prediction` | Loss near 0 when pred matches GT |
| `test_gsl_worst_prediction` | Loss near 1 when pred is inverted |
| `test_gsl_class_weights` | Different class weights produce different loss values |
| `test_gsl_requires_distance_map` | Raises `KeyError` if `distance_map` not in batch |

**File:** `tests/unit/test_compound_boundary_losses.py`

| Test | Assertion |
|------|-----------|
| `test_dice_ce_boundary_alpha_scheduling` | Alpha decays from 1.0 to alpha_min over schedule_epochs |
| `test_dice_ce_boundary_epoch_0_pure_dicece` | At epoch 0, compound equals DiceCE alone |
| `test_dice_ce_boundary_set_epoch` | `set_epoch()` updates internal state correctly |
| `test_dice_ce_gsl_no_scheduling` | GSL compound has no scheduling behavior |
| `test_dice_ce_gsl_gradient_flow` | Backward pass through compound produces valid gradients |

**File:** `tests/unit/test_spacing_aware_edt.py`

| Test | Assertion |
|------|-----------|
| `test_sdf_isotropic_matches_original` | With `sampling=(1,1,1)`, matches voxel-unit SDF |
| `test_sdf_anisotropic_differs` | With `sampling=(1,1,3)`, distances differ from voxel-unit |
| `test_sdf_physical_units` | Distance at (0,0,5) from surface with spacing (1,1,2) is 10 um |
| `test_sdf_backward_compatible` | `sampling=None` produces identical output to current code |

**File:** `tests/unit/test_loss_prerequisites.py`

| Test | Assertion |
|------|-----------|
| `test_validate_prerequisites_boundary_missing` | Raises `LossPrerequisiteError` |
| `test_validate_prerequisites_boundary_present` | No error when `distance_map` in aux configs |
| `test_validate_prerequisites_dice_ce_no_prereqs` | Standard losses have no prerequisites |
| `test_validate_prerequisites_gsl_missing` | Raises `LossPrerequisiteError` |

**File:** `tests/unit/test_anisotropy_guard.py`

| Test | Assertion |
|------|-----------|
| `test_anisotropy_below_threshold` | Returns ratio, no error logged |
| `test_anisotropy_above_threshold` | Returns ratio, error IS logged |
| `test_anisotropy_exact_threshold` | At exactly 5.0, no error (strictly greater) |

#### Integration Tests

**File:** `tests/integration/test_distance_loss_training_step.py`

| Test | Assertion |
|------|-----------|
| `test_full_training_step_boundary` | Forward + backward through DiceCEBoundaryLoss with synthetic batch |
| `test_full_training_step_gsl` | Forward + backward through DiceCEGSLLoss with synthetic batch |
| `test_factory_builds_boundary` | `build_loss_function("dice_ce_boundary")` returns `DiceCEBoundaryLoss` |
| `test_factory_builds_gsl` | `build_loss_function("dice_ce_gsl")` returns `DiceCEGSLLoss` |
| `test_trainer_dispatch_batch_aware` | `_compute_loss` calls `criterion(logits, labels, batch)` for BatchAwareLoss |

---

## 9. Config Integration (Phase 6)

### 9.1 Example YAML Config

**File:** `configs/experiments/dynunet_boundary_loss.yaml`

```yaml
# Boundary Loss experiment: DiceCE + BoundaryLoss with linear alpha scheduling.
# Requires distance_map auxiliary target in data pipeline.

experiment_name: dynunet_boundary_loss_v1
description: "DynUNet with DiceCE+BoundaryLoss (Kervadec 2019), alpha scheduling over 100 epochs"

model:
  family: dynunet
  in_channels: 1
  out_channels: 2

data:
  dataset_name: minivess
  patch_size: [128, 128, 32]
  voxel_spacing: [0, 0, 0]  # Native resolution (mandatory for correct spacing-aware EDT)
  aux_targets:
    - name: distance_map
      suffix: distance_map
      compute_fn: compute_distance_map_with_spacing  # Uses per-volume physical spacing

training:
  loss_name: dice_ce_boundary
  num_epochs: 100
  learning_rate: 0.001
  batch_size: 2

loss_config:
  alpha_init: 1.0
  alpha_min: 0.01
  schedule_epochs: 100
```

### 9.2 GSL YAML Config

**File:** `configs/experiments/dynunet_gsl.yaml`

```yaml
experiment_name: dynunet_gsl_v1
description: "DynUNet with DiceCE+GSL (Celaya 2024), no scheduling needed"

model:
  family: dynunet
  in_channels: 1
  out_channels: 2

data:
  dataset_name: minivess
  patch_size: [128, 128, 32]
  voxel_spacing: [0, 0, 0]
  aux_targets:
    - name: distance_map
      suffix: distance_map
      compute_fn: compute_distance_map_with_spacing

training:
  loss_name: dice_ce_gsl
  num_epochs: 100
  learning_rate: 0.001
  batch_size: 2

loss_config:
  lambda_dice_ce: 0.5
  lambda_gsl: 0.5
  # class_weights computed automatically from dataset if not specified
```

---

## 10. Implementation Order

| Phase | Scope | Files Modified | Files Created | Estimated Tests |
|-------|-------|---------------|--------------|-----------------|
| **Phase 1** | Shared infrastructure | `sdf_generation.py`, `multitask_targets.py`, `loss_functions.py`, `validation.py` | `spacing_utils.py` | 12 |
| **Phase 2** | Loss interface extension | `trainer.py` | `loss_protocols.py` | 4 |
| **Phase 3** | Boundary Loss (#100) | `loss_functions.py` | `vendored_losses/boundary_loss.py` | 12 |
| **Phase 4** | GSL (#101) | `loss_functions.py` | `vendored_losses/gsl_loss.py`, `class_weights.py` | 10 |
| **Phase 5** | Compound losses + scheduling | `trainer.py` (set_epoch hook) | (in boundary_loss.py, gsl_loss.py) | 5 |
| **Phase 6** | Config + integration tests | | config YAMLs, integration tests | 5 |
| **Total** | | 5 modified | 5 new | ~48 tests |

### Dependency Graph

```
Phase 1: Shared Infrastructure
    |
    +-- spacing_utils.py (new)
    |       |
    +-- sdf_generation.py (modify: add sampling param)
    |       |
    +-- multitask_targets.py (modify: add compute_fn_with_meta)
    |       |
    +-- loss_functions.py (add _LOSS_PREREQUISITES)
    |       |
    +-- validation.py (add validate_loss_prerequisites)
    |
    v
Phase 2: Loss Interface
    |
    +-- loss_protocols.py (new: BatchAwareLoss ABC)
    |       |
    +-- trainer.py (modify: _compute_loss dispatch)
    |
    v
Phase 3: Boundary Loss (#100)          Phase 4: GSL (#101)
    |                                       |
    +-- boundary_loss.py (new)              +-- gsl_loss.py (new)
    |                                       |
    +-- BoundaryLoss                        +-- GeneralizedSurfaceLoss
    |                                       |
    +-- DiceCEBoundaryLoss                  +-- DiceCEGSLLoss
    |                                       |
    +-- loss_functions.py (register)        +-- class_weights.py (new)
    |                                       |
    +-- loss_functions.py (register)        +-- loss_functions.py (register)
    |                                       |
    v                                       v
Phase 5: Scheduling + Integration
    |
    +-- trainer.py (set_epoch hook)
    |
    v
Phase 6: Config + Integration Tests
    |
    +-- configs/experiments/*.yaml
    +-- tests/integration/test_distance_loss_training_step.py
```

Phases 3 and 4 can be done in parallel once Phases 1-2 are complete.

---

## 11. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Distance map not propagated through spatial transforms | High | Verified: MONAI's `RandCropByPosNegLabeld` propagates all `spatial_keys`. Distance maps are included via `aux_keys`. |
| Sign convention mismatch | High | Boundary Loss uses negative inside GT. `compute_sdf_from_mask` already uses this convention (`dist_outside - dist_inside`). Verified against Kervadec reference implementation. |
| BoundaryLoss unbounded output breaks training | Medium | Always use compound `dice_ce_boundary` with alpha scheduling. Standalone `boundary` loss is available but should only be used by experts. |
| Anisotropic spacing corrupts distance fields | Medium | `check_anisotropy()` emits `logger.error()` for ratio > 5x. EDT with `sampling=` handles anisotropy correctly in the math, but extreme ratios amplify discretization error. |
| Distance map computation speed | Low | EDT is O(n) per volume. For MiniVess (512x512x110 max), takes ~50ms per volume. Precomputation during data loading is negligible compared to forward pass. |
| Class weight precomputation for GSL | Low | One-time scan of all label files. For 70 MiniVess volumes, takes < 5 seconds. |
| `set_epoch()` not called | Medium | Duck-typing approach (`hasattr`) ensures no crash if omitted, but alpha scheduling will be stuck at epoch 0 (= pure DiceCE). Add integration test to verify. |

---

## 12. Summary of New Files

| File | Purpose |
|------|---------|
| `src/minivess/data/spacing_utils.py` | Spacing extraction from MONAI meta dicts + anisotropy guard |
| `src/minivess/pipeline/loss_protocols.py` | `BatchAwareLoss` ABC for losses needing batch dict |
| `src/minivess/pipeline/vendored_losses/boundary_loss.py` | `BoundaryLoss` + `DiceCEBoundaryLoss` |
| `src/minivess/pipeline/vendored_losses/gsl_loss.py` | `GeneralizedSurfaceLoss` + `DiceCEGSLLoss` |
| `src/minivess/data/class_weights.py` | Dataset-level class weight computation for GSL |

## 13. Summary of Modified Files

| File | Change |
|------|--------|
| `src/minivess/pipeline/sdf_generation.py` | Add `sampling` parameter to `compute_sdf_from_mask()` |
| `src/minivess/data/multitask_targets.py` | Add `compute_fn_with_meta` field to `AuxTargetConfig` |
| `src/minivess/pipeline/loss_functions.py` | Add `_LOSS_PREREQUISITES`, register 4 new losses, update classifications |
| `src/minivess/data/validation.py` | Add `LossPrerequisiteError` + `validate_loss_prerequisites()` |
| `src/minivess/pipeline/trainer.py` | Add `BatchAwareLoss` dispatch in `_compute_loss()` + `set_epoch()` hook |
