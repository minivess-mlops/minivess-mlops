"""WeightWatcher post-training spectral analysis.

Analyzes trained model layers using WeightWatcher's alpha metric.
Filters to trainable layers only (excludes frozen SAM3 ViT backbone).

Metrics (RC8 — diag_ww_ prefix):
- diag_ww_alpha_mean, diag_ww_alpha_std
- diag_ww_num_layers_analyzed

Artifact: diagnostics/weightwatcher_per_layer.json (RC12).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

# Artifact path for MLflow logging (RC12)
ARTIFACT_PATH = "diagnostics"


def run_weightwatcher(
    model: nn.Module,
    *,
    filter_frozen: bool = True,
) -> dict[str, Any]:
    """Run WeightWatcher spectral analysis on a trained model.

    Parameters
    ----------
    model:
        The trained PyTorch model to analyze.
    filter_frozen:
        If True, exclude layers where all parameters have
        requires_grad=False (e.g., frozen SAM3 backbone).

    Returns
    -------
    dict with diag_ww_ prefixed summary metrics.
    """
    import weightwatcher as ww

    # Detect Conv3d layers and warn — WeightWatcher silently skips them.
    has_conv3d = _has_conv3d_layers(model)
    if has_conv3d:
        logger.warning(
            "Model contains Conv3d layers which WeightWatcher cannot analyze. "
            "Only Linear and Conv2d layers will produce spectral metrics."
        )

    watcher = ww.WeightWatcher(model=model)

    # WeightWatcher analyzes Linear/Conv2d layers by default.
    # 3D conv layers are silently skipped (empty DataFrame).
    # Some exotic architectures may crash WeightWatcher entirely.
    # Catch broadly to ensure diagnostics never kill the pipeline.
    _nan_result: dict[str, Any] = {
        "diag_ww_alpha_mean": float("nan"),
        "diag_ww_alpha_std": float("nan"),
        "diag_ww_num_layers_analyzed": 0,
    }
    try:
        details = watcher.analyze()
    except Exception:
        logger.warning(
            "WeightWatcher crashed on model layers (likely 3D conv with "
            "weight=None or unsupported layer type). Returning NaN metrics.",
            exc_info=True,
        )
        return _nan_result

    if filter_frozen and "layer_id" in details.columns:
        # Filter to trainable layers only
        trainable_layer_ids = _get_trainable_layer_ids(model)
        if trainable_layer_ids:
            details = details[details["layer_id"].isin(trainable_layer_ids)]

    num_layers = len(details)

    if num_layers == 0:
        msg = "WeightWatcher: no analyzable layers found"
        if has_conv3d:
            msg += " (model has Conv3d layers which WeightWatcher cannot analyze)"
        logger.warning(msg)
        return _nan_result

    alpha_values = details["alpha"].dropna()

    return {
        "diag_ww_alpha_mean": float(alpha_values.mean())
        if len(alpha_values) > 0
        else float("nan"),
        "diag_ww_alpha_std": float(alpha_values.std())
        if len(alpha_values) > 1
        else 0.0,
        "diag_ww_num_layers_analyzed": num_layers,
    }


def _has_conv3d_layers(model: nn.Module) -> bool:
    """Check if model contains any Conv3d modules."""
    import torch

    return any(
        isinstance(m, torch.nn.Conv3d) for m in model.modules()
    )


def _get_trainable_layer_ids(model: nn.Module) -> set[int]:
    """Get layer IDs for modules with at least one trainable parameter."""

    trainable_ids: set[int] = set()
    for idx, (_, module) in enumerate(model.named_modules()):
        # Check if this module has any trainable (non-frozen) weight parameters
        has_trainable = any(p.requires_grad for p in module.parameters(recurse=False))
        if has_trainable and _has_weight_matrix(module):
            trainable_ids.add(idx)
    return trainable_ids


def _has_weight_matrix(module: nn.Module) -> bool:
    """Check if module has a weight parameter (Conv or Linear)."""
    import torch

    return hasattr(module, "weight") and isinstance(
        getattr(module, "weight", None), torch.nn.Parameter
    )
