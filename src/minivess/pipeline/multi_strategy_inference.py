"""MONAI-native multi-strategy sliding-window inference for all model adapters.

Enforces identical MONAI sliding_window_inference parameters across ALL models
for scientific comparability in paper tables. Model-specific ROI sizes come
from ModelAdapter.get_eval_roi_size() — NOT hardcoded in this module.

Rule #3 (Library-First): monai.inferers.sliding_window_inference is the ONLY
allowed sliding-window inference path. No custom inference loops.
Rule #16 (No Regex): metric key prefixing uses str.partition().
Rule #9 (Task-Agnostic): no if model_family == "..." branches here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import torch
from monai.inferers import sliding_window_inference  # type: ignore[attr-defined]
from torch import Tensor

from minivess.adapters.base import SegmentationOutput

if TYPE_CHECKING:
    from collections.abc import Callable

    from minivess.config.evaluation_config import InferenceStrategyConfig

logger = logging.getLogger(__name__)

_DEFAULT_EVAL_ROI = (128, 128, 16)


class MultiStrategyInferenceRunner:
    """Run multiple sliding-window inference strategies on a single volume.

    Each strategy in ``strategies`` is run independently via
    ``monai.inferers.sliding_window_inference``.  The primary strategy (
    ``is_primary=True``) produces bare metric keys; all others are prefixed
    with the strategy name (e.g., ``fast/dsc``).

    Parameters
    ----------
    strategies:
        List of InferenceStrategyConfig objects.  May be empty (no eval).
    num_classes:
        Number of segmentation classes.
    """

    def __init__(
        self,
        strategies: list[InferenceStrategyConfig],
        num_classes: int = 2,
    ) -> None:
        self.strategies = strategies
        self.num_classes = num_classes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all_strategies(
        self,
        model: torch.nn.Module,
        volume: Tensor,
    ) -> dict[str, dict[str, Any]]:
        """Run all strategies on a single volume tensor.

        Parameters
        ----------
        model:
            Any nn.Module (ModelAdapter preferred for get_eval_roi_size()).
        volume:
            Input tensor of shape (B, C, H, W, D) — MONAI convention.

        Returns
        -------
        ``{strategy_name: {metric_name: value}}`` for each strategy.
        """
        volume_shape = tuple(volume.shape)
        predictor = self._make_predictor(model)
        results: dict[str, dict[str, Any]] = {}

        for strategy in self.strategies:
            roi_size = self._resolve_roi_size(strategy, model, volume_shape)
            logger.debug(
                "Running strategy=%r roi_size=%r overlap=%.2f",
                strategy.name,
                roi_size,
                strategy.overlap,
            )
            with torch.no_grad():
                raw_out = sliding_window_inference(
                    inputs=volume,
                    roi_size=roi_size,
                    sw_batch_size=strategy.sw_batch_size,
                    predictor=predictor,
                    overlap=strategy.overlap,
                    mode=strategy.aggregation_mode,
                )
                output = cast("Tensor", raw_out)
            # Argmax to hard labels for metric computation
            pred = torch.argmax(output, dim=1, keepdim=True).float()
            results[strategy.name] = {"raw_output": output, "prediction": pred}

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_roi_size(
        self,
        strategy: InferenceStrategyConfig,
        model: torch.nn.Module,
        volume_shape: tuple[int, ...],
    ) -> tuple[int, int, int]:
        """Resolve the actual roi_size tuple for this strategy + model.

        - ``"per_model"``: call model.get_eval_roi_size() if available,
          else fall back to the global default (128, 128, 16).
        - list with ``-1``: replace -1 with the corresponding volume
          dimension (depth = dim 4 in MONAI (B,C,H,W,D) convention).
        - literal list: return as tuple unchanged.
        """
        roi = strategy.roi_size

        if roi == "per_model":
            if hasattr(model, "get_eval_roi_size"):
                model_any: Any = model
                result: tuple[int, int, int] = model_any.get_eval_roi_size()
                return result
            logger.warning(
                "Strategy roi_size='per_model' but model has no get_eval_roi_size(). "
                "Falling back to default %r.",
                _DEFAULT_EVAL_ROI,
            )
            return _DEFAULT_EVAL_ROI

        if isinstance(roi, list):
            # volume_shape = (B, C, H, W, D) — spatial dims are [2, 3, 4]
            spatial_dims = list(volume_shape[2:])  # [H, W, D]
            resolved = [
                spatial_dims[i] if roi[i] == -1 else roi[i] for i in range(len(roi))
            ]
            return (resolved[0], resolved[1], resolved[2])

        return _DEFAULT_EVAL_ROI

    def _make_predictor(self, model: torch.nn.Module) -> Callable[[Tensor], Tensor]:
        """Wrap model so it always returns a plain Tensor for sliding_window_inference.

        SegmentationOutput (from ModelAdapter) is unwrapped to its prediction field.
        Plain Tensor outputs are returned unchanged. Uses hasattr() — no regex.
        """

        def predictor(x: Tensor) -> Tensor:
            out = model(x)
            if hasattr(out, "prediction") and isinstance(out, SegmentationOutput):
                return out.prediction
            if isinstance(out, Tensor):
                return out
            msg = (
                f"Model output type {type(out)} is not supported. "
                "Expected SegmentationOutput or Tensor."
            )
            raise TypeError(msg)

        return predictor
