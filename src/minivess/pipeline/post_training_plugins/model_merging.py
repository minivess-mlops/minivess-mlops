"""Model merging post-training plugin.

Wraps ``minivess.ensemble.model_merging`` functions for linear,
SLERP, and layer-wise merging of champion model state dicts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from minivess.ensemble.model_merging import layer_wise_merge, linear_merge, slerp_merge
from minivess.pipeline.post_training_plugin import PluginInput, PluginOutput

logger = logging.getLogger(__name__)

_VALID_METHODS = {"linear", "slerp", "layer_wise"}


class ModelMergingPlugin:
    """Model merging plugin — linear, SLERP, or layer-wise interpolation."""

    @property
    def name(self) -> str:
        return "model_merging"

    @property
    def requires_calibration_data(self) -> bool:
        return False

    def validate_inputs(self, plugin_input: PluginInput) -> list[str]:
        errors: list[str] = []
        method = plugin_input.config.get("method", "slerp")
        if method not in _VALID_METHODS:
            errors.append(
                f"Invalid merge method: {method!r}. Must be one of {sorted(_VALID_METHODS)}"
            )

        # Always require at least 2 checkpoints regardless of merge_pairs config.
        # Previously returned early when merge_pairs was truthy, silently skipping
        # this check — causing IndexError in execute(). Fix for #535.
        if len(plugin_input.checkpoint_paths) < 2:
            errors.append(
                f"Model merging needs at least 2 checkpoints (one pair), "
                f"got {len(plugin_input.checkpoint_paths)}"
            )
        return errors

    def execute(self, plugin_input: PluginInput) -> PluginOutput:
        if len(plugin_input.checkpoint_paths) < 2:
            msg = (
                f"Model merging requires at least 2 checkpoints, "
                f"got {len(plugin_input.checkpoint_paths)}. "
                "Call validate_inputs() before execute()."
            )
            raise ValueError(msg)
        config = plugin_input.config
        method: str = config.get("method", "slerp")
        t: float = config.get("t", 0.5)
        output_dir = Path(config.get("output_dir", "/tmp"))
        output_dir.mkdir(parents=True, exist_ok=True)
        merge_pairs: list[list[str]] = config.get("merge_pairs", [])

        model_paths: list[Path] = []
        metrics: dict[str, float] = {}

        if merge_pairs and plugin_input.run_metadata:
            # Category-based merging: match metadata to checkpoints
            category_to_sd: dict[str, dict[str, Any]] = {}
            for i, ckpt_path in enumerate(plugin_input.checkpoint_paths):
                meta = (
                    plugin_input.run_metadata[i]
                    if i < len(plugin_input.run_metadata)
                    else {}
                )
                cat = meta.get("champion_category", f"unknown_{i}")
                ckpt = torch.load(ckpt_path, weights_only=False)
                category_to_sd[cat] = ckpt["state_dict"]

            for pair in merge_pairs:
                cat1, cat2 = pair[0], pair[1]
                if cat1 not in category_to_sd or cat2 not in category_to_sd:
                    logger.warning(
                        "Merge pair (%s, %s) not found in checkpoints, skipping",
                        cat1,
                        cat2,
                    )
                    continue

                merged = _do_merge(
                    category_to_sd[cat1], category_to_sd[cat2], method=method, t=t
                )
                out_path = output_dir / f"merged_{method}_{cat1}_{cat2}.pt"
                torch.save(merged, out_path)
                model_paths.append(out_path)
                metrics[f"merge_{cat1}_{cat2}_method"] = 0.0  # tag only
                logger.info("Merged %s + %s via %s (t=%.2f)", cat1, cat2, method, t)
        else:
            # Simple: merge first two checkpoints
            ckpt1 = torch.load(plugin_input.checkpoint_paths[0], weights_only=False)
            ckpt2 = torch.load(plugin_input.checkpoint_paths[1], weights_only=False)
            merged = _do_merge(
                ckpt1["state_dict"], ckpt2["state_dict"], method=method, t=t
            )
            out_path = output_dir / f"merged_{method}.pt"
            torch.save(merged, out_path)
            model_paths.append(out_path)

        return PluginOutput(
            artifacts={"method": f"model_merging_{method}"},
            metrics=metrics,
            model_paths=model_paths,
        )


def _do_merge(
    sd1: dict[str, Any],
    sd2: dict[str, Any],
    *,
    method: str,
    t: float,
) -> dict[str, Any]:
    """Dispatch to the appropriate merge function."""
    if method == "slerp":
        return slerp_merge(sd1, sd2, t=t)
    if method == "layer_wise":
        return layer_wise_merge(sd1, sd2, layer_weights={}, method="linear")
    return linear_merge(sd1, sd2, t=t)
