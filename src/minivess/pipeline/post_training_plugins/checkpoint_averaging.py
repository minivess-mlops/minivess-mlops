"""Checkpoint averaging post-training plugin.

Wraps ``minivess.ensemble.model_soup.uniform_checkpoint_average()`` for per-loss
and cross-loss weight averaging of training checkpoints.

References:
    - Wortsman et al. (2022), "Model Soups: Averaging Weights of Multiple
      Fine-tuned Models Improves Accuracy without Increasing Inference Time"
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from minivess.ensemble.model_soup import uniform_checkpoint_average
from minivess.pipeline.post_training_plugin import PluginInput, PluginOutput

logger = logging.getLogger(__name__)


class CheckpointAveragingPlugin:
    """Checkpoint averaging plugin — uniform weight averaging of checkpoints."""

    @property
    def name(self) -> str:
        return "checkpoint_averaging"

    @property
    def requires_calibration_data(self) -> bool:
        return False

    def validate_inputs(self, plugin_input: PluginInput) -> list[str]:
        errors: list[str] = []
        if not plugin_input.checkpoint_paths:
            errors.append("No checkpoint paths provided")
        return errors

    def execute(self, plugin_input: PluginInput) -> PluginOutput:
        config = plugin_input.config
        per_loss = config.get("per_loss", True)
        cross_loss = config.get("cross_loss", False)
        output_dir = Path(config.get("output_dir", "/tmp"))
        output_dir.mkdir(parents=True, exist_ok=True)

        model_paths: list[Path] = []
        metrics: dict[str, float] = {}

        # Load all state dicts
        ckpt_data: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for i, ckpt_path in enumerate(plugin_input.checkpoint_paths):
            # SECURITY: weights_only=False -- self-produced checkpoint (model_state_dict,
            # state_dict). See trivy-litellm-secops-double-checking.md
            ckpt = torch.load(ckpt_path, weights_only=False)
            meta = (
                plugin_input.run_metadata[i]
                if i < len(plugin_input.run_metadata)
                else {}
            )
            ckpt_data.append((ckpt, meta))

        if per_loss:
            # Group checkpoints by loss type
            by_loss: dict[str, list[dict[str, torch.Tensor]]] = defaultdict(list)
            for ckpt, meta in ckpt_data:
                loss_type = meta.get("loss_type", "unknown")
                by_loss[loss_type].append(
                    ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                )

            for loss_type, state_dicts in sorted(by_loss.items()):
                averaged = uniform_checkpoint_average(state_dicts)
                out_path = output_dir / f"avg_{loss_type}.pt"
                torch.save(averaged, out_path)
                model_paths.append(out_path)
                metrics[f"avg_{loss_type}_n_checkpoints"] = float(len(state_dicts))
                logger.info(
                    "Checkpoint averaging per-loss: %s (%d checkpoints)",
                    loss_type,
                    len(state_dicts),
                )

        if cross_loss:
            all_sds = [
                ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                for ckpt, _ in ckpt_data
            ]
            averaged = uniform_checkpoint_average(all_sds)
            out_path = output_dir / "avg_cross_loss.pt"
            torch.save(averaged, out_path)
            model_paths.append(out_path)
            metrics["avg_cross_loss_n_checkpoints"] = float(len(all_sds))
            logger.info("Checkpoint averaging cross-loss: %d checkpoints", len(all_sds))

        return PluginOutput(
            artifacts={"method": "checkpoint_averaging"},
            metrics=metrics,
            model_paths=model_paths,
        )
