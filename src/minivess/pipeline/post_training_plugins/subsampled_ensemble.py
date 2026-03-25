"""Subsampled ensemble post-training plugin.

Produces M independent averaged models by subsampling checkpoints.
Each model averages a different random subset, then predictions
are ensembled across basins at inference time.

This is purely post-hoc — no training-time modifications needed.

References:
    - Wortsman et al. (2022), "Model Soups: Averaging Weights of Multiple
      Fine-tuned Models Improves Accuracy without Increasing Inference Time"
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path

import torch

from minivess.ensemble.model_soup import uniform_checkpoint_average
from minivess.pipeline.post_training_plugin import PluginInput, PluginOutput

logger = logging.getLogger(__name__)


class SubsampledEnsemblePlugin:
    """Subsampled ensemble plugin — M independent averaged models from checkpoint subsets."""

    @property
    def name(self) -> str:
        return "subsampled_ensemble"

    @property
    def requires_calibration_data(self) -> bool:
        return False

    def validate_inputs(self, plugin_input: PluginInput) -> list[str]:
        errors: list[str] = []
        n_ckpts = len(plugin_input.checkpoint_paths)
        fraction = plugin_input.config.get("subsample_fraction", 0.7)
        subset_size = max(1, math.floor(n_ckpts * fraction))

        if n_ckpts < 2:
            errors.append(
                f"Subsampled ensemble needs at least 2 checkpoints, got {n_ckpts}"
            )
        if n_ckpts > 0 and subset_size < 1:
            errors.append(
                f"Subsample fraction {fraction} with {n_ckpts} checkpoints "
                f"yields subset_size=0"
            )
        return errors

    def execute(self, plugin_input: PluginInput) -> PluginOutput:
        config = plugin_input.config
        n_models: int = config.get("n_models", 3)
        fraction: float = config.get("subsample_fraction", 0.7)
        seed: int = config.get("seed", 42)
        output_dir = Path(config.get("output_dir", "/tmp"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load all state dicts once
        all_sds = []
        for ckpt_path in plugin_input.checkpoint_paths:
            # SECURITY: weights_only=False -- self-produced checkpoint
            # (model_state_dict, state_dict). See trivy-litellm-secops-double-checking.md
            ckpt = torch.load(ckpt_path, weights_only=False)
            all_sds.append(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))

        n_total = len(all_sds)
        subset_size = max(1, math.floor(n_total * fraction))

        rng = random.Random(seed)
        model_paths: list[Path] = []
        metrics: dict[str, float] = {}

        for i in range(n_models):
            # Subsample indices
            if subset_size >= n_total:
                indices = list(range(n_total))
            else:
                indices = sorted(rng.sample(range(n_total), subset_size))

            subset_sds = [all_sds[j] for j in indices]
            averaged = uniform_checkpoint_average(subset_sds)

            out_path = output_dir / f"subsampled_ensemble_model_{i}.pt"
            torch.save(averaged, out_path)
            model_paths.append(out_path)
            metrics[f"subsampled_ensemble_{i}_n_checkpoints"] = float(len(subset_sds))
            logger.info(
                "Subsampled ensemble model %d/%d: averaged %d/%d checkpoints (indices=%s)",
                i + 1,
                n_models,
                len(subset_sds),
                n_total,
                indices,
            )

        metrics["subsampled_ensemble_n_models"] = float(n_models)
        return PluginOutput(
            artifacts={"method": "subsampled_ensemble"},
            metrics=metrics,
            model_paths=model_paths,
        )
