"""SWAG post-training plugin (Maddox et al. 2019).

Resumes training with SWALR schedule to collect SWAG posterior statistics.
Unlike checkpoint averaging (post-hoc), SWAG requires active training
with gradient computation and second-moment tracking.

Reference: Maddox et al. (2019), "A Simple Baseline for Bayesian Inference
in Deep Learning" (https://arxiv.org/abs/1902.02476)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.optim.swa_utils import SWALR

from minivess.ensemble.swag import SWAGModel
from minivess.pipeline.post_training_plugin import PluginInput, PluginOutput

logger = logging.getLogger(__name__)


@torch.no_grad()
def _update_bn_with_dict_loader(
    loader: Any,
    model: torch.nn.Module,
    *,
    device: torch.device | str | None = None,
) -> None:
    """Like ``torch.optim.swa_utils.update_bn`` but handles dict-based loaders.

    MONAI's ``ThreadDataLoader`` yields ``dict[str, Tensor]`` with an ``"image"``
    key, whereas PyTorch's ``update_bn`` expects plain tensor batches.
    """
    from torch.nn.modules.batchnorm import _BatchNorm

    momenta: dict[_BatchNorm, float | None] = {}
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            if module.running_mean is not None:
                module.running_mean = torch.zeros_like(module.running_mean)
            if module.running_var is not None:
                module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta:
        module.momentum = None  # cumulative moving average
        if module.num_batches_tracked is not None:
            module.num_batches_tracked.zero_()

    for batch in loader:
        if isinstance(batch, dict):
            inp = batch["image"]
        elif isinstance(batch, list | tuple):
            inp = batch[0]
        else:
            inp = batch
        if device is not None:
            inp = inp.to(device)
        model(inp)  # output ignored — only updating BN running stats

    for module, momentum in momenta.items():
        module.momentum = momentum
    model.train(was_training)


class SWAGPlugin:
    """SWAG plugin — Bayesian posterior approximation via resumed training."""

    @property
    def name(self) -> str:
        return "swag"

    @property
    def requires_calibration_data(self) -> bool:
        return True

    def validate_inputs(self, plugin_input: PluginInput) -> list[str]:
        errors: list[str] = []
        if not plugin_input.checkpoint_paths:
            errors.append("SWAG requires at least one checkpoint path")
        if plugin_input.calibration_data is None:
            errors.append("SWAG requires calibration_data with 'train_loader' key")
        elif "train_loader" not in plugin_input.calibration_data:
            errors.append(
                "SWAG calibration_data must contain 'train_loader' "
                "(PyTorch DataLoader for resumed training)"
            )
        return errors

    def execute(self, plugin_input: PluginInput) -> PluginOutput:
        config = plugin_input.config
        swa_lr: float = config.get("swa_lr", 0.01)
        swa_epochs: int = config.get("swa_epochs", 10)
        max_rank: int = config.get("max_rank", 20)
        do_update_bn: bool = config.get("update_bn", True)
        output_dir = Path(config.get("output_dir", "/tmp"))
        output_dir.mkdir(parents=True, exist_ok=True)

        assert plugin_input.calibration_data is not None
        train_loader = plugin_input.calibration_data["train_loader"]

        # Load base model from first checkpoint
        ckpt_path = plugin_input.checkpoint_paths[0]
        ckpt = torch.load(ckpt_path, weights_only=False)

        # Get the model - expect either a full model or state_dict
        if "model" in ckpt:
            base_model = ckpt["model"]
        elif "model_state_dict" in ckpt or "state_dict" in ckpt:
            # Need an architecture to load state_dict into
            # The caller must provide it via calibration_data["model"]
            if "model" not in plugin_input.calibration_data:
                raise ValueError(
                    "SWAG requires calibration_data['model'] (nn.Module) when "
                    "checkpoint contains only state_dict"
                )
            base_model = plugin_input.calibration_data["model"]
            sd_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
            base_model.load_state_dict(ckpt[sd_key])
        else:
            # Assume the checkpoint IS the state dict
            if "model" not in plugin_input.calibration_data:
                raise ValueError(
                    "SWAG requires calibration_data['model'] (nn.Module) when "
                    "checkpoint is a raw state_dict"
                )
            base_model = plugin_input.calibration_data["model"]
            base_model.load_state_dict(ckpt)

        device = next(base_model.parameters()).device

        # Create SWAG wrapper
        swag = SWAGModel(base_model, max_rank=max_rank)

        # Set up optimizer + SWALR schedule
        optimizer = torch.optim.SGD(base_model.parameters(), lr=swa_lr)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

        # Resume training for swa_epochs
        base_model.train()
        for epoch in range(swa_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                # Expect batch to be (inputs, targets) or dict with "image"/"label"
                if isinstance(batch, dict):
                    inputs = batch["image"].to(device)
                    targets = batch["label"].to(device)
                elif isinstance(batch, list | tuple):
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                else:
                    raise TypeError(f"Unexpected batch type: {type(batch)}")

                optimizer.zero_grad()
                raw_outputs = base_model(inputs)
                # ModelAdapter returns SegmentationOutput; extract logits
                outputs = (
                    raw_outputs.logits
                    if hasattr(raw_outputs, "logits")
                    else raw_outputs
                )

                # Handle multi-class (out_channels>1) vs binary output
                if outputs.shape == targets.shape:
                    # Same shape (e.g. both (B, C, D, H, W)) → BCE
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        outputs, targets.float()
                    )
                elif outputs.ndim >= 3 and outputs.shape[1] > 1:
                    # Multi-class logits vs integer labels → cross_entropy
                    if targets.ndim == outputs.ndim:
                        targets_ce = targets.squeeze(1).long()
                    else:
                        targets_ce = targets.long()
                    loss = torch.nn.functional.cross_entropy(outputs, targets_ce)
                else:
                    # Binary logits (B, 1, ...) → BCE with float labels
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        outputs.squeeze(1) if outputs.ndim > targets.ndim else outputs,
                        targets.squeeze(1).float()
                        if targets.ndim > 1
                        else targets.float(),
                    )
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            swa_scheduler.step()

            # Collect SWAG statistics after each epoch
            swag.collect_model(base_model)

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(
                "SWAG epoch %d/%d: loss=%.4f, lr=%.6f",
                epoch + 1,
                swa_epochs,
                avg_loss,
                optimizer.param_groups[0]["lr"],
            )

        # Update BatchNorm statistics
        if do_update_bn:
            swag._load_mean()  # Load mean weights first
            # PyTorch's update_bn expects (tensor,) batches, but MONAI loaders
            # yield dicts.  Wrap with an adapter that extracts the "image" key.
            _update_bn_with_dict_loader(train_loader, base_model, device=device)
            logger.info("SWAG: BatchNorm statistics updated")

        # Save SWAG model
        swag_path = output_dir / "swag_model.pt"
        swag.save(swag_path)

        metrics: dict[str, float] = {
            "swag_epochs": float(swa_epochs),
            "swag_lr": swa_lr,
            "swag_max_rank": float(max_rank),
            "swag_n_models_collected": float(swag.n_models_collected),
            "swag_n_deviations": float(len(swag._deviations)),
        }

        return PluginOutput(
            artifacts={"method": "swag", "max_rank": max_rank},
            metrics=metrics,
            model_paths=[swag_path],
        )
