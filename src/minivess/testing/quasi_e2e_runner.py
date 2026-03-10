"""Quasi-E2E test runner infrastructure.

Provides helper functions for building models, losses, and running single
forward-backward passes for each (model, loss) combination. Used by
pytest parametrized tests to verify training mechanics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from minivess.testing.capability_discovery import TestCombination

logger = logging.getLogger(__name__)

# Model-specific minimum test patch sizes.
# SAM3: 14×14 kernel on 2D slices → all dims >= 16
# VesselFM: divisible by 32
# Default: (32, 32, 8) is fine for dynunet/mamba
_SAM3_MODELS = frozenset({"sam3_vanilla", "sam3_topolora", "sam3_hybrid"})
_VESSELFM_MODELS = frozenset({"vesselfm"})

_PATCH_MAP: dict[str, tuple[int, int, int]] = {
    # SAM3: 2D slices need >= 14 in both H,W; D >= 16 for 3D context
    "sam3_vanilla": (16, 32, 32),
    "sam3_topolora": (16, 32, 32),
    "sam3_hybrid": (16, 32, 32),
    # VesselFM: 6 encoder levels → needs >= 64 per dim (divisor=32)
    "vesselfm": (64, 64, 64),
    # SwinUNETR: 5 stride-2 levels → divisor=32, all dims >= 32
    "swinunetr": (32, 32, 32),
    # divisor=16 models: 4 stride-2 levels → D >= 16
    "attentionunet": (32, 32, 16),
    "unetr": (32, 32, 16),
    "mamba": (32, 32, 16),
    "ulike_mamba": (32, 32, 16),
}
_DEFAULT_PATCH: tuple[int, int, int] = (32, 32, 8)


def _get_patch_size_for_model(model_name: str) -> tuple[int, int, int]:
    """Return appropriate test patch size for a model family."""
    return _PATCH_MAP.get(model_name, _DEFAULT_PATCH)


def generate_test_ids(combos: list[TestCombination]) -> list[str]:
    """Generate human-readable test IDs from combinations.

    Format: ``"model__loss"`` — deterministic and unique.
    """
    return [f"{c.model}__{c.loss}" for c in combos]


def _model_name_to_family(model_name: str) -> str:
    """Map model name string to ModelFamily enum value.

    In the current design, model names in method_capabilities.yaml
    exactly match ModelFamily enum values (both are lowercase strings).
    """
    from minivess.config.models import ModelFamily

    for member in ModelFamily:
        if member.value == model_name:
            return member.value
    msg = f"No ModelFamily enum for model name {model_name!r}"
    raise ValueError(msg)


def build_model_for_test(
    model_name: str,
    *,
    in_channels: int = 1,
    out_channels: int = 2,
) -> nn.Module:
    """Instantiate a model adapter's nn.Module for testing.

    Uses ``build_adapter()`` with appropriate config. SAM3 models
    require real pretrained weights — if SAM3 is not installed,
    ``RuntimeError`` is raised by ``build_adapter()``.

    Parameters
    ----------
    model_name:
        Model name (e.g., ``"dynunet"``, ``"sam3_vanilla"``).
    in_channels:
        Input channels (default 1 for grayscale).
    out_channels:
        Output channels/classes (default 2 for binary seg).

    Returns
    -------
    nn.Module from the adapter.

    Raises
    ------
    ValueError
        If model name is not a valid ModelFamily.
    """
    from minivess.adapters.model_builder import build_adapter
    from minivess.config.models import ModelConfig, ModelFamily

    # Validate model name exists in enum
    family_value = _model_name_to_family(model_name)
    family = ModelFamily(family_value)

    config = ModelConfig(
        family=family,
        name=model_name,
        in_channels=in_channels,
        out_channels=out_channels,
    )

    return build_adapter(config)


def build_loss_for_test(
    loss_name: str,
    *,
    num_classes: int = 2,
) -> nn.Module:
    """Build a loss function module for testing.

    Parameters
    ----------
    loss_name:
        Loss name (e.g., ``"cbdice_cldice"``, ``"dice_ce"``).
    num_classes:
        Number of output classes (default 2).

    Returns
    -------
    nn.Module loss function.
    """
    from minivess.pipeline.loss_functions import build_loss_function

    return build_loss_function(loss_name, num_classes=num_classes)


def run_single_forward_backward(
    *,
    model: nn.Module,
    loss_fn: nn.Module,
    patch_size: tuple[int, int, int] = (32, 32, 8),
    batch_size: int = 1,
    in_channels: int = 1,
    num_classes: int = 2,
) -> dict[str, Any]:
    """Run one forward + backward pass with random data.

    Creates random input/target tensors, runs forward pass through
    the model, computes loss, and runs backward pass. Returns
    diagnostic information.

    Parameters
    ----------
    model:
        The model to test.
    loss_fn:
        The loss function to test.
    patch_size:
        3D patch size (H, W, D).
    batch_size:
        Batch size (default 1).
    in_channels:
        Input channels.
    num_classes:
        Number of classes.

    Returns
    -------
    Dict with ``loss_value``, ``output_shape``, ``grad_norm``.
    """
    device = torch.device("cpu")
    model = model.to(device)
    model.train()

    # Random input: (B, C, H, W, D)
    x = torch.randn(
        batch_size, in_channels, *patch_size, device=device, dtype=torch.float32
    )

    # Random target: (B, 1, H, W, D) integer labels
    target = torch.randint(
        0, num_classes, (batch_size, 1, *patch_size), device=device
    ).long()

    # Forward pass — adapter returns SegmentationOutput or similar
    output = model(x)

    # Extract logits tensor from SegmentationOutput (if applicable)
    if hasattr(output, "logits"):
        logits = output.logits
    elif isinstance(output, torch.Tensor):
        logits = output
    else:
        msg = f"Unexpected output type: {type(output)}"
        raise TypeError(msg)

    # Compute loss — MONAI losses expect (B, C, H, W, D) pred + (B, 1, H, W, D) target
    loss = loss_fn(logits, target)

    # Backward pass
    loss.backward()

    # Collect gradient norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = total_norm**0.5

    return {
        "loss_value": loss.item(),
        "output_shape": tuple(logits.shape),
        "grad_norm": grad_norm,
    }
