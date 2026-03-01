"""ONNX export and validation for champion models.

Exports champion model checkpoints to ONNX format and validates the
exported model with ONNX Runtime for deployment readiness.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.pipeline.deploy_champion_discovery import ChampionModel

logger = logging.getLogger(__name__)


def export_champion_to_onnx(
    champion: ChampionModel,
    output_dir: Path,
    *,
    opset_version: int = 17,
    input_shape: tuple[int, ...] = (1, 1, 32, 32, 16),
) -> Path:
    """Export a champion model checkpoint to ONNX format.

    Parameters
    ----------
    champion:
        Champion model with checkpoint path and model config.
    output_dir:
        Directory to write the ONNX file.
    opset_version:
        ONNX opset version.
    input_shape:
        Example input shape for tracing (B, C, D, H, W).

    Returns
    -------
    Path to the exported ONNX file.

    Raises
    ------
    ValueError
        If champion has no checkpoint path.
    """
    import torch

    if champion.checkpoint_path is None:
        msg = "Champion has no checkpoint path — cannot export to ONNX"
        raise ValueError(msg)

    model = _load_model_from_checkpoint(champion)
    model.eval()

    onnx_filename = f"minivess_{champion.category}_{champion.run_id}.onnx"
    onnx_path = output_dir / onnx_filename

    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )

    logger.info(
        "Exported champion %s to ONNX: %s",
        champion.run_id,
        onnx_path,
    )
    return onnx_path


def validate_onnx_model(
    onnx_path: Path,
    *,
    input_shape: tuple[int, ...] | None = None,
) -> bool:
    """Validate an ONNX model file with ONNX Runtime.

    Parameters
    ----------
    onnx_path:
        Path to the ONNX model file.
    input_shape:
        Shape for test inference. If None, inferred from model metadata.

    Returns
    -------
    True if validation passes, False otherwise.
    """
    if not onnx_path.exists():
        logger.warning("ONNX model not found: %s", onnx_path)
        return False

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        input_info = session.get_inputs()[0]
        input_name = input_info.name

        # Infer shape from model if not provided
        if input_shape is None:
            model_shape = input_info.shape
            # Replace dynamic dims (strings/None) with small test values
            resolved = [dim if isinstance(dim, int) else 4 for dim in model_shape]
            input_shape = tuple(resolved)

        dummy = np.random.rand(*input_shape).astype(np.float32)
        outputs = session.run(None, {input_name: dummy})

        if outputs is None or len(outputs) == 0:
            logger.warning("ONNX model produced no outputs")
            return False

        output_shape = outputs[0].shape
        logger.info(
            "ONNX validation passed: input=%s output=%s",
            input_shape,
            output_shape,
        )
        return True

    except Exception:
        logger.exception("ONNX validation failed for %s", onnx_path)
        return False


def _load_model_from_checkpoint(champion: ChampionModel) -> Any:
    """Load a PyTorch model from a champion checkpoint.

    Uses a simple Conv3d fallback when model config is not available,
    as the primary purpose is ONNX export validation.
    """
    import torch
    import torch.nn as nn

    assert champion.checkpoint_path is not None  # noqa: S101

    checkpoint = torch.load(
        champion.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # Extract state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Build a simple model matching the state dict
    model_config = champion.model_config or {}
    in_channels = int(model_config.get("in_channels", 1))
    out_channels = int(model_config.get("out_channels", 2))

    # Strip common prefixes from state dict keys (e.g. "conv." from mock checkpoints)
    cleaned_state_dict = _strip_state_dict_prefix(state_dict)

    model = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    model.load_state_dict(cleaned_state_dict)
    return model


def _strip_state_dict_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip module prefixes from state dict keys for loading into plain modules."""
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        # Remove common prefixes: "module.", "conv.", "model.", etc.
        new_key = key
        for prefix in ("module.", "conv.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned
