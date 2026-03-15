from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class SegmentationOutput:
    """Standardized output from any segmentation model adapter."""

    prediction: Tensor  # (B, C, D, H, W) class probabilities
    logits: Tensor  # (B, C, D, H, W) raw logits before softmax
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterConfigInfo:
    """Typed configuration snapshot returned by ModelAdapter.get_config().

    Common fields are typed attributes; adapter-specific fields go in ``extras``.
    """

    family: str
    name: str
    in_channels: int | None = None
    out_channels: int | None = None
    trainable_params: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a serializable dict (common fields + extras merged)."""
        d: dict[str, Any] = {
            "family": self.family,
            "name": self.name,
        }
        if self.in_channels is not None:
            d["in_channels"] = self.in_channels
        if self.out_channels is not None:
            d["out_channels"] = self.out_channels
        if self.trainable_params is not None:
            d["trainable_params"] = self.trainable_params
        d.update(self.extras)
        return d


class ModelAdapter(ABC, nn.Module):
    """Abstract base class for pluggable segmentation model adapters.

    All segmentation models in the pipeline must implement this interface.
    This enables model-agnostic training, evaluation, and serving.
    """

    @abstractmethod
    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        """Run inference on a batch of 3D volumes.

        Parameters
        ----------
        images:
            Input tensor of shape (B, C, D, H, W).
        **kwargs:
            Model-specific parameters (e.g., prompts for SAMv3).

        Returns
        -------
        SegmentationOutput with predictions and raw logits.
        """
        ...

    @abstractmethod
    def get_config(self) -> AdapterConfigInfo:
        """Return model configuration as a typed dataclass."""
        ...

    def get_eval_roi_size(self) -> tuple[int, int, int]:
        """Return the sliding-window ROI size for full-volume evaluation.

        Override this method in adapters that require a non-standard evaluation
        patch size (e.g., SAM3 uses (512, 512, 3) to minimise encoder calls).
        The default (128, 128, 16) is appropriate for DynUNet, Mamba, VesselFM,
        and any adapter that does not override this method.

        Returns
        -------
        (H, W, D) tuple used for monai.inferers.sliding_window_inference.
        """
        return (128, 128, 16)

    def _build_output(self, logits: Tensor, architecture: str) -> SegmentationOutput:
        """Build a standardized SegmentationOutput from raw logits.

        Applies softmax along dim=1 and wraps the result with metadata.
        Subclasses with standard forward logic can call this instead of
        manually constructing SegmentationOutput.

        Parameters
        ----------
        logits:
            Raw model output tensor (B, C, D, H, W).
        architecture:
            Architecture name for metadata.

        Returns
        -------
        SegmentationOutput with softmax predictions and raw logits.
        """
        prediction = torch.softmax(logits, dim=1)
        return SegmentationOutput(
            prediction=prediction,
            logits=logits,
            metadata={"architecture": architecture},
        )

    def _build_config(self, **extras: Any) -> AdapterConfigInfo:
        """Build an AdapterConfigInfo from self.config and extras.

        Auto-populates family, name, in_channels, out_channels, and
        trainable_params from ``self.config`` and ``self.trainable_parameters()``.
        Any keyword arguments are placed in the ``extras`` dict.

        Parameters
        ----------
        **extras:
            Adapter-specific configuration fields.

        Returns
        -------
        AdapterConfigInfo with common fields auto-populated.
        """
        cfg: Any = self.config
        return AdapterConfigInfo(
            family=str(cfg.family.value),
            name=str(cfg.name),
            in_channels=int(cfg.in_channels) if cfg.in_channels is not None else None,
            out_channels=int(cfg.out_channels)
            if cfg.out_channels is not None
            else None,
            trainable_params=self.trainable_parameters(),
            extras=dict(extras),
        )

    def load_checkpoint(self, path: Path) -> None:
        """Load model weights from a checkpoint file.

        Handles both checkpoint formats:

        * **New format** (from :func:`~minivess.pipeline.multi_metric_tracker.save_metric_checkpoint`):
          A dict with keys ``"model_state_dict"``, ``"optimizer_state_dict"``,
          ``"scheduler_state_dict"``, and ``"checkpoint_metadata"``.  The
          ``"model_state_dict"`` value is the full adapter state dict (i.e.
          keys are prefixed with ``"net."`` because the trainer calls
          ``model.state_dict()``).
        * **Legacy format**: A bare ``state_dict`` where keys map directly to
          ``self.net`` parameters (no ``"net."`` prefix).

        Default implementation loads into ``self.net`` (legacy) or into
        ``self`` (new format). Override for adapters that manage weights
        differently (e.g., LoRA, CommaAdapter).

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist.
        """
        if not path.exists():
            msg = f"No checkpoint found at {path}"
            raise FileNotFoundError(msg)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            # New format: state_dict was produced by model.state_dict()
            # (full adapter), so load into self (the nn.Module adapter).
            state_dict = payload["model_state_dict"]
            self.load_state_dict(state_dict)
        else:
            # Legacy format: bare net state dict, load into self.net
            net = self.net
            assert isinstance(net, nn.Module)
            net.load_state_dict(payload)

    def save_checkpoint(self, path: Path) -> None:
        """Save model weights to a checkpoint file.

        Saves the full adapter state dict wrapped in the standard
        ``{"model_state_dict": ...}`` format. Works for both legacy
        adapters (with ``self.net``) and SAM3-style adapters that
        ARE the nn.Module.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.state_dict()}, path)

    def trainable_parameters(self) -> int:
        """Return the count of trainable parameters.

        Default implementation counts ``self.net`` parameters with
        ``requires_grad=True``.
        """
        net = self.net
        assert isinstance(net, nn.Module)
        return sum(p.numel() for p in net.parameters() if p.requires_grad)

    def export_onnx(self, path: Path, example_input: Tensor) -> None:
        """Export the model to ONNX format.

        Wraps the adapter in a thin module that returns raw logits tensor
        instead of SegmentationOutput (which ONNX tracing cannot handle).
        Uses legacy TorchScript exporter with opset 17.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.eval()

        adapter = self

        class _LogitsWrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.adapter = adapter

            def forward(self, x: Tensor) -> Tensor:
                result: Tensor = self.adapter(x).logits
                return result

        wrapper = _LogitsWrapper()
        wrapper.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                wrapper,
                (example_input,),
                str(path),
                input_names=["images"],
                output_names=["logits"],
                opset_version=17,
                dynamo=False,
            )
