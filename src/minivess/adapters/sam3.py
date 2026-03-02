"""SAM3 adapter stub — DEPRECATED, use sam3_vanilla/sam3_topolora/sam3_hybrid.

MedSAM3 (Liu et al., 2025) is a medical image variant of the Segment
Anything Model with concept-aware prompting. This adapter is a
deprecated placeholder. Use the concrete SAM3 variant adapters instead:

- ``Sam3VanillaAdapter`` — frozen SAM2 encoder + trainable decoder
- ``Sam3TopoLoraAdapter`` — SAM2 + LoRA + topology-aware loss
- ``Sam3HybridAdapter`` — SAM2 features + DynUNet 3D decoder

.. deprecated:: 2.0.0
    Use :class:`~minivess.adapters.sam3_vanilla.Sam3VanillaAdapter` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from minivess.adapters.base import AdapterConfigInfo, ModelAdapter, SegmentationOutput

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from minivess.config.models import ModelConfig


class Sam3Adapter(ModelAdapter):
    """SAM3/MedSAM3 adapter (DEPRECATED).

    .. deprecated:: 2.0.0
        Use :class:`Sam3VanillaAdapter`, :class:`Sam3TopoLoraAdapter`,
        or :class:`Sam3HybridAdapter` instead.

    Parameters
    ----------
    config:
        ModelConfig with SAM3_LORA family and LoRA fields.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        import warnings

        warnings.warn(
            "Sam3Adapter is deprecated. Use Sam3VanillaAdapter, "
            "Sam3TopoLoraAdapter, or Sam3HybridAdapter instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        msg = (
            "Sam3Adapter is deprecated — use Sam3VanillaAdapter, "
            "Sam3TopoLoraAdapter, or Sam3HybridAdapter instead."
        )
        raise RuntimeError(msg)

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        msg = "SAM3 forward not implemented"
        raise NotImplementedError(msg)

    def get_config(self) -> AdapterConfigInfo:
        return AdapterConfigInfo(
            family=self.config.family.value,
            name=self.config.name,
            extras={"lora_rank": self.config.lora_rank},
        )

    def load_checkpoint(self, path: Path) -> None:
        pass

    def save_checkpoint(self, path: Path) -> None:
        pass

    def trainable_parameters(self) -> int:
        return 0

    def export_onnx(self, path: Path, example_input: Tensor) -> None:
        msg = "SAM3 ONNX export not implemented"
        raise NotImplementedError(msg)
