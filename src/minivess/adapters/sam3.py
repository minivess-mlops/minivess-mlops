"""SAM3 adapter stub — exploratory, requires segment-anything package.

MedSAM3 (Liu et al., 2025) is a medical image variant of the Segment
Anything Model with concept-aware prompting. This adapter is a
placeholder that requires the external `segment-anything` package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from minivess.adapters.base import ModelAdapter, SegmentationOutput

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from minivess.config.models import ModelConfig


class Sam3Adapter(ModelAdapter):
    """SAM3/MedSAM3 adapter (exploratory).

    Raises ImportError if `segment-anything` is not installed.

    Parameters
    ----------
    config:
        ModelConfig with SAM3_LORA family and LoRA fields.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        try:
            import segment_anything  # noqa: F401
        except ImportError:
            msg = (
                "SAM3 adapter requires the 'segment-anything' package. "
                "Install with: pip install segment-anything"
            )
            raise ImportError(msg) from None

        msg = "SAM3 adapter is exploratory — full implementation pending"
        raise RuntimeError(msg)

    def forward(self, images: Tensor, **kwargs: Any) -> SegmentationOutput:
        msg = "SAM3 forward not implemented"
        raise NotImplementedError(msg)

    def get_config(self) -> dict[str, Any]:
        return {
            "family": self.config.family.value,
            "name": self.config.name,
            "lora_rank": self.config.lora_rank,
        }

    def load_checkpoint(self, path: Path) -> None:
        pass

    def save_checkpoint(self, path: Path) -> None:
        pass

    def trainable_parameters(self) -> int:
        return 0

    def export_onnx(self, path: Path, example_input: Tensor) -> None:
        msg = "SAM3 ONNX export not implemented"
        raise NotImplementedError(msg)
