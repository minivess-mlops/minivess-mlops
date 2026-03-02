"""Model builder factory — dispatches ModelConfig to concrete adapters.

Provides a single ``build_adapter(config)`` function that returns the
correct ``ModelAdapter`` implementation based on ``config.family``.

Lazy imports ensure that SAM2 dependencies are only loaded when a
SAM3 variant is actually requested.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minivess.adapters.base import ModelAdapter
    from minivess.config.models import ModelConfig

logger = logging.getLogger(__name__)


def build_adapter(config: ModelConfig) -> ModelAdapter:
    """Build a ModelAdapter from a ModelConfig.

    Dispatches on ``config.family`` to the correct adapter class.
    Uses lazy imports so that SAM2 dependencies are not required
    when only using MONAI models.

    Parameters
    ----------
    config:
        Model configuration specifying family and architecture params.

    Returns
    -------
    Concrete ModelAdapter instance.

    Raises
    ------
    ValueError
        If the model family is not supported.
    """
    from minivess.config.models import ModelFamily

    family = config.family

    if family == ModelFamily.MONAI_DYNUNET:
        from minivess.adapters.dynunet import DynUNetAdapter

        return DynUNetAdapter(config)

    if family == ModelFamily.MONAI_SEGRESNET:
        from minivess.adapters.segresnet import SegResNetAdapter

        return SegResNetAdapter(config)

    if family == ModelFamily.MONAI_SWINUNETR:
        from minivess.adapters.swinunetr import SwinUNETRAdapter

        return SwinUNETRAdapter(config)

    if family == ModelFamily.MONAI_VISTA3D:
        from minivess.adapters.vista3d import Vista3dAdapter

        return Vista3dAdapter(config)

    if family == ModelFamily.VESSEL_FM:
        from minivess.adapters.vesselfm import VesselFMAdapter

        return VesselFMAdapter(config)

    if family == ModelFamily.COMMA_MAMBA:
        from minivess.adapters.comma import CommaAdapter

        return CommaAdapter(config)

    if family == ModelFamily.SAM3_VANILLA:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        return Sam3VanillaAdapter(config)

    if family == ModelFamily.SAM3_TOPOLORA:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        return Sam3TopoLoraAdapter(config)

    if family == ModelFamily.SAM3_HYBRID:
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        return Sam3HybridAdapter(config)

    msg = (
        f"Unsupported model family '{family}'. "
        f"Available: {[f.value for f in ModelFamily]}"
    )
    raise ValueError(msg)
