"""Model builder factory and wrapper composition.

Provides:
- ``build_adapter(config)`` — dispatches ModelConfig to concrete adapters.
- ``apply_wrappers(model, wrappers)`` — applies config-driven wrappers (TFFM, multi-task).

Lazy imports ensure that SAM3 dependencies are only loaded when a
SAM3 variant is actually requested.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch.nn as nn  # noqa: TC002 — used at runtime in function signature

if TYPE_CHECKING:
    from minivess.adapters.base import ModelAdapter
    from minivess.config.models import ModelConfig

logger = logging.getLogger(__name__)


def _sam3_package_available() -> bool:
    """Check if SAM3 or compatible transformers is installed."""
    try:
        import sam3  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        from transformers import Sam3Model  # noqa: F401

        return True
    except (ImportError, AttributeError):
        pass
    return False


_SAM3_INSTALL_INSTRUCTIONS = """
════════════════════════════════════════════════════════════════
 SAM3 IS NOT INSTALLED — real pretrained weights required
════════════════════════════════════════════════════════════════

 Step 1: Request model access (Meta gated model — usually instant):
         https://huggingface.co/facebook/sam3
         → click "Agree and access repository"

 Step 2: Authenticate your HuggingFace token:
         uv run huggingface-cli login

 Step 3a: Install via Transformers (recommended):
          uv add "transformers>=4.50"
          uv run python -c "from transformers import Sam3Model; print('OK')"

 Step 3b: Or install from Meta's GitHub source:
          uv add "sam3 @ git+https://github.com/facebookresearch/sam3.git"

 ── For pipeline testing WITHOUT real SAM3 weights ─────────────
 Set in your model YAML:  architecture_params.pretrained: false
 This uses a random-weight stub. Results will be meaningless.
════════════════════════════════════════════════════════════════
"""


def _auto_stub_sam3(config: ModelConfig, kwargs: dict[str, Any]) -> None:
    """Validate SAM3 availability; raise loudly if pretrained weights are needed.

    Stub mode is ONLY activated when explicitly requested:
    - ``use_stub=True`` passed to ``build_adapter()`` (test fixtures)
    - ``architecture_params.pretrained: false`` in the model config (deliberate baseline)

    Silent fallback to a random-weight stub is NEVER acceptable when a
    pretrained SAM3 is expected — it produces meaningless training metrics
    while appearing to succeed.

    Raises
    ------
    RuntimeError
        When SAM3 package is not installed and pretrained weights are required.
    """
    if "use_stub" in kwargs:
        # Explicit caller request (e.g. test fixture) — honour it
        return

    pretrained = config.architecture_params.get("pretrained", True)
    if not pretrained:
        kwargs["use_stub"] = True
        logger.info(
            "SAM3 pretrained=false: using stub encoder (random weights, deliberate)"
        )
        return

    # pretrained=True (default) — real SAM3 package required
    if not _sam3_package_available():
        logger.error(_SAM3_INSTALL_INSTRUCTIONS)
        msg = (
            "SAM3 package not installed. "
            "See installation instructions logged above (ERROR level)."
        )
        raise RuntimeError(msg)

    # SAM3 is available — verify HF token before triggering a download
    from minivess.utils.hf_auth import require_hf_token

    require_hf_token("facebook/sam3")


def build_adapter(config: ModelConfig, **kwargs: Any) -> ModelAdapter:
    """Build a ModelAdapter from a ModelConfig.

    Dispatches on ``config.family`` to the correct adapter class.
    Uses lazy imports so that SAM3 dependencies are not required
    when only using MONAI models.

    Parameters
    ----------
    config:
        Model configuration specifying family and architecture params.
    **kwargs:
        Extra keyword arguments forwarded to the adapter constructor
        (e.g., ``use_stub=True`` for SAM3 adapters in tests).

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

    if family == ModelFamily.VESSEL_FM:
        from minivess.adapters.vesselfm import VesselFMAdapter

        pretrained = config.architecture_params.get("pretrained", False)
        return VesselFMAdapter(config, pretrained=pretrained)

    if family == ModelFamily.COMMA_MAMBA:
        from minivess.adapters.comma import CommaAdapter

        return CommaAdapter(config)

    if family == ModelFamily.ULIKE_MAMBA:
        from minivess.adapters.mamba import MambaAdapter

        return MambaAdapter(config)

    if family == ModelFamily.SAM3_VANILLA:
        from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

        _auto_stub_sam3(config, kwargs)
        return Sam3VanillaAdapter(config, **kwargs)

    if family == ModelFamily.SAM3_TOPOLORA:
        from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

        _auto_stub_sam3(config, kwargs)
        return Sam3TopoLoraAdapter(config, **kwargs)

    if family == ModelFamily.SAM3_HYBRID:
        from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

        _auto_stub_sam3(config, kwargs)
        return Sam3HybridAdapter(config, **kwargs)

    msg = (
        f"Unsupported model family '{family}'. "
        f"Available: {[f.value for f in ModelFamily]}"
    )
    raise ValueError(msg)


def apply_wrappers(
    model: nn.Module,
    wrappers: list[dict[str, Any]],
) -> nn.Module:
    """Apply a sequence of config-driven wrappers to a base model.

    Each wrapper dict must have a "type" key. Supported types:
    - "tffm": TFFMWrapper (grid_size, hidden_dim, n_heads, k_neighbors)
    - "multitask": MultiTaskAdapter (auxiliary_heads list)

    Args:
        model: Base model to wrap.
        wrappers: List of wrapper config dicts, applied in order.

    Returns:
        Wrapped model (or original if no wrappers).
    """
    for wrapper_cfg in wrappers:
        wrapper_type = wrapper_cfg["type"]

        if wrapper_type == "tffm":
            from minivess.adapters.tffm_wrapper import TFFMWrapper

            model = TFFMWrapper(
                base_model=model,  # type: ignore[arg-type]
                grid_size=wrapper_cfg.get("grid_size", 8),
                hidden_dim=wrapper_cfg.get("hidden_dim", 32),
                n_heads=wrapper_cfg.get("n_heads", 4),
                k_neighbors=wrapper_cfg.get("k_neighbors", 8),
            )
            logger.info(
                "Applied TFFMWrapper (grid=%d)", wrapper_cfg.get("grid_size", 8)
            )

        elif wrapper_type == "multitask":
            from minivess.adapters.multitask_adapter import (
                AuxHeadConfig,
                MultiTaskAdapter,
            )

            aux_heads = wrapper_cfg.get("auxiliary_heads", [])
            aux_configs = [
                AuxHeadConfig(
                    name=h["name"],
                    head_type=h["type"],
                    out_channels=h.get("out_channels", 1),
                )
                for h in aux_heads
            ]
            model = MultiTaskAdapter(base_model=model, aux_head_configs=aux_configs)
            logger.info("Applied MultiTaskAdapter (%d aux heads)", len(aux_configs))

        else:
            msg = f"Unknown wrapper type: {wrapper_type!r}. Supported: tffm, multitask"
            raise ValueError(msg)

    return model
