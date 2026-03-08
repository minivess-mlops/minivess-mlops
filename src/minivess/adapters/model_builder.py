"""Model builder factory and wrapper composition.

Provides:
- ``build_adapter(config)`` — dispatches ModelConfig to concrete adapters via registry.
- ``apply_wrappers(model, wrappers)`` — applies config-driven wrappers (TFFM, multi-task).

Lazy imports ensure that SAM3 dependencies are only loaded when a
SAM3 variant is actually requested. Each registration factory function
uses a local import so MONAI models never trigger SAM3 imports and vice versa.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch.nn as nn  # noqa: TC002 — used at runtime in function signature

if TYPE_CHECKING:
    from collections.abc import Callable

    from minivess.adapters.base import ModelAdapter
    from minivess.config.models import ModelConfig, ModelFamily

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps ModelFamily → zero-arg factory that returns a (config, **kwargs) → adapter callable.
# Populated by _register() calls below. Lazily imported adapter modules live inside
# each factory closure so only the requested model family is imported.
_MODEL_REGISTRY: dict[ModelFamily, Callable[..., ModelAdapter]] = {}


def _register(family: ModelFamily) -> Callable:  # type: ignore[type-arg]
    """Decorator that registers a builder function in _MODEL_REGISTRY."""

    def decorator(fn: Callable[..., ModelAdapter]) -> Callable[..., ModelAdapter]:
        _MODEL_REGISTRY[family] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# SAM3 helpers (unchanged)
# ---------------------------------------------------------------------------


def _sam3_package_available() -> bool:
    """Check if SAM3 or compatible transformers is installed."""
    try:
        import sam3  # noqa: F401

        return True
    except ImportError:
        logger.debug("Native sam3 package not found")
    try:
        from transformers import Sam3Model  # noqa: F401

        return True
    except (ImportError, AttributeError):
        logger.debug("HuggingFace transformers with Sam3Model not found")
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

════════════════════════════════════════════════════════════════
"""


def _require_sam3(config: ModelConfig, *, encoder_frozen: bool) -> None:
    """Validate SAM3 availability; raise loudly if not installed or GPU insufficient.

    SAM3 ALWAYS requires real pretrained weights. There is no stub or
    pretrained:false fallback — they were removed to prevent silent
    random-weight training (see .claude/metalearning/2026-03-02-sam3-implementation-fuckup.md).

    VRAM gate depends on whether the SAM3 encoder is frozen:

    - ``encoder_frozen=True`` (V1 Vanilla, V3 Hybrid): The encoder forward pass runs
      inside ``torch.no_grad()``, so no activation graph is stored for the 848M encoder
      params. Peak VRAM ≈ inference peak (~6 GB BF16). Gate: ``mode="inference"`` (6 GB).

    - ``encoder_frozen=False`` (V2 TopoLoRA): LoRA adapters are applied to the encoder
      FFN layers; gradients flow through all 32 ViT transformer blocks. Full activation
      graphs must be retained for backprop. Gate: ``mode="training"`` (16 GB).

    Checks (in order):
    1. SAM3 package is installed.
    2. GPU VRAM meets the mode-appropriate threshold.
    3. HuggingFace token is present (for gated model download).

    Raises
    ------
    RuntimeError
        When SAM3 package is not installed, VRAM is insufficient, or HF token missing.
    """
    if not _sam3_package_available():
        logger.error(_SAM3_INSTALL_INSTRUCTIONS)
        msg = (
            "SAM3 package not installed. "
            "See installation instructions logged above (ERROR level)."
        )
        raise RuntimeError(msg)

    from minivess.adapters.sam3_vram_check import check_sam3_vram

    vram_mode = "inference" if encoder_frozen else "training"
    check_sam3_vram(variant=config.name, mode=vram_mode)

    # SAM3 is available — verify HF token before triggering a download
    from minivess.utils.hf_auth import require_hf_token

    require_hf_token("facebook/sam3")


# ---------------------------------------------------------------------------
# MONAI adapter registrations
# ---------------------------------------------------------------------------


def _build_dynunet(config: ModelConfig, **kwargs: Any) -> ModelAdapter:
    from minivess.adapters.dynunet import DynUNetAdapter

    return DynUNetAdapter(config)


def _build_segresnet(config: ModelConfig, **kwargs: Any) -> ModelAdapter:
    from minivess.adapters.segresnet import SegResNetAdapter

    return SegResNetAdapter(config)


def _build_swinunetr(config: ModelConfig, **kwargs: Any) -> ModelAdapter:
    from minivess.adapters.swinunetr import SwinUNETRAdapter

    return SwinUNETRAdapter(config)


def _build_unetr(config: ModelConfig, **kwargs: Any) -> ModelAdapter:
    from minivess.adapters.unetr import UNETRAdapter

    return UNETRAdapter(config)


def _build_attentionunet(config: ModelConfig, **kwargs: Any) -> ModelAdapter:
    from minivess.adapters.attentionunet import AttentionUnetAdapter

    return AttentionUnetAdapter(config)


def _build_vesselfm(config: ModelConfig, **kwargs: Any) -> ModelAdapter:
    from minivess.adapters.vesselfm import VesselFMAdapter

    pretrained = config.architecture_params.get("pretrained", False)
    return VesselFMAdapter(config, pretrained=pretrained)


def _build_comma(config: ModelConfig, **kwargs: Any) -> ModelAdapter:
    from minivess.adapters.comma import CommaAdapter

    return CommaAdapter(config)


def _build_mamba(config: ModelConfig, **kwargs: Any) -> ModelAdapter:
    from minivess.adapters.mamba import MambaAdapter

    return MambaAdapter(config)


def _build_sam3_vanilla(config: ModelConfig, **kwargs: Any) -> ModelAdapter:
    from minivess.adapters.sam3_vanilla import Sam3VanillaAdapter

    # Encoder is frozen → no_grad during encoder pass → inference-level VRAM (~6 GB)
    _require_sam3(config, encoder_frozen=True)
    return Sam3VanillaAdapter(config)


def _build_sam3_topolora(config: ModelConfig, **kwargs: Any) -> ModelAdapter:
    from minivess.adapters.sam3_topolora import Sam3TopoLoraAdapter

    # LoRA adapters on encoder FFN → gradients flow → training-level VRAM (≥16 GB)
    _require_sam3(config, encoder_frozen=False)
    return Sam3TopoLoraAdapter(config)


def _build_sam3_hybrid(config: ModelConfig, **kwargs: Any) -> ModelAdapter:
    from minivess.adapters.sam3_hybrid import Sam3HybridAdapter

    # SAM encoder is frozen + detach() → inference-level VRAM for SAM part
    _require_sam3(config, encoder_frozen=True)
    return Sam3HybridAdapter(config)


# Populate registry — order does not matter; dict lookup is O(1)
def _populate_registry() -> None:
    from minivess.config.models import ModelFamily

    _MODEL_REGISTRY[ModelFamily.MONAI_DYNUNET] = _build_dynunet
    _MODEL_REGISTRY[ModelFamily.MONAI_SEGRESNET] = _build_segresnet
    _MODEL_REGISTRY[ModelFamily.MONAI_SWINUNETR] = _build_swinunetr
    _MODEL_REGISTRY[ModelFamily.MONAI_UNETR] = _build_unetr
    _MODEL_REGISTRY[ModelFamily.MONAI_ATTENTIONUNET] = _build_attentionunet
    _MODEL_REGISTRY[ModelFamily.VESSEL_FM] = _build_vesselfm
    _MODEL_REGISTRY[ModelFamily.COMMA_MAMBA] = _build_comma
    _MODEL_REGISTRY[ModelFamily.ULIKE_MAMBA] = _build_mamba
    _MODEL_REGISTRY[ModelFamily.SAM3_VANILLA] = _build_sam3_vanilla
    _MODEL_REGISTRY[ModelFamily.SAM3_TOPOLORA] = _build_sam3_topolora
    _MODEL_REGISTRY[ModelFamily.SAM3_HYBRID] = _build_sam3_hybrid


_populate_registry()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_adapter(config: ModelConfig, **kwargs: Any) -> ModelAdapter:
    """Build a ModelAdapter from a ModelConfig.

    Dispatches on ``config.family`` via ``_MODEL_REGISTRY``.
    Uses lazy imports so that SAM3 dependencies are not required
    when only using MONAI models.

    Parameters
    ----------
    config:
        Model configuration specifying family and architecture params.
    **kwargs:
        Extra keyword arguments forwarded to the adapter constructor.
        Note: SAM3 adapters no longer accept ``use_stub`` — pretrained
        weights are always required.

    Returns
    -------
    Concrete ModelAdapter instance.

    Raises
    ------
    ValueError
        If the model family is not registered.
    RuntimeError
        If a SAM3 family is requested but SAM3 is not installed.
    """
    family = config.family
    builder = _MODEL_REGISTRY.get(family)

    if builder is None:
        available = sorted(f.value for f in _MODEL_REGISTRY)
        msg = f"Unsupported model family '{family}'. Available: {available}"
        raise ValueError(msg)

    return builder(config, **kwargs)


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
