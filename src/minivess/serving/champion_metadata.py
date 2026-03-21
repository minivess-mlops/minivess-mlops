"""Champion metadata builder for manuscript reporting.

Builds comprehensive champion metadata for MLflow logging and
manuscript generation (TRIPOD-22 model specification).

PR-D T4 (Issue #828).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_champion_metadata(champion: dict[str, Any]) -> dict[str, Any]:
    """Build structured metadata dict for the champion model.

    Returns a nested dict with the following groups:
    - ``architecture``: param count, VRAM, FLOPs
    - ``training``: loss, epochs, optimizer, scheduler
    - ``performance``: metrics with CIs
    - ``post_training``: checkpoint averaging, recalibration
    - ``deployment``: ONNX opset, BentoML tag
    - ``factorial``: factorial factor values
    - ``tripod_compliance``: TRIPOD item references

    Parameters
    ----------
    champion:
        Champion run dict with all metadata fields.

    Returns
    -------
    Nested metadata dict.
    """
    return {
        "architecture": {
            "model": champion.get("model", "unknown"),
            "param_count": champion.get("param_count", 0),
            "vram_gb": champion.get("vram_gb", 0.0),
            "flops_estimate": champion.get("flops_estimate", 0),
        },
        "training": {
            "loss": champion.get("loss", "unknown"),
            "max_epochs": champion.get("max_epochs", 0),
            "batch_size": champion.get("batch_size", 0),
            "optimizer": champion.get("optimizer", "unknown"),
            "scheduler": champion.get("scheduler", "unknown"),
        },
        "performance": {
            "dsc": champion.get("dsc", 0.0),
            "cldice": champion.get("cldice", 0.0),
            "masd": champion.get("masd", 0.0),
            "compound_metric": champion.get("compound_metric", 0.0),
            "dsc_ci95_lo": champion.get("dsc_ci95_lo"),
            "dsc_ci95_hi": champion.get("dsc_ci95_hi"),
            "cldice_ci95_lo": champion.get("cldice_ci95_lo"),
            "cldice_ci95_hi": champion.get("cldice_ci95_hi"),
            "masd_ci95_lo": champion.get("masd_ci95_lo"),
            "masd_ci95_hi": champion.get("masd_ci95_hi"),
        },
        "post_training": {
            "swa_method": champion.get("swa_method", "none"),
            "recalibration": champion.get("recalibration", "none"),
        },
        "deployment": {
            "onnx_opset": champion.get("onnx_opset", 17),
            "bento_tag": champion.get("bento_tag", ""),
            "fold_strategy": champion.get("fold_strategy", "unknown"),
            "ensemble": champion.get("ensemble", "none"),
        },
        "factorial": {
            "model": champion.get("model", "unknown"),
            "loss": champion.get("loss", "unknown"),
            "aux_calib": champion.get("aux_calib", False),
        },
        "tripod_compliance": {
            "model_specification": "TRIPOD-22",
            "computational_cost": "TRIPOD-12",
            "performance_metrics": "TRIPOD-10",
        },
    }


def flatten_metadata_for_mlflow(metadata: dict[str, Any]) -> dict[str, str]:
    """Flatten nested metadata dict to slash-prefix MLflow params.

    Parameters
    ----------
    metadata:
        Nested metadata dict from :func:`build_champion_metadata`.

    Returns
    -------
    Flat dict with ``champion/{group}/{key}`` keys and string values.
    """
    _GROUP_PREFIX_MAP = {
        "architecture": "arch",
        "training": "train",
        "performance": "perf",
        "post_training": "post",
        "deployment": "deploy",
        "factorial": "factor",
        "tripod_compliance": "tripod",
    }

    flat: dict[str, str] = {}
    for group_name, fields in metadata.items():
        if not isinstance(fields, dict):
            continue
        prefix = _GROUP_PREFIX_MAP.get(group_name, group_name)
        for key, value in fields.items():
            flat[f"champion/{prefix}/{key}"] = str(value)

    return flat
