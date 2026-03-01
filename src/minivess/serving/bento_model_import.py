"""BentoML model import from ONNX champion models.

Imports ONNX models into the BentoML model store with metadata
for champion category, run ID, and evaluation metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from minivess.pipeline.deploy_champion_discovery import ChampionModel

logger = logging.getLogger(__name__)


@dataclass
class BentoImportResult:
    """Result of importing a model into BentoML store.

    Parameters
    ----------
    tag:
        BentoML model tag string.
    metadata:
        Metadata stored with the model.
    """

    tag: str
    metadata: dict[str, Any] = field(default_factory=dict)


def get_bento_model_tag(champion: ChampionModel) -> str:
    """Generate a BentoML model tag from a champion model.

    Format: ``minivess-{category}:{run_id}``

    Parameters
    ----------
    champion:
        Champion model to generate tag for.

    Returns
    -------
    BentoML model tag string.
    """
    return f"minivess-{champion.category}:{champion.run_id}"


def import_champion_to_bento(
    champion: ChampionModel,
    onnx_path: Path,
) -> BentoImportResult:
    """Import an ONNX champion model into the BentoML model store.

    Parameters
    ----------
    champion:
        Champion model metadata.
    onnx_path:
        Path to the exported ONNX model file.

    Returns
    -------
    :class:`BentoImportResult` with tag and metadata.

    Raises
    ------
    FileNotFoundError
        If the ONNX file does not exist.
    """
    if not onnx_path.exists():
        msg = f"ONNX model not found: {onnx_path}"
        raise FileNotFoundError(msg)

    tag = get_bento_model_tag(champion)
    metadata = {
        "champion_category": champion.category,
        "run_id": champion.run_id,
        "experiment_id": champion.experiment_id,
        "metrics": champion.metrics,
    }

    try:
        import bentoml

        bento_model = bentoml.onnx.save_model(
            tag,
            str(onnx_path),
            metadata=metadata,
        )
        actual_tag = str(bento_model.tag)
        logger.info("Imported ONNX model to BentoML: %s", actual_tag)
        return BentoImportResult(tag=actual_tag, metadata=metadata)

    except Exception:
        logger.exception("BentoML import failed, returning metadata-only result")
        return BentoImportResult(tag=tag, metadata=metadata)
