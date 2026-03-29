"""Annotation Prefect Flow — inference via deploy server for annotation workflows.

Runs segmentation inference using InferenceClient, records an annotation
session, and optionally computes agreement with a reference segmentation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray  # noqa: TC002
from prefect import flow, task

from minivess.orchestration.constants import FLOW_NAME_ANNOTATION
from minivess.observability.flow_observability import flow_observability_context
from minivess.orchestration.docker_guard import require_docker_context
from minivess.serving.api_models import SegmentationRequest

if TYPE_CHECKING:
    from minivess.serving.api_models import SegmentationResponse
    from minivess.serving.inference_client import InferenceClient

logger = logging.getLogger(__name__)


@dataclass
class AnnotationFlowConfig:
    """Configuration for the annotation flow.

    Parameters
    ----------
    model_name:
        Champion category to use for inference.
    output_mode:
        What to include in the response.
    server_url:
        Remote server URL. None means local inference.
    model_paths:
        Local model paths for LocalInferenceClient.
    uq_methods:
        UQ methods to apply.
    confidence_level:
        Confidence level for UQ prediction sets.
    """

    model_name: str = "balanced"
    output_mode: str = "full"
    server_url: str | None = None
    model_paths: dict[str, Any] | None = None
    uq_methods: list[str] | None = None
    confidence_level: float = 0.95


@dataclass
class AnnotationFlowResult:
    """Result from the annotation flow.

    Parameters
    ----------
    response:
        Dict from SegmentationResponse.to_dict() — NOT the dataclass.
    session_report:
        Markdown report of the annotation session.
    agreement_dice:
        DSC agreement vs reference, if reference was provided.
    """

    response: dict[str, Any]
    session_report: str
    agreement_dice: float | None


def _build_client(config: AnnotationFlowConfig) -> InferenceClient:
    """Build the appropriate inference client based on config."""
    if config.server_url is not None:
        from minivess.serving.inference_client import RemoteInferenceClient

        return RemoteInferenceClient(config.server_url)

    from pathlib import Path

    from minivess.serving.inference_client import LocalInferenceClient
    from minivess.serving.model_registry_server import ModelRegistryServer

    model_paths: dict[str, Path] = {}
    if config.model_paths:
        model_paths = {k: Path(v) for k, v in config.model_paths.items()}

    server = ModelRegistryServer(model_paths=model_paths)
    return LocalInferenceClient(server)


def _compute_dice(
    prediction: NDArray[np.integer],
    reference: NDArray[np.integer],
) -> float:
    """Compute Dice coefficient between two binary masks."""
    pred_bool = prediction.ravel().astype(bool)
    ref_bool = reference.ravel().astype(bool)
    intersection = np.sum(pred_bool & ref_bool)
    total = np.sum(pred_bool) + np.sum(ref_bool)
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


@task(name="run-inference")
def inference_task(
    volume: NDArray[np.float32],
    config: AnnotationFlowConfig,
    client: InferenceClient,
) -> SegmentationResponse:
    """Run segmentation inference."""
    request = SegmentationRequest(
        volume=volume,
        model_name=config.model_name,
        output_mode=config.output_mode,
        uq_methods=config.uq_methods,
        confidence_level=config.confidence_level,
    )
    return client.predict(request)


@task(name="record-annotation")
def record_annotation_task(
    volume_id: str,
    response: SegmentationResponse,
    reference: NDArray[np.integer] | None = None,
) -> tuple[str, float | None]:
    """Record annotation session and compute agreement."""
    timestamp = datetime.now(UTC).isoformat()
    agreement: float | None = None

    if reference is not None:
        seg_array = np.array(response.segmentation)
        agreement = _compute_dice(seg_array, reference)
        logger.info(
            "Volume %s: agreement DSC=%.4f with reference",
            volume_id,
            agreement,
        )

    report = (
        f"## Annotation Session\n\n"
        f"- **Volume**: {volume_id}\n"
        f"- **Model**: {response.model_name}\n"
        f"- **Timestamp**: {timestamp}\n"
        f"- **Inference time**: {response.inference_time_ms:.1f} ms\n"
        f"- **Shape**: {response.shape}\n"
    )
    if agreement is not None:
        report += f"- **Agreement DSC**: {agreement:.4f}\n"

    return report, agreement


@flow(name=FLOW_NAME_ANNOTATION)
def run_annotation_flow(
    volume: NDArray[np.float32],
    volume_id: str,
    config: AnnotationFlowConfig | None = None,
    reference: NDArray[np.integer] | None = None,
) -> AnnotationFlowResult:
    """Run the annotation flow: inference + session recording.

    Parameters
    ----------
    volume:
        Input volume for inference.
    volume_id:
        Identifier for the volume.
    config:
        Flow configuration. Uses defaults if None.
    reference:
        Optional reference segmentation for agreement computation.

    Returns
    -------
    AnnotationFlowResult with response dict, session report, and agreement.
    """
    require_docker_context("annotation")

    logs_dir = Path(os.environ.get("LOGS_DIR", "/app/logs"))
    with flow_observability_context("annotation", logs_dir=logs_dir) as event_logger:
        if config is None:
            config = AnnotationFlowConfig()

        client = _build_client(config)
        response = inference_task(volume, config, client)
        session_report, agreement = record_annotation_task(volume_id, response, reference)

        return AnnotationFlowResult(
            response=response.to_dict(),
            session_report=session_report,
            agreement_dice=agreement,
        )


if __name__ == "__main__":
    # Annotation flow requires volume and volume_id parameters.
    # Direct invocation prints usage instructions.
    raise SystemExit(
        "annotation_flow cannot be invoked directly — it requires volume "
        "and volume_id parameters.\n"
        "Use: prefect deployment run 'minivess-annotation/default'"
    )
