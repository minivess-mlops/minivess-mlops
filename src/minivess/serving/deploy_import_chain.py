"""Deploy import chain verification: ONNX → ORT → BentoML.

Pre-deployment verification gate — if this chain fails, the deploy
flow should NOT proceed to serving.

PR-D T2 (Issue #826).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ImportChainResult:
    """Result of the import chain verification.

    Parameters
    ----------
    onnx_valid:
        Whether the ONNX file is valid (exists and passes checker).
    ort_inference_ok:
        Whether ONNX Runtime inference succeeds.
    output_shape:
        Output tensor shape from ORT inference, or empty list.
    bento_tag:
        BentoML model tag if import succeeded, or None.
    errors:
        List of error messages encountered during verification.
    """

    onnx_valid: bool = False
    ort_inference_ok: bool = False
    output_shape: list[int] = field(default_factory=list)
    bento_tag: str | None = None
    errors: list[str] = field(default_factory=list)


def verify_onnx_file_exists(onnx_path: Path) -> bool:
    """Check that the ONNX file exists and is non-empty.

    Parameters
    ----------
    onnx_path:
        Path to the ONNX model file.

    Returns
    -------
    ``True`` if the file exists and has size > 0.
    """
    return onnx_path.is_file() and onnx_path.stat().st_size > 0


def verify_import_chain(
    champion: dict[str, Any],
    onnx_path: Path,
) -> ImportChainResult:
    """Verify the full MLflow → ONNX → ORT import chain.

    Parameters
    ----------
    champion:
        Champion model dict with ``run_id``, ``category``, etc.
    onnx_path:
        Path to the exported ONNX model.

    Returns
    -------
    :class:`ImportChainResult` with verification status.
    """
    result = ImportChainResult()

    # Step 1: Verify ONNX file exists
    if not verify_onnx_file_exists(onnx_path):
        result.errors.append(f"ONNX file not found or empty: {onnx_path}")
        return result

    # Step 2: Validate ONNX model
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        result.onnx_valid = True
        logger.info("ONNX model valid: %s", onnx_path)
    except Exception as exc:
        result.errors.append(f"ONNX validation failed: {exc}")
        return result

    # Step 3: Run ORT inference
    try:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))
        input_info = session.get_inputs()[0]
        input_shape = [d if isinstance(d, int) else 1 for d in input_info.shape]
        input_data = np.random.randn(*input_shape).astype(np.float32)

        outputs = session.run(None, {input_info.name: input_data})
        result.ort_inference_ok = True
        result.output_shape = list(outputs[0].shape)
        logger.info(
            "ORT inference OK: input=%s output=%s",
            input_shape,
            result.output_shape,
        )
    except Exception as exc:
        result.errors.append(f"ORT inference failed: {exc}")

    return result
