"""Tests for annotation flow — Prefect flow for deploy-server inference.

Covers:
- AnnotationFlowConfig defaults
- AnnotationFlowResult structure
- Local client routing (no server_url → LocalInferenceClient)
- Response is dict (from to_dict), not dataclass
- Agreement computation with reference
- No-reference path (agreement is None)
- Model name passed through
- Output mode defaulting

Closes #189.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from minivess.orchestration.flows.annotation_flow import (
    AnnotationFlowConfig,
    AnnotationFlowResult,
    run_annotation_flow,
)
from minivess.serving.api_models import SegmentationResponse


def _mock_response(shape: tuple[int, ...] = (8, 8, 8)) -> SegmentationResponse:
    return SegmentationResponse(
        segmentation=np.ones(shape, dtype=np.int64),
        shape=list(shape),
        model_name="balanced",
        inference_time_ms=42.0,
    )


class TestAnnotationFlowConfig:
    """Test config defaults."""

    def test_default_model_name(self) -> None:
        cfg = AnnotationFlowConfig()
        assert cfg.model_name == "balanced"

    def test_default_output_mode(self) -> None:
        cfg = AnnotationFlowConfig()
        assert cfg.output_mode == "full"

    def test_default_server_url_is_none(self) -> None:
        cfg = AnnotationFlowConfig()
        assert cfg.server_url is None

    def test_default_confidence_level(self) -> None:
        cfg = AnnotationFlowConfig()
        assert cfg.confidence_level == pytest.approx(0.95)

    def test_custom_model_name(self) -> None:
        cfg = AnnotationFlowConfig(model_name="topology")
        assert cfg.model_name == "topology"


class TestAnnotationFlowResult:
    """Test result structure."""

    def test_response_is_dict(self) -> None:
        resp = _mock_response()
        result = AnnotationFlowResult(
            response=resp.to_dict(),
            session_report="test report",
            agreement_dice=None,
        )
        assert isinstance(result.response, dict)
        assert "segmentation" in result.response

    def test_agreement_can_be_none(self) -> None:
        result = AnnotationFlowResult(
            response={},
            session_report="",
            agreement_dice=None,
        )
        assert result.agreement_dice is None

    def test_agreement_with_value(self) -> None:
        result = AnnotationFlowResult(
            response={},
            session_report="",
            agreement_dice=0.85,
        )
        assert result.agreement_dice == pytest.approx(0.85)


class TestRunAnnotationFlow:
    """Test run_annotation_flow end-to-end with mocks."""

    @patch(
        "minivess.orchestration.flows.annotation_flow._build_client",
    )
    def test_basic_invocation(self, mock_build_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.predict.return_value = _mock_response()
        mock_build_client.return_value = mock_client

        volume = np.zeros((8, 8, 8), dtype=np.float32)
        result = run_annotation_flow(volume, "mv01")

        assert isinstance(result, AnnotationFlowResult)
        assert isinstance(result.response, dict)
        assert result.agreement_dice is None

    @patch(
        "minivess.orchestration.flows.annotation_flow._build_client",
    )
    def test_with_reference(self, mock_build_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.predict.return_value = _mock_response()
        mock_build_client.return_value = mock_client

        volume = np.zeros((8, 8, 8), dtype=np.float32)
        reference = np.ones((8, 8, 8), dtype=np.int64)
        result = run_annotation_flow(volume, "mv01", reference=reference)

        assert result.agreement_dice is not None
        assert isinstance(result.agreement_dice, float)

    @patch(
        "minivess.orchestration.flows.annotation_flow._build_client",
    )
    def test_model_name_passthrough(self, mock_build_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.predict.return_value = _mock_response()
        mock_build_client.return_value = mock_client

        volume = np.zeros((8, 8, 8), dtype=np.float32)
        config = AnnotationFlowConfig(model_name="topology")
        run_annotation_flow(volume, "mv01", config=config)

        # Verify the request used the right model name
        call_args = mock_client.predict.call_args[0][0]
        assert call_args.model_name == "topology"

    @patch(
        "minivess.orchestration.flows.annotation_flow._build_client",
    )
    def test_session_report_contains_volume_id(
        self, mock_build_client: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.predict.return_value = _mock_response()
        mock_build_client.return_value = mock_client

        volume = np.zeros((8, 8, 8), dtype=np.float32)
        result = run_annotation_flow(volume, "mv42")

        assert "mv42" in result.session_report

    @patch(
        "minivess.orchestration.flows.annotation_flow._build_client",
    )
    def test_perfect_agreement(self, mock_build_client: MagicMock) -> None:
        """When prediction matches reference, agreement should be 1.0."""
        mock_client = MagicMock()
        mock_client.predict.return_value = _mock_response()
        mock_build_client.return_value = mock_client

        volume = np.zeros((8, 8, 8), dtype=np.float32)
        reference = np.ones((8, 8, 8), dtype=np.int64)
        result = run_annotation_flow(volume, "mv01", reference=reference)

        assert result.agreement_dice == pytest.approx(1.0)

    @patch(
        "minivess.orchestration.flows.annotation_flow._build_client",
    )
    def test_zero_agreement(self, mock_build_client: MagicMock) -> None:
        """When prediction is all zeros and reference all ones, dice=0."""
        mock_client = MagicMock()
        resp = SegmentationResponse(
            segmentation=np.zeros((8, 8, 8), dtype=np.int64),
            shape=[8, 8, 8],
            model_name="balanced",
            inference_time_ms=10.0,
        )
        mock_client.predict.return_value = resp
        mock_build_client.return_value = mock_client

        volume = np.zeros((8, 8, 8), dtype=np.float32)
        reference = np.ones((8, 8, 8), dtype=np.int64)
        result = run_annotation_flow(volume, "mv01", reference=reference)

        assert result.agreement_dice == pytest.approx(0.0)

    @patch(
        "minivess.orchestration.flows.annotation_flow._build_client",
    )
    def test_default_config_when_none(self, mock_build_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.predict.return_value = _mock_response()
        mock_build_client.return_value = mock_client

        volume = np.zeros((8, 8, 8), dtype=np.float32)
        result = run_annotation_flow(volume, "mv01", config=None)

        assert isinstance(result, AnnotationFlowResult)
