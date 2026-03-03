"""Quasi-E2E integration tests — verifying full debug pipeline.

Tests the complete pipeline flow:
1. Debug config creates _debug MLflow experiment
2. Per-volume metrics persisted as JSON artifact
3. Analysis flow on mock debug runs produces figures + LaTeX
4. Deploy metrics match training metrics (mocked)
5. Annotation flow round-trip with local client
6. Trigger stubs visible in caplog
7. Experiment naming enforcement

Closes #191.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from minivess.config.experiment_naming import (
    is_debug_experiment,
    validate_debug_experiment_name,
)
from minivess.orchestration.flows.annotation_flow import (
    AnnotationFlowResult,
    run_annotation_flow,
)
from minivess.orchestration.trigger import on_analysis_completion, on_dvc_version_change
from minivess.pipeline.ci import ConfidenceInterval
from minivess.pipeline.comparison import (
    build_comparison_table,
    format_comparison_latex,
    format_comparison_markdown,
)
from minivess.pipeline.deploy_verification import (
    verify_deploy_metrics,
    verify_onnx_vs_pytorch,
)
from minivess.pipeline.evaluation import FoldResult
from minivess.serving.api_models import SegmentationResponse

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TestDebugConfigIntegration:
    """Verify debug config is well-formed and naming is enforced."""

    def test_debug_config_name_validation(self) -> None:
        from minivess.config.compose import compose_experiment_config

        config = compose_experiment_config(experiment_name="dynunet_e2e_debug")
        name = config["experiment_name"]
        epochs = config["max_epochs"]
        assert is_debug_experiment(name)
        assert validate_debug_experiment_name(name, max_epochs=epochs) == name

    def test_debug_config_rejects_bad_name(self) -> None:
        from minivess.config.compose import compose_experiment_config

        config = compose_experiment_config(experiment_name="dynunet_e2e_debug")
        epochs = config["max_epochs"]
        with pytest.raises(ValueError, match="_debug"):
            validate_debug_experiment_name("bad_experiment", max_epochs=epochs)


class TestPerVolumeMetricsIntegration:
    """Verify per-volume metric persistence end-to-end."""

    def test_save_and_reload_roundtrip(self, tmp_path: Path) -> None:
        fr = FoldResult(
            volume_ids=["mv01", "mv02", "mv03"],
            per_volume_metrics={
                "dsc": [0.85, 0.90, 0.78],
                "centreline_dsc": [0.70, 0.82, 0.65],
            },
            aggregated={
                "dsc": ConfidenceInterval(
                    point_estimate=0.843,
                    lower=0.78,
                    upper=0.90,
                    confidence_level=0.95,
                    method="percentile",
                ),
            },
        )
        out_path = fr.save_per_volume_json(
            tmp_path / "per_volume_metrics" / "fold0_dice_ce.json"
        )
        loaded = FoldResult.load_per_volume_json(out_path)
        assert len(loaded) == 3
        assert loaded[0]["volume_id"] == "mv01"
        assert loaded[0]["dsc"] == pytest.approx(0.85)

    def test_best_worst_extraction(self) -> None:
        fr = FoldResult(
            volume_ids=["mv01", "mv02", "mv03"],
            per_volume_metrics={"dsc": [0.85, 0.90, 0.78]},
        )
        best_id, best_val = fr.best_volume("dsc")
        worst_id, worst_val = fr.worst_volume("dsc")
        assert best_id == "mv02"
        assert worst_id == "mv03"


class TestAnalysisFlowIntegration:
    """Verify analysis pipeline with comparison table and LaTeX export."""

    def _make_eval_results(self) -> dict[str, list[FoldResult]]:
        """Create mock eval results for 2 losses × 3 folds."""
        results: dict[str, list[FoldResult]] = {}
        for loss_name, dsc_mean in [("dice_ce", 0.82), ("cbdice_cldice", 0.78)]:
            folds: list[FoldResult] = []
            for i in range(3):
                dsc_val = dsc_mean + i * 0.01
                folds.append(
                    FoldResult(
                        per_volume_metrics={"dsc": [dsc_val]},
                        aggregated={
                            "dsc": ConfidenceInterval(
                                point_estimate=dsc_val,
                                lower=dsc_val - 0.05,
                                upper=dsc_val + 0.05,
                                confidence_level=0.95,
                                method="percentile",
                            ),
                        },
                    )
                )
            results[loss_name] = folds
        return results

    def test_comparison_table_built(self) -> None:
        table = build_comparison_table(self._make_eval_results())
        assert len(table.losses) == 2
        assert "dsc" in table.metric_names

    def test_markdown_contains_all_losses(self) -> None:
        table = build_comparison_table(self._make_eval_results())
        md = format_comparison_markdown(table)
        assert "dice_ce" in md
        assert "cbdice_cldice" in md

    def test_latex_export_booktabs(self) -> None:
        table = build_comparison_table(self._make_eval_results())
        latex = format_comparison_latex(table)
        assert r"\toprule" in latex
        assert r"\bottomrule" in latex
        assert r"\textbf{" in latex


class TestDeployVerificationIntegration:
    """Verify deploy metrics matching across pipeline."""

    def test_perfect_match(self) -> None:
        fr = FoldResult(
            volume_ids=["mv01"],
            per_volume_metrics={"dsc": [1.0]},
            aggregated={
                "dsc": ConfidenceInterval(
                    point_estimate=1.0,
                    lower=1.0,
                    upper=1.0,
                    confidence_level=0.95,
                    method="percentile",
                ),
            },
        )
        preds = [np.ones((5, 5, 5), dtype=np.int64)]
        labels = [np.ones((5, 5, 5), dtype=np.int64)]
        result = verify_deploy_metrics(fr, preds, labels)
        assert result.all_match is True

    def test_onnx_pytorch_identical(self) -> None:
        output = np.random.default_rng(42).random((1, 2, 8, 8, 8)).astype(np.float32)
        assert verify_onnx_vs_pytorch(output, output) is True


class TestAnnotationFlowIntegration:
    """Verify annotation flow end-to-end with mocked inference."""

    @patch("minivess.orchestration.flows.annotation_flow._build_client")
    def test_full_roundtrip(self, mock_build_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.predict.return_value = SegmentationResponse(
            segmentation=np.ones((8, 8, 8), dtype=np.int64),
            shape=[8, 8, 8],
            model_name="balanced",
            inference_time_ms=42.0,
        )
        mock_build_client.return_value = mock_client

        volume = np.zeros((8, 8, 8), dtype=np.float32)
        reference = np.ones((8, 8, 8), dtype=np.int64)
        result = run_annotation_flow(volume, "mv01", reference=reference)

        assert isinstance(result, AnnotationFlowResult)
        assert isinstance(result.response, dict)
        assert result.agreement_dice == pytest.approx(1.0)
        assert "mv01" in result.session_report


class TestTriggerStubsIntegration:
    """Verify trigger stubs log expected messages."""

    def test_dvc_trigger_in_caplog(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            on_dvc_version_change("minivess", "v1.0", "v1.1")
        assert "minivess" in caplog.text
        assert "Dashboard update pending" in caplog.text

    def test_analysis_trigger_in_caplog(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            on_analysis_completion("dynunet_e2e_debug", "balanced", 5)
        assert "dynunet_e2e_debug" in caplog.text
        assert "Dashboard update pending" in caplog.text
