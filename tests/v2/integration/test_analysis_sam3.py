"""Tests for SAM3 analysis pipeline (T8).

Tests gate evaluation with synthetic metrics, comparison table building,
and output artifact generation. CI-compatible (no MLflow or GPU required).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class TestGateEvaluation:
    """Test Go/No-Go gate evaluation with synthetic metrics."""

    def test_g1_pass_with_high_dsc(self) -> None:
        from minivess.pipeline.sam3_gates import evaluate_g1_baseline_viability

        result = evaluate_g1_baseline_viability(0.35)
        assert result.passed is True
        assert result.gate_name == "G1"

    def test_g1_fail_with_low_dsc(self) -> None:
        from minivess.pipeline.sam3_gates import evaluate_g1_baseline_viability

        result = evaluate_g1_baseline_viability(0.05)
        assert result.passed is False

    def test_g1_edge_case_exactly_threshold(self) -> None:
        from minivess.pipeline.sam3_gates import evaluate_g1_baseline_viability

        result = evaluate_g1_baseline_viability(0.10, threshold=0.10)
        assert result.passed is True

    def test_g2_pass_with_improvement(self) -> None:
        from minivess.pipeline.sam3_gates import evaluate_g2_topology_improvement

        result = evaluate_g2_topology_improvement(0.30, 0.40)
        assert result.passed is True

    def test_g2_fail_no_improvement(self) -> None:
        from minivess.pipeline.sam3_gates import evaluate_g2_topology_improvement

        result = evaluate_g2_topology_improvement(0.40, 0.41)
        assert result.passed is False  # Only 0.01, need 0.02

    def test_g3_pass_hybrid_better(self) -> None:
        from minivess.pipeline.sam3_gates import evaluate_g3_hybrid_value

        result = evaluate_g3_hybrid_value(0.40, 0.45)
        assert result.passed is True

    def test_g3_fail_hybrid_worse(self) -> None:
        from minivess.pipeline.sam3_gates import evaluate_g3_hybrid_value

        result = evaluate_g3_hybrid_value(0.45, 0.40)
        assert result.passed is False

    def test_evaluate_all_gates_returns_three(self) -> None:
        from minivess.pipeline.sam3_gates import evaluate_all_gates

        results = evaluate_all_gates(
            vanilla_dsc=0.35,
            vanilla_cldice=0.30,
            topolora_dsc=0.52,
            topolora_cldice=0.55,
            hybrid_dsc=0.61,
        )
        assert len(results) == 3
        assert results[0].gate_name == "G1"
        assert results[1].gate_name == "G2"
        assert results[2].gate_name == "G3"

    def test_all_gates_pass_with_good_metrics(self) -> None:
        from minivess.pipeline.sam3_gates import evaluate_all_gates

        results = evaluate_all_gates(
            vanilla_dsc=0.35,
            vanilla_cldice=0.30,
            topolora_dsc=0.52,
            topolora_cldice=0.55,
            hybrid_dsc=0.61,
        )
        assert all(r.passed for r in results)

    def test_all_gates_fail_with_bad_metrics(self) -> None:
        from minivess.pipeline.sam3_gates import evaluate_all_gates

        results = evaluate_all_gates(
            vanilla_dsc=0.01,  # G1 fail
            vanilla_cldice=0.50,
            topolora_dsc=0.03,
            topolora_cldice=0.50,  # G2 fail (no improvement)
            hybrid_dsc=0.02,  # G3 fail (worse than topolora)
        )
        assert results[0].passed is False  # G1
        assert results[1].passed is False  # G2
        assert results[2].passed is False  # G3


class TestAnalysisScriptHelpers:
    """Test the analyze_sam3_experiment helper functions."""

    def test_collect_sam3_run_metrics_from_mlruns(self, tmp_path: Path) -> None:
        """Test metric extraction from MLflow filesystem structure."""
        from scripts.analyze_sam3_experiment import collect_sam3_metrics

        # Create mock MLflow run structure
        exp_dir = tmp_path / "mlruns" / "123"
        for run_name, loss_name in [
            ("run1", "dice_ce"),
            ("run2", "cbdice_cldice"),
        ]:
            run_dir = exp_dir / run_name
            (run_dir / "metrics").mkdir(parents=True)
            (run_dir / "tags").mkdir(parents=True)
            (run_dir / "params").mkdir(parents=True)

            # Write metric files (MLflow format: timestamp value step)
            (run_dir / "metrics" / "eval_fold0_dsc").write_text(
                "1709000000 0.3500 0\n", encoding="utf-8"
            )
            (run_dir / "metrics" / "eval_fold1_dsc").write_text(
                "1709000000 0.4000 0\n", encoding="utf-8"
            )
            (run_dir / "tags" / "loss_function").write_text(loss_name, encoding="utf-8")
            (run_dir / "tags" / "mlflow.runName").write_text(
                f"{loss_name}_20260302", encoding="utf-8"
            )
            (run_dir / "params" / "model_family").write_text(
                "sam3_vanilla" if loss_name == "dice_ce" else "sam3_topolora",
                encoding="utf-8",
            )

        result = collect_sam3_metrics(
            mlruns_dir=tmp_path / "mlruns",
            experiment_id="123",
            metric_names=["dsc"],
            num_folds=2,
        )
        assert "sam3_vanilla" in result or "dice_ce" in result

    def test_build_sam3_comparison_table(self) -> None:
        """Test comparison table building from per-model metrics."""
        from scripts.analyze_sam3_experiment import build_sam3_comparison

        # Mock per-model metrics: {model_name: {metric: [fold0, fold1, fold2]}}
        model_metrics: dict[str, dict[str, list[float]]] = {
            "sam3_vanilla": {
                "dsc": [0.04, 0.05, 0.03],
                "cldice": [0.03, 0.04, 0.02],
            },
            "sam3_topolora": {
                "dsc": [0.10, 0.12, 0.09],
                "cldice": [0.08, 0.10, 0.07],
            },
            "sam3_hybrid": {
                "dsc": [0.11, 0.13, 0.10],
                "cldice": [0.09, 0.11, 0.08],
            },
        }
        table = build_sam3_comparison(model_metrics)
        assert len(table.losses) == 3
        assert "dsc" in table.metric_names

    def test_write_analysis_artifacts(self, tmp_path: Path) -> None:
        """Test artifact writing (MD, JSON, gate results)."""
        from scripts.analyze_sam3_experiment import write_analysis_artifacts

        model_metrics: dict[str, dict[str, list[float]]] = {
            "sam3_vanilla": {"dsc": [0.04, 0.05, 0.03]},
            "sam3_topolora": {"dsc": [0.10, 0.12, 0.09]},
            "sam3_hybrid": {"dsc": [0.11, 0.13, 0.10]},
        }
        write_analysis_artifacts(
            model_metrics=model_metrics,
            output_dir=tmp_path,
        )
        assert (tmp_path / "analysis_status.json").exists()
        status = json.loads(
            (tmp_path / "analysis_status.json").read_text(encoding="utf-8")
        )
        assert status["status"] in ("success", "partial")

    def test_gate_evaluation_with_training_results(self) -> None:
        """Test gate evaluation receives correct metrics from training."""
        from scripts.analyze_sam3_experiment import evaluate_gates_from_metrics

        model_metrics: dict[str, dict[str, list[float]]] = {
            "sam3_vanilla": {"dsc": [0.35, 0.40, 0.38], "cldice": [0.30, 0.35, 0.28]},
            "sam3_topolora": {
                "dsc": [0.52, 0.55, 0.50],
                "cldice": [0.55, 0.60, 0.53],
            },
            "sam3_hybrid": {"dsc": [0.61, 0.63, 0.58], "cldice": [0.50, 0.55, 0.48]},
        }
        gates = evaluate_gates_from_metrics(model_metrics)
        assert len(gates) == 3
        assert gates[0].passed is True  # G1: 0.377 >= 0.10
