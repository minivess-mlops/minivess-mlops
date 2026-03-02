"""Tests for SAM3 go/no-go gate evaluation (SAM-17).

Verifies that gate evaluation logic correctly identifies pass/fail
conditions for G1, G2, G3 gates.
"""

from __future__ import annotations

from minivess.pipeline.sam3_gates import (
    evaluate_all_gates,
    evaluate_g1_baseline_viability,
    evaluate_g2_topology_improvement,
    evaluate_g3_hybrid_value,
)


class TestG1BaselineViability:
    """G1: vanilla SAM3 DSC >= 0.10."""

    def test_passes_when_dsc_above_threshold(self) -> None:
        result = evaluate_g1_baseline_viability(dsc=0.35)
        assert result.passed is True
        assert result.gate_name == "G1"

    def test_fails_when_dsc_below_threshold(self) -> None:
        result = evaluate_g1_baseline_viability(dsc=0.05)
        assert result.passed is False

    def test_custom_threshold(self) -> None:
        result = evaluate_g1_baseline_viability(dsc=0.15, threshold=0.20)
        assert result.passed is False


class TestG2TopologyImprovement:
    """G2: TopoLoRA clDice improvement over vanilla >= 2%."""

    def test_passes_with_sufficient_improvement(self) -> None:
        result = evaluate_g2_topology_improvement(
            vanilla_cldice=0.38,
            topolora_cldice=0.55,
        )
        assert result.passed is True
        assert result.gate_name == "G2"

    def test_fails_with_insufficient_improvement(self) -> None:
        result = evaluate_g2_topology_improvement(
            vanilla_cldice=0.38,
            topolora_cldice=0.39,
        )
        assert result.passed is False

    def test_custom_threshold(self) -> None:
        result = evaluate_g2_topology_improvement(
            vanilla_cldice=0.38,
            topolora_cldice=0.43,
            threshold=0.10,
        )
        assert result.passed is False


class TestG3HybridValue:
    """G3: hybrid DSC > TopoLoRA DSC."""

    def test_passes_when_hybrid_better(self) -> None:
        result = evaluate_g3_hybrid_value(
            topolora_dsc=0.52,
            hybrid_dsc=0.61,
        )
        assert result.passed is True
        assert result.gate_name == "G3"

    def test_fails_when_hybrid_worse(self) -> None:
        result = evaluate_g3_hybrid_value(
            topolora_dsc=0.52,
            hybrid_dsc=0.48,
        )
        assert result.passed is False


class TestEvaluateAllGates:
    """evaluate_all_gates() runs all three gates."""

    def test_all_pass(self) -> None:
        results = evaluate_all_gates(
            vanilla_dsc=0.35,
            vanilla_cldice=0.38,
            topolora_dsc=0.52,
            topolora_cldice=0.55,
            hybrid_dsc=0.61,
        )
        assert len(results) == 3
        assert all(r.passed for r in results)

    def test_partial_failure(self) -> None:
        results = evaluate_all_gates(
            vanilla_dsc=0.05,  # fails G1
            vanilla_cldice=0.04,
            topolora_dsc=0.06,
            topolora_cldice=0.07,
            hybrid_dsc=0.08,
        )
        assert not results[0].passed  # G1 fails
        assert results[1].passed  # G2 passes (0.07 - 0.04 = 0.03 >= 0.02)
