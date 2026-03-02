"""SAM3 go/no-go gate evaluation.

Implements the three decision gates from the SAM3 variants plan:
- G1: Baseline viability — vanilla SAM3 DSC >= threshold
- G2: Topology improvement — TopoLoRA clDice improvement >= threshold
- G3: Hybrid value — hybrid DSC > TopoLoRA DSC
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateResult:
    """Result of a single go/no-go gate evaluation.

    Parameters
    ----------
    gate_name:
        Gate identifier (e.g., ``"G1"``).
    passed:
        Whether the gate condition was met.
    description:
        Human-readable description of the gate.
    observed_value:
        The actual measured value.
    threshold:
        The threshold that was applied.
    action_if_fail:
        What to do if the gate fails.
    """

    gate_name: str
    passed: bool
    description: str
    observed_value: float
    threshold: float
    action_if_fail: str


def evaluate_g1_baseline_viability(
    dsc: float,
    *,
    threshold: float = 0.10,
) -> GateResult:
    """G1: Is vanilla SAM3 minimally viable?

    Parameters
    ----------
    dsc:
        Vanilla SAM3 mean DSC across folds.
    threshold:
        Minimum acceptable DSC (default: 0.10).

    Returns
    -------
    GateResult
    """
    passed = dsc >= threshold
    return GateResult(
        gate_name="G1",
        passed=passed,
        description=f"Vanilla SAM3 DSC ({dsc:.4f}) >= {threshold}",
        observed_value=dsc,
        threshold=threshold,
        action_if_fail="Abandon SAM for segmentation entirely",
    )


def evaluate_g2_topology_improvement(
    vanilla_cldice: float,
    topolora_cldice: float,
    *,
    threshold: float = 0.02,
) -> GateResult:
    """G2: Does topology-aware loss improve SAM?

    Parameters
    ----------
    vanilla_cldice:
        Vanilla SAM3 mean clDice.
    topolora_cldice:
        TopoLoRA SAM3 mean clDice.
    threshold:
        Minimum absolute clDice improvement (default: 0.02 = 2%).

    Returns
    -------
    GateResult
    """
    improvement = topolora_cldice - vanilla_cldice
    passed = improvement >= threshold
    return GateResult(
        gate_name="G2",
        passed=passed,
        description=(
            f"TopoLoRA clDice ({topolora_cldice:.4f}) - "
            f"Vanilla clDice ({vanilla_cldice:.4f}) = "
            f"{improvement:.4f} >= {threshold}"
        ),
        observed_value=improvement,
        threshold=threshold,
        action_if_fail="Topology loss does not transfer to SAM LoRA",
    )


def evaluate_g3_hybrid_value(
    topolora_dsc: float,
    hybrid_dsc: float,
) -> GateResult:
    """G3: Does the hybrid architecture add value?

    Parameters
    ----------
    topolora_dsc:
        TopoLoRA SAM3 mean DSC.
    hybrid_dsc:
        Hybrid SAM3 mean DSC.

    Returns
    -------
    GateResult
    """
    passed = hybrid_dsc > topolora_dsc
    return GateResult(
        gate_name="G3",
        passed=passed,
        description=(
            f"Hybrid DSC ({hybrid_dsc:.4f}) > TopoLoRA DSC ({topolora_dsc:.4f})"
        ),
        observed_value=hybrid_dsc - topolora_dsc,
        threshold=0.0,
        action_if_fail="SAM features provide no complementary value",
    )


def evaluate_all_gates(
    vanilla_dsc: float,
    vanilla_cldice: float,
    topolora_dsc: float,
    topolora_cldice: float,
    hybrid_dsc: float,
    *,
    g1_threshold: float = 0.10,
    g2_threshold: float = 0.02,
) -> list[GateResult]:
    """Run all three SAM3 go/no-go gates.

    Parameters
    ----------
    vanilla_dsc:
        Vanilla SAM3 mean DSC.
    vanilla_cldice:
        Vanilla SAM3 mean clDice.
    topolora_dsc:
        TopoLoRA SAM3 mean DSC.
    topolora_cldice:
        TopoLoRA SAM3 mean clDice.
    hybrid_dsc:
        Hybrid SAM3 mean DSC.
    g1_threshold:
        G1 minimum DSC threshold.
    g2_threshold:
        G2 minimum clDice improvement threshold.

    Returns
    -------
    list[GateResult]
        Results for G1, G2, G3 in order.
    """
    results = [
        evaluate_g1_baseline_viability(vanilla_dsc, threshold=g1_threshold),
        evaluate_g2_topology_improvement(
            vanilla_cldice,
            topolora_cldice,
            threshold=g2_threshold,
        ),
        evaluate_g3_hybrid_value(topolora_dsc, hybrid_dsc),
    ]

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        logger.info("Gate %s: %s — %s", r.gate_name, status, r.description)

    return results
