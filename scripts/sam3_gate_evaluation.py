#!/usr/bin/env python3
"""SAM3 go/no-go gate evaluation script (SAM-17).

Evaluates the three decision gates from the SAM3 variants plan using
metrics from completed training runs.

Usage:
    uv run python scripts/sam3_gate_evaluation.py \
        --vanilla-dsc 0.35 --vanilla-cldice 0.38 \
        --topolora-dsc 0.52 --topolora-cldice 0.55 \
        --hybrid-dsc 0.61

    # Or load from evaluation results:
    uv run python scripts/sam3_gate_evaluation.py \
        --results-file outputs/analysis/sam3/evaluation_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate SAM3 go/no-go gates",
    )
    parser.add_argument("--vanilla-dsc", type=float, help="Vanilla SAM3 DSC")
    parser.add_argument("--vanilla-cldice", type=float, help="Vanilla SAM3 clDice")
    parser.add_argument("--topolora-dsc", type=float, help="TopoLoRA SAM3 DSC")
    parser.add_argument("--topolora-cldice", type=float, help="TopoLoRA SAM3 clDice")
    parser.add_argument("--hybrid-dsc", type=float, help="Hybrid SAM3 DSC")
    parser.add_argument(
        "--results-file",
        type=Path,
        help="JSON results file (alternative to manual metric args)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/sam3/gate_results.json"),
        help="Output JSON file for gate results",
    )
    return parser.parse_args()


def main() -> int:
    """Run gate evaluation."""
    args = parse_args()

    from minivess.pipeline.sam3_gates import evaluate_all_gates

    # Load metrics from file or CLI args
    if args.results_file and args.results_file.is_file():
        data = json.loads(args.results_file.read_text(encoding="utf-8"))
        vanilla_dsc = data["sam3_vanilla"]["dsc"]
        vanilla_cldice = data["sam3_vanilla"]["cldice"]
        topolora_dsc = data["sam3_topolora"]["dsc"]
        topolora_cldice = data["sam3_topolora"]["cldice"]
        hybrid_dsc = data["sam3_hybrid"]["dsc"]
    elif all(
        v is not None
        for v in [
            args.vanilla_dsc,
            args.vanilla_cldice,
            args.topolora_dsc,
            args.topolora_cldice,
            args.hybrid_dsc,
        ]
    ):
        vanilla_dsc = args.vanilla_dsc
        vanilla_cldice = args.vanilla_cldice
        topolora_dsc = args.topolora_dsc
        topolora_cldice = args.topolora_cldice
        hybrid_dsc = args.hybrid_dsc
    else:
        logger.error(
            "Provide either --results-file or all five metric args "
            "(--vanilla-dsc, --vanilla-cldice, --topolora-dsc, "
            "--topolora-cldice, --hybrid-dsc)"
        )
        return 1

    logger.info("Evaluating SAM3 go/no-go gates...")
    results = evaluate_all_gates(
        vanilla_dsc=vanilla_dsc,
        vanilla_cldice=vanilla_cldice,
        topolora_dsc=topolora_dsc,
        topolora_cldice=topolora_cldice,
        hybrid_dsc=hybrid_dsc,
    )

    # Display results
    print("\n" + "=" * 60)
    print("SAM3 Go/No-Go Gate Evaluation")
    print("=" * 60)

    all_passed = True
    gate_data = []
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        marker = "[+]" if r.passed else "[-]"
        print(f"\n  {marker} {r.gate_name}: {status}")
        print(f"      {r.description}")
        if not r.passed:
            print(f"      Action: {r.action_if_fail}")
            all_passed = False

        gate_data.append(
            {
                "gate": r.gate_name,
                "passed": r.passed,
                "description": r.description,
                "observed_value": r.observed_value,
                "threshold": r.threshold,
                "action_if_fail": r.action_if_fail,
            }
        )

    print(f"\n{'=' * 60}")
    overall = "ALL GATES PASSED" if all_passed else "SOME GATES FAILED"
    print(f"  Overall: {overall}")
    print(f"{'=' * 60}\n")

    # Save results
    output_data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "all_passed": all_passed,
        "gates": gate_data,
        "inputs": {
            "vanilla_dsc": vanilla_dsc,
            "vanilla_cldice": vanilla_cldice,
            "topolora_dsc": topolora_dsc,
            "topolora_cldice": topolora_cldice,
            "hybrid_dsc": hybrid_dsc,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_data, indent=2),
        encoding="utf-8",
    )
    logger.info("Gate results written to %s", args.output)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
