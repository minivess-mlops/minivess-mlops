#!/usr/bin/env python3
"""SAM3 comparison report generator (SAM-14).

Generates a markdown report comparing SAM3 variants against the DynUNet
baseline using evaluation results from evaluate_sam3_comparison.py.

Usage:
    uv run python scripts/generate_sam3_report.py \
        --results-dir outputs/analysis/sam3 \
        --output docs/results/sam3_comparison_report.md
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
        description="Generate SAM3 comparison report",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/analysis/sam3"),
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/results/sam3_comparison_report.md"),
        help="Output markdown report path",
    )
    return parser.parse_args()


REPORT_TEMPLATE = """\
# SAM3 Variants Comparison Report

**Generated:** {timestamp}

## Overview

This report compares three SAM2-based segmentation variants against the
DynUNet baseline on the MiniVess microvessel dataset (70 volumes, 3-fold CV).

| Variant | Architecture | Loss | Trainable Params |
|---------|-------------|------|-----------------|
| DynUNet (baseline) | 3D DynUNet | cbdice_cldice | ~3.5M |
| SAM3-Vanilla (V1) | Frozen SAM2 + decoder | dice_ce | ~2M |
| SAM3-TopoLoRA (V2) | SAM2 + LoRA + decoder | cbdice_cldice | ~6.9M |
| SAM3-Hybrid (V3) | SAM2 + DynUNet + fusion | cbdice_cldice | ~7M+ |

## Expected Results

Based on the literature (Ravi et al. 2024, Xiang et al. 2025, Li et al. 2025):

- **V1 (Vanilla):** DSC ~0.35-0.55, clDice ~0.3-0.5 (SAM2 struggles with 3D microvasculature)
- **V2 (TopoLoRA):** +10-20% clDice over V1 (topology-aware loss guides LoRA adaptation)
- **V3 (Hybrid):** Best SAM variant, but likely still below DynUNet standalone

## Metric Suite

| Metric | Direction | Category |
|--------|-----------|----------|
| DSC (Dice) | Higher better | Overlap |
| HD95 | Lower better | Boundary |
| ASSD | Lower better | Boundary |
| NSD (Surface Dice) | Higher better | Boundary |
| clDice | Higher better | Topology |
| Betti Error 0 | Lower better | Topology |
| Betti Error 1 | Lower better | Topology |
| Junction F1 | Higher better | Topology |

## Results

{results_section}

## Go/No-Go Gate Evaluation

{gates_section}

## Conclusions

{conclusions_section}

## References

- Ravi et al. (2024). "SAM 2: Segment Anything in Images and Videos."
- Xiang et al. (2025). "TopoLoRA-SAM." arXiv:2601.02273.
- Li et al. (2025). "nnSAM: Plug-and-Play Segment Anything Model."
"""


def main() -> int:
    """Generate comparison report."""
    args = parse_args()

    logger.info("SAM3 Report Generator")
    logger.info("Results dir: %s", args.results_dir)
    logger.info("Output: %s", args.output)

    # Check for evaluation results
    results_section = (
        "*No evaluation results available yet. Train models and run evaluation first.*"
    )
    gates_section = "*Gates not evaluated yet.*"
    conclusions_section = "*Pending training and evaluation.*"

    template_path = args.results_dir / "evaluation_template.json"
    results_path = args.results_dir / "evaluation_results.json"

    if results_path.is_file():
        results_data = json.loads(results_path.read_text(encoding="utf-8"))
        results_section = f"```json\n{json.dumps(results_data, indent=2)}\n```"
        conclusions_section = "See gate evaluation above for actionable conclusions."
    elif template_path.is_file():
        results_section = (
            "*Evaluation template generated but no trained models yet.*\n\n"
            "Run training first:\n"
            "```bash\n"
            "uv run python scripts/run_experiment.py "
            "--config configs/experiments/sam3_all_debug.yaml\n"
            "```"
        )

    # Generate report
    report = REPORT_TEMPLATE.format(
        timestamp=datetime.now(UTC).isoformat(),
        results_section=results_section,
        gates_section=gates_section,
        conclusions_section=conclusions_section,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    logger.info("Report written to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
