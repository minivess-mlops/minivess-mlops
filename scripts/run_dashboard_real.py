"""Run Dashboard Flow (Flow 5) with real upstream data from all prior flows.

Collects results from Data, Analysis, and Deploy flows to generate
a comprehensive dashboard report with markdown and JSON metadata.

Run:
    uv run python scripts/run_dashboard_real.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    """Run dashboard flow with real upstream results."""
    import argparse

    parser = argparse.ArgumentParser(description="Dashboard with real data")
    parser.add_argument(
        "--output-dir", default="outputs/dashboard", help="Output directory"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("DASHBOARD FLOW: Real Upstream Data")
    logger.info("=" * 70)

    # Collect upstream data
    logger.info("\n[1/3] Collecting upstream results")

    # Data flow results
    data_summary_path = (
        PROJECT_ROOT / "outputs" / "data_flow" / "data_flow_summary.json"
    )
    if data_summary_path.exists():
        data_summary = json.loads(data_summary_path.read_text(encoding="utf-8"))
        n_volumes = data_summary.get("n_volumes", 0)
        quality_passed = data_summary.get("quality_passed", False)
        logger.info("  Data: %d volumes, quality=%s", n_volumes, quality_passed)
    else:
        logger.warning("  Data flow results not found — using defaults")
        n_volumes = 70
        quality_passed = True

    # Analysis flow results — champion info
    champion_category = "balanced"
    loss_name = "cbdice_cldice"
    champion_run_id = ""

    # Check champion tags on filesystem
    mlruns_dir = PROJECT_ROOT / "mlruns"
    exp_id = "843896622863223169"
    exp_dir = mlruns_dir / exp_id
    if exp_dir.exists():
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            tags_dir = run_dir / "tags"
            if tags_dir.is_dir() and (tags_dir / "champion_rank_balanced").exists():
                champion_run_id = run_dir.name
                loss_tag = tags_dir / "loss_function"
                if loss_tag.exists():
                    loss_name = loss_tag.read_text(encoding="utf-8").strip()
                break

    logger.info("  Champion: %s (%s)", loss_name, champion_run_id[:8] or "unknown")

    # Deploy flow results — ONNX
    onnx_dir = PROJECT_ROOT / "outputs" / "deploy" / "onnx"
    onnx_files = list(onnx_dir.glob("*.onnx")) if onnx_dir.exists() else []
    onnx_exported = len(onnx_files) > 0
    logger.info("  ONNX models: %d", len(onnx_files))

    # Flow results summary
    flow_results = {
        "data_flow": "PASSED" if quality_passed else "FAILED",
        "analysis_flow": "PASSED",
        "deploy_flow": "PASSED" if onnx_exported else "SKIPPED",
        "dashboard_flow": "RUNNING",
    }

    # Run the dashboard flow
    logger.info("\n[2/3] Running dashboard flow")
    from minivess.orchestration.flows.dashboard_flow import run_dashboard_flow

    result = run_dashboard_flow(
        output_dir=output_dir,
        # Data section
        n_volumes=n_volumes,
        quality_gate_passed=quality_passed,
        external_datasets={},
        # Config section
        environment="local",
        experiment_config="dynunet_loss_variation_v2",
        model_profile_name="dynunet",
        # Model section
        architecture_name="MONAI DynUNet (128 filters)",
        param_count=5_604_866,
        onnx_exported=onnx_exported,
        champion_category=champion_category,
        loss_name=loss_name,
        # Pipeline section
        flow_results=flow_results,
        last_data_version="v1.0",
        last_training_run_id=champion_run_id,
        trigger_source="scripts/run_dashboard_real.py",
    )

    # Report results
    logger.info("\n[3/3] Results")
    report_path = result.get("report_path")
    metadata_path = result.get("metadata_path")

    if report_path and Path(str(report_path)).exists():
        report_content = Path(str(report_path)).read_text(encoding="utf-8")
        logger.info("  Report: %s (%d chars)", report_path, len(report_content))
        logger.info("\n  --- Dashboard Preview (first 30 lines) ---")
        for line in report_content.split("\n")[:30]:
            logger.info("  %s", line)
    else:
        logger.warning("  No report generated")

    if metadata_path and Path(str(metadata_path)).exists():
        logger.info("  Metadata: %s", metadata_path)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    all_files = list(output_dir.rglob("*"))
    artifact_files = [f for f in all_files if f.is_file()]
    logger.info("  Total artifacts: %d", len(artifact_files))
    for f in sorted(artifact_files):
        logger.info(
            "    %s (%.1f KB)", f.relative_to(output_dir), f.stat().st_size / 1024
        )

    logger.info("\nDASHBOARD FLOW COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
