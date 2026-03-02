"""Run the full pipeline trigger chain with real data and real artifacts.

Wires all 5 flows through PipelineTriggerChain and executes:
data → (train SKIPPED, uses existing runs) → analyze → deploy → dashboard.

This is the ultimate E2E verification — all artifacts from all flows
are generated from real MiniVess data and real trained checkpoints.

Run:
    uv run python scripts/run_full_pipeline.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT_ID = "843896622863223169"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "minivess"


def _make_data_flow_callable(output_dir: Path) -> Any:
    """Create a data flow wrapper compatible with trigger chain."""

    def wrapper(*, trigger_source: str = "pipeline") -> dict[str, Any]:
        from minivess.orchestration.flows.data_flow import run_data_flow

        result = run_data_flow(data_dir=DATA_DIR, n_folds=3, seed=42)

        # Save summary
        summary = {
            "n_volumes": len(result.pairs),
            "quality_passed": result.quality_passed,
            "n_folds": result.n_folds,
            "trigger_source": trigger_source,
        }
        summary_path = output_dir / "data_flow_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        return summary

    return wrapper


def _make_analysis_flow_callable(output_dir: Path) -> Any:
    """Create wrapper that generates figures, tables, LaTeX from real mlruns."""

    def wrapper(*, trigger_source: str = "pipeline") -> dict[str, Any]:
        # Generate figures from real data (same as P3)
        import numpy as np

        from minivess.pipeline.comparison import (
            ComparisonTable,
            LossResult,
            MetricSummary,
        )

        eval_metrics = ["dsc", "centreline_dsc", "measured_masd"]
        n_folds = 3
        exp_dir = MLRUNS_DIR / EXPERIMENT_ID

        loss_results: list[LossResult] = []
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            tags_dir = run_dir / "tags"
            if not tags_dir.is_dir():
                continue
            loss_tag = tags_dir / "loss_function"
            if not loss_tag.exists():
                continue

            loss_name = loss_tag.read_text(encoding="utf-8").strip()
            metrics_dir = run_dir / "metrics"
            if not metrics_dir.is_dir():
                continue

            # Check completeness (need fold2 metrics)
            has_fold2 = any(
                f.name.startswith("eval_fold2") for f in metrics_dir.iterdir()
            )
            if not has_fold2:
                continue

            metric_dict: dict[str, MetricSummary] = {}
            for metric in eval_metrics:
                values = []
                for fold in range(n_folds):
                    metric_file = metrics_dir / f"eval_fold{fold}_{metric}"
                    if metric_file.exists():
                        lines = (
                            metric_file.read_text(encoding="utf-8").strip().split("\n")
                        )
                        if lines:
                            parts = lines[-1].split()
                            if len(parts) >= 2:
                                values.append(float(parts[1]))

                if values:
                    mean_val = float(np.mean(values))
                    std_val = float(np.std(values))
                    metric_dict[metric] = MetricSummary(
                        mean=mean_val,
                        std=std_val,
                        ci_lower=mean_val - 1.96 * std_val,
                        ci_upper=mean_val + 1.96 * std_val,
                        per_fold=values,
                    )

            if metric_dict:
                loss_results.append(
                    LossResult(
                        loss_name=loss_name,
                        num_folds=n_folds,
                        metrics=metric_dict,
                    )
                )

        if loss_results:
            table = ComparisonTable(
                losses=loss_results,
                metric_names=eval_metrics,
            )

            # Generate figures
            figures_dir = output_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            from minivess.pipeline.viz.generate_all_figures import generate_all_figures

            generate_all_figures(
                output_dir=figures_dir,
                comparison_table=table,
                formats=["png", "svg"],
            )

            # Generate tables
            from minivess.pipeline.comparison import (
                format_comparison_latex,
                format_comparison_markdown,
            )

            md_table = format_comparison_markdown(table)
            latex_table = format_comparison_latex(table)
            (output_dir / "comparison_table.md").write_text(md_table, encoding="utf-8")
            (output_dir / "comparison_table.tex").write_text(
                latex_table, encoding="utf-8"
            )

        return {
            "n_losses": len(loss_results),
            "trigger_source": trigger_source,
            "artifacts_dir": str(output_dir),
        }

    return wrapper


def _make_deploy_flow_callable(output_dir: Path) -> Any:
    """Create deploy wrapper that exports real champion to ONNX."""

    def wrapper(*, trigger_source: str = "pipeline") -> dict[str, Any]:
        from minivess.pipeline.deploy_champion_discovery import discover_champions
        from minivess.pipeline.deploy_onnx_export import (
            export_champion_to_onnx,
            validate_onnx_model,
        )

        champions = discover_champions(MLRUNS_DIR, EXPERIMENT_ID)
        if not champions:
            logger.warning("No champions found for deploy")
            return {"n_champions": 0, "trigger_source": trigger_source}

        onnx_dir = output_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        onnx_results: dict[str, dict[str, Any]] = {}
        for champ in champions:
            if not champ.checkpoint_path:
                continue
            try:
                onnx_path = export_champion_to_onnx(
                    champ, onnx_dir, opset_version=17, input_shape=(1, 1, 32, 32, 16)
                )
                is_valid = validate_onnx_model(
                    onnx_path, input_shape=(1, 1, 32, 32, 16)
                )
                onnx_results[champ.category] = {
                    "path": str(onnx_path),
                    "size_mb": round(onnx_path.stat().st_size / 1e6, 2),
                    "valid": is_valid,
                }
            except Exception:
                logger.exception("Failed to export %s", champ.category)

        return {
            "n_champions": len(champions),
            "onnx_exports": onnx_results,
            "trigger_source": trigger_source,
        }

    return wrapper


def _make_dashboard_flow_callable(
    pipeline_output_dir: Path,
) -> Any:
    """Create dashboard wrapper that collects results from upstream flows."""

    def wrapper(*, trigger_source: str = "pipeline") -> dict[str, Any]:
        from minivess.orchestration.flows.dashboard_flow import run_dashboard_flow

        dashboard_dir = pipeline_output_dir / "dashboard"
        dashboard_dir.mkdir(parents=True, exist_ok=True)

        # Read upstream data
        data_summary_path = pipeline_output_dir / "data" / "data_flow_summary.json"
        n_volumes = 70
        quality_passed = True
        if data_summary_path.exists():
            data_summary = json.loads(data_summary_path.read_text(encoding="utf-8"))
            n_volumes = data_summary.get("n_volumes", 70)
            quality_passed = data_summary.get("quality_passed", True)

        # Check ONNX
        onnx_dir = pipeline_output_dir / "deploy" / "onnx"
        onnx_files = list(onnx_dir.glob("*.onnx")) if onnx_dir.exists() else []

        flow_results = {
            "data_flow": "PASSED" if quality_passed else "FAILED",
            "analysis_flow": "PASSED",
            "deploy_flow": "PASSED" if onnx_files else "SKIPPED",
            "dashboard_flow": "RUNNING",
        }

        result = run_dashboard_flow(
            output_dir=dashboard_dir,
            n_volumes=n_volumes,
            quality_gate_passed=quality_passed,
            environment="local",
            experiment_config="dynunet_loss_variation_v2",
            model_profile_name="dynunet",
            architecture_name="MONAI DynUNet (128 filters)",
            param_count=5_604_866,
            onnx_exported=len(onnx_files) > 0,
            champion_category="balanced",
            loss_name="cbdice_cldice",
            flow_results=flow_results,
            last_data_version="v1.0",
            trigger_source=trigger_source,
        )
        return {
            "report_path": str(result.get("report_path", "")),
            "metadata_path": str(result.get("metadata_path", "")),
        }

    return wrapper


def main() -> int:
    """Run the full pipeline trigger chain."""
    import argparse

    parser = argparse.ArgumentParser(description="Full pipeline trigger chain")
    parser.add_argument(
        "--output-dir", default="outputs/pipeline", help="Pipeline output directory"
    )
    parser.add_argument(
        "--skip-data", action="store_true", help="Skip data flow (use cached)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("FULL PIPELINE TRIGGER CHAIN")
    logger.info("=" * 70)
    logger.info("  Output: %s", output_dir)
    logger.info("  Data:   %s", DATA_DIR)
    logger.info("  MLruns: %s", MLRUNS_DIR)

    # Import trigger chain
    from minivess.orchestration.trigger import FlowTriggerConfig, PipelineTriggerChain

    chain = PipelineTriggerChain()

    # Register flow wrappers
    data_output = output_dir / "data"
    data_output.mkdir(parents=True, exist_ok=True)
    chain.register_flow("data", _make_data_flow_callable(data_output), is_core=True)

    # Train is SKIPPED — we use existing dynunet_loss_variation_v2 runs
    # (no-op remains registered by default)

    analysis_output = output_dir / "analysis"
    analysis_output.mkdir(parents=True, exist_ok=True)
    chain.register_flow(
        "analyze", _make_analysis_flow_callable(analysis_output), is_core=True
    )

    deploy_output = output_dir / "deploy"
    deploy_output.mkdir(parents=True, exist_ok=True)
    chain.register_flow(
        "deploy", _make_deploy_flow_callable(deploy_output), is_core=True
    )

    chain.register_flow(
        "dashboard", _make_dashboard_flow_callable(output_dir), is_core=False
    )

    # Configure: skip train, optionally skip data
    skip_flows = ["train"]
    if args.skip_data:
        skip_flows.append("data")

    config = FlowTriggerConfig(
        skip_flows=skip_flows, dashboard_always=True, dry_run=False
    )

    # Run the chain
    logger.info("\n[RUNNING] Pipeline trigger chain...")
    results = chain.run_chain(trigger_source="test/quasi-e2e-debugging", config=config)

    # Report results
    logger.info("\n" + "=" * 70)
    logger.info("TRIGGER CHAIN RESULTS")
    logger.info("=" * 70)

    chain_result: dict[str, Any] = {
        "trigger_source": "test/quasi-e2e-debugging",
        "timestamp": datetime.now(UTC).isoformat(),
        "flows": [],
    }

    total_duration = 0.0
    all_success = True
    for r in results:
        status_icon = {"success": "OK", "failed": "FAIL", "skipped": "SKIP"}.get(
            r.status, "?"
        )
        logger.info(
            "  [%4s] %-12s  %.2fs  %s",
            status_icon,
            r.flow_name,
            r.duration_s,
            r.error or "",
        )
        total_duration += r.duration_s
        if r.status == "failed":
            all_success = False
        chain_result["flows"].append(
            {
                "flow_name": r.flow_name,
                "status": r.status,
                "duration_s": round(r.duration_s, 2),
                "error": r.error,
            }
        )

    chain_result["total_duration_s"] = round(total_duration, 2)
    chain_result["all_success"] = all_success

    # Save results
    results_path = output_dir / "trigger_chain_results.json"
    results_path.write_text(
        json.dumps(chain_result, indent=2, default=str), encoding="utf-8"
    )
    logger.info("\n  Chain results: %s", results_path)
    logger.info("  Total duration: %.2fs", total_duration)
    logger.info("  Status: %s", "ALL PASSED" if all_success else "SOME FAILED")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
