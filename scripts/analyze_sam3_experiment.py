#!/usr/bin/env python3
"""SAM3 experiment analysis — metrics extraction, comparison, gate evaluation.

Reads SAM3 training results from MLflow filesystem, builds per-variant
comparison tables, evaluates go/no-go gates, and writes output artifacts.

Usage::

    uv run python scripts/analyze_sam3_experiment.py \\
        --mlruns-dir mlruns \\
        --output-dir outputs/analysis/sam3

    # With reference DynUNet baseline:
    uv run python scripts/analyze_sam3_experiment.py \\
        --mlruns-dir mlruns \\
        --reference-experiment dynunet_loss_variation_v2 \\
        --output-dir outputs/analysis/sam3
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Metrics to extract from MLflow eval results
DEFAULT_METRICS = ["dsc", "measured_masd", "centreline_dsc"]

# Extended metrics (when available from full evaluation)
FULL_METRICS = [
    "dsc",
    "hd95",
    "assd",
    "nsd",
    "cldice",
    "betti_error_0",
    "betti_error_1",
    "junction_f1",
]


def _read_mlflow_metric(metric_file: Path) -> float | None:
    """Read last value from an MLflow metric file.

    MLflow stores metrics as: ``timestamp value step`` per line.
    Returns the value from the last line, or None if file missing/empty.
    """
    if not metric_file.exists():
        return None
    text = metric_file.read_text(encoding="utf-8").strip()
    if not text:
        return None
    last_line = text.split("\n")[-1]
    parts = last_line.split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def _read_mlflow_tag(tag_file: Path) -> str | None:
    """Read a tag value from MLflow filesystem."""
    if not tag_file.exists():
        return None
    return tag_file.read_text(encoding="utf-8").strip()


def _read_mlflow_param(param_file: Path) -> str | None:
    """Read a param value from MLflow filesystem."""
    if not param_file.exists():
        return None
    return param_file.read_text(encoding="utf-8").strip()


def _find_experiment_id(mlruns_dir: Path, experiment_name: str) -> str | None:
    """Find MLflow experiment ID by name.

    Scans experiment directories for matching meta.yaml name field.
    """
    for exp_dir in mlruns_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        meta_file = exp_dir / "meta.yaml"
        if meta_file.exists():
            text = meta_file.read_text(encoding="utf-8")
            for line in text.split("\n"):
                if line.startswith("name:"):
                    name = line.split(":", 1)[1].strip()
                    if name == experiment_name:
                        return exp_dir.name
    return None


def collect_sam3_metrics(
    *,
    mlruns_dir: Path,
    experiment_id: str,
    metric_names: list[str] | None = None,
    num_folds: int = 3,
) -> dict[str, dict[str, list[float]]]:
    """Collect per-model, per-metric fold values from MLflow runs.

    Returns
    -------
    dict mapping model_family → {metric_name → [fold0_val, fold1_val, ...]}
    """
    if metric_names is None:
        metric_names = DEFAULT_METRICS

    exp_dir = mlruns_dir / experiment_id
    if not exp_dir.is_dir():
        logger.warning("Experiment dir not found: %s", exp_dir)
        return {}

    # Collect all runs → group by model_family
    model_runs: dict[str, list[Path]] = {}
    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name in ("meta.yaml", ".trash"):
            continue
        metrics_dir = run_dir / "metrics"
        params_dir = run_dir / "params"
        tags_dir = run_dir / "tags"
        if not metrics_dir.exists():
            continue

        # Get model_family from params or tags
        model_family = _read_mlflow_param(params_dir / "model_family")
        if not model_family:
            model_family = _read_mlflow_tag(tags_dir / "model_family")
        if not model_family:
            # Infer from loss_function tag
            loss_fn = _read_mlflow_tag(tags_dir / "loss_function")
            if loss_fn:
                model_family = f"unknown_{loss_fn}"
            else:
                continue

        if model_family not in model_runs:
            model_runs[model_family] = []
        model_runs[model_family].append(run_dir)

    # Extract per-fold metrics for each model
    results: dict[str, dict[str, list[float]]] = {}
    for model_name, runs in model_runs.items():
        metrics_by_name: dict[str, list[float]] = {m: [] for m in metric_names}

        for run_dir in runs:
            metrics_dir = run_dir / "metrics"
            for fold_id in range(num_folds):
                for metric in metric_names:
                    # Try eval_fold{N}_{metric} naming convention
                    metric_key = f"eval_fold{fold_id}_{metric}"
                    value = _read_mlflow_metric(metrics_dir / metric_key)
                    if value is not None:
                        metrics_by_name[metric].append(value)

        # Only include models with at least one metric value
        if any(len(v) > 0 for v in metrics_by_name.values()):
            results[model_name] = metrics_by_name

    return results


def build_sam3_comparison(
    model_metrics: dict[str, dict[str, list[float]]],
) -> Any:
    """Build a ComparisonTable from per-model metrics.

    Parameters
    ----------
    model_metrics:
        {model_name: {metric_name: [fold0, fold1, ...]}}

    Returns
    -------
    ComparisonTable (from minivess.pipeline.comparison)
    """
    from minivess.pipeline.comparison import ComparisonTable, LossResult, MetricSummary

    losses: list[LossResult] = []
    all_metric_names: set[str] = set()

    for model_name, metrics in model_metrics.items():
        metric_summaries: dict[str, MetricSummary] = {}
        for metric_name, values in metrics.items():
            if not values:
                continue
            all_metric_names.add(metric_name)
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            metric_summaries[metric_name] = MetricSummary(
                mean=mean_val,
                std=std_val,
                ci_lower=mean_val - 1.96 * std_val,
                ci_upper=mean_val + 1.96 * std_val,
                per_fold=values,
            )
        losses.append(
            LossResult(
                loss_name=model_name,
                num_folds=max(len(v) for v in metrics.values()) if metrics else 0,
                metrics=metric_summaries,
            )
        )

    return ComparisonTable(
        losses=losses,
        metric_names=sorted(all_metric_names),
    )


def evaluate_gates_from_metrics(
    model_metrics: dict[str, dict[str, list[float]]],
) -> list[Any]:
    """Evaluate Go/No-Go gates from per-model metric values.

    Parameters
    ----------
    model_metrics:
        {model_name: {metric_name: [fold0, fold1, ...]}}

    Returns
    -------
    list of GateResult
    """
    from minivess.pipeline.sam3_gates import evaluate_all_gates

    # Compute cross-fold means for gate evaluation
    def _mean(model: str, metric: str) -> float:
        values = model_metrics.get(model, {}).get(metric, [])
        return statistics.mean(values) if values else 0.0

    # Map model keys — support both "sam3_vanilla" and variations
    vanilla_key = next(
        (k for k in model_metrics if "vanilla" in k.lower()), "sam3_vanilla"
    )
    topolora_key = next(
        (k for k in model_metrics if "topolora" in k.lower()), "sam3_topolora"
    )
    hybrid_key = next(
        (k for k in model_metrics if "hybrid" in k.lower()), "sam3_hybrid"
    )

    # Gate metrics — use dsc and cldice (or centreline_dsc as fallback)
    vanilla_dsc = _mean(vanilla_key, "dsc")
    vanilla_cldice = _mean(vanilla_key, "cldice") or _mean(
        vanilla_key, "centreline_dsc"
    )
    topolora_dsc = _mean(topolora_key, "dsc")
    topolora_cldice = _mean(topolora_key, "cldice") or _mean(
        topolora_key, "centreline_dsc"
    )
    hybrid_dsc = _mean(hybrid_key, "dsc")

    return evaluate_all_gates(
        vanilla_dsc=vanilla_dsc,
        vanilla_cldice=vanilla_cldice,
        topolora_dsc=topolora_dsc,
        topolora_cldice=topolora_cldice,
        hybrid_dsc=hybrid_dsc,
    )


def write_analysis_artifacts(
    *,
    model_metrics: dict[str, dict[str, list[float]]],
    output_dir: Path,
    gate_results: list[Any] | None = None,
) -> dict[str, Path]:
    """Write analysis output artifacts to disk.

    Parameters
    ----------
    model_metrics:
        Per-model, per-metric fold values.
    output_dir:
        Output directory.
    gate_results:
        Optional pre-computed gate results.

    Returns
    -------
    dict mapping artifact name to file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    # 1. Build comparison table
    table = build_sam3_comparison(model_metrics)
    from minivess.pipeline.comparison import (
        format_comparison_markdown,
    )

    md_text = format_comparison_markdown(table)
    md_path = output_dir / "sam3_comparison.md"
    md_path.write_text(md_text, encoding="utf-8")
    artifacts["comparison_md"] = md_path

    # 2. Gate evaluation
    if gate_results is None:
        gate_results = evaluate_gates_from_metrics(model_metrics)

    gate_data = [
        {
            "gate": r.gate_name,
            "passed": r.passed,
            "description": r.description,
            "observed_value": r.observed_value,
            "threshold": r.threshold,
        }
        for r in gate_results
    ]
    gates_path = output_dir / "gate_results.json"
    gates_path.write_text(
        json.dumps(gate_data, indent=2),
        encoding="utf-8",
    )
    artifacts["gate_results"] = gates_path

    # 3. Raw metrics (for downstream analysis)
    metrics_path = output_dir / "per_model_metrics.json"
    metrics_path.write_text(
        json.dumps(model_metrics, indent=2),
        encoding="utf-8",
    )
    artifacts["metrics_json"] = metrics_path

    # 4. Status file
    status = {
        "status": "success" if len(model_metrics) >= 3 else "partial",
        "timestamp": datetime.now(UTC).isoformat(),
        "models_analyzed": len(model_metrics),
        "model_names": sorted(model_metrics.keys()),
        "artifacts": [str(p.relative_to(output_dir)) for p in artifacts.values()],
    }
    status_path = output_dir / "analysis_status.json"
    status_path.write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    artifacts["status"] = status_path

    return artifacts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze SAM3 experiment results from MLflow",
    )
    parser.add_argument(
        "--mlruns-dir",
        type=Path,
        default=Path("mlruns"),
        help="Path to MLflow mlruns directory",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        nargs="+",
        default=[
            "sam3_vanilla_debug",
            "sam3_topolora_debug",
            "sam3_hybrid_debug",
        ],
        help="MLflow experiment name(s) to analyze",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/sam3"),
        help="Output directory for analysis artifacts",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=3,
        help="Number of folds to extract",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run SAM3 analysis pipeline."""
    args = parse_args(argv)

    logger.info("=" * 70)
    logger.info("SAM3 Experiment Analysis")
    logger.info("=" * 70)
    logger.info("MLflow dir: %s", args.mlruns_dir)
    logger.info("Experiments: %s", args.experiment_name)

    if not args.mlruns_dir.is_dir():
        logger.error("MLflow directory not found: %s", args.mlruns_dir)
        return 1

    # Collect metrics from all experiment directories
    all_model_metrics: dict[str, dict[str, list[float]]] = {}
    for exp_name in args.experiment_name:
        exp_id = _find_experiment_id(args.mlruns_dir, exp_name)
        if exp_id is None:
            logger.warning("Experiment '%s' not found in mlruns", exp_name)
            continue

        logger.info("Found experiment '%s' → ID %s", exp_name, exp_id)
        metrics = collect_sam3_metrics(
            mlruns_dir=args.mlruns_dir,
            experiment_id=exp_id,
            num_folds=args.num_folds,
        )
        all_model_metrics.update(metrics)

    if not all_model_metrics:
        logger.error("No metrics found. Train SAM3 models first.")
        return 1

    logger.info(
        "Collected metrics for %d models: %s",
        len(all_model_metrics),
        sorted(all_model_metrics.keys()),
    )

    # Log summary
    for model_name in sorted(all_model_metrics):
        model_data = all_model_metrics[model_name]
        for metric_name in sorted(model_data):
            values = model_data[metric_name]
            if values:
                mean_val = statistics.mean(values)
                logger.info(
                    "  %s / %s: %.4f (%d folds)",
                    model_name,
                    metric_name,
                    mean_val,
                    len(values),
                )

    # Evaluate gates
    gate_results = evaluate_gates_from_metrics(all_model_metrics)

    # Write artifacts
    artifacts = write_analysis_artifacts(
        model_metrics=all_model_metrics,
        output_dir=args.output_dir,
        gate_results=gate_results,
    )

    logger.info("Analysis artifacts written to %s:", args.output_dir)
    for name, path in artifacts.items():
        logger.info("  %s: %s", name, path)

    # Print gate summary
    print("\n" + "=" * 60)
    print("SAM3 Go/No-Go Gate Evaluation")
    print("=" * 60)
    for r in gate_results:
        status = "PASS" if r.passed else "FAIL"
        marker = "[+]" if r.passed else "[-]"
        print(f"\n  {marker} {r.gate_name}: {status}")
        print(f"      {r.description}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
