"""Cross-experiment comparison: full-width vs half-width DynUNet.

Reads eval metrics from both experiments, builds comparison tables,
and generates side-by-side analysis figures and combined LaTeX.

Run:
    uv run python scripts/compare_experiments.py
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

EXPERIMENTS = {
    "full_width": {
        "id": "843896622863223169",
        "name": "dynunet_loss_variation_v2",
        "label": "Full-width (128 filters)",
    },
    "half_width": {
        "id": "859817033030295110",
        "name": "dynunet_half_width_v1",
        "label": "Half-width (64 filters)",
    },
}

N_FOLDS = 3
EVAL_METRICS = ["dsc", "centreline_dsc", "measured_masd"]
METRIC_LABELS = {
    "dsc": "Dice (DSC)",
    "centreline_dsc": "Centreline DSC",
    "measured_masd": "MASD",
}
MAXIMIZE_METRICS = {"dsc", "centreline_dsc"}


def read_metric(exp_id: str, run_id: str, metric_name: str) -> float:
    """Read the last metric value from MLflow filesystem."""
    metric_file = MLRUNS_DIR / exp_id / run_id / "metrics" / metric_name
    if not metric_file.exists():
        return float("nan")
    lines = metric_file.read_text(encoding="utf-8").strip().split("\n")
    if lines:
        parts = lines[-1].split()
        if len(parts) >= 2:
            return float(parts[1])
    return float("nan")


def read_tag(exp_id: str, run_id: str, tag_key: str) -> str:
    """Read a tag value from MLflow filesystem."""
    tag_file = MLRUNS_DIR / exp_id / run_id / "tags" / tag_key
    if not tag_file.exists():
        return ""
    return tag_file.read_text(encoding="utf-8").strip()


def find_complete_runs(exp_id: str) -> list[str]:
    """Find complete runs with 3-fold evaluation."""
    exp_dir = MLRUNS_DIR / exp_id
    if not exp_dir.exists():
        return []
    complete: list[str] = []
    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "meta.yaml").exists():
            continue
        metrics_dir = run_dir / "metrics"
        if metrics_dir.exists():
            has_fold2 = any(
                f.name.startswith("eval_fold2") for f in metrics_dir.iterdir()
            )
            if has_fold2:
                complete.append(run_dir.name)
    return complete


def collect_experiment_data(
    exp_id: str,
) -> dict[str, dict[str, list[float]]]:
    """Collect per-fold metrics for all losses in an experiment."""
    runs = find_complete_runs(exp_id)
    data: dict[str, dict[str, list[float]]] = {}

    for run_id in runs:
        loss = read_tag(exp_id, run_id, "loss_function")
        if not loss:
            continue
        data[loss] = {}
        for metric in EVAL_METRICS:
            values = [
                read_metric(exp_id, run_id, f"eval_fold{fold}_{metric}")
                for fold in range(N_FOLDS)
            ]
            data[loss][metric] = values

    return data


def generate_comparison_figure(
    full_data: dict[str, dict[str, list[float]]],
    half_data: dict[str, dict[str, list[float]]],
    output_dir: Path,
) -> None:
    """Generate side-by-side comparison figure."""
    from minivess.pipeline.viz.figure_export import save_figure

    # Find common losses
    common_losses = sorted(set(full_data.keys()) & set(half_data.keys()))
    if not common_losses:
        logger.warning("No common losses between experiments")
        return

    fig, axes = plt.subplots(1, len(EVAL_METRICS), figsize=(5 * len(EVAL_METRICS), 5))
    if len(EVAL_METRICS) == 1:
        axes = [axes]

    x = np.arange(len(common_losses))
    width = 0.35

    for ax, metric in zip(axes, EVAL_METRICS, strict=False):
        full_means = []
        full_stds = []
        half_means = []
        half_stds = []

        for loss in common_losses:
            fv = [v for v in full_data[loss].get(metric, []) if not math.isnan(v)]
            hv = [v for v in half_data[loss].get(metric, []) if not math.isnan(v)]

            full_means.append(np.mean(fv) if fv else 0)
            full_stds.append(np.std(fv) if len(fv) > 1 else 0)
            half_means.append(np.mean(hv) if hv else 0)
            half_stds.append(np.std(hv) if len(hv) > 1 else 0)

        ax.bar(
            x - width / 2,
            full_means,
            width,
            yerr=full_stds,
            label="Full-width",
            color="#4C72B0",
            capsize=3,
        )
        ax.bar(
            x + width / 2,
            half_means,
            width,
            yerr=half_stds,
            label="Half-width",
            color="#DD8452",
            capsize=3,
        )

        ax.set_xlabel("Loss Function")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xticks(x)
        ax.set_xticklabels(common_losses, rotation=45, ha="right", fontsize=8)
        ax.legend()

    fig.suptitle("Full-width vs Half-width DynUNet", fontsize=14, fontweight="bold")
    fig.tight_layout()

    save_figure(
        fig,
        "cross_experiment_comparison",
        output_dir=output_dir,
        formats=["png", "svg"],
    )
    plt.close(fig)
    logger.info("  Saved cross-experiment comparison figure")


def generate_delta_table(
    full_data: dict[str, dict[str, list[float]]],
    half_data: dict[str, dict[str, list[float]]],
) -> str:
    """Generate markdown table showing delta between experiments."""
    common_losses = sorted(set(full_data.keys()) & set(half_data.keys()))

    lines = ["| Loss | Metric | Full-width | Half-width | Delta | Winner |"]
    lines.append("| --- | --- | --- | --- | --- | --- |")

    for loss in common_losses:
        for metric in EVAL_METRICS:
            fv = [v for v in full_data[loss].get(metric, []) if not math.isnan(v)]
            hv = [v for v in half_data[loss].get(metric, []) if not math.isnan(v)]

            f_mean = np.mean(fv) if fv else float("nan")
            h_mean = np.mean(hv) if hv else float("nan")

            if math.isnan(f_mean) or math.isnan(h_mean):
                delta_str = "N/A"
                winner = "N/A"
            else:
                delta = f_mean - h_mean
                delta_str = f"{delta:+.4f}"
                maximize = metric in MAXIMIZE_METRICS
                if maximize:
                    winner = "Full" if delta > 0 else "Half"
                else:
                    winner = "Full" if delta < 0 else "Half"

            lines.append(
                f"| {loss} | {metric} | {f_mean:.4f} | {h_mean:.4f} | {delta_str} | {winner} |"
            )

    return "\n".join(lines)


def main() -> int:
    """Run cross-experiment comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-experiment comparison")
    parser.add_argument(
        "--output-dir", default="outputs/comparison", help="Output directory"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("CROSS-EXPERIMENT COMPARISON: FULL vs HALF WIDTH")
    logger.info("=" * 70)

    # Collect data
    full_data = collect_experiment_data(EXPERIMENTS["full_width"]["id"])
    half_data = collect_experiment_data(EXPERIMENTS["half_width"]["id"])

    logger.info("\n  Full-width losses: %s", sorted(full_data.keys()))
    logger.info("  Half-width losses: %s", sorted(half_data.keys()))

    # Print summary per experiment
    for exp_key, exp_info in EXPERIMENTS.items():
        data = full_data if exp_key == "full_width" else half_data
        logger.info("\n  [%s]", exp_info["label"])
        for loss, metrics in sorted(data.items()):
            dsc_vals = [v for v in metrics.get("dsc", []) if not math.isnan(v)]
            cldsc_vals = [
                v for v in metrics.get("centreline_dsc", []) if not math.isnan(v)
            ]
            logger.info(
                "    %s: DSC=%.4f, clDSC=%.4f",
                loss,
                np.mean(dsc_vals) if dsc_vals else float("nan"),
                np.mean(cldsc_vals) if cldsc_vals else float("nan"),
            )

    # Generate comparison figure
    logger.info("\n  Generating comparison figure...")
    generate_comparison_figure(full_data, half_data, output_dir)

    # Generate delta table
    delta_md = generate_delta_table(full_data, half_data)
    delta_path = output_dir / "cross_experiment_delta.md"
    delta_path.write_text(delta_md, encoding="utf-8")
    logger.info("  Delta table: %s", delta_path)

    # Print delta table
    logger.info("\n  --- Delta Table ---")
    for line in delta_md.split("\n"):
        logger.info("  %s", line)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 70)

    all_files = list(output_dir.rglob("*"))
    artifact_files = [f for f in all_files if f.is_file()]
    for f in sorted(artifact_files):
        logger.info(
            "  %s (%.1f KB)", f.relative_to(output_dir), f.stat().st_size / 1024
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
