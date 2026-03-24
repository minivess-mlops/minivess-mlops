"""Cross-approach topology comparison evaluator.

Compares all conditions from topology-aware segmentation experiments.
Computes mean +/- std across folds, runs paired bootstrap tests,
evaluates predictions P1/P3/P4, and exports to Markdown + LaTeX.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TopologyComparisonEvaluator:
    """Evaluates and compares topology-aware segmentation approaches.

    Args:
        results: Dict mapping condition name -> fold results list.
            Each fold result is a dict of metric_name -> float.
        metric_names: List of metric names to compare.
    """

    def __init__(
        self,
        results: dict[str, list[dict[str, float]]],
        metric_names: list[str] | None = None,
    ) -> None:
        self.results = results
        self.metric_names = metric_names or [
            "dice",
            "hd95",
            "assd",
            "nsd",
            "cldice",
            "betti_error_0",
            "betti_error_1",
            "junction_f1",
        ]

    def compute_summary(self) -> dict[str, dict[str, dict[str, float]]]:
        """Compute mean and std for each condition x metric.

        Returns:
            Nested dict: condition -> metric -> {"mean": float, "std": float}
        """
        summary: dict[str, dict[str, dict[str, float]]] = {}
        for condition, fold_results in self.results.items():
            summary[condition] = {}
            for metric in self.metric_names:
                values = [fr.get(metric, 0.0) for fr in fold_results]
                arr = np.array(values)
                summary[condition][metric] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                }
        return summary

    def paired_bootstrap(
        self,
        condition_a: str,
        condition_b: str,
        metric: str,
        n_bootstrap: int = 10000,
        *,
        seed: int,
    ) -> dict[str, float]:
        """Paired bootstrap test between two conditions.

        Returns:
            Dict with "p_value" and "mean_diff" (a - b).
        """
        rng = np.random.default_rng(seed)
        values_a = np.array([fr.get(metric, 0.0) for fr in self.results[condition_a]])
        values_b = np.array([fr.get(metric, 0.0) for fr in self.results[condition_b]])

        observed_diff = float(np.mean(values_a) - np.mean(values_b))
        diffs = values_a - values_b
        n = len(diffs)

        count_extreme = 0
        for _ in range(n_bootstrap):
            boot_idx = rng.integers(0, n, size=n)
            boot_diff = float(np.mean(diffs[boot_idx]))
            if boot_diff <= 0:
                count_extreme += 1

        p_value = count_extreme / n_bootstrap
        return {"p_value": p_value, "mean_diff": observed_diff}

    def evaluate_predictions(self) -> dict[str, dict[str, Any]]:
        """Evaluate predictions P1, P3, P4 as pass/fail.

        P1: D2C improves clDice by > 3pp (or secondary gate: any metric > 2pp)
        P3: Multi-task improves clDice by > 1pp
        P4: TFFM — reports diff (informational)

        Returns:
            Dict with prediction results.
        """
        summary = self.compute_summary()
        predictions: dict[str, dict[str, Any]] = {}

        # P1: D2C effect
        if "baseline" in summary and "d2c_only" in summary:
            baseline_cldice = summary["baseline"].get("cldice", {}).get("mean", 0.0)
            d2c_cldice = summary["d2c_only"].get("cldice", {}).get("mean", 0.0)
            diff_pp = (d2c_cldice - baseline_cldice) * 100
            primary_pass = diff_pp > 3.0

            # Secondary gate: any metric improves > 2pp
            secondary_pass = False
            for metric in self.metric_names:
                base_val = summary["baseline"].get(metric, {}).get("mean", 0.0)
                d2c_val = summary["d2c_only"].get(metric, {}).get("mean", 0.0)
                if (d2c_val - base_val) * 100 > 2.0:
                    secondary_pass = True
                    break

            predictions["P1"] = {
                "pass": primary_pass or secondary_pass,
                "cldice_diff_pp": diff_pp,
                "primary_pass": primary_pass,
                "secondary_pass": secondary_pass,
            }

        # P3: Multi-task effect
        if "baseline" in summary and "multitask" in summary:
            baseline_cldice = summary["baseline"].get("cldice", {}).get("mean", 0.0)
            mt_cldice = summary["multitask"].get("cldice", {}).get("mean", 0.0)
            diff_pp = (mt_cldice - baseline_cldice) * 100
            predictions["P3"] = {
                "pass": diff_pp > 1.0,
                "cldice_diff_pp": diff_pp,
            }

        # P4: TFFM effect (informational)
        if "baseline" in summary and "tffm" in summary:
            baseline_cldice = summary["baseline"].get("cldice", {}).get("mean", 0.0)
            tffm_cldice = summary["tffm"].get("cldice", {}).get("mean", 0.0)
            diff_pp = (tffm_cldice - baseline_cldice) * 100
            predictions["P4"] = {
                "pass": None,  # Informational only
                "cldice_diff_pp": diff_pp,
                "note": "Informational — no pass/fail gate",
            }

        return predictions

    def export_markdown(self) -> str:
        """Export comparison table as Markdown."""
        summary = self.compute_summary()
        conditions = list(summary.keys())

        header = "| Condition | " + " | ".join(self.metric_names) + " |"
        separator = "|---|" + "|".join(["---"] * len(self.metric_names)) + "|"
        rows = [header, separator]

        for condition in conditions:
            cells = []
            for metric in self.metric_names:
                stats = summary[condition].get(metric, {"mean": 0.0, "std": 0.0})
                cells.append(f"{stats['mean']:.3f} +/- {stats['std']:.3f}")
            rows.append(f"| {condition} | " + " | ".join(cells) + " |")

        return "\n".join(rows)

    def export_latex(self) -> str:
        """Export comparison table as LaTeX."""
        summary = self.compute_summary()
        conditions = list(summary.keys())

        col_spec = "l" + "c" * len(self.metric_names)
        lines = [
            "\\begin{tabular}{" + col_spec + "}",
            "\\toprule",
            "Condition & " + " & ".join(self.metric_names) + " \\\\",
            "\\midrule",
        ]

        for condition in conditions:
            cells = []
            for metric in self.metric_names:
                stats = summary[condition].get(metric, {"mean": 0.0, "std": 0.0})
                cells.append(f"${stats['mean']:.3f} \\pm {stats['std']:.3f}$")
            lines.append(f"{condition} & " + " & ".join(cells) + " \\\\")

        lines.extend(["\\bottomrule", "\\end{tabular}"])
        return "\n".join(lines)
