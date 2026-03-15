"""Scan MLflow experiment outputs and update knowledge-graph/experiments/*.yaml.

Reads run metrics from Parquet exports (outputs/duckdb/parquet/) or directly
from mlruns/ and writes/updates knowledge-graph/experiments/*.yaml with
champion metrics and per-run summaries.

Usage:
    uv run python scripts/scan_experiments.py
    uv run python scripts/scan_experiments.py --dry-run

CLAUDE.md Rule #16: import re is BANNED.
CLAUDE.md Rule #6:  Use pathlib.Path, never string concatenation.
CLAUDE.md Rule #7:  Use datetime.now(UTC) — never datetime.now().
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
PARQUET_DIR = REPO_ROOT / "outputs" / "duckdb" / "parquet"
KG_EXPERIMENTS_DIR = REPO_ROOT / "knowledge-graph" / "experiments"


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


@dataclass
class RunMetrics:
    """Metrics from a single training run."""

    run_id: str
    experiment_name: str
    model_family: str
    loss_function: str
    fold: int
    dsc_mean: float
    dsc_std: float = 0.0
    cldice_mean: float = 0.0
    cldice_std: float = 0.0
    epochs: int = 0


# ---------------------------------------------------------------------------
# Champion selection
# ---------------------------------------------------------------------------


def find_champion(runs: list[RunMetrics], metric: str) -> RunMetrics | None:
    """Return the run with the highest value for the given metric.

    Args:
        runs: List of RunMetrics to compare.
        metric: Attribute name on RunMetrics (e.g., 'dsc_mean', 'cldice_mean').

    Returns:
        The RunMetrics with the maximum metric value, or None if runs is empty.
    """
    if not runs:
        return None
    return max(runs, key=lambda r: getattr(r, metric, 0.0))


# ---------------------------------------------------------------------------
# Data loading: Parquet files
# ---------------------------------------------------------------------------


def load_runs_from_parquet(parquet_dir: Path) -> list[RunMetrics]:
    """Load run metrics from Parquet export directory.

    Reads JSON sidecar files ('.json' extension) alongside Parquet files.
    If no files exist, returns an empty list — no error raised.

    Args:
        parquet_dir: Directory containing per-run JSON/Parquet exports.

    Returns:
        List of RunMetrics for all parseable runs found.
    """
    if not parquet_dir.exists():
        return []

    runs: list[RunMetrics] = []
    for json_file in sorted(parquet_dir.glob("*.json")):
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        try:
            runs.append(
                RunMetrics(
                    run_id=str(raw.get("run_id", json_file.stem)),
                    experiment_name=str(raw.get("experiment_name", "")),
                    model_family=str(raw.get("model_family", "")),
                    loss_function=str(raw.get("loss_function", "")),
                    fold=int(raw.get("fold", 0)),
                    dsc_mean=float(raw.get("dsc_mean", 0.0)),
                    dsc_std=float(raw.get("dsc_std", 0.0)),
                    cldice_mean=float(raw.get("cldice_mean", 0.0)),
                    cldice_std=float(raw.get("cldice_std", 0.0)),
                    epochs=int(raw.get("epochs", 0)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    return runs


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------


def write_experiments_yaml(
    runs: list[RunMetrics],
    experiment_name: str,
    out_path: Path,
) -> None:
    """Write experiment runs and champions to a YAML file (idempotent).

    Args:
        runs: List of RunMetrics from this experiment.
        experiment_name: Experiment identifier (e.g., 'dynunet_loss_variation_v2').
        out_path: Destination YAML file path.
    """
    champion_cldice = find_champion(runs, "cldice_mean")
    champion_dsc = find_champion(runs, "dsc_mean")

    def _champion_dict(r: RunMetrics | None) -> dict[str, Any]:
        if r is None:
            return {}
        return {
            "run_id": r.run_id,
            "loss_function": r.loss_function,
            "model_family": r.model_family,
            "dsc_mean": r.dsc_mean,
            "dsc_std": r.dsc_std,
            "cldice_mean": r.cldice_mean,
            "cldice_std": r.cldice_std,
        }

    runs_list = sorted(
        [asdict(r) for r in runs],
        key=lambda r: (r["loss_function"], r["fold"]),
    )

    data: dict[str, Any] = {
        "_meta": {
            "generated_by": "scripts/scan_experiments.py",
            "experiment_name": experiment_name,
            "last_updated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_runs": len(runs),
            "note": "Auto-generated — do not edit manually. Run /kg-sync to regenerate.",
        },
        "champion": {
            "by_cldice": _champion_dict(champion_cldice),
            "by_dsc": _champion_dict(champion_dsc),
        },
        "runs": runs_list,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan MLflow Parquet exports and update knowledge-graph/experiments/*.yaml"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=PARQUET_DIR,
        help="Directory with per-run JSON/Parquet exports",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=KG_EXPERIMENTS_DIR,
        help="Output directory for experiment YAMLs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    runs = load_runs_from_parquet(args.parquet_dir)
    print(f"Loaded {len(runs)} runs from {args.parquet_dir}")

    if not runs:
        print(
            "No runs found — nothing to write. Run scripts/export_duckdb_parquet.py first."
        )
        return

    # Group by experiment name
    by_experiment: dict[str, list[RunMetrics]] = {}
    for run in runs:
        by_experiment.setdefault(run.experiment_name, []).append(run)

    for exp_name, exp_runs in sorted(by_experiment.items()):
        out_path = args.out_dir / f"{exp_name}_generated.yaml"
        if args.dry_run:
            print(f"[dry-run] Would write {len(exp_runs)} runs → {out_path}")
            continue
        write_experiments_yaml(exp_runs, exp_name, out_path)
        print(f"Written: {out_path} ({len(exp_runs)} runs)")


if __name__ == "__main__":
    main()
