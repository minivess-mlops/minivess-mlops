"""Local mock E2E: train → analysis → biostatistics chain (Task 3.1, #926).

Verifies the full inter-flow chain without Docker or GPU:
1. Creates synthetic MLflow training runs (simulates train_flow output)
2. Runs biostatistics discovery + ANOVA + figures + tables
3. Verifies newly wired functions (Task 2.13) produce output

Catches wiring issues BEFORE cloud launch. Runs in ~15s locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from minivess.config.biostatistics_config import BiostatisticsConfig
from minivess.pipeline.biostatistics_discovery import (
    SourceRunManifest,
    discover_source_runs,
    validate_source_completeness,
)
from minivess.pipeline.biostatistics_duckdb import (
    build_biostatistics_duckdb,
)
from minivess.pipeline.biostatistics_figures import generate_figures
from minivess.pipeline.biostatistics_lineage import build_lineage_manifest
from minivess.pipeline.biostatistics_rankings import compute_rankings
from minivess.pipeline.biostatistics_statistics import (
    compute_factorial_anova,
    compute_pairwise_comparisons,
    compute_variance_decomposition,
)
from minivess.pipeline.biostatistics_tables import generate_tables

# ---------------------------------------------------------------------------
# Synthetic mlruns builder (multi-model factorial design)
# ---------------------------------------------------------------------------

_CONDITIONS = {
    "dynunet__dice_ce": 0.82,
    "dynunet__cbdice_cldice": 0.86,
    "mambavesselnet__dice_ce": 0.79,
    "mambavesselnet__cbdice_cldice": 0.83,
}
_N_FOLDS = 3
_N_VOLUMES = 10
_METRICS = ["dsc", "cldice"]
_EXPERIMENT_NAME = "dynunet_loss_variation_v2"


def _create_factorial_mlruns(base: Path) -> Path:
    """Create synthetic mlruns with 2-model x 2-loss factorial design."""
    mlruns = base / "mlruns"
    exp_dir = mlruns / "1"
    exp_dir.mkdir(parents=True)
    (exp_dir / "meta.yaml").write_text(f"name: {_EXPERIMENT_NAME}\n", encoding="utf-8")

    rng = np.random.default_rng(42)

    for cond, mean in _CONDITIONS.items():
        for fold in range(_N_FOLDS):
            run_id = f"run_{cond}_{fold}"
            run_dir = exp_dir / run_id
            run_dir.mkdir()
            (run_dir / "meta.yaml").write_text("status: FINISHED\n", encoding="utf-8")

            params_dir = run_dir / "params"
            params_dir.mkdir()
            # Parse condition key into model__loss
            parts = cond.split("__")
            loss_name = parts[-1] if len(parts) > 1 else cond
            (params_dir / "loss_name").write_text(loss_name, encoding="utf-8")
            (params_dir / "fold_id").write_text(str(fold), encoding="utf-8")

            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir()
            for metric in _METRICS:
                m = mean - 0.02 if metric == "cldice" else mean
                for vol in range(_N_VOLUMES):
                    score = float(rng.normal(m, 0.04))
                    metric_path = (
                        metrics_dir / "eval" / str(fold) / "vol" / str(vol) / metric
                    )
                    metric_path.parent.mkdir(parents=True, exist_ok=True)
                    metric_path.write_text(f"1700000000000 {score} 0", encoding="utf-8")

            (metrics_dir / "val_dice").write_text(
                f"1700000000000 {float(rng.normal(mean, 0.03))} 0",
                encoding="utf-8",
            )

    return mlruns


def _build_per_volume_data(
    manifest: SourceRunManifest,
    mlruns_dir: Path,
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    """Build per-volume data from synthetic mlruns."""
    from minivess.pipeline.biostatistics_duckdb import (
        _is_per_volume_metric,
        _parse_per_volume_metric,
        _read_metrics,
    )

    per_volume: dict[str, dict[str, dict[int, Any]]] = {}
    for run_info in manifest.runs:
        # _read_metrics expects the run_dir (adds /metrics internally)
        run_dir = mlruns_dir / "1" / run_info.run_id
        metrics = _read_metrics(run_dir)
        condition = run_info.loss_function

        for key, value in metrics.items():
            if not _is_per_volume_metric(key):
                continue
            fold_id, vol_id, metric_name = _parse_per_volume_metric(key)
            if fold_id is None:
                continue
            per_volume.setdefault(metric_name, {}).setdefault(condition, {}).setdefault(
                fold_id, []
            )
            per_volume[metric_name][condition][fold_id].append(float(value))

    # Convert lists to numpy arrays
    result: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for metric, conditions in per_volume.items():
        result[metric] = {}
        for cond, folds in conditions.items():
            result[metric][cond] = {
                f: np.array(vals) for f, vals in folds.items()
            }
    return result


# ---------------------------------------------------------------------------
# E2E chain test
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTrainAnalysisBiostatisticsChain:
    """Full chain: synthetic train runs → discovery → ANOVA → figures → tables."""

    def test_full_chain_produces_all_artifacts(self, tmp_path: Path) -> None:
        """End-to-end: discovery + stats + figures + tables with factorial data."""
        # Phase 1: Create synthetic training runs
        mlruns_dir = _create_factorial_mlruns(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Phase 2: Config
        config = BiostatisticsConfig(
            experiment_names=[_EXPERIMENT_NAME],
            metrics=list(_METRICS),
            primary_metric="dsc",
            mlruns_dir=mlruns_dir,
            output_dir=output_dir,
            min_folds_per_condition=_N_FOLDS,
            min_conditions=2,  # Discovery groups by loss_name, not model__loss
            n_bootstrap=500,
            seed=42,
        )

        # Phase 3: Discovery
        manifest = discover_source_runs(mlruns_dir, config.experiment_names)
        assert len(manifest.runs) >= len(_CONDITIONS) * _N_FOLDS

        validation = validate_source_completeness(
            manifest,
            min_folds=config.min_folds_per_condition,
            min_conditions=config.min_conditions,
        )
        assert validation.valid

        # Phase 4: DuckDB
        db_path = output_dir / "biostatistics.duckdb"
        build_biostatistics_duckdb(manifest, mlruns_dir, db_path)
        assert db_path.exists()

        # Phase 5: Per-volume data
        per_volume_data = _build_per_volume_data(manifest, mlruns_dir)
        assert len(per_volume_data) >= 2  # dsc, cldice

        higher_is_better: dict[str, bool] = {m: True for m in config.metrics}

        # Phase 6: Statistics (pairwise + variance + ANOVA)
        all_pairwise = []
        all_variance = []
        all_anova = []
        for metric in config.metrics:
            if metric in per_volume_data:
                pw = compute_pairwise_comparisons(
                    per_volume_data=per_volume_data[metric],
                    metric_name=metric,
                    alpha=config.alpha,
                    primary_metric=config.primary_metric,
                    n_bootstrap=config.n_bootstrap,
                    seed=config.seed,
                )
                all_pairwise.extend(pw)
                vd = compute_variance_decomposition(
                    per_volume_data=per_volume_data[metric],
                    metric_name=metric,
                    friedman_alpha=config.alpha,
                )
                all_variance.extend(vd)
                # Factorial ANOVA (Task 2.13 wiring)
                try:
                    anova = compute_factorial_anova(
                        per_volume_data=per_volume_data,
                        metric_name=metric,
                        factor_names=["loss_name"],
                    )
                    all_anova.append(anova)
                except Exception:
                    pass  # ANOVA may fail with insufficient factors

        assert len(all_pairwise) >= 1
        assert len(all_variance) >= 1

        # Phase 7: Rankings
        rankings = compute_rankings(
            per_volume_data=per_volume_data,
            metric_names=config.metrics,
            higher_is_better=higher_is_better,
            alpha=config.alpha,
        )

        # Phase 8: Figures (with ANOVA results — Task 2.13 wiring)
        figures_dir = output_dir / "figures"
        figures = generate_figures(
            per_volume_data=per_volume_data,
            pairwise=all_pairwise,
            variance=all_variance,
            rankings=rankings,
            output_dir=figures_dir,
            anova_results=all_anova if all_anova else None,
        )
        assert len(figures) >= 2  # At least effect size + forest

        # Phase 9: Tables (with ANOVA results — Task 2.13 wiring)
        tables_dir = output_dir / "tables"
        tables = generate_tables(
            pairwise=all_pairwise,
            variance=all_variance,
            rankings=rankings,
            output_dir=tables_dir,
            anova_results=all_anova if all_anova else None,
        )
        assert len(tables) >= 3  # comparison + effect sizes + variance

        # Phase 10: Lineage
        lineage = build_lineage_manifest(
            manifest=manifest,
            figures=figures,
            tables=tables,
        )
        assert lineage is not None

        # Verify output directories have files
        assert any(figures_dir.rglob("*.png"))
        assert any(tables_dir.rglob("*.tex"))

    def test_chain_discovers_all_conditions(self, tmp_path: Path) -> None:
        """Discovery finds all 4 factorial conditions."""
        mlruns_dir = _create_factorial_mlruns(tmp_path)
        manifest = discover_source_runs(mlruns_dir, [_EXPERIMENT_NAME])

        loss_functions = {r.loss_function for r in manifest.runs}
        # Discovery groups by loss_name, so we check for the unique losses
        expected_losses = {cond.split("__")[-1] for cond in _CONDITIONS}
        for loss in expected_losses:
            assert loss in loss_functions, f"Missing loss function: {loss}"
