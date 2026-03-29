"""End-to-end tests for the biostatistics pipeline on fixture DuckDB.

Phase 4: Run the full statistical pipeline on synthetic data with known
effects and verify all expected outputs are produced. Uses DuckDB-only
data loading (Phase 2 build_per_volume_data_from_duckdb).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pytest

from minivess.config.biostatistics_config import BiostatisticsConfig
from minivess.pipeline.biostatistics_duckdb import (
    _ALL_DDL,
    build_per_volume_data_from_duckdb,
)
from minivess.pipeline.biostatistics_statistics import (
    apply_hierarchical_gatekeeping,
    bootstrap_ci,
    compute_pairwise_comparisons,
    compute_variance_decomposition,
    sensitivity_concordance,
    stratified_permutation_test,
)

_CFG = BiostatisticsConfig()


@pytest.fixture()
def fixture_db(tmp_path: Path) -> Path:
    """Create a fixture DuckDB with 2 conditions, 3 folds, 23 volumes."""
    db_path = tmp_path / "biostatistics.duckdb"
    conn = duckdb.connect(str(db_path))
    for ddl in _ALL_DDL:
        conn.execute(ddl)

    rng = np.random.default_rng(_CFG.seed)

    for loss in ["dice_ce", "cbdice_cldice"]:
        base_dsc = 0.80 if loss == "cbdice_cldice" else 0.77
        for fold_id in range(3):
            run_id = f"run_{loss}_f{fold_id}"
            conn.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    run_id, "exp1", "smoke_mini_eval", loss, fold_id,
                    "dynunet", False, "FINISHED", "2026-03-29",
                    "none", "none", "per_loss_single_best", False,
                ],
            )
            for vol_idx in range(23):
                for metric, base in [("dsc", base_dsc), ("cldice", base_dsc + 0.05), ("masd", 1.5)]:
                    val = float(rng.normal(base, 0.03)) if metric != "masd" else float(rng.exponential(base))
                    conn.execute(
                        "INSERT INTO per_volume_metrics VALUES (?, ?, ?, ?, ?)",
                        [run_id, fold_id, f"mv{vol_idx:02d}", metric, val],
                    )
    conn.close()
    return db_path


class TestFullPipelineProducesResults:
    """The complete biostatistics pipeline runs without error on fixture data."""

    def test_data_loading(self, fixture_db: Path) -> None:
        data = build_per_volume_data_from_duckdb(fixture_db, metrics=["dsc", "cldice", "masd"])
        assert len(data) == 3  # 3 metrics
        for metric in ["dsc", "cldice", "masd"]:
            assert metric in data
            assert len(data[metric]) == 2  # 2 conditions

    def test_pairwise_comparisons(self, fixture_db: Path) -> None:
        data = build_per_volume_data_from_duckdb(fixture_db, metrics=["dsc"])
        results = compute_pairwise_comparisons(
            per_volume_data=data["dsc"],
            metric_name="dsc",
            alpha=_CFG.alpha,
            primary_metric="cldice",
            n_bootstrap=100,
            seed=_CFG.seed,
        )
        assert len(results) == 1  # 2 conditions → 1 pair
        assert results[0].metric == "dsc"
        assert 0.0 <= results[0].p_value <= 1.0

    def test_stratified_permutation(self, fixture_db: Path) -> None:
        data = build_per_volume_data_from_duckdb(fixture_db, metrics=["dsc"])
        conditions = sorted(data["dsc"].keys())
        result = stratified_permutation_test(
            fold_data_a=data["dsc"][conditions[0]],
            fold_data_b=data["dsc"][conditions[1]],
            n_permutations=199,
            seed=_CFG.seed,
        )
        assert result.n_folds == 3
        assert result.n_permutations == 199

    def test_variance_decomposition(self, fixture_db: Path) -> None:
        data = build_per_volume_data_from_duckdb(fixture_db, metrics=["dsc"])
        results = compute_variance_decomposition(
            per_volume_data=data["dsc"],
            metric_name="dsc",
            friedman_alpha=_CFG.alpha,
        )
        assert len(results) == 1
        assert results[0].metric == "dsc"
        assert results[0].icc_type == "ICC2"

    def test_bootstrap_ci_on_real_data(self, fixture_db: Path) -> None:
        data = build_per_volume_data_from_duckdb(fixture_db, metrics=["dsc"])
        conditions = sorted(data["dsc"].keys())
        # Pool one condition's fold-0 data for CI
        scores = data["dsc"][conditions[0]][0]
        lo, hi, method = bootstrap_ci(scores, n_bootstrap=200, seed=_CFG.seed)
        assert lo < hi
        assert method == "bca"  # N=23 >= bca_min_n=20

    def test_sensitivity_concordance_on_real_data(self, fixture_db: Path) -> None:
        data = build_per_volume_data_from_duckdb(fixture_db, metrics=["dsc"])
        conditions = sorted(data["dsc"].keys())
        a = np.concatenate([data["dsc"][conditions[0]][f] for f in sorted(data["dsc"][conditions[0]])])
        b = np.concatenate([data["dsc"][conditions[1]][f] for f in sorted(data["dsc"][conditions[1]])])
        n = min(len(a), len(b))
        result = sensitivity_concordance(a[:n], b[:n], alpha=_CFG.alpha)
        assert "concordant" in result
        assert isinstance(result["concordant"], bool)

    def test_hierarchical_gatekeeping_on_real_pvalues(self, fixture_db: Path) -> None:
        data = build_per_volume_data_from_duckdb(fixture_db, metrics=["dsc", "cldice", "masd"])
        # Compute p-values per metric
        p_values = {}
        for metric in ["dsc", "cldice", "masd"]:
            results = compute_pairwise_comparisons(
                per_volume_data=data[metric],
                metric_name=metric,
                alpha=_CFG.alpha,
                primary_metric="cldice",
                n_bootstrap=100,
                seed=_CFG.seed,
            )
            if results:
                p_values[metric] = results[0].p_value

        gatekeep = apply_hierarchical_gatekeeping(
            co_primary_p={m: p_values[m] for m in ["cldice", "masd"] if m in p_values},
            secondary_p={m: p_values[m] for m in ["dsc"] if m in p_values},
            alpha=_CFG.alpha,
        )
        assert "cldice" in gatekeep
        assert "tested" in gatekeep["cldice"]


class TestPowerAnalysis:
    """Phase 4.2: Power analysis produces quantified results."""

    def test_bootstrap_ci_width_decreases_with_n(self) -> None:
        """Larger N → narrower CI (sanity check)."""
        rng = np.random.default_rng(42)
        data_small = rng.normal(0.8, 0.03, size=10)
        data_large = rng.normal(0.8, 0.03, size=100)

        lo_s, hi_s, _ = bootstrap_ci(data_small, n_bootstrap=500, seed=42)
        lo_l, hi_l, _ = bootstrap_ci(data_large, n_bootstrap=500, seed=42)

        width_small = hi_s - lo_s
        width_large = hi_l - lo_l
        assert width_large < width_small, "Larger N should give narrower CI"


class TestMultipleComparisonsDisclosure:
    """Phase 4.3: Total test count and effective alpha reporting."""

    def test_total_test_count(self) -> None:
        """With 2 conditions and 3 metrics, pairwise count = C(2,2) × 3 = 3."""
        from itertools import combinations

        n_conditions = 2
        n_metrics = 3
        n_pairwise = len(list(combinations(range(n_conditions), 2))) * n_metrics
        assert n_pairwise == 3

    def test_effective_alpha_after_bonferroni(self) -> None:
        """Co-primary Bonferroni: alpha/n_coprimary."""
        n_coprimary = 2
        effective = _CFG.alpha / n_coprimary
        assert effective == pytest.approx(0.025)
