"""Nature Protocols compliance artifacts for the biostatistics flow.

Generates:
- CONSORT flow diagram data (programmatic, not hand-drawn)
- Timing annotations from MLflow perf/* metrics
- Provenance chain (DuckDB → stats → figures/tables)
- Limitations paragraph from power analysis
- Pre-registration link
- TRIPOD+AI item mapping

All outputs are JSON files consumed by downstream rendering (R, LaTeX, Mermaid).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def generate_consort_data(
    n_total_volumes: int = 70,
    n_folds: int = 3,
    n_val_per_fold: int = 23,
    n_train_per_fold: int = 47,
    n_losses: int = 2,
    n_post_training_methods: int = 2,
    n_ensemble_strategies: int = 1,
    n_deepvess_volumes: int = 7,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate CONSORT-style flow diagram data.

    Programmatically derived from experiment parameters — not hand-drawn.
    Satisfies TRIPOD+AI Items 6 and 20a.

    Returns dict and optionally writes to output_dir/consort_data.json.
    """
    n_training_runs = n_losses * n_folds
    n_post_training_runs = n_training_runs * n_post_training_methods
    n_conditions = n_losses * n_post_training_methods * n_ensemble_strategies

    consort = {
        "generated_at": datetime.now(UTC).isoformat(),
        "enrollment": {
            "total_volumes": n_total_volumes,
            "dataset": "MiniVess",
            "modality": "multiphoton microscopy",
            "organ": "mouse brain cortex",
        },
        "allocation": {
            "n_folds": n_folds,
            "train_per_fold": n_train_per_fold,
            "val_per_fold": n_val_per_fold,
            "split_config": "configs/splits/3fold_seed42.json",
            "seed": 42,
        },
        "training": {
            "n_losses": n_losses,
            "loss_names": ["dice_ce", "cbdice_cldice"],
            "n_training_runs": n_training_runs,
            "model_family": "dynunet",
            "max_epochs": 20,
        },
        "post_training": {
            "n_methods": n_post_training_methods,
            "method_names": ["none", "checkpoint_averaging"],
            "n_post_training_runs": n_post_training_runs,
        },
        "analysis": {
            "n_ensemble_strategies": n_ensemble_strategies,
            "strategy_names": ["per_loss_single_best"],
            "n_total_conditions": n_conditions,
        },
        "external_test": {
            "dataset": "DeepVess",
            "n_volumes": n_deepvess_volumes,
            "source": "Cornell University",
            "modality": "multiphoton",
        },
        "biostatistics": {
            "n_figures": 10,
            "n_tables": 8,
            "statistical_tests": [
                "stratified_permutation",
                "mixed_effects_anova",
                "bayesian_signed_rank",
                "friedman",
                "specification_curve",
            ],
        },
    }

    if output_dir is not None:
        path = output_dir / "consort_data.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(consort, f, indent=2)
        logger.info("CONSORT data written to %s", path)

    return consort


def generate_tripod_mapping(
    output_dir: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Map biostatistics artifacts to TRIPOD+AI items.

    Returns dict: artifact_id -> {tripod_item, description, evidence}.
    """
    mapping = {
        "F2_consort": {
            "tripod_item": "6, 20a",
            "description": "Participant flow diagram (CONSORT-style)",
            "evidence": "consort_data.json → fig_consort_flow.R → F2.pdf",
        },
        "T1_comparison": {
            "tripod_item": "23a",
            "description": "Performance with confidence intervals",
            "evidence": "pairwise_results.json → T1_comparison.tex",
        },
        "T4_variance": {
            "tripod_item": "23b",
            "description": "Heterogeneity (inter-fold agreement)",
            "evidence": "variance_decomposition.json → T4_variance.tex",
        },
        "T8_power": {
            "tripod_item": "9a",
            "description": "Sample size justification",
            "evidence": "power_analysis.json → T8_power.tex",
        },
        "F7_generalization": {
            "tripod_item": "24",
            "description": "External validation (DeepVess)",
            "evidence": "test_generalization.json → fig_generalization_gap.R → F7.pdf",
        },
        "F4_forest": {
            "tripod_item": "12",
            "description": "Statistical methods (effect sizes with CIs)",
            "evidence": "pairwise_results.json → fig_forest_plot.R → F4.pdf",
        },
        "F6b_speccurve": {
            "tripod_item": "12",
            "description": "Sensitivity analysis (specification curve)",
            "evidence": "spec_curve.json → fig_specification_curve.R → F6b.pdf",
        },
        "pre_registration": {
            "tripod_item": "3",
            "description": "Pre-registration via git tag",
            "evidence": "git tag pre-registration/mini-experiment-v1",
        },
    }

    if output_dir is not None:
        path = output_dir / "tripod_mapping.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
        logger.info("TRIPOD mapping written to %s", path)

    return mapping


def generate_limitations_paragraph(
    power_results: dict[str, Any] | None = None,
    n_folds: int = 3,
    n_volumes_per_fold: int = 23,
    output_dir: Path | None = None,
) -> str:
    """Generate auto-populated limitations paragraph from power analysis.

    Returns LaTeX-ready paragraph text.
    """
    paragraph = (
        f"With K={n_folds} folds as blocks, the Friedman test has limited power "
        f"to detect small-to-medium effect sizes. Per-volume stratified tests "
        f"(N={n_volumes_per_fold} per fold) provide substantially more statistical power. "
        f"This mini-experiment serves as a platform validation; the full factorial "
        f"(720 conditions × 3 folds) will provide definitive statistical power for "
        f"all planned comparisons."
    )

    if power_results and "detectable_d" in power_results:
        d = power_results["detectable_d"]
        paragraph += (
            f" At the per-volume level, the stratified permutation test achieves "
            f"80\\% power to detect Cohen's $d \\geq {d:.3f}$."
        )

    if output_dir is not None:
        path = output_dir / "limitations_paragraph.tex"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(paragraph)
        logger.info("Limitations paragraph written to %s", path)

    return paragraph


def generate_provenance_chain_mermaid(
    duckdb_sha: str = "",
    git_sha: str = "",
    n_figures: int = 10,
    n_tables: int = 8,
    output_dir: Path | None = None,
) -> str:
    """Generate Mermaid diagram of the data provenance chain.

    Shows: MLflow → DuckDB → Statistical Engine → Figures/Tables → Manuscript.
    """
    mermaid = f"""graph LR
    A[MLflow mlruns/] -->|build_biostatistics_duckdb| B[DuckDB<br/>SHA: {duckdb_sha[:12]}...]
    B -->|build_per_volume_data_from_duckdb| C[Statistical Engine]
    C -->|7 analyses| D[Results JSON]
    D -->|export_all_r_data| E[R Data JSON]
    E -->|Rscript generate_all_biostat_figures.R| F[{n_figures} Figures<br/>PDF + PNG]
    D -->|generate_tables| G[{n_tables} Tables<br/>LaTeX .tex]
    F -->|copy| H[Manuscript]
    G -->|copy| H
    F ---|JSON sidecar| I[Reproducibility<br/>Artifacts]
    G ---|JSON sidecar| I

    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style F fill:#e8f5e9
    style G fill:#e8f5e9
    style H fill:#fce4ec
    style I fill:#f5f5f5

    classDef sha fill:#fff9c4,stroke:#f9a825
"""

    if output_dir is not None:
        path = output_dir / "provenance_chain.mmd"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(mermaid)
        logger.info("Provenance chain Mermaid written to %s", path)

    return mermaid
