"""Biostatistics Prefect flow — cross-run statistical analysis.

Imports directly from prefect. If Prefect is not installed,
ImportError is the CORRECT behavior.

Entry points:
  - Prefect: prefect deployment run 'biostatistics-flow/default'
  - Docker:  docker compose -f deployment/docker-compose.flows.yml run biostatistics
  - Tests:   MINIVESS_ALLOW_HOST=1 pytest (escape hatch for Docker context check)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from prefect import flow, task

from minivess.config.biostatistics_config import BiostatisticsConfig
from minivess.observability.lineage import LineageEmitter, emit_flow_lineage
from minivess.orchestration.constants import FLOW_NAME_BIOSTATISTICS
from minivess.pipeline.biostatistics_discovery import (
    discover_source_runs,
    validate_source_completeness,
)
from minivess.pipeline.biostatistics_duckdb import (
    build_biostatistics_duckdb,
    export_parquet,
)
from minivess.pipeline.biostatistics_figures import generate_figures
from minivess.pipeline.biostatistics_lineage import build_lineage_manifest
from minivess.pipeline.biostatistics_mlflow import log_biostatistics_run
from minivess.pipeline.biostatistics_rankings import compute_rankings
from minivess.pipeline.biostatistics_statistics import (
    compute_bayesian_comparisons,
    compute_pairwise_comparisons,
    compute_variance_decomposition,
)
from minivess.pipeline.biostatistics_tables import generate_tables
from minivess.pipeline.biostatistics_types import BiostatisticsResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Docker context gate
# ---------------------------------------------------------------------------


def _require_docker_context() -> None:
    """Require Docker container context or MINIVESS_ALLOW_HOST=1.

    Raises
    ------
    RuntimeError
        If not running inside Docker and MINIVESS_ALLOW_HOST is not set.
    """
    if os.environ.get("MINIVESS_ALLOW_HOST") == "1":
        return
    if os.environ.get("DOCKER_CONTAINER"):
        return
    if Path("/.dockerenv").exists():
        return
    raise RuntimeError(
        "Biostatistics flow must run inside a Docker container.\n"
        "Run: docker compose -f deployment/docker-compose.flows.yml "
        "run biostatistics\n"
        "Escape hatch for tests: MINIVESS_ALLOW_HOST=1"
    )


# ---------------------------------------------------------------------------
# Prefect tasks (thin wrappers around pure functions)
# ---------------------------------------------------------------------------


@task(name="discover-source-runs")
def task_discover_source_runs(
    mlruns_dir: str,
    experiment_names: list[str],
) -> Any:
    """Discover FINISHED runs from MLflow mlruns directory."""
    return discover_source_runs(Path(mlruns_dir), experiment_names)


@task(name="validate-source-completeness")
def task_validate_source_completeness(
    manifest: Any,
    min_folds: int,
    min_conditions: int,
) -> Any:
    """Validate source data completeness."""
    return validate_source_completeness(manifest, min_folds, min_conditions)


@task(name="build-biostatistics-duckdb")
def task_build_duckdb(
    manifest: Any,
    mlruns_dir: str,
    output_dir: str,
) -> Path:
    """Build biostatistics DuckDB database."""
    db_path = Path(output_dir) / "biostatistics.duckdb"
    build_biostatistics_duckdb(manifest, Path(mlruns_dir), db_path)
    parquet_dir = Path(output_dir) / "parquet"
    export_parquet(db_path, parquet_dir)
    return db_path


@task(name="compute-pairwise")
def task_compute_pairwise(
    per_volume_data: dict[str, dict[int, Any]],
    metric_name: str,
    alpha: float,
    primary_metric: str,
    n_bootstrap: int,
    seed: int,
) -> list[Any]:
    """Compute pairwise statistical comparisons."""
    return compute_pairwise_comparisons(
        per_volume_data=per_volume_data,
        metric_name=metric_name,
        alpha=alpha,
        primary_metric=primary_metric,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


@task(name="compute-bayesian")
def task_compute_bayesian(
    per_volume_data: dict[str, dict[int, Any]],
    metric_name: str,
    rope: float,
) -> list[Any]:
    """Compute Bayesian signed-rank comparisons."""
    return compute_bayesian_comparisons(
        per_volume_data=per_volume_data,
        metric_name=metric_name,
        rope=rope,
    )


@task(name="compute-variance-decomposition")
def task_compute_variance(
    per_volume_data: dict[str, dict[int, Any]],
    metric_name: str,
) -> list[Any]:
    """Compute variance decomposition (Friedman + ICC)."""
    return compute_variance_decomposition(
        per_volume_data=per_volume_data,
        metric_name=metric_name,
    )


@task(name="compute-rankings")
def task_compute_rankings(
    per_volume_data: dict[str, dict[str, dict[int, Any]]],
    metric_names: list[str],
    higher_is_better: dict[str, bool],
) -> list[Any]:
    """Compute multi-metric rankings."""
    return compute_rankings(
        per_volume_data=per_volume_data,
        metric_names=metric_names,
        higher_is_better=higher_is_better,
    )


@task(name="generate-figures")
def task_generate_figures(
    per_volume_data: dict[str, dict[str, dict[int, Any]]],
    pairwise: list[Any],
    variance: list[Any],
    rankings: list[Any],
    output_dir: str,
) -> list[Any]:
    """Generate publication-quality figures."""
    return generate_figures(
        per_volume_data=per_volume_data,
        pairwise=pairwise,
        variance=variance,
        rankings=rankings,
        output_dir=Path(output_dir) / "figures",
    )


@task(name="generate-tables")
def task_generate_tables(
    pairwise: list[Any],
    variance: list[Any],
    rankings: list[Any],
    output_dir: str,
) -> list[Any]:
    """Generate LaTeX tables."""
    return generate_tables(
        pairwise=pairwise,
        variance=variance,
        rankings=rankings,
        output_dir=Path(output_dir) / "tables",
    )


@task(name="build-lineage-manifest")
def task_build_lineage(
    manifest: Any, figures: list[Any], tables: list[Any]
) -> dict[str, Any]:
    """Build lineage manifest."""
    return build_lineage_manifest(manifest=manifest, figures=figures, tables=tables)


@task(name="log-biostatistics-run")
def task_log_mlflow(
    manifest: Any,
    lineage: dict[str, Any],
    figures: list[Any],
    tables: list[Any],
    db_path: Any,
) -> str:
    """Log biostatistics run to MLflow."""
    return log_biostatistics_run(
        manifest=manifest,
        lineage=lineage,
        figures=figures,
        tables=tables,
        db_path=db_path,
    )


@task(name="narrate-figures")
def narrate_figures(
    figures: list[Any],
    n_conditions: int = 0,
    use_agent: bool | None = None,
) -> list[dict[str, Any]]:
    """Generate captions for figures using Pydantic AI agent or deterministic fallback.

    Parameters
    ----------
    figures:
        List of figure metadata dicts (each with ``figure_type``, ``metric``, ``path``).
    n_conditions:
        Number of experimental conditions compared.
    use_agent:
        Explicitly enable/disable agent. Defaults to ``MINIVESS_USE_AGENTS`` env var.

    Returns
    -------
    List of caption dicts, one per figure.
    """
    if not figures:
        return []

    if use_agent is None:
        use_agent = os.environ.get("MINIVESS_USE_AGENTS") == "1"

    captions: list[dict[str, Any]] = []
    for fig in figures:
        fig_type = (
            fig.get("figure_type", "unknown") if isinstance(fig, dict) else "unknown"
        )
        metric = fig.get("metric", "unknown") if isinstance(fig, dict) else "unknown"

        if use_agent:
            try:
                from pydantic_ai.models.test import TestModel

                from minivess.agents.figure_narrator import FigureContext, _build_agent

                agent = _build_agent(model="test")
                ctx = FigureContext(
                    figure_type=fig_type,
                    n_conditions=n_conditions,
                    primary_metric=metric,
                )
                test_output = {
                    "caption": (
                        f"Comparison of {n_conditions} experimental conditions. "
                        f"Primary metric: {metric}."
                    ),
                    "alt_text": f"{fig_type} showing {metric} scores",
                    "statistical_note": None,
                }
                result = agent.run_sync(
                    "Generate a caption.",
                    deps=ctx,
                    model=TestModel(custom_output_args=test_output, call_tools=[]),
                )
                captions.append(
                    {
                        "caption": result.output.caption,
                        "alt_text": result.output.alt_text,
                        "figure_type": fig_type,
                    }
                )
                continue
            except ImportError:
                logger.debug("pydantic-ai not available, using deterministic fallback")

        # Deterministic fallback
        from minivess.orchestration.agent_interface import DeterministicFigureNarration

        stub_result = DeterministicFigureNarration().decide(
            context={
                "figure_type": fig_type,
                "n_conditions": n_conditions,
                "primary_metric": metric,
            }
        )
        captions.append(
            {
                "caption": stub_result["caption"],
                "figure_type": fig_type,
            }
        )

    return captions


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


@flow(name=FLOW_NAME_BIOSTATISTICS, validate_parameters=False)
def run_biostatistics_flow(
    config_path: str = "/app/configs/biostatistics/default.yaml",
    trigger_source: str = "manual",
) -> BiostatisticsResult:
    """Run the biostatistics flow: cross-run statistical analysis.

    Parameters
    ----------
    config_path:
        Path to biostatistics config YAML.
    trigger_source:
        How the flow was triggered (manual, post-training, schedule).

    Returns
    -------
    BiostatisticsResult with all analysis artifacts.
    """
    _require_docker_context()

    logger.info("Starting biostatistics flow (source: %s)", trigger_source)

    # Load config
    config = _load_config(config_path)

    mlruns_dir_str = str(config.mlruns_dir)
    output_dir_str = str(config.output_dir)

    # Phase 2: Discovery + validation
    manifest: Any = task_discover_source_runs(
        mlruns_dir=mlruns_dir_str,
        experiment_names=config.experiment_names,
    )
    _validation = task_validate_source_completeness(
        manifest=manifest,
        min_folds=config.min_folds_per_condition,
        min_conditions=config.min_conditions,
    )

    # Phase 2: DuckDB
    db_path = task_build_duckdb(
        manifest=manifest,
        mlruns_dir=mlruns_dir_str,
        output_dir=output_dir_str,
    )

    # Build per-volume data from manifest for statistics
    per_volume_data = _build_per_volume_data(manifest, Path(mlruns_dir_str))

    # Derive higher_is_better from metric names (lower is better for HD95, ASSD)
    higher_is_better: dict[str, bool] = {}
    for m in config.metrics:
        higher_is_better[m] = m not in ("hd95", "assd", "be_0", "be_1")

    # Phase 3: Statistical engine
    all_pairwise: list[Any] = []
    all_variance: list[Any] = []
    for metric in config.metrics:
        if metric in per_volume_data:
            pairwise = task_compute_pairwise(
                per_volume_data=per_volume_data[metric],
                metric_name=metric,
                alpha=config.alpha,
                primary_metric=config.primary_metric,
                n_bootstrap=config.n_bootstrap,
                seed=config.seed,
            )
            all_pairwise.extend(pairwise)

            task_compute_bayesian(
                per_volume_data=per_volume_data[metric],
                metric_name=metric,
                rope=config.rope_values.get(metric, 0.01),
            )

            variance = task_compute_variance(
                per_volume_data=per_volume_data[metric],
                metric_name=metric,
            )
            all_variance.extend(variance)

    # Phase 4: Rankings
    rankings = task_compute_rankings(
        per_volume_data=per_volume_data,
        metric_names=config.metrics,
        higher_is_better=higher_is_better,
    )

    # Phase 5: Figures
    figures = task_generate_figures(
        per_volume_data=per_volume_data,
        pairwise=all_pairwise,
        variance=all_variance,
        rankings=rankings,
        output_dir=output_dir_str,
    )

    # Phase 6: Tables
    tables = task_generate_tables(
        pairwise=all_pairwise,
        variance=all_variance,
        rankings=rankings,
        output_dir=output_dir_str,
    )

    # Phase 7: Lineage + MLflow
    lineage = task_build_lineage(
        manifest=manifest,
        figures=figures,
        tables=tables,
    )
    mlflow_run_id = task_log_mlflow(
        manifest=manifest,
        lineage=lineage,
        figures=figures,
        tables=tables,
        db_path=db_path,
    )

    logger.info("Biostatistics flow complete. MLflow run: %s", mlflow_run_id)

    # OpenLineage lineage emission (Issue #799 — IEC 62304 §8 traceability)
    try:
        _emitter = LineageEmitter(namespace="minivess")
        emit_flow_lineage(
            emitter=_emitter,
            job_name="biostatistics-flow",
            inputs=[{"namespace": "minivess", "name": "mlflow_runs"}],
            outputs=[
                {"namespace": "minivess", "name": "figures"},
                {"namespace": "minivess", "name": "tables"},
            ],
        )
    except Exception:
        logger.warning("OpenLineage emission failed (non-blocking)", exc_info=True)

    return BiostatisticsResult(
        manifest=manifest,
        db_path=db_path,
        pairwise=all_pairwise,
        variance=all_variance,
        rankings=rankings,
        figures=figures,
        tables=tables,
        mlflow_run_id=mlflow_run_id,
    )


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------


def _load_config(config_path: str) -> BiostatisticsConfig:
    """Load BiostatisticsConfig from YAML file or return defaults."""
    import yaml

    path = Path(config_path)
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        return BiostatisticsConfig(**raw)
    logger.warning("Config file %s not found, using defaults", config_path)
    return BiostatisticsConfig()


def _build_per_volume_data(
    manifest: Any,
    mlruns_dir: Path,
    split: str = "trainval",
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    """Build per-volume data structure from mlruns.

    Parameters
    ----------
    manifest:
        SourceRunManifest with discovered runs.
    mlruns_dir:
        Path to the mlruns directory.
    split:
        Which split to extract: ``"trainval"`` reads eval/ prefix metrics
        (MiniVess cross-validated), ``"test"`` reads test/ prefix metrics
        (external test datasets like DeepVess).

    Returns
    -------
    ``{metric_name: {condition: {fold_id: scores_array}}}``
    """
    from minivess.pipeline.biostatistics_duckdb import (
        _is_test_metric,
        _parse_per_volume_metric,
        _parse_test_metric,
        _read_metrics,
    )

    data: dict[str, dict[str, dict[int, list[float]]]] = {}

    for run in manifest.runs:
        run_dir = mlruns_dir / run.experiment_id / run.run_id
        metrics = _read_metrics(run_dir)

        for metric_name, value in metrics.items():
            if split == "test":
                # For test split, look for test/{dataset}/vol_{id}/{metric}
                # or test/{dataset}/{subset}/{metric}
                if not _is_test_metric(metric_name):
                    continue
                dataset_name, subset_or_vol, base_metric = _parse_test_metric(
                    metric_name,
                )
                # Per-volume test metrics: test/{dataset}/vol_{id}/{metric}
                if subset_or_vol.startswith("vol_"):
                    # Use fold_id=0 for test sets (no folds)
                    data.setdefault(base_metric, {}).setdefault(
                        run.loss_function, {}
                    ).setdefault(0, []).append(value)
                # Aggregate or subset-level: store as summary
                elif subset_or_vol in ("all", ""):
                    data.setdefault(base_metric, {}).setdefault(
                        run.loss_function, {}
                    ).setdefault(0, []).append(value)
            else:
                # For trainval split, use eval/ prefix per-volume metrics
                # Look for per-volume metrics: eval/{fold}/vol/{id}/{metric}
                parts = metric_name.split("/")
                if len(parts) >= 5 and parts[0] == "eval" and parts[2] == "vol":
                    fold_id, volume_id, base_metric = _parse_per_volume_metric(
                        metric_name,
                    )
                    if fold_id is not None:
                        data.setdefault(base_metric, {}).setdefault(
                            run.loss_function, {}
                        ).setdefault(fold_id, []).append(value)

    # Convert lists to numpy arrays
    result: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for metric, conditions in data.items():
        result[metric] = {}
        for cond, folds in conditions.items():
            result[metric][cond] = {}
            for fold, values in folds.items():
                result[metric][cond][fold] = np.array(values)

    return result


if __name__ == "__main__":
    run_biostatistics_flow()
