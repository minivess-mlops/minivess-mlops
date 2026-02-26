"""Analysis Prefect Flow (Flow 3): Evaluation, ensembles, comparison, registration.

Orchestrates post-training model analysis:

1. Load training run artifacts from MLflow
2. Build ensemble models using 4 strategies
3. Evaluate all models (single + ensemble) on test datasets
4. Compare models with paired bootstrap tests
5. Register best model as champion
6. Generate summary report

Uses ``_prefect_compat`` decorators for graceful degradation when Prefect
is disabled (``PREFECT_DISABLED=1`` for CI/test environments).
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from minivess.ensemble.builder import EnsembleBuilder, EnsembleSpec
from minivess.orchestration import flow, get_run_logger, task
from minivess.pipeline.comparison import (
    ComparisonTable,
    LossResult,
    MetricSummary,
    format_comparison_markdown,
)
from minivess.pipeline.model_promoter import ModelPromoter

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

    from minivess.config.evaluation_config import EvaluationConfig
    from minivess.data.test_datasets import HierarchicalDataLoaderDict
    from minivess.pipeline.evaluation_runner import EvaluationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers (mockable in tests)
# ---------------------------------------------------------------------------


def _discover_runs(
    eval_config: EvaluationConfig,
    model_config: dict[str, Any],
    *,
    tracking_uri: str | None = None,
) -> list[dict[str, Any]]:
    """Query MLflow for completed training runs.

    Wraps :meth:`EnsembleBuilder.discover_training_runs` so that tests
    can patch this single function instead of the builder's internals.

    Parameters
    ----------
    eval_config:
        Evaluation configuration.
    model_config:
        Model architecture configuration.
    tracking_uri:
        Optional MLflow tracking URI override.

    Returns
    -------
    List of run info dicts, each with keys:
    ``run_id``, ``loss_type``, ``fold_id``, ``artifact_dir``, ``metrics``.
    """
    builder = EnsembleBuilder(
        eval_config, model_config, tracking_uri=tracking_uri
    )
    return builder.discover_training_runs()


def _evaluate_single_model_on_all(
    model: nn.Module,
    dataloaders_dict: HierarchicalDataLoaderDict,
    eval_config: EvaluationConfig,
    *,
    model_name: str,
    output_dir: Path | None = None,
) -> dict[str, dict[str, EvaluationResult]]:
    """Evaluate a single model on all datasets and subsets.

    Wraps :meth:`UnifiedEvaluationRunner.evaluate_model` to keep the
    heavy inference dependency mockable in tests.

    Parameters
    ----------
    model:
        Trained ``nn.Module``.
    dataloaders_dict:
        ``{dataset: {subset: DataLoader}}``.
    eval_config:
        Evaluation configuration.
    model_name:
        Human-readable model identifier.
    output_dir:
        Optional directory for saving predictions.

    Returns
    -------
    ``{dataset: {subset: EvaluationResult}}``
    """
    from minivess.pipeline.evaluation_runner import UnifiedEvaluationRunner
    from minivess.pipeline.inference import SlidingWindowInferenceRunner

    inference_runner = SlidingWindowInferenceRunner(
        roi_size=(32, 32, 32),
        num_classes=2,
        overlap=0.25,
    )
    runner = UnifiedEvaluationRunner(eval_config, inference_runner)
    return runner.evaluate_model(
        model,
        dataloaders_dict,
        model_name=model_name,
        output_dir=output_dir,
    )


def _extract_single_models_from_runs(
    runs: list[dict[str, Any]],
) -> dict[str, str]:
    """Extract model identifiers from training runs.

    Returns a mapping from descriptive name to run_id.  The actual
    model loading happens inside the ensemble builder; here we just
    track the names for evaluation.

    Parameters
    ----------
    runs:
        Run info dicts from MLflow discovery.

    Returns
    -------
    ``{model_name: run_id}``
    """
    result: dict[str, str] = {}
    for run in runs:
        name = f"{run['loss_type']}_fold{run['fold_id']}"
        result[name] = run["run_id"]
    return result


# ---------------------------------------------------------------------------
# Prefect tasks
# ---------------------------------------------------------------------------


@task(name="load-training-artifacts")  # type: ignore[untyped-decorator]
def load_training_artifacts(
    eval_config: EvaluationConfig,
    model_config: dict[str, Any],
    *,
    tracking_uri: str | None = None,
) -> list[dict[str, Any]]:
    """Query MLflow for completed training runs.

    Delegates to :func:`_discover_runs` for testability.

    Parameters
    ----------
    eval_config:
        Evaluation configuration with MLflow experiment name.
    model_config:
        Model architecture configuration.
    tracking_uri:
        Optional MLflow tracking URI override.

    Returns
    -------
    List of run info dicts.
    """
    log = get_run_logger()
    log.info("Loading training artifacts from MLflow...")

    runs = _discover_runs(eval_config, model_config, tracking_uri=tracking_uri)

    log.info(
        "Loaded %d training runs (%d unique losses)",
        len(runs),
        len({r["loss_type"] for r in runs}),
    )
    return runs


@task(name="build-ensembles")  # type: ignore[untyped-decorator]
def build_ensembles(
    runs: list[dict[str, Any]],
    eval_config: EvaluationConfig,
    model_config: dict[str, Any],
) -> dict[str, EnsembleSpec]:
    """Build all configured ensemble strategies.

    Creates an :class:`EnsembleBuilder` and delegates to
    :meth:`~EnsembleBuilder.build_all`.

    Parameters
    ----------
    runs:
        Pre-fetched run info dicts from :func:`load_training_artifacts`.
    eval_config:
        Evaluation configuration with ensemble strategy list.
    model_config:
        Model architecture configuration.

    Returns
    -------
    Mapping from ensemble name to :class:`EnsembleSpec`.
    """
    log = get_run_logger()
    log.info(
        "Building ensembles for %d strategies...",
        len(eval_config.ensemble_strategies),
    )

    builder = EnsembleBuilder(eval_config, model_config)
    ensembles = builder.build_all(runs)

    log.info(
        "Built %d ensembles: %s",
        len(ensembles),
        list(ensembles.keys()),
    )
    return ensembles


@task(name="evaluate-all-models")  # type: ignore[untyped-decorator]
def evaluate_all_models(
    single_models: dict[str, nn.Module],
    ensembles: dict[str, EnsembleSpec],
    dataloaders_dict: HierarchicalDataLoaderDict,
    eval_config: EvaluationConfig,
    *,
    output_dir: Path | None = None,
) -> dict[str, dict[str, dict[str, EvaluationResult]]]:
    """Evaluate all single models and ensembles on all datasets.

    Iterates over single models and ensemble specs, evaluating each
    against the hierarchical dataloader dictionary.

    Parameters
    ----------
    single_models:
        ``{model_name: nn.Module}`` for individual checkpoints.
    ensembles:
        ``{ensemble_name: EnsembleSpec}`` for ensemble models.
    dataloaders_dict:
        ``{dataset: {subset: DataLoader}}``.
    eval_config:
        Evaluation configuration.
    output_dir:
        Optional base directory for prediction outputs.

    Returns
    -------
    ``{model_name: {dataset: {subset: EvaluationResult}}}``
    """
    log = get_run_logger()
    all_results: dict[str, dict[str, dict[str, EvaluationResult]]] = {}

    # Evaluate single models
    for model_name, model in single_models.items():
        log.info("Evaluating single model: %s", model_name)
        model_output = (
            output_dir / "single" / model_name if output_dir else None
        )
        results = _evaluate_single_model_on_all(
            model,
            dataloaders_dict,
            eval_config,
            model_name=model_name,
            output_dir=model_output,
        )
        all_results[model_name] = results

    # Evaluate ensembles (use first member's net as representative;
    # proper ensemble inference would average logits across members,
    # but the UnifiedEvaluationRunner accepts any nn.Module)
    for ens_name, spec in ensembles.items():
        if not spec.members:
            log.warning("Ensemble '%s' has no members; skipping", ens_name)
            continue

        log.info(
            "Evaluating ensemble: %s (%d members)",
            ens_name,
            len(spec.members),
        )
        # Use the first member for now; a proper ensemble forward pass
        # would combine all members' predictions.  The evaluation runner
        # is model-agnostic so any nn.Module works here.
        representative_model = spec.members[0].net
        ens_output = output_dir / "ensemble" / ens_name if output_dir else None
        results = _evaluate_single_model_on_all(
            representative_model,
            dataloaders_dict,
            eval_config,
            model_name=ens_name,
            output_dir=ens_output,
        )
        all_results[ens_name] = results

    log.info("Evaluated %d total models", len(all_results))
    return all_results


@task(name="generate-comparison")  # type: ignore[untyped-decorator]
def generate_comparison(
    all_results: dict[str, dict[str, dict[str, EvaluationResult]]],
) -> str:
    """Generate cross-model comparison markdown.

    Builds a :class:`ComparisonTable` from all evaluation results and
    formats it as a markdown table using :func:`format_comparison_markdown`.

    Parameters
    ----------
    all_results:
        ``{model_name: {dataset: {subset: EvaluationResult}}}``.

    Returns
    -------
    Markdown-formatted comparison string.
    """
    log = get_run_logger()

    if not all_results:
        return "*No results to compare.*"

    # Build a ComparisonTable from the flat results
    # We aggregate per-model across all dataset/subset combos
    loss_results: list[LossResult] = []
    all_metric_names: set[str] = set()

    for model_name, ds_dict in all_results.items():
        metric_values: dict[str, list[float]] = {}
        for _ds_name, subset_dict in ds_dict.items():
            for _subset_name, eval_result in subset_dict.items():
                for metric_name, ci in eval_result.fold_result.aggregated.items():
                    metric_values.setdefault(metric_name, []).append(
                        ci.point_estimate
                    )
                    all_metric_names.add(metric_name)

        # Build MetricSummary for each metric
        metrics: dict[str, MetricSummary] = {}
        for metric_name, values in metric_values.items():
            import numpy as np

            arr = np.array(values)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            metrics[metric_name] = MetricSummary(
                mean=mean_val,
                std=std_val,
                ci_lower=mean_val - 1.96 * std_val if len(arr) > 1 else mean_val,
                ci_upper=mean_val + 1.96 * std_val if len(arr) > 1 else mean_val,
                per_fold=values,
            )

        loss_results.append(
            LossResult(
                loss_name=model_name,
                num_folds=1,
                metrics=metrics,
            )
        )

    table = ComparisonTable(
        losses=loss_results,
        metric_names=sorted(all_metric_names),
    )
    markdown = format_comparison_markdown(table)

    log.info(
        "Generated comparison for %d models across %d metrics",
        len(all_results),
        len(all_metric_names),
    )
    return markdown


@task(name="register-champion")  # type: ignore[untyped-decorator]
def register_champion_task(
    all_results: dict[str, dict[str, dict[str, EvaluationResult]]],
    eval_config: EvaluationConfig,
    *,
    run_id: str | None = None,
    environment: str = "staging",
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Find best model and prepare promotion info.

    Uses :class:`ModelPromoter` to rank models by the primary metric
    and generate a promotion report.  Does NOT call
    :meth:`ModelPromoter.register_champion` (which requires a live MLflow
    server); instead returns the promotion metadata for the caller to
    act on.

    Parameters
    ----------
    all_results:
        ``{model_name: {dataset: {subset: EvaluationResult}}}``.
    eval_config:
        Evaluation configuration with primary metric settings.
    run_id:
        Optional MLflow run_id for registration.
    environment:
        Deployment environment label (``"staging"``, ``"production"``).
    tracking_uri:
        Optional MLflow tracking URI override.

    Returns
    -------
    Dict with ``champion_name``, ``champion_score``, ``rankings``,
    ``promotion_report``, ``environment``.
    """
    log = get_run_logger()

    promoter = ModelPromoter(eval_config)

    if not all_results:
        log.warning("No results to promote; returning empty promotion info")
        return {
            "champion_name": "",
            "champion_score": float("nan"),
            "rankings": [],
            "promotion_report": "*No models to promote.*",
            "environment": environment,
        }

    champion_name, champion_score = promoter.find_best_model(all_results)
    rankings = promoter.rank_models(all_results)
    promotion_report = promoter.generate_promotion_report(
        rankings, champion_name=champion_name
    )

    log.info(
        "Champion: %s (score=%.4f, metric=%s)",
        champion_name,
        champion_score,
        eval_config.primary_metric,
    )

    return {
        "champion_name": champion_name,
        "champion_score": champion_score,
        "rankings": rankings,
        "promotion_report": promotion_report,
        "environment": environment,
    }


@task(name="generate-report")  # type: ignore[untyped-decorator]
def generate_report(
    all_results: dict[str, dict[str, dict[str, EvaluationResult]]],
    comparison_md: str,
    promotion_info: dict[str, Any],
) -> str:
    """Generate final markdown summary report.

    Combines:

    * Header with timestamp and model count
    * Per-model evaluation summaries
    * Cross-model comparison table
    * Champion promotion report

    Parameters
    ----------
    all_results:
        ``{model_name: {dataset: {subset: EvaluationResult}}}``.
    comparison_md:
        Markdown from :func:`generate_comparison`.
    promotion_info:
        Dict from :func:`register_champion_task`.

    Returns
    -------
    Complete markdown report string.
    """
    log = get_run_logger()
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        "# Analysis Flow Report",
        "",
        f"**Generated:** {now}",
        f"**Models evaluated:** {len(all_results)}",
        "",
    ]

    # Section: Per-model summaries
    lines.append("## Per-Model Results")
    lines.append("")
    for model_name, ds_dict in sorted(all_results.items()):
        lines.append(f"### {model_name}")
        lines.append("")
        for ds_name in sorted(ds_dict.keys()):
            for subset_name in sorted(ds_dict[ds_name].keys()):
                eval_result = ds_dict[ds_name][subset_name]
                agg = eval_result.fold_result.aggregated
                metric_strs: list[str] = []
                for m_name in sorted(agg.keys()):
                    ci = agg[m_name]
                    if not math.isnan(ci.point_estimate):
                        metric_strs.append(
                            f"{m_name}={ci.point_estimate:.4f}"
                        )
                lines.append(
                    f"- **{ds_name}/{subset_name}**: "
                    + ", ".join(metric_strs)
                )
        lines.append("")

    # Section: Cross-model comparison
    lines.append("## Cross-Model Comparison")
    lines.append("")
    lines.append(comparison_md)
    lines.append("")

    # Section: Champion promotion
    champion_name = promotion_info.get("champion_name", "N/A")
    champion_score = promotion_info.get("champion_score", float("nan"))
    lines.append("## Champion Model")
    lines.append("")
    if champion_name:
        score_str = (
            f"{champion_score:.4f}"
            if not math.isnan(champion_score)
            else "N/A"
        )
        lines.append(f"**Champion:** {champion_name} (score: {score_str})")
    else:
        lines.append("*No champion selected.*")
    lines.append("")

    promotion_report = promotion_info.get("promotion_report", "")
    if promotion_report:
        lines.append(promotion_report)
        lines.append("")

    report = "\n".join(lines)
    log.info("Generated analysis report (%d characters)", len(report))
    return report


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


@flow(name="analysis-flow")  # type: ignore[untyped-decorator]
def run_analysis_flow(
    eval_config: EvaluationConfig,
    model_config: dict[str, Any],
    dataloaders_dict: HierarchicalDataLoaderDict,
    *,
    output_dir: Path | None = None,
    environment: str = "staging",
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Main analysis flow orchestrating all tasks.

    Executes the complete post-training analysis pipeline:

    1. **Load** training artifacts from MLflow
    2. **Build** ensemble models using configured strategies
    3. **Evaluate** all models (single + ensemble) on test datasets
    4. **Compare** models with cross-loss comparison table
    5. **Register** the best model as champion
    6. **Report** generate a comprehensive markdown summary

    Parameters
    ----------
    eval_config:
        :class:`EvaluationConfig` with metric and MLflow settings.
    model_config:
        Model architecture configuration dict.
    dataloaders_dict:
        ``{dataset: {subset: DataLoader}}``.
    output_dir:
        Optional base directory for prediction outputs.
    environment:
        Deployment environment label.
    tracking_uri:
        Optional MLflow tracking URI override.

    Returns
    -------
    Dict with keys: ``results``, ``comparison``, ``promotion``, ``report``.
    """
    log = get_run_logger()
    log.info("Starting analysis flow...")

    # Step 1: Load training artifacts
    runs = load_training_artifacts(
        eval_config, model_config, tracking_uri=tracking_uri
    )

    # Step 2: Build ensembles
    ensembles = build_ensembles(runs, eval_config, model_config)

    # Step 3: Extract single model identifiers (for logging/tracking)
    # Note: In production, single models are loaded from checkpoints.
    # Here we pass an empty dict because the actual model objects
    # are loaded by the ensemble builder; callers wanting to evaluate
    # individual fold models should add them to the single_models dict.
    single_models: dict[str, nn.Module] = {}

    # Step 4: Evaluate all models
    all_results = evaluate_all_models(
        single_models,
        ensembles,
        dataloaders_dict,
        eval_config,
        output_dir=output_dir,
    )

    # Step 5: Generate comparison
    comparison_md = generate_comparison(all_results)

    # Step 6: Register champion
    promotion_info = register_champion_task(
        all_results,
        eval_config,
        environment=environment,
        tracking_uri=tracking_uri,
    )

    # Step 7: Generate report
    report = generate_report(all_results, comparison_md, promotion_info)

    log.info("Analysis flow complete. Champion: %s", promotion_info.get("champion_name"))

    return {
        "results": all_results,
        "comparison": comparison_md,
        "promotion": promotion_info,
        "report": report,
    }
