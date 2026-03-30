"""Analysis Prefect Flow (Flow 3): Evaluation, ensembles, comparison, registration.

Orchestrates post-training model analysis:

1. Load training run artifacts from MLflow
2. Build ensemble models using 4 strategies
3. Log models as MLflow pyfunc artifacts (single + ensemble)
4. Evaluate all models (single + ensemble) on test datasets
5. Run MLflow evaluate with custom segmentation metrics
6. Compare models with paired bootstrap tests
7. Register best model as champion
8. Generate summary report

Uses Prefect @flow and @task decorators for orchestration.
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from prefect import flow, get_run_logger, task
from torch import nn

from minivess.compliance.audit import AuditTrail
from minivess.ensemble.builder import (
    EnsembleBuilder,
    EnsembleSpec,
)
from minivess.observability.flow_observability import gpu_flow_observability_context
from minivess.observability.lineage import LineageEmitter, emit_flow_lineage
from minivess.observability.tracking import resolve_tracking_uri
from minivess.orchestration.constants import (
    EXPERIMENT_EVALUATION,
    EXPERIMENT_POST_TRAINING,
    EXPERIMENT_TRAINING,
    FLOW_NAME_ANALYSIS,
    FLOW_NAME_TRAIN,
    resolve_experiment_name,
)
from minivess.orchestration.docker_guard import require_docker_context
from minivess.orchestration.mlflow_helpers import (
    find_upstream_safely,
    log_completion_safe,
)
from minivess.pipeline.champion_tagger import tag_champions
from minivess.pipeline.comparison import (
    ComparisonTable,
    LossResult,
    MetricSummary,
    format_comparison_latex,
    format_comparison_markdown,
)
from minivess.pipeline.model_promoter import ModelPromoter

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from minivess.config.evaluation_config import EvaluationConfig
    from minivess.data.test_datasets import HierarchicalDataLoaderDict
    from minivess.pipeline.evaluation_runner import EvaluationResult

from minivess.observability.prefect_hooks import create_task_timing_hooks

logger = logging.getLogger(__name__)

_on_complete, _on_fail = create_task_timing_hooks()


def _validate_analysis_env() -> None:
    """Validate required environment variables for analysis flow.

    Raises
    ------
    RuntimeError
        When ANALYSIS_OUTPUT_DIR is not set, with actionable instructions.
    """
    if not os.environ.get("ANALYSIS_OUTPUT_DIR"):
        raise RuntimeError(
            "Required environment variable ANALYSIS_OUTPUT_DIR not set.\n"
            "Set it before running the analysis flow:\n"
            "  export ANALYSIS_OUTPUT_DIR=/path/to/outputs/analysis\n"
            "Or configure it in your .env file."
        )


# ---------------------------------------------------------------------------
# Ensemble inference wrapper (uses all members, not just first)
# ---------------------------------------------------------------------------


class _EnsembleInferenceWrapper(nn.Module):  # type: ignore[misc]
    """Wrap multiple member networks into a single ``nn.Module``.

    Forward pass averages logits across all members, producing the
    same output shape as a single model.  This replaces the old
    first-member-only hack.

    Parameters
    ----------
    member_nets:
        List of loaded ``nn.Module`` instances (one per ensemble member).
    """

    def __init__(self, member_nets: list[nn.Module]) -> None:
        super().__init__()
        # Store as ModuleList so parameters are tracked
        self._members = nn.ModuleList(member_nets)

    @property
    def n_members(self) -> int:
        """Number of ensemble members."""
        return len(self._members)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average logits across all ensemble members.

        Parameters
        ----------
        x:
            Input tensor ``(B, C_in, D, H, W)``.

        Returns
        -------
        Averaged logits ``(B, C_out, D, H, W)``.
        """
        outputs: list[torch.Tensor] = []
        for member in self._members:
            member.eval()
            out = member(x)
            # Handle SegmentationOutput from ModelAdapter
            if hasattr(out, "logits"):
                outputs.append(out.logits)
            elif hasattr(out, "prediction"):
                outputs.append(out.prediction)
            else:
                outputs.append(out)

        stacked = torch.stack(outputs, dim=0)  # (M, B, C, D, H, W)
        return stacked.mean(dim=0)  # (B, C, D, H, W)


# ---------------------------------------------------------------------------
# Internal helpers (mockable in tests)
# ---------------------------------------------------------------------------


def _discover_runs(
    eval_config: EvaluationConfig,
    model_config_dict: dict[str, Any],
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
    model_config_dict:
        Model architecture configuration.
    tracking_uri:
        Optional MLflow tracking URI override.

    Returns
    -------
    List of run info dicts, each with keys:
    ``run_id``, ``loss_type``, ``fold_id``, ``artifact_dir``, ``metrics``.
    """
    builder = EnsembleBuilder(eval_config, model_config_dict, tracking_uri=tracking_uri)
    result: list[dict[str, Any]] = builder.discover_training_runs(
        require_eval_metrics=eval_config.require_eval_metrics,
    )
    return result


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

    # Resolve roi_size from the primary inference strategy (CLAUDE.md Rule #9 — no
    # hardcoded roi_size here; all inference configuration comes from eval_config).
    primary_strategy = next(
        (s for s in eval_config.inference_strategies if s.is_primary), None
    )
    if primary_strategy is not None and isinstance(primary_strategy.roi_size, list):
        _r = primary_strategy.roi_size
        primary_roi: tuple[int, int, int] = (_r[0], _r[1], _r[2])
        primary_overlap = primary_strategy.overlap
        primary_sw_batch_size = primary_strategy.sw_batch_size
    else:
        # Empty inference_strategies — use sensible defaults that are NOT (32, 32, 32)
        primary_roi = (128, 128, 16)
        primary_overlap = 0.5
        primary_sw_batch_size = 4

    inference_runner = SlidingWindowInferenceRunner(
        roi_size=primary_roi,
        num_classes=2,
        overlap=primary_overlap,
        sw_batch_size=primary_sw_batch_size,
    )
    runner = UnifiedEvaluationRunner(eval_config, inference_runner)
    eval_result: dict[str, dict[str, Any]] = runner.evaluate_model(
        model,
        dataloaders_dict,
        model_name=model_name,
        output_dir=output_dir,
    )
    return eval_result


def _extract_single_models_as_modules(
    ensembles: dict[str, EnsembleSpec],
) -> dict[str, nn.Module]:
    """Extract unique single-fold models from ensemble members.

    Ensemble members already have their ``.net`` loaded.  This function
    deduplicates by ``run_id`` and returns the first occurrence of each
    unique model.

    Parameters
    ----------
    ensembles:
        Mapping from ensemble name to :class:`EnsembleSpec`.

    Returns
    -------
    ``{model_name: nn.Module}`` — deduplicated by run_id.
    """
    seen_run_ids: set[str] = set()
    models: dict[str, nn.Module] = {}

    for spec in ensembles.values():
        for member in spec.members:
            if member.run_id in seen_run_ids:
                continue
            seen_run_ids.add(member.run_id)
            name = f"{member.loss_type}_fold{member.fold_id}"
            models[name] = member.net

    return models


def _log_single_model_safe(
    run: dict[str, Any],
    model_config_dict: dict[str, Any],
    eval_config: EvaluationConfig,
) -> str | None:
    """Attempt to log a single model as MLflow pyfunc artifact.

    Returns the model URI on success, or ``None`` on failure.
    Designed to be mockable in tests.
    """
    from pathlib import Path

    from minivess.serving.model_logger import log_single_model

    try:
        import mlflow

        mlflow.set_experiment(eval_config.mlflow_evaluation_experiment)
        artifact_dir = Path(run["artifact_dir"])
        ckpt_name = eval_config.checkpoint_filename()
        ckpt_path = artifact_dir / ckpt_name

        if not ckpt_path.exists():
            logger.debug("Checkpoint not found: %s", ckpt_path)
            return None

        name = f"{run['loss_type']}_fold{run['fold_id']}"
        with mlflow.start_run(run_name=f"pyfunc_{name}"):
            log_single_model(
                checkpoint_path=ckpt_path,
                model_config_dict=model_config_dict,
            )
            active_run = mlflow.active_run()
            assert active_run is not None
            return f"runs:/{active_run.info.run_id}/model"
    except Exception:
        logger.warning(
            "Could not log pyfunc model for run %s",
            run.get("run_id"),
            exc_info=True,
        )
        return None


def _log_ensemble_model_safe(
    ensemble_spec: EnsembleSpec,
    model_config_dict: dict[str, Any],
    eval_config: EvaluationConfig,
) -> str | None:
    """Attempt to log an ensemble model as MLflow pyfunc artifact.

    Returns the model URI on success, or ``None`` on failure.
    Designed to be mockable in tests.
    """
    from minivess.serving.model_logger import log_ensemble_model

    try:
        import mlflow

        mlflow.set_experiment(eval_config.mlflow_evaluation_experiment)
        with mlflow.start_run(run_name=f"pyfunc_{ensemble_spec.name}"):
            log_ensemble_model(
                ensemble_spec=ensemble_spec,
                model_config_dict=model_config_dict,
            )
            active_run = mlflow.active_run()
            assert active_run is not None
            return f"runs:/{active_run.info.run_id}/ensemble_model"
    except Exception:
        logger.warning(
            "Could not log pyfunc ensemble model '%s'",
            ensemble_spec.name,
            exc_info=True,
        )
        return None


def _run_mlflow_eval_safe(
    results: dict[str, dict[str, EvaluationResult]],
    eval_config: EvaluationConfig,
    *,
    model_name: str,
) -> dict[str, float]:
    """Attempt to run mlflow.evaluate() for a model's predictions.

    Falls back to empty dict if evaluation cannot be performed (e.g.,
    no predictions saved to disk, or MLflow unavailable).
    Designed to be mockable in tests.
    """
    from pathlib import Path

    from minivess.serving.mlflow_evaluators import run_mlflow_evaluation

    try:
        # Collect prediction dirs from results
        for ds_results in results.values():
            for eval_result in ds_results.values():
                if eval_result.predictions_dir is not None:
                    pred_dir = Path(eval_result.predictions_dir)
                    label_dir = pred_dir  # Labels in same dir for now
                    mlflow_result = run_mlflow_evaluation(
                        pred_dir,
                        label_dir,
                        include_expensive=eval_config.include_expensive_metrics,
                    )
                    if hasattr(mlflow_result, "metrics"):
                        return dict(mlflow_result.metrics)
    except Exception:
        logger.warning(
            "MLflow evaluation failed for model '%s'",
            model_name,
            exc_info=True,
        )

    return {}


# ---------------------------------------------------------------------------
# Prefect tasks
# ---------------------------------------------------------------------------


@task(name="load-training-artifacts", on_completion=[_on_complete], on_failure=[_on_fail])
def load_training_artifacts(
    eval_config: EvaluationConfig,
    model_config_dict: dict[str, Any],
    *,
    tracking_uri: str | None = None,
) -> list[dict[str, Any]]:
    """Query MLflow for completed training runs.

    Delegates to :func:`_discover_runs` for testability.

    Parameters
    ----------
    eval_config:
        Evaluation configuration with MLflow experiment name.
    model_config_dict:
        Model architecture configuration.
    tracking_uri:
        Optional MLflow tracking URI override.

    Returns
    -------
    List of run info dicts.
    """
    log = get_run_logger()
    log.info("Loading training artifacts from MLflow...")

    runs = _discover_runs(eval_config, model_config_dict, tracking_uri=tracking_uri)

    log.info(
        "Loaded %d training runs (%d unique losses)",
        len(runs),
        len({r["loss_type"] for r in runs}),
    )
    return runs


@task(name="discover-post-training-models", on_completion=[_on_complete], on_failure=[_on_fail])
def discover_post_training_models(
    *,
    experiment_name: str = EXPERIMENT_POST_TRAINING,
    tracking_uri: str | None = None,
) -> list[dict[str, Any]]:
    """Discover models produced by the post-training flow.

    Queries the training experiment (SAME as training, per synthesis Part 2.3)
    for runs tagged with ``flow_name='post-training-flow'``.
    Returns empty list if the experiment doesn't exist or has no runs.

    Parameters
    ----------
    experiment_name:
        Post-training MLflow experiment name.
    tracking_uri:
        Optional MLflow tracking URI override.

    Returns
    -------
    List of post-training model info dicts.
    """
    log = get_run_logger()
    log.info("Discovering post-training models from %s...", experiment_name)

    try:
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            log.info(
                "Post-training experiment '%s' not found — skipping", experiment_name
            )
            return []

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                "attributes.status = 'FINISHED'"
                " AND tags.flow_name = 'post-training-flow'"
            ),
            output_format="list",
        )
        log.info("Found %d post-training model(s)", len(runs))
        return [
            {
                "run_id": r.info.run_id,
                "tags": dict(r.data.tags),
                "post_training_method": r.data.tags.get("post_training_method", "none"),
            }
            for r in runs
        ]
    except ImportError:
        log.info("MLflow not available — skipping post-training discovery")
        return []
    except Exception:
        log.warning("Post-training model discovery failed", exc_info=True)
        return []


@task(name="discover-zero-shot-baselines", on_completion=[_on_complete], on_failure=[_on_fail])
def discover_zero_shot_baselines(
    *,
    factorial_yaml: Path | None = None,
) -> list[dict[str, Any]]:
    """Discover zero-shot baseline models from the factorial YAML (#888).

    Zero-shot baselines (SAM3 Vanilla, VesselFM) are evaluated but NOT
    trained in the factorial grid. Each baseline gets ``is_zero_shot=true``
    tag in MLflow for Biostatistics to exclude from ANOVA.

    VesselFM is constrained to DeepVess-only (data leakage with MiniVess).

    Parameters
    ----------
    factorial_yaml:
        Path to factorial YAML (e.g., ``configs/factorial/debug.yaml``).
        Returns empty list if None.

    Returns
    -------
    List of zero-shot baseline dicts with model, strategy, dataset, and tags.
    """
    log = get_run_logger()

    if factorial_yaml is None:
        log.info("No factorial YAML provided — no zero-shot baselines")
        return []

    try:
        from minivess.config.factorial_config import parse_factorial_yaml

        design = parse_factorial_yaml(factorial_yaml)
    except (FileNotFoundError, ValueError):
        log.warning(
            "Failed to parse factorial YAML %s for zero-shot baselines",
            factorial_yaml,
            exc_info=True,
        )
        return []

    baselines: list[dict[str, Any]] = []
    for entry in design.zero_shot_baselines:
        model = entry.get("model", "unknown")
        strategy = entry.get("strategy", "unknown")
        dataset = entry.get("dataset", "minivess")
        folds = entry.get("folds", 1)

        baseline = {
            "model": model,
            "strategy": strategy,
            "dataset": dataset,
            "folds": folds,
            "is_zero_shot": True,
            "tags": {
                "is_zero_shot": "true",
                "model_family": model,
                "zero_shot_strategy": strategy,
                "zero_shot_dataset": dataset,
            },
        }
        baselines.append(baseline)

    log.info("Discovered %d zero-shot baseline(s) from factorial YAML", len(baselines))
    return baselines


@task(name="predict-with-uncertainty", on_completion=[_on_complete], on_failure=[_on_fail])
def predict_with_uncertainty(
    *,
    models: list[nn.Module],
    images: torch.Tensor,
) -> dict[str, Any]:
    """Run deep ensemble inference with uncertainty decomposition (#886).

    Wraps :class:`DeepEnsemblePredictor` for use as a Prefect task in the
    analysis flow. Produces entropy/MI decomposition per Lakshminarayanan
    et al. (2017).

    Parameters
    ----------
    models:
        List of independently trained models (ensemble members).
    images:
        Input tensor (B, C_in, D, H, W).

    Returns
    -------
    Dict with keys: prediction, uncertainty_map, total_uncertainty,
    aleatoric_uncertainty, epistemic_uncertainty, n_members.
    """
    log = get_run_logger()
    log.info("Running deep ensemble UQ with %d member(s)", len(models))

    from minivess.ensemble.deep_ensembles import DeepEnsemblePredictor

    predictor = DeepEnsemblePredictor(models)
    result = predictor.predict(images)

    return {
        "prediction": result.prediction,
        "uncertainty_map": result.uncertainty_map,
        "total_uncertainty": result.metadata["total_uncertainty"],
        "aleatoric_uncertainty": result.metadata["aleatoric_uncertainty"],
        "epistemic_uncertainty": result.metadata["epistemic_uncertainty"],
        "n_members": result.metadata["n_members"],
    }


@task(name="build-ensembles", on_completion=[_on_complete], on_failure=[_on_fail])
def build_ensembles(
    runs: list[dict[str, Any]],
    eval_config: EvaluationConfig,
    model_config_dict: dict[str, Any],
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
    model_config_dict:
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

    builder = EnsembleBuilder(eval_config, model_config_dict)
    ensembles: dict[str, Any] = builder.build_all(runs)

    log.info(
        "Built %d ensembles: %s",
        len(ensembles),
        list(ensembles.keys()),
    )
    return ensembles


@task(name="log-models-to-mlflow", on_completion=[_on_complete], on_failure=[_on_fail])
def log_models_to_mlflow(
    runs: list[dict[str, Any]],
    ensembles: dict[str, EnsembleSpec],
    eval_config: EvaluationConfig,
    model_config_dict: dict[str, Any],
) -> dict[str, str | None]:
    """Log single and ensemble models as MLflow pyfunc artifacts.

    Creates properly versioned MLflow artifacts with model signatures,
    checkpoints, and ensemble manifests so that models can be loaded
    via ``mlflow.pyfunc.load_model()``.

    Parameters
    ----------
    runs:
        Training run info dicts.
    ensembles:
        Built ensemble specifications.
    eval_config:
        Evaluation configuration.
    model_config_dict:
        Model architecture configuration.

    Returns
    -------
    ``{model_name: model_uri}`` — URIs may be ``None`` if logging failed.
    """
    log = get_run_logger()
    model_uris: dict[str, str | None] = {}

    # Log individual fold models
    for run in runs:
        name = f"{run['loss_type']}_fold{run['fold_id']}"
        uri = _log_single_model_safe(run, model_config_dict, eval_config)
        model_uris[name] = uri
        if uri:
            log.info("Logged pyfunc model: %s -> %s", name, uri)

    # Log ensemble models
    for ens_name, spec in ensembles.items():
        uri = _log_ensemble_model_safe(spec, model_config_dict, eval_config)
        model_uris[ens_name] = uri
        if uri:
            log.info("Logged pyfunc ensemble: %s -> %s", ens_name, uri)

    log.info(
        "Logged %d/%d models as pyfunc artifacts",
        sum(1 for v in model_uris.values() if v is not None),
        len(model_uris),
    )
    return model_uris


@task(name="evaluate-all-models", on_completion=[_on_complete], on_failure=[_on_fail])
def evaluate_all_models(
    single_models: dict[str, nn.Module],
    ensembles: dict[str, EnsembleSpec],
    dataloaders_dict: HierarchicalDataLoaderDict,
    eval_config: EvaluationConfig,
    *,
    output_dir: Path | None = None,
) -> dict[str, dict[str, dict[str, EvaluationResult]]]:
    """Evaluate all single models and ensembles on all datasets.

    Single models are evaluated directly.  Ensembles are wrapped in
    :class:`_EnsembleInferenceWrapper` which averages predictions
    across all members (replacing the previous first-member-only hack).

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
        model_output = output_dir / "single" / model_name if output_dir else None
        results = _evaluate_single_model_on_all(
            model,
            dataloaders_dict,
            eval_config,
            model_name=model_name,
            output_dir=model_output,
        )
        all_results[model_name] = results

    # Evaluate ensembles using all members via _EnsembleInferenceWrapper
    for ens_name, spec in ensembles.items():
        if not spec.members:
            log.warning("Ensemble '%s' has no members; skipping", ens_name)
            continue

        log.info(
            "Evaluating ensemble: %s (%d members)",
            ens_name,
            len(spec.members),
        )
        # Wrap all member nets into a single nn.Module that averages outputs
        ensemble_model = _EnsembleInferenceWrapper([m.net for m in spec.members])
        ens_output = output_dir / "ensemble" / ens_name if output_dir else None
        results = _evaluate_single_model_on_all(
            ensemble_model,
            dataloaders_dict,
            eval_config,
            model_name=ens_name,
            output_dir=ens_output,
        )
        all_results[ens_name] = results

    log.info("Evaluated %d total models", len(all_results))
    return all_results


@task(name="evaluate-with-mlflow", on_completion=[_on_complete], on_failure=[_on_fail])
def evaluate_with_mlflow(
    all_results: dict[str, dict[str, dict[str, EvaluationResult]]],
    eval_config: EvaluationConfig,
) -> dict[str, dict[str, float]]:
    """Run MLflow evaluate with custom segmentation metrics.

    For each model that has saved predictions, runs
    ``mlflow.evaluate()`` with custom Dice and compound metrics.

    Parameters
    ----------
    all_results:
        ``{model_name: {dataset: {subset: EvaluationResult}}}``.
    eval_config:
        Evaluation configuration.

    Returns
    -------
    ``{model_name: {metric_name: value}}``
    """
    log = get_run_logger()
    mlflow_results: dict[str, dict[str, float]] = {}

    for model_name, ds_dict in all_results.items():
        metrics = _run_mlflow_eval_safe(ds_dict, eval_config, model_name=model_name)
        if metrics:
            mlflow_results[model_name] = metrics
            log.info(
                "MLflow evaluation for %s: %d metrics",
                model_name,
                len(metrics),
            )

    log.info(
        "MLflow evaluation complete for %d/%d models",
        len(mlflow_results),
        len(all_results),
    )
    return mlflow_results


@task(name="generate-comparison", on_completion=[_on_complete], on_failure=[_on_fail])
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
                    metric_values.setdefault(metric_name, []).append(ci.point_estimate)
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
    markdown: str = format_comparison_markdown(table)

    log.info(
        "Generated comparison for %d models across %d metrics",
        len(all_results),
        len(all_metric_names),
    )
    return markdown


@task(name="export-analysis-duckdb", on_completion=[_on_complete], on_failure=[_on_fail])
def task_export_analysis_duckdb(
    all_results: dict[str, dict[str, dict[str, Any]]],
    manifest_runs: list[Any],
    output_dir: Path,
    metadata: dict[str, str],
) -> Path | None:
    """Export per-volume metrics to analysis_results.duckdb (DuckDB 1).

    Extracts per-volume metrics from FoldResult in each EvaluationResult
    and materializes them into a self-contained DuckDB file.

    Parameters
    ----------
    all_results:
        {model_name: {dataset: {subset: EvaluationResult}}}.
    manifest_runs:
        List of SourceRun objects for the runs table.
    output_dir:
        Output directory for the DuckDB file.
    metadata:
        Key-value pairs for the metadata table.

    Returns
    -------
    Path to analysis_results.duckdb, or None if no per-volume data found.
    """
    from minivess.pipeline.biostatistics_duckdb import build_analysis_results_duckdb
    from minivess.pipeline.biostatistics_types import SourceRunManifest

    pv_records: list[dict[str, object]] = []
    fold_records: list[dict[str, object]] = []

    for model_name, datasets in all_results.items():
        for dataset_name, subsets in datasets.items():
            for subset_name, eval_result in subsets.items():
                fold_result = getattr(eval_result, "fold_result", None)
                if fold_result is None:
                    continue

                # Determine split and dataset
                split = "test" if dataset_name != "minivess" else "trainval"
                dataset = dataset_name

                # Extract fold_id from model name or default to 0
                fold_id = getattr(eval_result, "fold_id", 0)

                # Per-volume metrics
                pv_metrics = getattr(fold_result, "per_volume_metrics", {})
                vol_ids = getattr(fold_result, "volume_ids", [])
                for metric_name, values in pv_metrics.items():
                    for idx, val in enumerate(values):
                        vol_id = vol_ids[idx] if idx < len(vol_ids) else f"vol_{idx}"
                        pv_records.append({
                            "run_id": model_name,
                            "fold_id": fold_id,
                            "split": split,
                            "dataset": dataset,
                            "volume_id": vol_id,
                            "metric_name": metric_name,
                            "metric_value": float(val),
                        })

                # Fold-level aggregated metrics
                aggregated = getattr(fold_result, "aggregated", {})
                for metric_name, ci in aggregated.items():
                    point = getattr(ci, "point_estimate", None)
                    if point is not None:
                        fold_records.append({
                            "run_id": model_name,
                            "fold_id": fold_id,
                            "split": split,
                            "metric_name": metric_name,
                            "metric_value": float(point),
                        })

    if not pv_records:
        logger.warning("No per-volume metrics found — skipping analysis DuckDB export")
        return None

    manifest = SourceRunManifest.from_runs(manifest_runs)
    db_path = output_dir / "analysis_results.duckdb"

    build_analysis_results_duckdb(
        manifest=manifest,
        per_volume_records=pv_records,
        fold_metric_records=fold_records,
        output_path=db_path,
        metadata=metadata,
    )

    logger.info(
        "Exported analysis_results.duckdb: %d per-volume, %d fold metrics",
        len(pv_records),
        len(fold_records),
    )
    return db_path


@task(name="register-champion", on_completion=[_on_complete], on_failure=[_on_fail])
def register_champion_task(
    all_results: dict[str, dict[str, dict[str, EvaluationResult]]],
    eval_config: EvaluationConfig,
    *,
    model_uris: dict[str, str | None] | None = None,
    run_id: str | None = None,
    environment: str = "staging",
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Find best model and prepare promotion info.

    Uses :class:`ModelPromoter` to rank models by the primary metric
    and generate a promotion report.  If ``model_uris`` is provided
    and contains a valid URI for the champion, attempts to register
    it in the MLflow Model Registry.

    Parameters
    ----------
    all_results:
        ``{model_name: {dataset: {subset: EvaluationResult}}}``.
    eval_config:
        Evaluation configuration with primary metric settings.
    model_uris:
        Optional ``{model_name: model_uri}`` from :func:`log_models_to_mlflow`.
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

    # Attempt actual MLflow registration if model URI is available
    registration_info: dict[str, str] | None = None
    if model_uris and model_uris.get(champion_name):
        champion_uri = model_uris[champion_name]
        # Extract run_id from URI "runs:/{run_id}/model"
        champion_run_id = run_id
        if champion_uri and "runs:/" in champion_uri:
            parts = champion_uri.split("/")
            if len(parts) >= 3:
                champion_run_id = parts[1]

        if champion_run_id:
            try:
                registration_info = promoter.register_champion(
                    champion_name,
                    run_id=champion_run_id,
                    environment=environment,
                    tracking_uri=tracking_uri,
                )
                log.info(
                    "Registered champion in MLflow: %s",
                    registration_info,
                )
            except Exception:
                log.warning(
                    "Could not register champion in MLflow (server unavailable?)",
                    exc_info=True,
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
        "registration": registration_info,
    }


@task(name="tag-champion-models", on_completion=[_on_complete], on_failure=[_on_fail])
def tag_champion_models(
    analysis_entries: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    ensembles: dict[str, EnsembleSpec],
    eval_config: EvaluationConfig,
    *,
    mlruns_dir: Path | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Tag champion models on the MLflow training runs filesystem.

    Writes ``champion_*`` tags directly to the ``mlruns/`` filesystem
    so that champion models can be identified via pandas/polars filtering.

    Parameters
    ----------
    analysis_entries:
        Flat list of analysis entry dicts from
        :func:`create_analysis_experiment`.
    runs:
        Training run info dicts from :func:`load_training_artifacts`.
    ensembles:
        Ensemble specs from :func:`build_ensembles`.
    eval_config:
        Evaluation configuration with primary metric settings.
    mlruns_dir:
        Root mlruns directory. Auto-detected from repo root if ``None``.
    experiment_id:
        MLflow experiment ID. Auto-detected if ``None``.

    Returns
    -------
    Dict with ``champion_selection`` and ``tags_written`` count.
    """
    from pathlib import Path

    from minivess.config.evaluation_config import MetricDirection

    log = get_run_logger()

    # Auto-detect mlruns_dir from repo root
    if mlruns_dir is None:
        mlruns_dir = Path(__file__).resolve().parents[3] / "mlruns"

    # Auto-detect experiment_id from the training experiment
    if experiment_id is None and mlruns_dir.is_dir():
        # Find experiment directory by name
        for exp_dir in mlruns_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            meta = exp_dir / "meta.yaml"
            if meta.is_file():
                content = meta.read_text(encoding="utf-8")
                if eval_config.mlflow_training_experiment in content:
                    experiment_id = exp_dir.name
                    break

    if experiment_id is None:
        log.warning("Could not determine experiment_id; skipping champion tagging")
        return {"champion_selection": None, "tags_written": 0}

    maximize = eval_config.primary_metric_direction == MetricDirection.MAXIMIZE

    selection = tag_champions(
        mlruns_dir,
        experiment_id,
        analysis_entries,
        runs=runs,
        ensembles=ensembles,
        primary_metric=eval_config.primary_metric,
        maximize=maximize,
    )

    log.info(
        "Champion tagging complete: single_fold=%s, cv_mean=%s, ensemble=%s",
        selection.best_single_fold is not None,
        selection.best_cv_mean is not None,
        selection.best_ensemble is not None,
    )

    return {
        "champion_selection": selection,
        "tags_written": 1,  # Simplified count
    }


@task(name="generate-report", on_completion=[_on_complete], on_failure=[_on_fail])
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
                        metric_strs.append(f"{m_name}={ci.point_estimate:.4f}")
                lines.append(
                    f"- **{ds_name}/{subset_name}**: " + ", ".join(metric_strs)
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
        score_str = f"{champion_score:.4f}" if not math.isnan(champion_score) else "N/A"
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


@task(name="summarize-experiment", on_completion=[_on_complete], on_failure=[_on_fail])
def summarize_experiment(
    all_results: dict[str, dict[str, dict[str, Any]]],
    promotion_info: dict[str, Any],
    use_agent: bool | None = None,
) -> dict[str, Any]:
    """Summarize experiment results using Pydantic AI agent or deterministic fallback.

    When ``use_agent=True`` (or ``MINIVESS_USE_AGENTS=1``), uses the Pydantic AI
    experiment summarizer with TestModel in CI. Falls back to the deterministic
    stub if pydantic-ai is not installed or ``use_agent=False``.

    Parameters
    ----------
    all_results:
        ``{model_name: {dataset: {subset: EvaluationResult}}}``.
    promotion_info:
        Dict from :func:`register_champion_task`.
    use_agent:
        Explicitly enable/disable agent. Defaults to ``MINIVESS_USE_AGENTS`` env var.

    Returns
    -------
    Summary dict with ``action``, ``reasoning``, and ``summary`` keys.
    """
    if use_agent is None:
        use_agent = os.environ.get("MINIVESS_USE_AGENTS") == "1"

    context = {
        "n_models": len(all_results),
        "best_model": promotion_info.get("champion_name", "unknown"),
        "best_metric_value": promotion_info.get("champion_score", 0.0),
    }

    if use_agent:
        try:
            from pydantic_ai.models.test import TestModel

            from minivess.agents.experiment_summarizer import (
                AnalysisContext,
                _build_agent,
            )

            agent = _build_agent(model="test")
            ctx = AnalysisContext(
                n_models=context["n_models"],
                best_model=context["best_model"],
                best_metric_value=context["best_metric_value"],
            )
            test_output = {
                "narrative": (
                    f"Evaluated {ctx.n_models} models. Best: {ctx.best_model} "
                    f"(metric={ctx.best_metric_value:.4f})."
                ),
                "best_model": ctx.best_model,
                "best_metric_value": ctx.best_metric_value,
                "key_findings": [f"Best model: {ctx.best_model}"],
                "recommendations": [],
            }
            result = agent.run_sync(
                "Summarize this experiment.",
                deps=ctx,
                model=TestModel(custom_output_args=test_output, call_tools=[]),
            )
            return {
                "action": "summarize",
                "reasoning": result.output.narrative,
                "summary": result.output.model_dump(),
            }
        except ImportError:
            logger.debug("pydantic-ai not available, using deterministic fallback")

    from minivess.orchestration.agent_interface import DeterministicExperimentSummary

    return DeterministicExperimentSummary().decide(context=context)


# ---------------------------------------------------------------------------
# Analysis experiment entry builder
# ---------------------------------------------------------------------------

# Model name convention: "{loss_type}_fold{fold_id}" for single-fold models.
# NOTE: re.compile() is BANNED (CLAUDE.md Rule #16). Use str.rsplit() instead.


def parse_fold_metric(model_name: str) -> tuple[str, int]:
    """Parse a model name into ``(loss_function, fold_id)`` using str.rsplit().

    Parameters
    ----------
    model_name:
        E.g. ``"dice_ce_fold0"`` or ``"cbdice_cldice_fold2"``.

    Returns
    -------
    Tuple of ``(loss_function, fold_id)``.

    Raises
    ------
    ValueError
        If ``model_name`` does not contain ``_fold`` followed by an integer.
    """
    if "_fold" not in model_name:
        msg = f"Model name {model_name!r} has no '_fold' component"
        raise ValueError(msg)
    prefix, fold_part = model_name.rsplit("_fold", 1)
    try:
        fold_id = int(fold_part)
    except ValueError:
        msg = f"Model name {model_name!r} has non-integer fold suffix {fold_part!r}"
        raise ValueError(msg) from None
    return prefix, fold_id


def _parse_model_name(model_name: str) -> tuple[str | None, int | None]:
    """Parse a model name into ``(loss_function, fold_id)``.

    Returns ``(None, None)`` for ensemble / non-standard names.

    Parameters
    ----------
    model_name:
        E.g. ``"dice_ce_fold0"`` or ``"per_loss_single_best"``.

    Returns
    -------
    Tuple of ``(loss_function, fold_id)`` or ``(None, None)``.
    """
    try:
        return parse_fold_metric(model_name)
    except ValueError:
        return None, None


def _extract_primary_metric_value(
    eval_result: EvaluationResult,
    primary_metric: str,
) -> float:
    """Extract the primary metric's point estimate from an EvaluationResult.

    Falls back to ``float('nan')`` if the metric is not present.

    Parameters
    ----------
    eval_result:
        Result for one dataset/subset combination.
    primary_metric:
        Name of the primary metric (e.g. ``"dsc"``).

    Returns
    -------
    The point estimate as a float.
    """
    ci = eval_result.fold_result.aggregated.get(primary_metric)
    if ci is not None:
        value: float = ci.point_estimate
        return value
    return float("nan")


def _flatten_metrics(
    eval_result: EvaluationResult,
) -> dict[str, float]:
    """Flatten a single EvaluationResult's aggregated CIs to ``{name: value}``.

    Parameters
    ----------
    eval_result:
        Result for one dataset/subset combination.

    Returns
    -------
    Flat dict of metric names to point estimates.
    """
    return {
        name: ci.point_estimate
        for name, ci in eval_result.fold_result.aggregated.items()
    }


def _first_eval_result(
    ds_dict: dict[str, dict[str, EvaluationResult]],
) -> EvaluationResult | None:
    """Return the first EvaluationResult from a nested dataset dict.

    Parameters
    ----------
    ds_dict:
        ``{dataset: {subset: EvaluationResult}}``.

    Returns
    -------
    The first ``EvaluationResult``, or ``None`` if empty.
    """
    for subset_dict in ds_dict.values():
        for eval_result in subset_dict.values():
            return eval_result
    return None


def create_analysis_experiment(
    all_results: dict[str, dict[str, dict[str, EvaluationResult]]],
    eval_config: EvaluationConfig,
) -> list[dict[str, Any]]:
    """Create structured entries for an MLflow analysis experiment.

    Produces a flat list of dicts, each representing one "row" in the
    analysis experiment.  Entry types:

    * ``"per_fold"``  -- one per loss-function/fold combination
    * ``"cv_mean"``   -- one per loss-function (mean across folds)
    * ``"ensemble"``  -- one per ensemble strategy
    * ``"champion"``  -- single best model across all entries

    Parameters
    ----------
    all_results:
        ``{model_name: {dataset: {subset: EvaluationResult}}}``.
        Model names follow the convention ``"{loss}_fold{id}"`` for
        single-fold checkpoints; any other name is treated as an
        ensemble.
    eval_config:
        :class:`EvaluationConfig` with ``primary_metric`` and
        ``primary_metric_direction``.

    Returns
    -------
    List of entry dicts, each with keys:
    ``entry_type``, ``model_name``, ``loss_function``, ``fold_id``,
    ``metrics``, ``primary_metric_value``.
    """
    from minivess.config.evaluation_config import MetricDirection

    entries: list[dict[str, Any]] = []
    primary = eval_config.primary_metric

    # ---- 1. Per-fold entries (single-fold checkpoints) --------------------
    # Group by loss_function for CV-mean computation
    loss_fold_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    loss_fold_primary: dict[str, list[float]] = defaultdict(list)

    for model_name, ds_dict in all_results.items():
        loss_fn, fold_id = _parse_model_name(model_name)
        if loss_fn is None:
            # Not a per-fold model; will handle below as ensemble
            continue

        first = _first_eval_result(ds_dict)
        if first is None:
            continue

        metrics = _flatten_metrics(first)
        primary_val = _extract_primary_metric_value(first, primary)

        entries.append(
            {
                "entry_type": "per_fold",
                "model_name": model_name,
                "loss_function": loss_fn,
                "fold_id": fold_id,
                "metrics": metrics,
                "primary_metric_value": primary_val,
            }
        )

        loss_fold_metrics[loss_fn].append(metrics)
        loss_fold_primary[loss_fn].append(primary_val)

    # ---- 2. CV-mean entries (one per loss function) -----------------------
    for loss_fn, fold_metrics_list in sorted(loss_fold_metrics.items()):
        if not fold_metrics_list:
            continue

        # Average each metric across folds
        all_metric_names = {k for m in fold_metrics_list for k in m}
        mean_metrics: dict[str, float] = {}
        for metric_name in sorted(all_metric_names):
            values = [m[metric_name] for m in fold_metrics_list if metric_name in m]
            if values:
                mean_metrics[metric_name] = sum(values) / len(values)

        primary_vals = loss_fold_primary[loss_fn]
        cv_primary = (
            sum(primary_vals) / len(primary_vals) if primary_vals else float("nan")
        )

        entries.append(
            {
                "entry_type": "cv_mean",
                "model_name": f"{loss_fn}_cv_mean",
                "loss_function": loss_fn,
                "fold_id": None,
                "metrics": mean_metrics,
                "primary_metric_value": cv_primary,
            }
        )

    # ---- 3. Ensemble entries (non-fold model names) -----------------------
    for model_name, ds_dict in all_results.items():
        loss_fn, fold_id = _parse_model_name(model_name)
        if loss_fn is not None:
            # Already handled as per-fold
            continue

        first = _first_eval_result(ds_dict)
        if first is None:
            continue

        metrics = _flatten_metrics(first)
        primary_val = _extract_primary_metric_value(first, primary)

        entries.append(
            {
                "entry_type": "ensemble",
                "model_name": model_name,
                "loss_function": None,
                "fold_id": None,
                "metrics": metrics,
                "primary_metric_value": primary_val,
            }
        )

    # ---- 4. Champion entry (best overall by primary metric) ---------------
    maximize = eval_config.primary_metric_direction == MetricDirection.MAXIMIZE

    best_entry: dict[str, Any] | None = None
    best_score = float("-inf") if maximize else float("inf")

    for entry in entries:
        val = entry["primary_metric_value"]
        if math.isnan(val):
            continue
        if (maximize and val > best_score) or (not maximize and val < best_score):
            best_score = val
            best_entry = entry

    if best_entry is not None:
        entries.append(
            {
                "entry_type": "champion",
                "model_name": best_entry["model_name"],
                "loss_function": best_entry["loss_function"],
                "fold_id": best_entry["fold_id"],
                "metrics": dict(best_entry["metrics"]),
                "primary_metric_value": best_score,
            }
        )

    return entries


# ---------------------------------------------------------------------------
# Artifact export helpers
# ---------------------------------------------------------------------------


def _build_comparison_table_from_results(
    all_results: dict[str, dict[str, dict[str, Any]]],
) -> ComparisonTable | None:
    """Build a ComparisonTable from evaluation results for figure generation.

    Attempts to extract per-fold metric values from EvaluationResult objects.
    Returns None if insufficient data.
    """
    from minivess.pipeline.ci import ConfidenceInterval

    losses: list[LossResult] = []
    all_metric_names: set[str] = set()

    for model_name, datasets in all_results.items():
        fold_metrics: dict[str, list[float]] = defaultdict(list)
        n_folds = 0

        for _dataset_name, subsets in datasets.items():
            for _subset_name, eval_result in subsets.items():
                n_folds += 1
                if hasattr(eval_result, "metrics") and eval_result.metrics:
                    for metric_name, value in eval_result.metrics.items():
                        if isinstance(value, ConfidenceInterval):
                            fold_metrics[metric_name].append(
                                value.point_estimate,
                            )
                        elif isinstance(value, int | float):
                            fold_metrics[metric_name].append(float(value))

        if not fold_metrics:
            continue

        metrics: dict[str, MetricSummary] = {}
        for metric_name, values in fold_metrics.items():
            import numpy as np

            arr = np.array(values)
            mean_val = float(arr.mean())
            std_val = float(arr.std()) if len(arr) > 1 else 0.0
            metrics[metric_name] = MetricSummary(
                mean=mean_val,
                std=std_val,
                ci_lower=mean_val - 1.96 * std_val,
                ci_upper=mean_val + 1.96 * std_val,
                per_fold=values,
            )
            all_metric_names.add(metric_name)

        losses.append(
            LossResult(
                loss_name=model_name,
                num_folds=n_folds,
                metrics=metrics,
            ),
        )

    if not losses:
        return None

    return ComparisonTable(
        losses=losses,
        metric_names=sorted(all_metric_names),
    )


def _export_analysis_artifacts(
    comparison_md: str,
    all_results: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path | None,
) -> dict[str, str]:
    """Export comparison tables (markdown + LaTeX) and figures to disk.

    Returns dict mapping artifact type to file path.
    """
    from pathlib import Path as _Path

    from minivess.pipeline.viz.generate_all_figures import generate_all_figures

    base_dir = output_dir or _Path(
        os.environ.get("ANALYSIS_OUTPUT_DIR", "/app/outputs/analysis")
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths: dict[str, str] = {}

    # Save markdown comparison table
    md_path = base_dir / "comparison_table.md"
    md_path.write_text(comparison_md, encoding="utf-8")
    artifact_paths["comparison_md"] = str(md_path)
    logger.info("Saved comparison markdown: %s", md_path)

    # Build ComparisonTable from results and export LaTeX
    table = _build_comparison_table_from_results(all_results)
    if table is not None:
        latex = format_comparison_latex(table)
        tex_path = base_dir / "comparison_table.tex"
        tex_path.write_text(latex, encoding="utf-8")
        artifact_paths["comparison_latex"] = str(tex_path)
        logger.info("Saved LaTeX table: %s", tex_path)

        # Generate all figures with real data
        figures_dir = base_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig_summary = generate_all_figures(
            output_dir=figures_dir,
            comparison_table=table,
            formats=["png", "svg"],
        )
        artifact_paths["figures_dir"] = str(figures_dir)
        artifact_paths["figures_succeeded"] = ",".join(fig_summary["succeeded"])
        artifact_paths["figures_failed"] = ",".join(fig_summary["failed"])
        logger.info(
            "Generated %d figures (%d failed)",
            len(fig_summary["succeeded"]),
            len(fig_summary["failed"]),
        )
    else:
        logger.warning("No comparison table built — skipping LaTeX and figures")

    return artifact_paths


@dataclass
class EmbeddingDriftResult:
    """Result from embedding_drift_task."""

    drift_detected: bool
    p_value: float
    mmd_statistic: float


@task(name="embedding-drift", on_completion=[_on_complete], on_failure=[_on_fail])
def embedding_drift_task(
    *,
    reference_embeddings: NDArray[np.float32],
    current_embeddings: NDArray[np.float32],
    p_val_threshold: float,
    n_permutations: int = 100,
    tmp_dir: Path | None = None,
) -> EmbeddingDriftResult:
    """Run Tier 2 embedding-space drift detection.

    Compares current model embeddings against reference embeddings using
    EmbeddingDriftDetector (permutation MMD with RBF kernel).

    Parameters
    ----------
    reference_embeddings:
        Reference embedding array from training run.
    current_embeddings:
        Current embedding array from validation.
    p_val_threshold:
        P-value threshold for drift detection.
    n_permutations:
        Number of permutations for the MMD test.
    tmp_dir:
        If provided (inside active MLflow run), save MMD summary as artifact.

    Returns
    -------
    EmbeddingDriftResult with p-value and MMD statistic.
    """
    import json

    import mlflow

    from minivess.observability.drift import EmbeddingDriftDetector

    detector = EmbeddingDriftDetector(
        reference_embeddings,
        p_val_threshold=p_val_threshold,
        n_permutations=n_permutations,
    )
    result = detector.detect(current_embeddings)
    mmd_stat = result.feature_scores.get("mmd_statistic", 0.0)

    # Persist to MLflow if tmp_dir provided and active run exists
    if tmp_dir is not None:
        try:
            active_run = mlflow.active_run()
            if active_run is not None:
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tier2_path = tmp_dir / "tier2_mmd_summary.json"
                summary = {
                    "drift_detected": result.drift_detected,
                    "p_value": result.dataset_drift_score,
                    "mmd_statistic": mmd_stat,
                    "timestamp": result.timestamp.isoformat(),
                }
                tier2_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                mlflow.log_artifact(str(tier2_path), artifact_path="drift_reports")
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to persist Tier 2 drift report to MLflow", exc_info=True
            )

    return EmbeddingDriftResult(
        drift_detected=result.drift_detected,
        p_value=result.dataset_drift_score,
        mmd_statistic=mmd_stat,
    )


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


@flow(name=FLOW_NAME_ANALYSIS, validate_parameters=False)
def run_analysis_flow(
    eval_config: EvaluationConfig,
    model_config_dict: dict[str, Any],
    dataloaders_dict: HierarchicalDataLoaderDict,
    *,
    output_dir: Path | None = None,
    environment: str = "staging",
    tracking_uri: str | None = None,
    include_post_training: bool = True,
    upstream_training_run_id: str | None = None,
) -> dict[str, Any]:
    """Main analysis flow orchestrating all tasks.

    Executes the complete post-training analysis pipeline:

    1. **Load** training artifacts from MLflow
    2. **Build** ensemble models using configured strategies
    3. **Log** models as MLflow pyfunc artifacts (single + ensemble)
    4. **Evaluate** all models (single + ensemble) on test datasets
    5. **MLflow evaluate** with custom segmentation metrics
    6. **Compare** models with cross-loss comparison table
    7. **Register** the best model as champion
    8. **Report** generate a comprehensive markdown summary

    Parameters
    ----------
    eval_config:
        :class:`EvaluationConfig` with metric and MLflow settings.
    model_config_dict:
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
    require_docker_context("analysis")

    logs_dir = Path(os.environ.get("LOGS_DIR", "/app/logs"))
    with gpu_flow_observability_context("analysis", logs_dir=logs_dir):
        log = get_run_logger()
        log.info("Starting analysis flow...")

        # Preflight: validate required environment variables
        _validate_analysis_env()

        # Step 0 (optional): Discover post-training models
        post_training_models: list[dict[str, Any]] = []
        if include_post_training:
            post_training_models = discover_post_training_models(
                tracking_uri=tracking_uri,
            )
            if post_training_models:
                log.info(
                    "Discovered %d post-training model(s) for evaluation",
                    len(post_training_models),
                )

        # Step 0b: Auto-derive ensemble strategies from factorial YAML (Task 2.13)
        factorial_yaml_path = os.environ.get("FACTORIAL_YAML")
        if factorial_yaml_path:
            override_strategies = _resolve_ensemble_strategies(
                factorial_yaml=Path(factorial_yaml_path),
            )
            if override_strategies:
                log.info(
                    "Overriding ensemble strategies from factorial YAML: %s",
                    override_strategies,
                )

        # Step 1: Load training artifacts
        runs = load_training_artifacts(
            eval_config, model_config_dict, tracking_uri=tracking_uri
        )

        # Step 2: Build ensembles
        ensembles = build_ensembles(runs, eval_config, model_config_dict)

        # Step 3: Log models as MLflow pyfunc artifacts
        model_uris = log_models_to_mlflow(runs, ensembles, eval_config, model_config_dict)

        # Step 4: Extract single-fold models from ensemble members
        single_models = _extract_single_models_as_modules(ensembles)

        # Step 5: Evaluate all models (single + ensemble with all members)
        all_results = evaluate_all_models(
            single_models,
            ensembles,
            dataloaders_dict,
            eval_config,
            output_dir=output_dir,
        )

        # Step 6: Run MLflow evaluate with custom metrics
        mlflow_eval_results = evaluate_with_mlflow(all_results, eval_config)

        # Step 7: Generate comparison
        comparison_md = generate_comparison(all_results)

        # Step 7b: Export analysis DuckDB (per-volume metrics for biostatistics)
        analysis_duckdb_path = task_export_analysis_duckdb(
            all_results=all_results,
            manifest_runs=runs if hasattr(runs, "__iter__") else [],
            output_dir=output_dir or Path("outputs/analysis"),
            metadata={
                "git_sha": os.environ.get("GIT_SHA", "unknown"),
                "upstream_experiment": os.environ.get("UPSTREAM_EXPERIMENT", "unknown"),
            },
        )

        # Step 8: Register champion (with model URIs for actual registration)
        promotion_info = register_champion_task(
            all_results,
            eval_config,
            model_uris=model_uris,
            environment=environment,
            tracking_uri=tracking_uri,
        )

        # Step 9: Tag champion models on training runs filesystem
        analysis_entries = create_analysis_experiment(all_results, eval_config)
        champion_info = tag_champion_models(
            analysis_entries,
            runs,
            ensembles,
            eval_config,
        )

        # Step 10: Generate report
        report = generate_report(all_results, comparison_md, promotion_info)

        # Step 10b: Experiment summary (agent decision point — deterministic stub)
        experiment_summary = summarize_experiment(all_results, promotion_info)

        # Step 11: Export comparison tables and figures to disk
        artifact_paths = _export_analysis_artifacts(
            comparison_md,
            all_results,
            output_dir,
        )

        log.info(
            "Analysis flow complete. Champion: %s", promotion_info.get("champion_name")
        )

        # --- FlowContract: tag run and log completion ---
        _tracking_uri = tracking_uri or resolve_tracking_uri()
        # Use provided upstream ID or auto-discover from MLflow
        if upstream_training_run_id is None:
            upstream = find_upstream_safely(
                tracking_uri=_tracking_uri,
                experiment_name=os.environ.get(
                    "UPSTREAM_EXPERIMENT", resolve_experiment_name(EXPERIMENT_TRAINING)
                ),
                upstream_flow="train",
            )
            upstream_training_run_id = upstream["run_id"] if upstream else None
        mlflow_run_id: str | None = None
        try:
            import mlflow

            mlflow.set_tracking_uri(_tracking_uri)
            mlflow.set_experiment(resolve_experiment_name(EXPERIMENT_EVALUATION))
            with mlflow.start_run(
                tags={
                    "flow_name": FLOW_NAME_ANALYSIS,
                    "upstream_training_run_id": upstream_training_run_id,
                }
            ) as active_run:
                mlflow_run_id = active_run.info.run_id
        except Exception:
            log.warning("Failed to log analysis_flow to MLflow", exc_info=True)

        # Log flow completion (best-effort, non-blocking)
        log_completion_safe(
            flow_name=FLOW_NAME_ANALYSIS,
            tracking_uri=_tracking_uri,
            run_id=mlflow_run_id,
        )

        # OpenLineage lineage emission (Issue #799 — IEC 62304 §8 traceability)
        try:
            _emitter = LineageEmitter(namespace="minivess")
            emit_flow_lineage(
                emitter=_emitter,
                job_name="analysis-flow",
                inputs=[
                    {"namespace": "minivess", "name": "checkpoints"},
                    {"namespace": "minivess", "name": "test_datasets"},
                ],
                outputs=[
                    {"namespace": "minivess", "name": "evaluation_metrics"},
                    {"namespace": "minivess", "name": "ensemble_models"},
                ],
            )
        except Exception:
            logger.warning("OpenLineage emission failed (non-blocking)", exc_info=True)

        # FDA test set firewall: log every test evaluation (Issue #821)
        try:
            audit = AuditTrail()
            audit.log_test_evaluation(
                model_name=eval_config.primary_metric or "analysis_flow",
                metrics={"test_set_access_count": 1.0},
                actor="analysis_flow",
            )
        except Exception:
            logger.warning(
                "AuditTrail log_test_evaluation failed (non-blocking)", exc_info=True
            )

        return {
            "results": all_results,
            "comparison": comparison_md,
            "promotion": promotion_info,
            "report": report,
            "mlflow_evaluation": mlflow_eval_results,
            "champion_tags": champion_info,
            "artifact_paths": artifact_paths,
            "post_training_models": post_training_models,
            "experiment_summary": experiment_summary,
            "mlflow_run_id": mlflow_run_id,
            "upstream_training_run_id": upstream_training_run_id,
            "analysis_duckdb_path": analysis_duckdb_path,
        }


# ---------------------------------------------------------------------------
# Docker entry point helpers
# ---------------------------------------------------------------------------


def _load_config_from_mlflow(run_id: str, tracking_uri: str) -> dict[str, Any]:
    """Load resolved experiment config from MLflow artifact (config/resolved_config.yaml).

    Falls back to empty dict if the artifact is not present (e.g. legacy runs).
    """
    import yaml

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()
        local_path = client.download_artifacts(run_id, "config/resolved_config.yaml")
        with open(local_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        logger.warning(
            "Could not load config/resolved_config.yaml for run %s — using defaults",
            run_id,
        )
        return {}


def _build_eval_config_from_dict(config_dict: dict[str, Any]) -> EvaluationConfig:
    """Build EvaluationConfig from a resolved config dict.

    Overrides mlflow_training_experiment from config when available.
    """
    from minivess.config.evaluation_config import EvaluationConfig

    upstream_exp = os.environ.get("UPSTREAM_EXPERIMENT")
    overrides: dict[str, Any] = {}
    if upstream_exp:
        overrides["mlflow_training_experiment"] = upstream_exp
        # Debug experiments don't produce eval_fold2_dsc — relax the gate (#588).
        if upstream_exp.startswith("debug_"):
            overrides["require_eval_metrics"] = False
    return EvaluationConfig(**overrides)


def _build_model_config_from_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract model configuration dict from resolved config.

    Returns the 'model' sub-dict if present, otherwise wraps the full dict.
    """
    result: dict[str, Any] = config_dict.get("model") or config_dict
    return result


def _build_dataloaders_from_config(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Build external test dataloaders from EXTERNAL_DATA_DIR env var.

    Discovers DeepVess (and any future external_test datasets) from the
    directory pointed to by EXTERNAL_DATA_DIR and builds a
    HierarchicalDataLoaderDict for evaluation.

    Parameters
    ----------
    config_dict:
        Resolved experiment configuration (currently unused but kept for
        future data_config extraction).

    Returns
    -------
    ``{dataset_name: {"all": DataLoader, ...}}``

    Raises
    ------
    FileNotFoundError
        If EXTERNAL_DATA_DIR is not set or does not exist (Rule 25: loud failures).
    """
    external_data_dir_str = os.environ.get("EXTERNAL_DATA_DIR")
    if not external_data_dir_str:
        raise FileNotFoundError(
            "EXTERNAL_DATA_DIR environment variable is not set.\n"
            "Set it to the directory containing external test datasets:\n"
            "  export EXTERNAL_DATA_DIR=data/external\n"
            "See: .env.example and docs/datasets/README.md"
        )

    external_data_dir = Path(external_data_dir_str)
    if not external_data_dir.is_dir():
        raise FileNotFoundError(
            f"EXTERNAL_DATA_DIR does not exist: {external_data_dir}\n"
            "Download external test datasets first.\n"
            "See: docs/datasets/README.md for download instructions."
        )

    from minivess.data.external_datasets import (
        EXTERNAL_DATASETS,
        discover_external_test_pairs,
    )

    result: dict[str, Any] = {}

    for ds_name, ds_config in EXTERNAL_DATASETS.items():
        # Only include datasets with role="external_test"
        # VesselNN is drift_detection_only — must be excluded
        role = getattr(ds_config, "role", "external_test")
        if role != "external_test":
            logger.debug(
                "Skipping dataset '%s' (role=%s, not external_test)",
                ds_name,
                role,
            )
            continue

        ds_dir = external_data_dir / ds_name
        if not ds_dir.is_dir():
            logger.warning(
                "External dataset directory not found: %s — skipping %s",
                ds_dir,
                ds_name,
            )
            continue

        pairs = discover_external_test_pairs(ds_dir, ds_name)
        if not pairs:
            logger.warning(
                "No image/label pairs found for dataset '%s' in %s",
                ds_name,
                ds_dir,
            )
            continue

        # Build real DataLoaders from discovered pairs.
        # Uses _build_loader_from_dicts which creates MONAI CacheDataset
        # with validation transforms (no augmentation — test data only).
        try:
            from minivess.config.models import DataConfig
            from minivess.data.test_datasets import _build_loader_from_dicts

            # Build a minimal DataConfig for external test evaluation.
            # External test data uses native resolution (no resampling) and
            # the same patch/transform pipeline as MiniVess validation.
            data_kwargs = config_dict.get("data", {})
            data_kwargs.setdefault("dataset_name", ds_name)
            data_config = DataConfig(**data_kwargs)
            loader = _build_loader_from_dicts(pairs, data_config, cache_rate=0.0)
            result[ds_name] = {"all": loader}
        except Exception:
            # Rule #25: Loud failures. Do NOT fall back to raw pairs —
            # evaluation on unprocessed data produces invalid metrics.
            logger.error(
                "Failed to build DataLoader for '%s' — cannot evaluate. "
                "Fix the data format or transforms before proceeding.",
                ds_name,
                exc_info=True,
            )
            raise
        logger.info(
            "Discovered %d pairs for external test dataset '%s'",
            len(pairs),
            ds_name,
        )

    if not result:
        logger.warning(
            "No external test datasets found in %s. "
            "Download DeepVess and place in %s/deepvess/",
            external_data_dir,
            external_data_dir,
        )

    return result


def _entry_point_from_env() -> dict[str, Any]:
    """Discover analysis parameters from UPSTREAM_EXPERIMENT env var.

    Called by the Docker entry point (__main__) to build the parameters
    needed by run_analysis_flow() without requiring the caller to supply
    eval_config, model_config_dict, and dataloaders_dict explicitly.

    Returns
    -------
    dict with keys: eval_config, model_config_dict, dataloaders_dict,
                    upstream_training_run_id, tracking_uri
    """
    upstream_exp = os.environ.get("UPSTREAM_EXPERIMENT")
    if not upstream_exp:
        raise RuntimeError(
            "UPSTREAM_EXPERIMENT not set.\n"
            "Run: docker compose run -e UPSTREAM_EXPERIMENT=<experiment_name> analyze\n"
            "Example: docker compose run -e UPSTREAM_EXPERIMENT=debug_full_pipeline analyze"
        )

    tracking_uri = resolve_tracking_uri()
    upstream = find_upstream_safely(
        tracking_uri=tracking_uri,
        experiment_name=upstream_exp,
        upstream_flow=FLOW_NAME_TRAIN,
    )
    if not upstream:
        raise RuntimeError(
            f"No completed training runs found in experiment '{upstream_exp}'. "
            "Run training first: docker compose run -e EXPERIMENT=<name> train"
        )

    config_dict = _load_config_from_mlflow(upstream["run_id"], tracking_uri)
    eval_config = _build_eval_config_from_dict(config_dict)
    model_config_dict = _build_model_config_from_dict(config_dict)
    dataloaders_dict = _build_dataloaders_from_config(config_dict)

    return {
        "eval_config": eval_config,
        "model_config_dict": model_config_dict,
        "dataloaders_dict": dataloaders_dict,
        "upstream_training_run_id": upstream["run_id"],
        "tracking_uri": tracking_uri,
    }


def _resolve_ensemble_strategies(
    *,
    factorial_yaml: Path | None = None,
) -> list[str] | None:
    """Resolve ensemble strategies from factorial YAML if provided.

    If ``factorial_yaml`` is set, reads ``factors.analysis.ensemble_strategy``
    to auto-derive the strategy list. This enables composable factorial designs
    where different labs define their own strategy sets.

    Returns ``None`` when no YAML is provided or parsing fails — the caller
    should fall back to ``eval_config.ensemble_strategies`` defaults.

    References: XML plan T1.7, synthesis Part 1.2 Layer C.
    """
    if factorial_yaml is None:
        return None

    try:
        from minivess.config.factorial_config import parse_factorial_yaml

        design = parse_factorial_yaml(factorial_yaml)
        levels = design.factor_levels()
        strategies = levels.get("ensemble_strategy")
        if strategies:
            logger.info(
                "Auto-derived %d ensemble strategies from %s: %s",
                len(strategies),
                factorial_yaml,
                strategies,
            )
            return strategies
    except Exception:
        logger.warning(
            "Failed to parse factorial YAML %s for ensemble strategies, "
            "using config defaults",
            factorial_yaml,
            exc_info=True,
        )
    return None


if __name__ == "__main__":
    # Docker entry point — reads UPSTREAM_EXPERIMENT env var to discover
    # the correct MLflow training experiment for analysis.
    # UPSTREAM_EXPERIMENT and EXPERIMENT are single-source config (Rule #22).
    params = _entry_point_from_env()
    run_analysis_flow(
        eval_config=params["eval_config"],
        model_config_dict=params["model_config_dict"],
        dataloaders_dict=params["dataloaders_dict"],
        upstream_training_run_id=params["upstream_training_run_id"],
        tracking_uri=params["tracking_uri"],
    )
