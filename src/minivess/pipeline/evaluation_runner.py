"""Unified evaluation runner for single models and ensembles.

Provides :class:`UnifiedEvaluationRunner` which evaluates any ``nn.Module``
(single model or ensemble) against hierarchical dataloaders and logs results
to the ``minivess_evaluation`` MLflow experiment.

Key design decisions
--------------------
* **Model-agnostic** -- accepts any ``nn.Module``; same code path for a single
  fold checkpoint and a multi-member Deep Ensemble.
* **Metric-agnostic** -- delegates to :class:`EvaluationRunner` from
  ``evaluation.py`` for MetricsReloaded computation.
* **Observable** -- logs metrics, tags, and prediction artifacts to MLflow.
* **Composable** -- each method is independently callable for notebooks,
  CI, or Prefect tasks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np

from minivess.pipeline.evaluation import EvaluationRunner, FoldResult
from minivess.pipeline.prediction_store import save_volume_prediction
from minivess.pipeline.validation_metrics import compute_compound_masd_cldice

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

    from minivess.config.evaluation_config import EvaluationConfig
    from minivess.data.test_datasets import HierarchicalDataLoaderDict
    from minivess.pipeline.inference import SlidingWindowInferenceRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Result of evaluating one model on one dataset subset.

    Parameters
    ----------
    model_name:
        Descriptive name of the model being evaluated.
    dataset_name:
        Dataset identifier (e.g. ``"minivess"``).
    subset_name:
        Subset identifier (e.g. ``"all"``, ``"thin_vessels"``).
    fold_result:
        :class:`FoldResult` with per-volume and aggregated metrics.
    predictions_dir:
        Directory containing saved ``.npz`` prediction files, if any.
    uncertainty_maps_dir:
        Directory containing uncertainty maps (ensemble-only), if any.
    """

    model_name: str
    dataset_name: str
    subset_name: str
    fold_result: FoldResult
    predictions_dir: Path | None
    uncertainty_maps_dir: Path | None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class UnifiedEvaluationRunner:
    """Evaluate single models and ensembles uniformly.

    Delegates to :class:`SlidingWindowInferenceRunner` for sliding-window
    inference and to :class:`EvaluationRunner` for MetricsReloaded metrics
    with bootstrap CIs.

    Parameters
    ----------
    eval_config:
        :class:`EvaluationConfig` with metric and MLflow settings.
    inference_runner:
        Pre-configured :class:`SlidingWindowInferenceRunner`.
    """

    def __init__(
        self,
        eval_config: EvaluationConfig,
        inference_runner: SlidingWindowInferenceRunner,
    ) -> None:
        self.eval_config = eval_config
        self.inference_runner = inference_runner
        self._metrics_runner = EvaluationRunner(
            include_expensive=eval_config.include_expensive_metrics,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model: nn.Module,
        dataloaders_dict: HierarchicalDataLoaderDict,
        *,
        model_name: str,
        model_tags: dict[str, str] | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, dict[str, EvaluationResult]]:
        """Evaluate a model on all datasets and subsets.

        Parameters
        ----------
        model:
            Trained ``nn.Module`` (single or ensemble).
        dataloaders_dict:
            ``{dataset_name: {subset_name: DataLoader, ...}, ...}``.
        model_name:
            Human-readable model identifier.
        model_tags:
            Optional extra tags for MLflow.
        output_dir:
            If given, save predictions under
            ``output_dir / dataset / subset /``.

        Returns
        -------
        Nested dict: ``{dataset_name: {subset_name: EvaluationResult}}``.
        """
        results: dict[str, dict[str, EvaluationResult]] = {}

        for ds_name, subset_loaders in dataloaders_dict.items():
            ds_results: dict[str, EvaluationResult] = {}
            for subset_name, loader in subset_loaders.items():
                subset_output_dir = (
                    output_dir / ds_name / subset_name if output_dir else None
                )
                ds_results[subset_name] = self.evaluate_single_subset(
                    model,
                    loader,
                    model_name=model_name,
                    dataset_name=ds_name,
                    subset_name=subset_name,
                    output_dir=subset_output_dir,
                )
            results[ds_name] = ds_results

        return results

    def evaluate_single_subset(
        self,
        model: nn.Module,
        loader: Any,
        *,
        model_name: str,
        dataset_name: str,
        subset_name: str,
        output_dir: Path | None = None,
    ) -> EvaluationResult:
        """Evaluate on a single dataset subset.

        Runs sliding-window inference, computes MetricsReloaded metrics,
        and optionally saves predictions to disk.

        Parameters
        ----------
        model:
            Trained segmentation model.
        loader:
            DataLoader yielding ``{"image": ..., "label": ...}`` dicts.
        model_name:
            Model identifier.
        dataset_name:
            Dataset identifier.
        subset_name:
            Subset identifier.
        output_dir:
            If given, save ``.npz`` predictions here.

        Returns
        -------
        :class:`EvaluationResult`
        """
        # Step 1: Inference
        predictions, labels = self.inference_runner.infer_dataset(
            model, loader, device="cpu"
        )

        # Step 2: Save predictions if output_dir given
        predictions_dir: Path | None = None
        if output_dir is not None:
            predictions_dir = output_dir
            predictions_dir.mkdir(parents=True, exist_ok=True)
            for idx, pred in enumerate(predictions):
                save_volume_prediction(
                    output_dir=predictions_dir,
                    volume_name=f"vol_{idx:04d}",
                    hard_pred=pred,
                    soft_pred=pred.astype(np.float32),
                )

        # Step 3: MetricsReloaded evaluation
        fold_result = self._run_evaluation(predictions, labels)

        return EvaluationResult(
            model_name=model_name,
            dataset_name=dataset_name,
            subset_name=subset_name,
            fold_result=fold_result,
            predictions_dir=predictions_dir,
            uncertainty_maps_dir=None,
        )

    def log_results_to_mlflow(
        self,
        results: dict[str, dict[str, EvaluationResult]],
        *,
        model_name: str,
        model_tags: dict[str, str] | None = None,
    ) -> str | None:
        """Log all results as one MLflow run in minivess_evaluation experiment.

        Parameters
        ----------
        results:
            ``{dataset_name: {subset_name: EvaluationResult}}``.
        model_name:
            Model identifier for the run name.
        model_tags:
            Optional tags (e.g. ``model_type``, ``ensemble_strategy``).

        Returns
        -------
        MLflow run_id if logging succeeded, else ``None``.
        """
        try:
            mlflow.set_experiment(self.eval_config.mlflow_evaluation_experiment)
        except Exception:
            logger.warning(
                "Could not set MLflow experiment '%s'; skipping logging",
                self.eval_config.mlflow_evaluation_experiment,
                exc_info=True,
            )
            return None

        tags: dict[str, str] = {"model_name": model_name}
        if model_tags:
            tags.update(model_tags)

        with mlflow.start_run(run_name=f"eval_{model_name}") as run:
            mlflow.set_tags(tags)

            # Log metrics for each dataset/subset
            flat_metrics: dict[str, float] = {}
            for ds_name, subset_results in results.items():
                for subset_name, eval_result in subset_results.items():
                    prefix = f"eval_{ds_name}_{subset_name}_"
                    for metric_name, ci in eval_result.fold_result.aggregated.items():
                        flat_metrics[f"{prefix}{metric_name}"] = ci.point_estimate
                        flat_metrics[f"{prefix}{metric_name}_ci_lower"] = ci.lower
                        flat_metrics[f"{prefix}{metric_name}_ci_upper"] = ci.upper

                    # Compute and log compound metric
                    masd_ci = eval_result.fold_result.aggregated.get("measured_masd")
                    cldice_ci = eval_result.fold_result.aggregated.get("centreline_dsc")
                    if masd_ci is not None and cldice_ci is not None:
                        compound = compute_compound_masd_cldice(
                            masd=masd_ci.point_estimate,
                            cldice=cldice_ci.point_estimate,
                        )
                        flat_metrics[f"{prefix}compound_masd_cldice"] = compound

            # Filter out NaN values (MLflow rejects them)
            safe_metrics = {
                k: v for k, v in flat_metrics.items() if not math.isnan(v)
            }
            if safe_metrics:
                mlflow.log_metrics(safe_metrics)

            run_id: str = run.info.run_id

        logger.info(
            "Logged evaluation results to MLflow run %s (%d metrics)",
            run_id,
            len(safe_metrics),
        )
        return run_id

    def generate_summary_markdown(
        self,
        results: dict[str, dict[str, EvaluationResult]],
        *,
        model_name: str,
    ) -> str:
        """Generate a markdown summary table of evaluation results.

        Parameters
        ----------
        results:
            ``{dataset_name: {subset_name: EvaluationResult}}``.
        model_name:
            Model identifier for the table header.

        Returns
        -------
        Markdown-formatted string.
        """
        lines: list[str] = [
            f"## Evaluation Summary: {model_name}",
            "",
        ]

        # Collect all metric names
        all_metric_names: set[str] = set()
        for ds_results in results.values():
            for eval_result in ds_results.values():
                all_metric_names.update(eval_result.fold_result.aggregated.keys())
        sorted_metrics = sorted(all_metric_names)

        if not sorted_metrics:
            lines.append("*No metrics to display.*")
            return "\n".join(lines)

        # Build table header
        header = "| Dataset | Subset | " + " | ".join(sorted_metrics) + " |"
        separator = "| --- | --- | " + " | ".join(["---"] * len(sorted_metrics)) + " |"
        lines.extend([header, separator])

        # Build rows
        for ds_name in sorted(results.keys()):
            for subset_name in sorted(results[ds_name].keys()):
                eval_result = results[ds_name][subset_name]
                cells: list[str] = [ds_name, subset_name]
                for metric_name in sorted_metrics:
                    ci = eval_result.fold_result.aggregated.get(metric_name)
                    if ci is None or math.isnan(ci.point_estimate):
                        cells.append("N/A")
                    else:
                        cells.append(
                            f"{ci.point_estimate:.4f} "
                            f"[{ci.lower:.4f}, {ci.upper:.4f}]"
                        )
                lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers (public for mocking in tests)
    # ------------------------------------------------------------------

    def _run_evaluation(
        self,
        predictions: list[np.ndarray],
        labels: list[np.ndarray],
    ) -> FoldResult:
        """Run MetricsReloaded evaluation with bootstrap CIs.

        This is the expensive call that unit tests mock out.
        Delegates to :class:`EvaluationRunner.evaluate_fold`.

        Parameters
        ----------
        predictions:
            List of binary prediction arrays ``(D, H, W)``.
        labels:
            List of binary ground truth arrays ``(D, H, W)``.

        Returns
        -------
        :class:`FoldResult` with per-volume metrics and aggregated CIs.
        """
        return self._metrics_runner.evaluate_fold(
            predictions,
            labels,
            confidence_level=self.eval_config.confidence_level,
            n_resamples=self.eval_config.bootstrap_n_resamples,
        )
