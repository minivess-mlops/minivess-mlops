"""Ensemble builder from MLflow training runs.

Constructs ensembles from completed training runs using 4 strategies:

1. **per_loss_single_best** -- For each loss, ensemble its K folds using
   only the primary_metric checkpoint.
2. **all_loss_single_best** -- Ensemble ALL folds across ALL losses using
   the primary_metric checkpoint.
3. **per_loss_all_best** -- For each loss, ensemble its K folds using ALL
   tracked best-metric checkpoints.
4. **all_loss_all_best** -- Full Deep Ensemble: all folds x all losses x
   all metric checkpoints.

The builder queries MLflow for completed training runs via
:meth:`discover_training_runs`, but strategy methods accept pre-fetched run
info dicts to keep the MLflow dependency testable (mock-friendly).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from minivess.config.evaluation_config import EnsembleStrategyName
from minivess.serving.mlflow_wrapper import _build_net_from_config

if TYPE_CHECKING:
    from minivess.config.evaluation_config import EvaluationConfig

logger = logging.getLogger(__name__)

# The 6 tracked metrics from dynunet_losses.yaml checkpoint config.
# Each generates a ``best_{name}.pth`` checkpoint file.
_DEFAULT_TRACKED_METRICS: list[str] = [
    "val_loss",
    "val_dice",
    "val_f1_foreground",
    "val_cldice",
    "val_masd",
    "val_compound_masd_cldice",
]


@dataclass
class EnsembleMember:
    """Metadata and loaded model for one ensemble member.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.pth`` checkpoint file that was loaded.
    run_id:
        MLflow run ID that produced this checkpoint.
    loss_type:
        Loss function name used during training.
    fold_id:
        Cross-validation fold index (0-based).
    metric_name:
        Metric whose best value triggered this checkpoint save.
    net:
        Loaded and eval-mode network.
    """

    checkpoint_path: Path
    run_id: str
    loss_type: str
    fold_id: int
    metric_name: str
    net: nn.Module


@dataclass
class EnsembleSpec:
    """Specification for a built ensemble.

    Parameters
    ----------
    name:
        Human-readable identifier for this ensemble.
    strategy:
        Which :class:`EnsembleStrategyName` was used to build it.
    members:
        List of loaded ensemble members with provenance metadata.
    description:
        Human-readable description of what this ensemble contains.
    """

    name: str
    strategy: EnsembleStrategyName
    members: list[EnsembleMember]
    description: str


class EnsembleBuilder:
    """Build ensembles from MLflow training runs.

    The builder is designed for testability: strategy methods accept
    pre-fetched run info dicts so unit tests can supply mock data
    without needing an MLflow server.

    Parameters
    ----------
    eval_config:
        Evaluation configuration (primary metric, strategies, etc.).
    model_config:
        Model architecture config dict for :func:`_build_net_from_config`.
    tracking_uri:
        Optional MLflow tracking URI. If ``None``, uses the default
        resolution logic from :mod:`minivess.observability.tracking`.
    """

    def __init__(
        self,
        eval_config: EvaluationConfig,
        model_config: dict[str, Any],
        *,
        tracking_uri: str | None = None,
    ) -> None:
        self.eval_config = eval_config
        self.model_config = model_config
        self.tracking_uri = tracking_uri

    # ------------------------------------------------------------------
    # MLflow discovery (integration; not tested in unit tests)
    # ------------------------------------------------------------------

    def discover_training_runs(self) -> list[dict[str, Any]]:
        """Query MLflow for completed training runs with tags.

        Returns a list of run info dicts with the shape::

            {
                "run_id": str,
                "loss_type": str,
                "fold_id": int,
                "artifact_dir": str,
                "metrics": {"val_dice": 0.81, ...},
            }

        Requires a running MLflow tracking server.
        """
        import mlflow
        from mlflow.tracking import MlflowClient

        from minivess.observability.tracking import resolve_tracking_uri

        uri = resolve_tracking_uri(tracking_uri=self.tracking_uri)
        mlflow.set_tracking_uri(uri)
        client = MlflowClient(tracking_uri=uri)

        experiment = client.get_experiment_by_name(
            self.eval_config.mlflow_training_experiment,
        )
        if experiment is None:
            logger.warning(
                "MLflow experiment '%s' not found",
                self.eval_config.mlflow_training_experiment,
            )
            return []

        runs_data = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["start_time DESC"],
        )

        run_infos: list[dict[str, Any]] = []
        for run in runs_data:
            tags = run.data.tags
            loss_type = tags.get("loss_type")
            fold_id_str = tags.get("fold_id")
            if loss_type is None or fold_id_str is None:
                logger.debug(
                    "Skipping run %s: missing loss_type or fold_id tags",
                    run.info.run_id,
                )
                continue

            artifact_uri = run.info.artifact_uri
            run_infos.append(
                {
                    "run_id": run.info.run_id,
                    "loss_type": loss_type,
                    "fold_id": int(fold_id_str),
                    "artifact_dir": artifact_uri,
                    "metrics": dict(run.data.metrics),
                }
            )

        logger.info("Discovered %d completed training runs", len(run_infos))
        return run_infos

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def build_all(
        self,
        runs: list[dict[str, Any]],
    ) -> dict[str, EnsembleSpec]:
        """Build all configured ensemble strategies.

        Parameters
        ----------
        runs:
            Pre-fetched run info dicts (from :meth:`discover_training_runs`
            or mocked for testing).

        Returns
        -------
        Mapping from ensemble name to :class:`EnsembleSpec`.
        """
        strategies = self.eval_config.ensemble_strategies
        result: dict[str, EnsembleSpec] = {}

        if EnsembleStrategyName.PER_LOSS_SINGLE_BEST in strategies:
            result.update(self.build_per_loss_single_best(runs))

        if EnsembleStrategyName.ALL_LOSS_SINGLE_BEST in strategies:
            result.update(self.build_all_loss_single_best(runs))

        if EnsembleStrategyName.PER_LOSS_ALL_BEST in strategies:
            result.update(self.build_per_loss_all_best(runs))

        if EnsembleStrategyName.ALL_LOSS_ALL_BEST in strategies:
            result.update(self.build_all_loss_all_best(runs))

        logger.info(
            "Built %d ensembles across %d strategies",
            len(result),
            len(strategies),
        )
        return result

    # ------------------------------------------------------------------
    # Strategy 1: per_loss_single_best
    # ------------------------------------------------------------------

    def build_per_loss_single_best(
        self,
        runs: list[dict[str, Any]],
    ) -> dict[str, EnsembleSpec]:
        """For each loss, ensemble its K folds using only primary_metric.

        Produces one ensemble per unique loss type found in *runs*.

        Parameters
        ----------
        runs:
            Pre-fetched run info dicts.

        Returns
        -------
        Mapping from ensemble name to :class:`EnsembleSpec`.
        """
        primary = self.eval_config.primary_metric
        by_loss = _group_runs_by_loss(runs)
        result: dict[str, EnsembleSpec] = {}

        for loss_type, loss_runs in sorted(by_loss.items()):
            members = self._load_members_for_metric(loss_runs, primary)
            name = f"per_loss_single_best_{loss_type}"
            result[name] = EnsembleSpec(
                name=name,
                strategy=EnsembleStrategyName.PER_LOSS_SINGLE_BEST,
                members=members,
                description=(
                    f"{len(members)}-fold ensemble for {loss_type} "
                    f"using {primary} checkpoint"
                ),
            )

        return result

    # ------------------------------------------------------------------
    # Strategy 2: all_loss_single_best
    # ------------------------------------------------------------------

    def build_all_loss_single_best(
        self,
        runs: list[dict[str, Any]],
    ) -> dict[str, EnsembleSpec]:
        """Ensemble ALL folds across ALL losses using primary_metric.

        Produces one ensemble containing all runs.

        Parameters
        ----------
        runs:
            Pre-fetched run info dicts.

        Returns
        -------
        Single-entry mapping from ensemble name to :class:`EnsembleSpec`.
        """
        primary = self.eval_config.primary_metric
        members = self._load_members_for_metric(runs, primary)
        name = "all_loss_single_best"

        return {
            name: EnsembleSpec(
                name=name,
                strategy=EnsembleStrategyName.ALL_LOSS_SINGLE_BEST,
                members=members,
                description=(
                    f"{len(members)}-member ensemble across all losses "
                    f"using {primary} checkpoint"
                ),
            )
        }

    # ------------------------------------------------------------------
    # Strategy 3: per_loss_all_best
    # ------------------------------------------------------------------

    def build_per_loss_all_best(
        self,
        runs: list[dict[str, Any]],
    ) -> dict[str, EnsembleSpec]:
        """For each loss, ensemble its K folds using ALL tracked metrics.

        Each fold contributes one member per tracked metric checkpoint.

        Parameters
        ----------
        runs:
            Pre-fetched run info dicts.

        Returns
        -------
        Mapping from ensemble name to :class:`EnsembleSpec`.
        """
        by_loss = _group_runs_by_loss(runs)
        result: dict[str, EnsembleSpec] = {}

        for loss_type, loss_runs in sorted(by_loss.items()):
            members = self._load_members_all_metrics(loss_runs)
            name = f"per_loss_all_best_{loss_type}"
            result[name] = EnsembleSpec(
                name=name,
                strategy=EnsembleStrategyName.PER_LOSS_ALL_BEST,
                members=members,
                description=(
                    f"{len(members)}-member ensemble for {loss_type} "
                    f"using all metric checkpoints"
                ),
            )

        return result

    # ------------------------------------------------------------------
    # Strategy 4: all_loss_all_best
    # ------------------------------------------------------------------

    def build_all_loss_all_best(
        self,
        runs: list[dict[str, Any]],
    ) -> dict[str, EnsembleSpec]:
        """Full Deep Ensemble: all folds x all losses x all metrics.

        Parameters
        ----------
        runs:
            Pre-fetched run info dicts.

        Returns
        -------
        Single-entry mapping from ensemble name to :class:`EnsembleSpec`.
        """
        members = self._load_members_all_metrics(runs)
        name = "all_loss_all_best"

        return {
            name: EnsembleSpec(
                name=name,
                strategy=EnsembleStrategyName.ALL_LOSS_ALL_BEST,
                members=members,
                description=(
                    f"{len(members)}-member full deep ensemble "
                    f"across all losses and metrics"
                ),
            )
        }

    # ------------------------------------------------------------------
    # Checkpoint loading (public for testing)
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: Path) -> nn.Module:
        """Load a single checkpoint into a network.

        Handles both new format (dict with ``model_state_dict`` key) and
        legacy format (raw state_dict).

        Parameters
        ----------
        path:
            Path to the ``.pth`` checkpoint file.

        Returns
        -------
        Loaded network in eval mode.
        """
        net = _build_net_from_config(self.model_config)
        payload = torch.load(path, map_location="cpu", weights_only=True)

        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        else:
            state_dict = payload

        try:
            net.load_state_dict(state_dict)
        except RuntimeError:
            inner_net = getattr(net, "net", None)
            if inner_net is not None:
                inner_net.load_state_dict(state_dict)
            else:
                logger.warning(
                    "State dict keys do not match for %s; using initialized weights",
                    path,
                )

        net.eval()
        return net

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_members_for_metric(
        self,
        runs: list[dict[str, Any]],
        metric_name: str,
    ) -> list[EnsembleMember]:
        """Load one checkpoint per run for a specific metric.

        Parameters
        ----------
        runs:
            Run info dicts to process.
        metric_name:
            Metric whose ``best_{metric_name}.pth`` checkpoint to load.

        Returns
        -------
        List of successfully loaded members.
        """
        members: list[EnsembleMember] = []
        for run in runs:
            artifact_dir = Path(run["artifact_dir"])
            safe_name = metric_name.replace("/", "_")
            ckpt_path = artifact_dir / f"best_{safe_name}.pth"

            if not ckpt_path.exists():
                logger.warning(
                    "Checkpoint not found: %s (run %s, loss=%s, fold=%d)",
                    ckpt_path,
                    run["run_id"],
                    run["loss_type"],
                    run["fold_id"],
                )
                continue

            try:
                net = self.load_checkpoint(ckpt_path)
            except Exception:
                logger.warning(
                    "Failed to load checkpoint %s",
                    ckpt_path,
                    exc_info=True,
                )
                continue

            members.append(
                EnsembleMember(
                    checkpoint_path=ckpt_path,
                    run_id=run["run_id"],
                    loss_type=run["loss_type"],
                    fold_id=run["fold_id"],
                    metric_name=metric_name,
                    net=net,
                )
            )

        return members

    def _load_members_all_metrics(
        self,
        runs: list[dict[str, Any]],
    ) -> list[EnsembleMember]:
        """Load ALL tracked metric checkpoints for each run.

        For each run, loads ``best_{metric}.pth`` for every metric in
        :data:`_DEFAULT_TRACKED_METRICS` that exists on disk.

        Parameters
        ----------
        runs:
            Run info dicts to process.

        Returns
        -------
        List of successfully loaded members.
        """
        all_members: list[EnsembleMember] = []
        for metric_name in _DEFAULT_TRACKED_METRICS:
            members = self._load_members_for_metric(runs, metric_name)
            all_members.extend(members)
        return all_members


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _group_runs_by_loss(
    runs: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group run info dicts by their ``loss_type`` field.

    Parameters
    ----------
    runs:
        Flat list of run info dicts.

    Returns
    -------
    Mapping from loss_type to list of run info dicts.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        loss = run["loss_type"]
        grouped.setdefault(loss, []).append(run)
    return grouped
