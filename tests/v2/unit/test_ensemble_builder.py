"""Tests for ensemble builder from MLflow training runs.

Phase 5 of the evaluation plan: Ensemble Builder (#91).
Tests EnsembleMember, EnsembleSpec, and EnsembleBuilder with all 4 strategies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import fields
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path

from minivess.config.evaluation_config import (
    EnsembleStrategyName,
    EvaluationConfig,
)
from minivess.ensemble.builder import (
    EnsembleBuilder,
    EnsembleMember,
    EnsembleSpec,
    expand_runs_to_per_fold,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# The 6 tracked metrics from dynunet_losses.yaml checkpoint config
TRACKED_METRICS: list[str] = [
    "val_loss",
    "val_dice",
    "val_f1_foreground",
    "val_cldice",
    "val_masd",
    "val_compound_masd_cldice",
]

LOSS_TYPES: list[str] = ["dice_ce", "cbdice", "dice_ce_cldice", "cbdice_cldice"]

NUM_FOLDS = 3


class _MockNet(nn.Module):
    """Minimal network that produces 2-class output for testing."""

    def __init__(self, fg_prob: float = 0.8) -> None:
        super().__init__()
        self._fg_prob = fg_prob
        self._dummy = nn.Parameter(torch.tensor(fg_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, d, h, w = x.shape
        fg = torch.full((b, 1, d, h, w), self._fg_prob)
        bg = torch.full((b, 1, d, h, w), 1.0 - self._fg_prob)
        return torch.cat([bg, fg], dim=1)


def _create_checkpoint_new_format(
    path: Path,
    *,
    fg_prob: float = 0.8,
) -> Path:
    """Save a checkpoint in the new format with model_state_dict key."""
    net = _MockNet(fg_prob)
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "checkpoint_metadata": {
                "epoch": 50,
                "metric_name": "val_compound_masd_cldice",
            },
        },
        path,
    )
    return path


def _create_checkpoint_legacy_format(
    path: Path,
    *,
    fg_prob: float = 0.8,
) -> Path:
    """Save a checkpoint in legacy format (raw state_dict)."""
    net = _MockNet(fg_prob)
    torch.save(net.state_dict(), path)
    return path


def _create_model_config(tmp_path: Path) -> Path:
    """Save a model config JSON for _SimpleNet / _build_net_from_config."""
    config_path = tmp_path / "model_config.json"
    config_path.write_text(
        json.dumps(
            {
                "family": "test",
                "name": "test-model",
                "in_channels": 1,
                "out_channels": 2,
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _make_run_infos(
    tmp_path: Path,
    *,
    losses: list[str] | None = None,
    num_folds: int = NUM_FOLDS,
    metrics_per_fold: list[str] | None = None,
) -> list[dict]:
    """Create fake run info dicts with real checkpoint files.

    Each run has checkpoints for all specified metrics.

    Parameters
    ----------
    tmp_path:
        Temp directory for writing checkpoint files.
    losses:
        Loss types to create runs for.
    num_folds:
        Number of folds per loss.
    metrics_per_fold:
        Metric names for which to create best-metric checkpoints.

    Returns
    -------
    List of run info dicts.
    """
    if losses is None:
        losses = LOSS_TYPES
    if metrics_per_fold is None:
        metrics_per_fold = TRACKED_METRICS

    runs: list[dict] = []
    for loss_type in losses:
        for fold_id in range(num_folds):
            artifact_dir = tmp_path / f"run_{loss_type}_fold{fold_id}"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            # Create a checkpoint for each tracked metric
            for metric_name in metrics_per_fold:
                safe_name = metric_name.replace("/", "_")
                ckpt_path = artifact_dir / f"best_{safe_name}.pth"
                _create_checkpoint_new_format(
                    ckpt_path,
                    fg_prob=0.5 + 0.01 * fold_id,
                )

            run_info = {
                "run_id": f"run_{loss_type}_fold{fold_id}",
                "loss_type": loss_type,
                "fold_id": fold_id,
                "artifact_dir": str(artifact_dir),
                "metrics": {m: 0.7 + 0.01 * fold_id for m in metrics_per_fold},
            }
            runs.append(run_info)

    return runs


@pytest.fixture()
def model_config_dict() -> dict:
    """Model architecture config dict for _build_net_from_config."""
    return {
        "family": "test",
        "name": "test-model",
        "in_channels": 1,
        "out_channels": 2,
    }


@pytest.fixture()
def eval_config() -> EvaluationConfig:
    """Default evaluation config with all 4 strategies."""
    return EvaluationConfig()


@pytest.fixture()
def eval_config_single_strategy() -> EvaluationConfig:
    """Evaluation config with only per_loss_single_best strategy."""
    return EvaluationConfig(
        ensemble_strategies=[EnsembleStrategyName.PER_LOSS_SINGLE_BEST],
    )


@pytest.fixture()
def run_infos(tmp_path: Path) -> list[dict]:
    """Standard set of run infos: 4 losses x 3 folds x 6 metrics."""
    return _make_run_infos(tmp_path)


# ---------------------------------------------------------------------------
# Tests: EnsembleMember dataclass
# ---------------------------------------------------------------------------


class TestEnsembleMember:
    """EnsembleMember holds metadata and loaded model for one checkpoint."""

    def test_ensemble_member_fields(self) -> None:
        """EnsembleMember should have all required fields."""
        field_names = {f.name for f in fields(EnsembleMember)}
        expected = {
            "checkpoint_path",
            "run_id",
            "loss_type",
            "fold_id",
            "metric_name",
            "net",
        }
        assert expected == field_names

    def test_ensemble_member_construction(self, tmp_path: Path) -> None:
        """EnsembleMember should be constructable with all fields."""
        ckpt = tmp_path / "test.pth"
        ckpt.touch()
        net = _MockNet()
        member = EnsembleMember(
            checkpoint_path=ckpt,
            run_id="run_abc",
            loss_type="dice_ce",
            fold_id=0,
            metric_name="val_dice",
            net=net,
        )
        assert member.run_id == "run_abc"
        assert member.loss_type == "dice_ce"
        assert member.fold_id == 0
        assert member.metric_name == "val_dice"
        assert member.net is net
        assert member.checkpoint_path == ckpt


# ---------------------------------------------------------------------------
# Tests: EnsembleSpec dataclass
# ---------------------------------------------------------------------------


class TestEnsembleSpec:
    """EnsembleSpec holds a built ensemble's specification and members."""

    def test_ensemble_spec_fields(self) -> None:
        """EnsembleSpec should have name, strategy, members, description."""
        field_names = {f.name for f in fields(EnsembleSpec)}
        expected = {"name", "strategy", "members", "description"}
        assert expected == field_names

    def test_ensemble_spec_construction(self) -> None:
        """EnsembleSpec should be constructable with all fields."""
        spec = EnsembleSpec(
            name="per_loss_dice_ce",
            strategy=EnsembleStrategyName.PER_LOSS_SINGLE_BEST,
            members=[],
            description="3-fold ensemble for dice_ce using primary metric",
        )
        assert spec.name == "per_loss_dice_ce"
        assert spec.strategy == EnsembleStrategyName.PER_LOSS_SINGLE_BEST
        assert spec.members == []
        assert "dice_ce" in spec.description


# ---------------------------------------------------------------------------
# Tests: EnsembleBuilder init
# ---------------------------------------------------------------------------


class TestEnsembleBuilderInit:
    """EnsembleBuilder should accept config and model config."""

    def test_builder_init_with_config(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
    ) -> None:
        """Builder should store eval_config and model_config."""
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        assert builder.eval_config is eval_config
        assert builder.model_config is model_config_dict

    def test_builder_init_with_tracking_uri(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
    ) -> None:
        """Builder should accept optional tracking_uri."""
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
            tracking_uri="mlruns",
        )
        assert builder.tracking_uri == "mlruns"


# ---------------------------------------------------------------------------
# Tests: per_loss_single_best strategy
# ---------------------------------------------------------------------------


class TestPerLossSingleBest:
    """per_loss_single_best: K folds per loss using primary_metric only."""

    def test_creates_one_ensemble_per_loss(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        run_infos: list[dict],
    ) -> None:
        """Should produce 4 ensembles (one per loss type)."""
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_per_loss_single_best(run_infos)
        assert len(result) == 4
        # Each key should match the expected naming pattern exactly
        for loss in LOSS_TYPES:
            expected_key = f"per_loss_single_best_{loss}"
            assert expected_key in result, (
                f"Expected key {expected_key!r} in {list(result)}"
            )

    def test_member_count_equals_num_folds(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        run_infos: list[dict],
    ) -> None:
        """Each ensemble should have 3 members (one per fold)."""
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_per_loss_single_best(run_infos)
        for name, spec in result.items():
            assert len(spec.members) == NUM_FOLDS, (
                f"Ensemble {name} has {len(spec.members)} members, expected {NUM_FOLDS}"
            )

    def test_members_use_primary_metric_checkpoint(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        run_infos: list[dict],
    ) -> None:
        """All members should use the primary_metric checkpoint."""
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_per_loss_single_best(run_infos)
        primary = eval_config.primary_metric
        for _name, spec in result.items():
            for member in spec.members:
                assert member.metric_name == primary

    def test_strategy_tag_is_correct(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        run_infos: list[dict],
    ) -> None:
        """Each spec should have PER_LOSS_SINGLE_BEST strategy."""
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_per_loss_single_best(run_infos)
        for spec in result.values():
            assert spec.strategy == EnsembleStrategyName.PER_LOSS_SINGLE_BEST


# ---------------------------------------------------------------------------
# Tests: all_loss_single_best strategy
# ---------------------------------------------------------------------------


class TestAllLossSingleBest:
    """all_loss_single_best: all folds across all losses, primary_metric."""

    def test_creates_one_ensemble(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        run_infos: list[dict],
    ) -> None:
        """Should produce exactly 1 ensemble."""
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_all_loss_single_best(run_infos)
        assert len(result) == 1

    def test_member_count_is_losses_times_folds(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        run_infos: list[dict],
    ) -> None:
        """Should have 12 members: 4 losses x 3 folds."""
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_all_loss_single_best(run_infos)
        spec = next(iter(result.values()))
        assert len(spec.members) == len(LOSS_TYPES) * NUM_FOLDS  # 12

    def test_strategy_tag(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        run_infos: list[dict],
    ) -> None:
        """The spec should have ALL_LOSS_SINGLE_BEST strategy."""
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_all_loss_single_best(run_infos)
        spec = next(iter(result.values()))
        assert spec.strategy == EnsembleStrategyName.ALL_LOSS_SINGLE_BEST


# ---------------------------------------------------------------------------
# Tests: per_loss_all_best strategy
# ---------------------------------------------------------------------------


class TestPerLossAllBest:
    """per_loss_all_best: K folds per loss, ALL 6 best-metric checkpoints."""

    def test_creates_one_ensemble_per_loss(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        run_infos: list[dict],
    ) -> None:
        """Should produce 4 ensembles (one per loss type)."""
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_per_loss_all_best(run_infos)
        assert len(result) == 4

    def test_member_count_is_folds_times_metrics(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        run_infos: list[dict],
    ) -> None:
        """Each ensemble: 3 folds x 6 metrics = 18 members."""
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_per_loss_all_best(run_infos)
        expected = NUM_FOLDS * len(TRACKED_METRICS)  # 18
        for name, spec in result.items():
            assert len(spec.members) == expected, (
                f"Ensemble {name}: {len(spec.members)} members, expected {expected}"
            )

    def test_strategy_tag(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        run_infos: list[dict],
    ) -> None:
        """Each spec should have PER_LOSS_ALL_BEST strategy."""
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_per_loss_all_best(run_infos)
        for spec in result.values():
            assert spec.strategy == EnsembleStrategyName.PER_LOSS_ALL_BEST


# ---------------------------------------------------------------------------
# Tests: all_loss_all_best strategy
# ---------------------------------------------------------------------------


class TestAllLossAllBest:
    """all_loss_all_best: Full deep ensemble — all folds x all losses x all metrics."""

    def test_creates_one_ensemble(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        run_infos: list[dict],
    ) -> None:
        """Should produce exactly 1 ensemble."""
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_all_loss_all_best(run_infos)
        assert len(result) == 1

    def test_member_count_is_full_deep_ensemble(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        run_infos: list[dict],
    ) -> None:
        """Should have 72 members: 4 losses x 3 folds x 6 metrics."""
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_all_loss_all_best(run_infos)
        spec = next(iter(result.values()))
        expected = len(LOSS_TYPES) * NUM_FOLDS * len(TRACKED_METRICS)  # 72
        assert len(spec.members) == expected

    def test_strategy_tag(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        run_infos: list[dict],
    ) -> None:
        """The spec should have ALL_LOSS_ALL_BEST strategy."""
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_all_loss_all_best(run_infos)
        spec = next(iter(result.values()))
        assert spec.strategy == EnsembleStrategyName.ALL_LOSS_ALL_BEST


# ---------------------------------------------------------------------------
# Tests: build_all orchestration
# ---------------------------------------------------------------------------


class TestBuildAll:
    """build_all() delegates to configured strategies and returns combined dict."""

    def test_build_all_returns_dict_of_specs(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        run_infos: list[dict],
    ) -> None:
        """build_all should return a dict mapping name -> EnsembleSpec."""
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_all(run_infos)
        assert isinstance(result, dict)
        for name, spec in result.items():
            assert isinstance(name, str)
            assert isinstance(spec, EnsembleSpec)

    def test_build_all_includes_all_strategies(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        run_infos: list[dict],
    ) -> None:
        """With all 4 strategies, build_all produces:
        4 per_loss_single_best + 1 all_loss_single_best +
        4 per_loss_all_best + 1 all_loss_all_best = 10 ensembles.
        """
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_all(run_infos)
        assert len(result) == 10

    def test_strategy_filtering_respects_config(
        self,
        eval_config_single_strategy: EvaluationConfig,
        model_config_dict: dict,
        run_infos: list[dict],
    ) -> None:
        """With only PER_LOSS_SINGLE_BEST, build_all should return 4 ensembles."""
        builder = EnsembleBuilder(
            eval_config=eval_config_single_strategy,
            model_config=model_config_dict,
        )
        result = builder.build_all(run_infos)
        assert len(result) == 4
        for spec in result.values():
            assert spec.strategy == EnsembleStrategyName.PER_LOSS_SINGLE_BEST


# ---------------------------------------------------------------------------
# Tests: checkpoint loading
# ---------------------------------------------------------------------------


class TestCheckpointLoading:
    """Checkpoint loading should handle new and legacy formats."""

    def test_load_checkpoint_new_format(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        tmp_path: Path,
    ) -> None:
        """New format: dict with 'model_state_dict' key."""
        ckpt_path = tmp_path / "best_val_dice.pth"
        _create_checkpoint_new_format(ckpt_path)

        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        net = builder.load_checkpoint(ckpt_path)
        assert isinstance(net, nn.Module)

    def test_load_checkpoint_legacy_format(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        tmp_path: Path,
    ) -> None:
        """Legacy format: raw state_dict (no 'model_state_dict' key)."""
        ckpt_path = tmp_path / "legacy.pth"
        _create_checkpoint_legacy_format(ckpt_path)

        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        net = builder.load_checkpoint(ckpt_path)
        assert isinstance(net, nn.Module)

    def test_loaded_net_produces_output(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        tmp_path: Path,
    ) -> None:
        """A loaded network should produce valid output shape."""
        ckpt_path = tmp_path / "test.pth"
        _create_checkpoint_new_format(ckpt_path)

        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        net = builder.load_checkpoint(ckpt_path)
        x = torch.randn(1, 1, 4, 4, 4)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1, 2, 4, 4, 4)


# ---------------------------------------------------------------------------
# Tests: missing checkpoint handling
# ---------------------------------------------------------------------------


class TestMissingCheckpoints:
    """Missing checkpoints should be skipped with a warning, not crash."""

    def test_missing_checkpoint_skipped_with_warning(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If a checkpoint file is missing, the member should be skipped."""
        # Create run info pointing to a non-existent checkpoint dir
        run_info = {
            "run_id": "run_missing",
            "loss_type": "dice_ce",
            "fold_id": 0,
            "artifact_dir": str(tmp_path / "nonexistent"),
            "metrics": {"val_dice": 0.8},
        }

        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        with caplog.at_level(logging.WARNING):
            result = builder.build_per_loss_single_best([run_info])

        # Should have produced an ensemble with 0 members (checkpoint missing)
        # and logged a warning
        spec = next(iter(result.values()))
        assert len(spec.members) == 0
        assert any(
            "missing" in record.message.lower() or "not found" in record.message.lower()
            for record in caplog.records
        )

    def test_partial_missing_keeps_valid_members(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        tmp_path: Path,
    ) -> None:
        """If some checkpoints exist and some don't, only valid ones are loaded."""
        # Create one valid run and one with missing checkpoint
        valid_dir = tmp_path / "valid_run"
        valid_dir.mkdir(parents=True)
        primary = eval_config.primary_metric
        _create_checkpoint_new_format(
            valid_dir / f"best_{primary}.pth",
        )

        runs = [
            {
                "run_id": "run_valid",
                "loss_type": "dice_ce",
                "fold_id": 0,
                "artifact_dir": str(valid_dir),
                "metrics": {primary: 0.8},
            },
            {
                "run_id": "run_missing",
                "loss_type": "dice_ce",
                "fold_id": 1,
                "artifact_dir": str(tmp_path / "missing_dir"),
                "metrics": {primary: 0.7},
            },
        ]

        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_per_loss_single_best(runs)
        spec = next(iter(result.values()))
        # Only the valid checkpoint should be loaded
        assert len(spec.members) == 1
        assert spec.members[0].run_id == "run_valid"


# ---------------------------------------------------------------------------
# Tests: member metadata correctness
# ---------------------------------------------------------------------------


class TestMemberMetadata:
    """Loaded members should carry correct provenance metadata."""

    def test_member_preserves_run_id(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        tmp_path: Path,
    ) -> None:
        """Each member's run_id should match the source run."""
        runs = _make_run_infos(tmp_path, losses=["dice_ce"], num_folds=1)
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_per_loss_single_best(runs)
        spec = next(iter(result.values()))
        assert spec.members[0].run_id == "run_dice_ce_fold0"

    def test_member_preserves_fold_id(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        tmp_path: Path,
    ) -> None:
        """Each member's fold_id should match the source run."""
        runs = _make_run_infos(tmp_path, losses=["dice_ce"], num_folds=2)
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_per_loss_single_best(runs)
        spec = next(iter(result.values()))
        fold_ids = sorted(m.fold_id for m in spec.members)
        assert fold_ids == [0, 1]


# ---------------------------------------------------------------------------
# Tests: expand_runs_to_per_fold
# ---------------------------------------------------------------------------


def _make_raw_per_loss_runs() -> list[dict[str, Any]]:
    """Create raw per-loss run infos (1 entry per loss, no fold_id)."""
    return [
        {
            "run_id": f"run_{loss}",
            "loss_type": loss,
            "fold_id": 0,
            "artifact_dir": f"/tmp/mlruns/{loss}",
            "metrics": {"val_dice": 0.8, "eval_fold2_dsc": 0.79},
            "num_folds": 3,
        }
        for loss in LOSS_TYPES
    ]


class TestExpandRunsToPerFold:
    """Tests for expand_runs_to_per_fold() which synthesizes per-fold entries."""

    def test_expand_creates_3_entries_per_loss(self) -> None:
        """Each per-loss run should produce 3 per-fold entries."""
        raw = _make_raw_per_loss_runs()
        expanded = expand_runs_to_per_fold(raw)
        assert len(expanded) == len(LOSS_TYPES) * NUM_FOLDS

    def test_expanded_runs_share_artifact_dir(self) -> None:
        """All fold entries for the same loss share the artifact directory."""
        raw = _make_raw_per_loss_runs()
        expanded = expand_runs_to_per_fold(raw)
        by_loss: dict[str, set[str]] = {}
        for entry in expanded:
            by_loss.setdefault(entry["loss_type"], set()).add(entry["artifact_dir"])
        for loss, dirs in by_loss.items():
            assert len(dirs) == 1, f"Expected 1 artifact_dir for {loss}, got {dirs}"

    def test_expanded_runs_have_distinct_fold_ids(self) -> None:
        """Each loss should have fold_ids 0, 1, 2."""
        raw = _make_raw_per_loss_runs()
        expanded = expand_runs_to_per_fold(raw)
        by_loss: dict[str, list[int]] = {}
        for entry in expanded:
            by_loss.setdefault(entry["loss_type"], []).append(entry["fold_id"])
        for loss, folds in by_loss.items():
            assert sorted(folds) == [0, 1, 2], f"Bad fold_ids for {loss}: {folds}"

    def test_expanded_runs_share_run_id(self) -> None:
        """All fold entries for the same loss share the original run_id."""
        raw = _make_raw_per_loss_runs()
        expanded = expand_runs_to_per_fold(raw)
        by_loss: dict[str, set[str]] = {}
        for entry in expanded:
            by_loss.setdefault(entry["loss_type"], set()).add(entry["run_id"])
        for loss, run_ids in by_loss.items():
            assert len(run_ids) == 1, f"Expected 1 run_id for {loss}, got {run_ids}"

    def test_expanded_runs_preserve_metrics(self) -> None:
        """Expanded entries should carry the original metrics dict."""
        raw = _make_raw_per_loss_runs()
        expanded = expand_runs_to_per_fold(raw)
        for entry in expanded:
            assert "val_dice" in entry["metrics"]
            assert entry["metrics"]["val_dice"] == 0.8

    def test_empty_input_returns_empty(self) -> None:
        """No runs in, no runs out."""
        assert expand_runs_to_per_fold([]) == []

    def test_custom_num_folds(self) -> None:
        """Runs with num_folds=5 produce 5 entries each."""
        raw = [
            {
                "run_id": "run_test",
                "loss_type": "dice_ce",
                "fold_id": 0,
                "artifact_dir": "/tmp/test",
                "metrics": {},
                "num_folds": 5,
            }
        ]
        expanded = expand_runs_to_per_fold(raw)
        assert len(expanded) == 5
        assert sorted(e["fold_id"] for e in expanded) == [0, 1, 2, 3, 4]

    def test_ensemble_builder_works_with_expanded_runs(
        self,
        eval_config: EvaluationConfig,
        model_config_dict: dict,
        tmp_path: Path,
    ) -> None:
        """EnsembleBuilder strategies work correctly with expanded runs."""
        primary = eval_config.primary_metric
        # Create one artifact dir per loss with checkpoints
        raw_runs: list[dict[str, Any]] = []
        for loss in LOSS_TYPES:
            loss_dir = tmp_path / loss
            loss_dir.mkdir(parents=True)
            _create_checkpoint_new_format(loss_dir / f"best_{primary}.pth")
            raw_runs.append(
                {
                    "run_id": f"run_{loss}",
                    "loss_type": loss,
                    "fold_id": 0,
                    "artifact_dir": str(loss_dir),
                    "metrics": {"val_dice": 0.8},
                    "num_folds": 3,
                }
            )

        expanded = expand_runs_to_per_fold(raw_runs)
        builder = EnsembleBuilder(
            eval_config=eval_config,
            model_config=model_config_dict,
        )
        result = builder.build_per_loss_single_best(expanded)
        assert len(result) == len(LOSS_TYPES)
        for spec in result.values():
            # Each loss has 3 fold entries all pointing to same dir
            # so 3 members loaded from the same checkpoint
            assert len(spec.members) == 3
