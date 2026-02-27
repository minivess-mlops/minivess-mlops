"""Integration tests: MLflow artifact integrity for v2 experiment.

Verifies that the ``dynunet_loss_variation_v2`` MLflow experiment (ID
843896622863223169) contains exactly 4 production runs with complete
artifacts: metrics, checkpoints, evaluation fold results, bootstrap CIs,
and metric history JSON.

All tests read directly from the real ``mlruns/`` filesystem layout using
the ``mlruns_inspector`` helper module — no live MLflow tracking server or
client is required.

Run with::

    uv run pytest tests/v2/integration/test_mlruns_integrity.py -m integration -v

Skip automatically in CI when ``mlruns/`` is absent (e.g. no training has
been executed on the CI runner).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from minivess.pipeline.mlruns_inspector import (
    get_production_runs,
    get_run_metrics_list,
    get_run_params,
    get_run_tags,
    read_metric_last_value,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MLRUNS_DIR = Path(__file__).resolve().parents[3] / "mlruns"
V2_EXPERIMENT_ID = "843896622863223169"

EXPECTED_LOSSES: set[str] = {
    "dice_ce",
    "cbdice",
    "dice_ce_cldice",
    "cbdice_cldice",
}

EXPECTED_CHECKPOINTS: set[str] = {
    "best_val_loss.pth",
    "best_val_dice.pth",
    "best_val_f1_foreground.pth",
    "best_val_cldice.pth",
    "best_val_masd.pth",
    "best_val_compound_masd_cldice.pth",
    "last.pth",
}

# Required hyperparameter keys logged to params/
REQUIRED_PARAMS: set[str] = {
    "batch_size",
    "max_epochs",
    "model_family",
    "seed",
}

# Eval fold metrics (one per fold 0-2)
EVAL_FOLD_BASE_METRICS: list[str] = [
    "dsc",
    "centreline_dsc",
    "measured_masd",
]

# Approximate checkpoint size reference (bytes) — best_val_* checkpoints.
# These are DynUNet model weights; all best_* files should be within 5% of
# this reference.  last.pth is excluded because it may differ slightly.
_CHECKPOINT_REFERENCE_BYTES: int = 67_740_000  # ~67.7 MB
_CHECKPOINT_TOLERANCE: float = 0.05  # 5 %

# ---------------------------------------------------------------------------
# Module-level skip guard
# ---------------------------------------------------------------------------

_mlruns_missing = not (MLRUNS_DIR / V2_EXPERIMENT_ID).is_dir()
_skip_reason = (
    f"mlruns experiment directory not found: {MLRUNS_DIR / V2_EXPERIMENT_ID}. "
    "Run training first or check MLRUNS_DIR path."
)


# ---------------------------------------------------------------------------
# Helper: production run IDs (cached at module level to avoid repeated I/O)
# ---------------------------------------------------------------------------


def _get_cached_production_runs() -> list[str]:
    """Return production run IDs; empty list when mlruns absent."""
    if _mlruns_missing:
        return []
    return get_production_runs(MLRUNS_DIR, V2_EXPERIMENT_ID)


# Evaluated once at collection time; used in parametrize where possible.
_PRODUCTION_RUNS: list[str] = _get_cached_production_runs()


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(_mlruns_missing, reason=_skip_reason)
class TestMlrunsIntegrity:
    """Verify structural integrity of v2 MLflow experiment artifacts."""

    # ------------------------------------------------------------------
    # 1. Experiment directory exists
    # ------------------------------------------------------------------

    def test_v2_experiment_exists(self) -> None:
        """V2 experiment directory must exist under mlruns/."""
        exp_dir = MLRUNS_DIR / V2_EXPERIMENT_ID
        assert exp_dir.is_dir(), (
            f"Experiment directory not found: {exp_dir}. "
            "Expected dynunet_loss_variation_v2 experiment."
        )

    # ------------------------------------------------------------------
    # 2. Exactly 4 production runs identified
    # ------------------------------------------------------------------

    def test_production_runs_identified(self) -> None:
        """Exactly 4 production runs must be identifiable via eval_fold metrics."""
        runs = get_production_runs(MLRUNS_DIR, V2_EXPERIMENT_ID)
        assert len(runs) == 4, (
            f"Expected 4 production runs, found {len(runs)}: {runs}. "
            "Production runs are identified by having eval_fold* metrics."
        )

    # ------------------------------------------------------------------
    # 3. All production runs have loss_function tag
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_each_production_run_has_loss_function_tag(self, run_id: str) -> None:
        """Every production run must have a ``loss_function`` tag."""
        tags = get_run_tags(MLRUNS_DIR, V2_EXPERIMENT_ID, run_id)
        assert "loss_function" in tags, (
            f"Run {run_id} missing 'loss_function' tag. "
            f"Available tags: {sorted(tags.keys())}"
        )
        assert tags["loss_function"], f"Run {run_id} has empty 'loss_function' tag."

    # ------------------------------------------------------------------
    # 4. Loss function tags match expected set
    # ------------------------------------------------------------------

    def test_production_losses_cover_expected_set(self) -> None:
        """Collected loss_function tags must exactly equal EXPECTED_LOSSES."""
        runs = get_production_runs(MLRUNS_DIR, V2_EXPERIMENT_ID)
        found_losses: set[str] = set()
        for run_id in runs:
            tags = get_run_tags(MLRUNS_DIR, V2_EXPERIMENT_ID, run_id)
            loss = tags.get("loss_function", "")
            if loss:
                found_losses.add(loss)

        assert found_losses == EXPECTED_LOSSES, (
            f"Loss set mismatch.\n"
            f"  Expected: {sorted(EXPECTED_LOSSES)}\n"
            f"  Found:    {sorted(found_losses)}\n"
            f"  Missing:  {sorted(EXPECTED_LOSSES - found_losses)}\n"
            f"  Extra:    {sorted(found_losses - EXPECTED_LOSSES)}"
        )

    # ------------------------------------------------------------------
    # 5. Each production run has at least 45 metric files
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_each_production_run_has_45_plus_metrics(self, run_id: str) -> None:
        """Each production run must have >= 45 logged metric files."""
        metrics = get_run_metrics_list(MLRUNS_DIR, V2_EXPERIMENT_ID, run_id)
        tags = get_run_tags(MLRUNS_DIR, V2_EXPERIMENT_ID, run_id)
        loss = tags.get("loss_function", run_id)
        assert len(metrics) >= 45, (
            f"Run {run_id} ({loss}) has only {len(metrics)} metric files "
            f"(expected >= 45). Metrics found: {metrics}"
        )

    # ------------------------------------------------------------------
    # 6. Each production run has all 7 checkpoint files
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_each_production_run_has_7_checkpoints(self, run_id: str) -> None:
        """All 7 expected checkpoint files must exist in artifacts/checkpoints/."""
        checkpoints_dir = (
            MLRUNS_DIR / V2_EXPERIMENT_ID / run_id / "artifacts" / "checkpoints"
        )
        assert checkpoints_dir.is_dir(), (
            f"Checkpoints directory missing for run {run_id}: {checkpoints_dir}"
        )

        found: set[str] = {f.name for f in checkpoints_dir.iterdir() if f.is_file()}
        missing = EXPECTED_CHECKPOINTS - found
        assert not missing, (
            f"Run {run_id} is missing checkpoints: {sorted(missing)}. "
            f"Found: {sorted(found)}"
        )

    # ------------------------------------------------------------------
    # 7. Each production run has required hyperparameter params
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_each_production_run_has_required_params(self, run_id: str) -> None:
        """Required hyperparameter params must be logged for every production run."""
        params = get_run_params(MLRUNS_DIR, V2_EXPERIMENT_ID, run_id)
        missing = REQUIRED_PARAMS - set(params.keys())
        assert not missing, (
            f"Run {run_id} is missing required params: {sorted(missing)}. "
            f"Found params: {sorted(params.keys())}"
        )
        # All required params must be non-empty
        for key in REQUIRED_PARAMS:
            assert params[key], (
                f"Run {run_id} has empty value for required param '{key}'."
            )

    # ------------------------------------------------------------------
    # 8. Each production run has artifacts/history/metric_history.json
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_each_production_run_has_metric_history_json(self, run_id: str) -> None:
        """artifacts/history/metric_history.json must exist for every production run."""
        history_file = (
            MLRUNS_DIR
            / V2_EXPERIMENT_ID
            / run_id
            / "artifacts"
            / "history"
            / "metric_history.json"
        )
        assert history_file.is_file(), (
            f"metric_history.json not found for run {run_id}: {history_file}"
        )
        assert history_file.stat().st_size > 0, (
            f"metric_history.json is empty for run {run_id}: {history_file}"
        )

    # ------------------------------------------------------------------
    # 9. Each production run has eval_fold{0,1,2}_{dsc,centreline_dsc,measured_masd}
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_each_production_run_has_eval_fold_metrics(self, run_id: str) -> None:
        """Eval fold point-estimate metrics must exist for folds 0, 1, 2."""
        metrics_set = set(get_run_metrics_list(MLRUNS_DIR, V2_EXPERIMENT_ID, run_id))
        expected_eval_metrics: list[str] = [
            f"eval_fold{fold}_{base}"
            for fold in range(3)
            for base in EVAL_FOLD_BASE_METRICS
        ]
        missing = [m for m in expected_eval_metrics if m not in metrics_set]
        assert not missing, f"Run {run_id} is missing eval fold metrics: {missing}"

    # ------------------------------------------------------------------
    # 10. Each production run has bootstrap CI metrics
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_each_production_run_has_bootstrap_cis(self, run_id: str) -> None:
        """Bootstrap confidence interval metrics must exist for each fold and metric."""
        metrics_set = set(get_run_metrics_list(MLRUNS_DIR, V2_EXPERIMENT_ID, run_id))
        expected_ci_metrics: list[str] = [
            f"eval_fold{fold}_{base}_ci_{suffix}"
            for fold in range(3)
            for base in EVAL_FOLD_BASE_METRICS
            for suffix in ("level", "lower", "upper")
        ]
        missing = [m for m in expected_ci_metrics if m not in metrics_set]
        assert not missing, f"Run {run_id} is missing bootstrap CI metrics: {missing}"

    # ------------------------------------------------------------------
    # 11. Checkpoint sizes are consistent (best_*.pth within 5 % of reference)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_checkpoint_sizes_consistent(self, run_id: str) -> None:
        """best_*.pth checkpoint files must be within 5 % of the reference size."""
        checkpoints_dir = (
            MLRUNS_DIR / V2_EXPERIMENT_ID / run_id / "artifacts" / "checkpoints"
        )
        best_checkpoints = [
            f
            for f in checkpoints_dir.iterdir()
            if f.name.startswith("best_") and f.suffix == ".pth"
        ]
        assert best_checkpoints, (
            f"No best_*.pth checkpoints found for run {run_id} in {checkpoints_dir}"
        )

        low = _CHECKPOINT_REFERENCE_BYTES * (1.0 - _CHECKPOINT_TOLERANCE)
        high = _CHECKPOINT_REFERENCE_BYTES * (1.0 + _CHECKPOINT_TOLERANCE)

        size_violations: list[str] = []
        for ckpt in best_checkpoints:
            size = ckpt.stat().st_size
            if not (low <= size <= high):
                size_violations.append(
                    f"{ckpt.name}: {size:,} bytes "
                    f"(expected {int(low):,} – {int(high):,})"
                )

        assert not size_violations, (
            f"Run {run_id} checkpoint size out of tolerance (±5 % of "
            f"{_CHECKPOINT_REFERENCE_BYTES:,} bytes):\n"
            + "\n".join(f"  {v}" for v in size_violations)
        )

    # ------------------------------------------------------------------
    # 12. No NaN values in eval metric files
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_no_nan_in_eval_metrics(self, run_id: str) -> None:
        """Eval metric files must not contain NaN as their logged value."""
        nan_metrics: list[str] = []
        all_metrics = get_run_metrics_list(MLRUNS_DIR, V2_EXPERIMENT_ID, run_id)
        eval_metrics = [m for m in all_metrics if m.startswith("eval_fold")]

        for metric_name in eval_metrics:
            value = read_metric_last_value(
                MLRUNS_DIR, V2_EXPERIMENT_ID, run_id, metric_name
            )
            if math.isnan(value):
                nan_metrics.append(metric_name)

        assert not nan_metrics, (
            f"Run {run_id} has NaN in eval metrics: {sorted(nan_metrics)}"
        )

    # ------------------------------------------------------------------
    # 13. metric_history.json parses as valid JSON
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_metric_history_is_valid_json(self, run_id: str) -> None:
        """artifacts/history/metric_history.json must parse as valid JSON."""
        history_file = (
            MLRUNS_DIR
            / V2_EXPERIMENT_ID
            / run_id
            / "artifacts"
            / "history"
            / "metric_history.json"
        )
        try:
            content = history_file.read_text(encoding="utf-8")
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            pytest.fail(f"Run {run_id} metric_history.json is not valid JSON: {exc}")
        assert isinstance(data, dict), (
            f"Run {run_id} metric_history.json root must be a dict, "
            f"got {type(data).__name__}"
        )

    # ------------------------------------------------------------------
    # 14. metric_history.json has expected structure
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_metric_history_structure(self, run_id: str) -> None:
        """metric_history.json must have 'epochs' key with non-empty epoch entries.

        Each epoch entry must be a dict with at minimum 'epoch' and 'metrics'
        keys.  The 'metrics' sub-dict must contain training metrics such as
        ``train_loss`` and ``val_loss``.
        """
        history_file = (
            MLRUNS_DIR
            / V2_EXPERIMENT_ID
            / run_id
            / "artifacts"
            / "history"
            / "metric_history.json"
        )
        data = json.loads(history_file.read_text(encoding="utf-8"))

        assert "epochs" in data, (
            f"Run {run_id} metric_history.json missing 'epochs' key. "
            f"Top-level keys: {sorted(data.keys())}"
        )

        epochs: list[dict[str, object]] = data["epochs"]
        assert isinstance(epochs, list), (
            f"Run {run_id} 'epochs' must be a list, got {type(epochs).__name__}"
        )
        assert len(epochs) > 0, f"Run {run_id} 'epochs' list is empty."

        # Validate first and last epoch entries
        for idx in (0, -1):
            entry = epochs[idx]
            assert isinstance(entry, dict), (
                f"Run {run_id} epoch entry at index {idx} is not a dict: {entry!r}"
            )
            assert "epoch" in entry, (
                f"Run {run_id} epoch entry at index {idx} missing 'epoch' key."
            )
            assert "metrics" in entry, (
                f"Run {run_id} epoch entry at index {idx} missing 'metrics' key."
            )
            metrics_dict = entry["metrics"]
            assert isinstance(metrics_dict, dict), (
                f"Run {run_id} epoch entry 'metrics' must be a dict."
            )
            # Check that core training metrics are present in at least one entry
            for required_metric in ("train_loss", "val_loss"):
                assert required_metric in metrics_dict, (
                    f"Run {run_id} epoch entry at index {idx} missing "
                    f"'{required_metric}' in metrics. "
                    f"Found: {sorted(metrics_dict.keys())}"
                )

        # All epoch numbers must be non-negative integers
        epoch_numbers = [e.get("epoch") for e in epochs]
        assert all(isinstance(n, int) and n >= 0 for n in epoch_numbers), (
            f"Run {run_id} some epoch entries have invalid 'epoch' values: "
            f"{[n for n in epoch_numbers if not (isinstance(n, int) and n >= 0)]}"
        )
