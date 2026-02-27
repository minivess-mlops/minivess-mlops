"""Integration tests: checkpoint loadability and metric_history.json validation.

Verifies that every checkpoint saved during the ``dynunet_loss_variation_v2``
experiment (MLflow experiment ID 843896622863223169) is structurally sound and
loadable into the DynUNetAdapter — and that each run's metric_history.json is
internally consistent.

The 4 production runs cover these loss functions:
    - dice_ce        (af4adc1599c14538b334a9b2a57613aa)
    - cbdice         (3a9f3615207f47068c0f9650abcd9faf)
    - dice_ce_cldice (4b2451ac0a6a40cd80cbd992db6193c5)
    - cbdice_cldice  (01d904c61b1043a6b4d4630ec1506992)

Each run has 7 checkpoints in ``artifacts/checkpoints/`` and one
``artifacts/history/metric_history.json`` (~46 KB, 100 epoch entries).

Checkpoints are saved by :class:`~minivess.pipeline.trainer.SegmentationTrainer`
with the following top-level keys:
    - ``model_state_dict``
    - ``optimizer_state_dict``
    - ``scheduler_state_dict``
    - ``scaler_state_dict``
    - ``checkpoint_metadata``

State dict keys have the ``net.`` prefix because the adapter
(:class:`~minivess.adapters.dynunet.DynUNetAdapter`) wraps
``monai.networks.nets.DynUNet`` as ``self.net``.  Therefore the state dict
must be loaded into the *adapter* (not ``adapter.net``).

Run with::

    uv run pytest tests/v2/integration/test_checkpoint_loadability.py \\
        -m integration -v

All tests are skipped automatically when ``mlruns/`` is absent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from minivess.adapters.dynunet import DynUNetAdapter
from minivess.config.models import ModelConfig, ModelFamily
from minivess.pipeline.mlruns_inspector import get_production_runs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MLRUNS_DIR: Path = Path(__file__).resolve().parents[3] / "mlruns"
V2_EXPERIMENT_ID: str = "843896622863223169"

EXPECTED_CHECKPOINTS: set[str] = {
    "best_val_loss.pth",
    "best_val_dice.pth",
    "best_val_f1_foreground.pth",
    "best_val_cldice.pth",
    "best_val_masd.pth",
    "best_val_compound_masd_cldice.pth",
    "last.pth",
}

# Number of checkpoints per run × number of production runs
EXPECTED_TOTAL_CHECKPOINTS: int = len(EXPECTED_CHECKPOINTS) * 4  # 7 × 4 = 28

# Input shape used for forward-pass tests: (B, C, D, H, W)
_FORWARD_PASS_INPUT_SHAPE: tuple[int, ...] = (1, 1, 32, 32, 16)
_FORWARD_PASS_OUTPUT_SHAPE: tuple[int, ...] = (1, 2, 32, 32, 16)

# Training history expected structure
_MIN_EPOCHS_IN_HISTORY: int = 100
_NUM_EXPECTED_FOLDS: int = 3  # fold0, fold1, fold2

# ---------------------------------------------------------------------------
# Module-level skip guard
# ---------------------------------------------------------------------------

_mlruns_missing: bool = not (MLRUNS_DIR / V2_EXPERIMENT_ID).is_dir()
_skip_reason: str = (
    f"mlruns experiment directory not found: {MLRUNS_DIR / V2_EXPERIMENT_ID}. "
    "Run training first or verify the MLRUNS_DIR path."
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_cached_production_runs() -> list[str]:
    """Return production run IDs; empty list when mlruns absent."""
    if _mlruns_missing:
        return []
    return get_production_runs(MLRUNS_DIR, V2_EXPERIMENT_ID)


def _build_dynunet_adapter() -> DynUNetAdapter:
    """Instantiate a DynUNetAdapter with the default MiniVess architecture."""
    config = ModelConfig(
        family=ModelFamily.MONAI_DYNUNET,
        name="checkpoint-loadability-test",
        in_channels=1,
        out_channels=2,
    )
    return DynUNetAdapter(config)


# Evaluated once at collection time to allow parametrize decorators.
_PRODUCTION_RUNS: list[str] = _get_cached_production_runs()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(_mlruns_missing, reason=_skip_reason)
class TestCheckpointLoadability:
    """Verify checkpoint existence, structure, and loadability for all 4 runs."""

    # ------------------------------------------------------------------
    # 1. Count: 28 checkpoint files total (7 per run × 4 runs)
    # ------------------------------------------------------------------

    def test_all_production_checkpoints_exist(self) -> None:
        """All 28 checkpoint files must exist (7 per run × 4 production runs).

        Walks every production run's ``artifacts/checkpoints/`` directory and
        asserts that the union of found files matches the expected 28 total.
        """
        total_found: int = 0
        missing_by_run: dict[str, list[str]] = {}

        for run_id in _PRODUCTION_RUNS:
            checkpoints_dir = (
                MLRUNS_DIR / V2_EXPERIMENT_ID / run_id / "artifacts" / "checkpoints"
            )
            assert checkpoints_dir.is_dir(), (
                f"Checkpoints directory missing for run {run_id}: {checkpoints_dir}"
            )
            found: set[str] = {f.name for f in checkpoints_dir.iterdir() if f.is_file()}
            missing = sorted(EXPECTED_CHECKPOINTS - found)
            if missing:
                missing_by_run[run_id] = missing
            total_found += len(found & EXPECTED_CHECKPOINTS)

        assert not missing_by_run, (
            "Some runs are missing checkpoint files:\n"
            + "\n".join(
                f"  Run {rid}: {fnames}" for rid, fnames in missing_by_run.items()
            )
        )
        assert total_found == EXPECTED_TOTAL_CHECKPOINTS, (
            f"Expected {EXPECTED_TOTAL_CHECKPOINTS} checkpoint files total, "
            f"found {total_found}."
        )

    # ------------------------------------------------------------------
    # 2. model_state_dict key present in every checkpoint
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_checkpoint_has_model_state_dict_key(self, run_id: str) -> None:
        """Every checkpoint dict must contain the ``model_state_dict`` key.

        Loads each of the 7 .pth files for *run_id* with
        ``weights_only=True`` (safe, no arbitrary code execution) and verifies
        the required key is present.
        """
        checkpoints_dir = (
            MLRUNS_DIR / V2_EXPERIMENT_ID / run_id / "artifacts" / "checkpoints"
        )
        checkpoint_files = [checkpoints_dir / name for name in EXPECTED_CHECKPOINTS]
        missing_key: list[str] = []

        for ckpt_path in checkpoint_files:
            payload: dict[str, Any] = torch.load(
                ckpt_path, map_location="cpu", weights_only=True
            )
            if "model_state_dict" not in payload:
                missing_key.append(ckpt_path.name)

        assert not missing_key, (
            f"Run {run_id}: checkpoints missing 'model_state_dict' key: "
            f"{sorted(missing_key)}"
        )

    # ------------------------------------------------------------------
    # 3. State dict loads into DynUNetAdapter without errors
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_checkpoint_loadable_into_dynunet(self, run_id: str) -> None:
        """Every checkpoint's state dict must load into DynUNetAdapter cleanly.

        The state dict keys use the ``net.`` prefix (adapter wraps MONAI
        DynUNet as ``self.net``), so the state dict is loaded into the
        full *adapter*, not into ``adapter.net``.  The test verifies that
        ``load_state_dict`` succeeds with ``strict=True`` and that no keys
        are missing or unexpected.
        """
        checkpoints_dir = (
            MLRUNS_DIR / V2_EXPERIMENT_ID / run_id / "artifacts" / "checkpoints"
        )
        failed: list[str] = []

        for ckpt_name in EXPECTED_CHECKPOINTS:
            ckpt_path = checkpoints_dir / ckpt_name
            payload: dict[str, Any] = torch.load(
                ckpt_path, map_location="cpu", weights_only=True
            )
            state_dict: dict[str, Any] = payload["model_state_dict"]

            adapter = _build_dynunet_adapter()
            try:
                result = adapter.load_state_dict(state_dict, strict=True)
                # load_state_dict returns a NamedTuple with missing/unexpected keys
                if result.missing_keys or result.unexpected_keys:
                    failed.append(
                        f"{ckpt_name}: missing={result.missing_keys[:3]}, "
                        f"unexpected={result.unexpected_keys[:3]}"
                    )
            except RuntimeError as exc:
                failed.append(f"{ckpt_name}: {exc}")

        assert not failed, (
            f"Run {run_id}: checkpoint load_state_dict failures:\n"
            + "\n".join(f"  {msg}" for msg in failed)
        )

    # ------------------------------------------------------------------
    # 4. Forward pass on (1,1,32,32,16) produces (1,2,32,32,16)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_model_forward_pass_valid_shape(self, run_id: str) -> None:
        """Loaded model forward pass must produce the expected output shape.

        Uses ``last.pth`` as the representative checkpoint.  Input is a
        random tensor of shape ``(1, 1, 32, 32, 16)``; expected output shape
        is ``(1, 2, 32, 32, 16)`` (batch=1, classes=2, spatial dims).
        """
        ckpt_path = (
            MLRUNS_DIR
            / V2_EXPERIMENT_ID
            / run_id
            / "artifacts"
            / "checkpoints"
            / "last.pth"
        )
        payload: dict[str, Any] = torch.load(
            ckpt_path, map_location="cpu", weights_only=True
        )
        state_dict: dict[str, Any] = payload["model_state_dict"]

        adapter = _build_dynunet_adapter()
        adapter.load_state_dict(state_dict, strict=True)
        adapter.eval()

        x = torch.randn(*_FORWARD_PASS_INPUT_SHAPE)
        with torch.no_grad():
            output = adapter.net(x)

        assert output.shape == torch.Size(_FORWARD_PASS_OUTPUT_SHAPE), (
            f"Run {run_id}: expected output shape {_FORWARD_PASS_OUTPUT_SHAPE}, "
            f"got {tuple(output.shape)}"
        )

    # ------------------------------------------------------------------
    # 5. Model output is not all-zeros (weights are meaningful)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_model_output_not_all_zeros(self, run_id: str) -> None:
        """Loaded model output must have non-zero standard deviation.

        A freshly initialised or zeroed-out model would produce near-constant
        output.  Meaningful trained weights produce varied logits.  Uses
        ``last.pth`` and the same synthetic input as the shape test.
        """
        ckpt_path = (
            MLRUNS_DIR
            / V2_EXPERIMENT_ID
            / run_id
            / "artifacts"
            / "checkpoints"
            / "last.pth"
        )
        payload: dict[str, Any] = torch.load(
            ckpt_path, map_location="cpu", weights_only=True
        )
        state_dict: dict[str, Any] = payload["model_state_dict"]

        adapter = _build_dynunet_adapter()
        adapter.load_state_dict(state_dict, strict=True)
        adapter.eval()

        torch.manual_seed(0)
        x = torch.randn(*_FORWARD_PASS_INPUT_SHAPE)
        with torch.no_grad():
            output = adapter.net(x)

        std_value: float = output.std().item()
        assert std_value > 1e-6, (
            f"Run {run_id}: model output standard deviation is near zero "
            f"({std_value:.2e}), suggesting weights are trivial or zeroed-out."
        )

    # ------------------------------------------------------------------
    # 6. Best checkpoint weights differ from last.pth
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_best_checkpoints_differ_from_last(self, run_id: str) -> None:
        """best_val_compound_masd_cldice.pth weights must differ from last.pth.

        If early stopping triggered before epoch 100, or if the best epoch
        precedes the final epoch, the weight tensors will differ.  This test
        confirms that best-checkpoint saving is distinct from simple last-epoch
        saving.  We compare the first weight tensor in the state dict.
        """
        checkpoints_dir = (
            MLRUNS_DIR / V2_EXPERIMENT_ID / run_id / "artifacts" / "checkpoints"
        )
        last_path = checkpoints_dir / "last.pth"
        best_path = checkpoints_dir / "best_val_compound_masd_cldice.pth"

        last_payload: dict[str, Any] = torch.load(
            last_path, map_location="cpu", weights_only=True
        )
        best_payload: dict[str, Any] = torch.load(
            best_path, map_location="cpu", weights_only=True
        )

        last_state: dict[str, torch.Tensor] = last_payload["model_state_dict"]
        best_state: dict[str, torch.Tensor] = best_payload["model_state_dict"]

        # Compare the first parameter tensor in the state dict.  If best epoch
        # != last epoch the parameters will differ.
        first_key = next(iter(last_state))
        tensors_identical = torch.allclose(last_state[first_key], best_state[first_key])

        last_meta = last_payload.get("checkpoint_metadata", {})
        best_meta = best_payload.get("checkpoint_metadata", {})
        last_epoch = last_meta.get("epoch") if isinstance(last_meta, dict) else None
        best_epoch = best_meta.get("epoch") if isinstance(best_meta, dict) else None

        assert not tensors_identical, (
            f"Run {run_id}: best_val_compound_masd_cldice.pth and last.pth have "
            f"identical weights (both at epoch {best_epoch}). "
            f"Expected them to differ (last epoch={last_epoch}, "
            f"best epoch={best_epoch})."
        )


# ---------------------------------------------------------------------------
# metric_history.json validation tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(_mlruns_missing, reason=_skip_reason)
class TestMetricHistoryValidation:
    """Validate the ``artifacts/history/metric_history.json`` for each run."""

    def _load_history(self, run_id: str) -> dict[str, Any]:
        """Load and return the parsed metric_history.json for *run_id*."""
        history_file = (
            MLRUNS_DIR
            / V2_EXPERIMENT_ID
            / run_id
            / "artifacts"
            / "history"
            / "metric_history.json"
        )
        data: dict[str, Any] = json.loads(history_file.read_text(encoding="utf-8"))
        return data

    # ------------------------------------------------------------------
    # 7. Valid JSON
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_metric_history_is_valid_json(self, run_id: str) -> None:
        """metric_history.json must parse as valid JSON with a dict root."""
        history_file = (
            MLRUNS_DIR
            / V2_EXPERIMENT_ID
            / run_id
            / "artifacts"
            / "history"
            / "metric_history.json"
        )
        assert history_file.is_file(), (
            f"Run {run_id}: metric_history.json not found at {history_file}"
        )
        try:
            content = history_file.read_text(encoding="utf-8")
            data = json.loads(content)
        except (json.JSONDecodeError, OSError) as exc:
            pytest.fail(f"Run {run_id}: metric_history.json is not valid JSON: {exc}")

        assert isinstance(data, dict), (
            f"Run {run_id}: metric_history.json root must be a dict, "
            f"got {type(data).__name__}"
        )
        assert "epochs" in data, (
            f"Run {run_id}: metric_history.json missing 'epochs' key. "
            f"Top-level keys: {sorted(data.keys())}"
        )

    # ------------------------------------------------------------------
    # 8. Fold-level coverage: run has eval metrics for all 3 folds
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_metric_history_has_fold_entries(self, run_id: str) -> None:
        """Each production run must have evaluation data for all 3 folds.

        The metric_history.json itself covers one fold's training epochs.
        Cross-fold coverage is verified via the MLflow metrics directory:
        the presence of ``eval_fold0_dsc``, ``eval_fold1_dsc``, and
        ``eval_fold2_dsc`` confirms that training completed across all 3 folds.
        This is the definitive signal that the run is a complete, 3-fold
        cross-validated experiment.
        """
        metrics_dir = MLRUNS_DIR / V2_EXPERIMENT_ID / run_id / "metrics"
        assert metrics_dir.is_dir(), (
            f"Run {run_id}: metrics directory not found at {metrics_dir}"
        )

        found_folds: list[int] = []
        for fold_idx in range(_NUM_EXPECTED_FOLDS):
            fold_metric_file = metrics_dir / f"eval_fold{fold_idx}_dsc"
            if fold_metric_file.is_file():
                found_folds.append(fold_idx)

        assert len(found_folds) >= _NUM_EXPECTED_FOLDS, (
            f"Run {run_id}: expected evaluation data for {_NUM_EXPECTED_FOLDS} folds "
            f"(fold0–fold{_NUM_EXPECTED_FOLDS - 1}), "
            f"but only found data for folds: {found_folds}. "
            "Training may have been interrupted before completing all folds."
        )

    # ------------------------------------------------------------------
    # 9. 100 epochs recorded in metric_history.json
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_metric_history_has_100_epochs(self, run_id: str) -> None:
        """Each metric_history.json must contain exactly 100 epoch entries.

        Training was configured with ``max_epochs=100``.  The last epoch entry
        must have ``epoch == 99`` (0-based indexing).  This confirms training
        ran to completion without early stopping.
        """
        data = self._load_history(run_id)
        epochs: list[dict[str, Any]] = data["epochs"]

        assert isinstance(epochs, list), (
            f"Run {run_id}: 'epochs' must be a list, got {type(epochs).__name__}"
        )
        assert len(epochs) >= _MIN_EPOCHS_IN_HISTORY, (
            f"Run {run_id}: expected at least {_MIN_EPOCHS_IN_HISTORY} epoch entries "
            f"in metric_history.json, found {len(epochs)}."
        )

        last_epoch_num = epochs[-1].get("epoch")
        assert last_epoch_num == _MIN_EPOCHS_IN_HISTORY - 1, (
            f"Run {run_id}: last epoch number in history should be "
            f"{_MIN_EPOCHS_IN_HISTORY - 1} (0-based), got {last_epoch_num}."
        )

    # ------------------------------------------------------------------
    # 10. No NaN values in training / validation metrics
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_metric_history_no_nan_values(self, run_id: str) -> None:
        """No float metric value in metric_history.json may be NaN.

        NaN values indicate numerical instability (exploding loss, bad
        initialisation, or fp16 overflow).  The check covers all metric
        keys in every epoch entry.
        """
        data = self._load_history(run_id)
        epochs: list[dict[str, Any]] = data["epochs"]

        nan_locations: list[str] = []
        for entry in epochs:
            epoch_num = entry.get("epoch", "?")
            metrics_dict = entry.get("metrics", {})
            if not isinstance(metrics_dict, dict):
                continue
            for metric_name, value in metrics_dict.items():
                if isinstance(value, float) and math.isnan(value):
                    nan_locations.append(f"epoch={epoch_num}, metric={metric_name}")

        assert not nan_locations, (
            f"Run {run_id}: NaN values found in metric_history.json:\n"
            + "\n".join(f"  {loc}" for loc in nan_locations[:20])
        )

    # ------------------------------------------------------------------
    # 11. Training loss decreases: epoch 100 < epoch 1
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("run_id", _PRODUCTION_RUNS)
    def test_metric_history_loss_decreases(self, run_id: str) -> None:
        """Training loss at the final epoch must be lower than at epoch 1.

        This is a sanity check that the model actually learned: a monotonic
        decrease is not required (loss can oscillate), but the terminal value
        must be strictly below the value at the first epoch.  Uses
        ``train_loss`` which is recorded every epoch without exception.
        """
        data = self._load_history(run_id)
        epochs: list[dict[str, Any]] = data["epochs"]

        assert len(epochs) >= 2, (  # noqa: PLR2004
            f"Run {run_id}: need at least 2 epoch entries to compare loss trend, "
            f"found {len(epochs)}."
        )

        # epoch[0] is epoch 0 (index 0), epoch[1] would be epoch 1.
        # We compare the very first logged epoch to the very last.
        first_metrics = epochs[0].get("metrics", {})
        last_metrics = epochs[-1].get("metrics", {})

        first_loss = first_metrics.get("train_loss")
        last_loss = last_metrics.get("train_loss")

        assert isinstance(first_loss, (int, float)) and not math.isnan(first_loss), (
            f"Run {run_id}: epoch 0 'train_loss' is missing or NaN: {first_loss}"
        )
        assert isinstance(last_loss, (int, float)) and not math.isnan(last_loss), (
            f"Run {run_id}: final epoch 'train_loss' is missing or NaN: {last_loss}"
        )
        assert last_loss < first_loss, (
            f"Run {run_id}: training loss did not decrease. "
            f"epoch 0 train_loss={first_loss:.6f}, "
            f"epoch {epochs[-1].get('epoch')} train_loss={last_loss:.6f}. "
            "Model may not have trained correctly."
        )
