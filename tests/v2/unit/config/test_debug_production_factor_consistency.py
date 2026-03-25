"""Debug=Production factor consistency guard (Issue #942).

Regression tests ensuring configs/factorial/debug.yaml and
configs/factorial/paper_full.yaml have IDENTICAL factors, model overrides,
and zero-shot baselines. Debug may differ ONLY in:
  1. Epochs (max_epochs)
  2. Data (max_train_volumes, max_val_volumes)
  3. Folds (num_folds, zero_shot_baselines[*].folds)

See: CLAUDE.md Rule #27 — Debug Run = Full Production.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
DEBUG_PATH = REPO_ROOT / "configs" / "factorial" / "debug.yaml"
PAPER_FULL_PATH = REPO_ROOT / "configs" / "factorial" / "paper_full.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file with utf-8 encoding."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def debug_cfg() -> dict[str, Any]:
    return _load_yaml(DEBUG_PATH)


@pytest.fixture(scope="module")
def paper_cfg() -> dict[str, Any]:
    return _load_yaml(PAPER_FULL_PATH)


# ── Paths into the factor tree to compare ──────────────────────────────
COMPARED_PATHS = [
    ("factors", "training", "model_family"),
    ("factors", "training", "loss_name"),
    ("factors", "training", "aux_calibration"),
    ("factors", "post_training", "method"),
    ("factors", "post_training", "recalibration"),
    ("factors", "analysis", "ensemble_strategy"),
]


def _resolve(cfg: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Walk nested dict by key tuple, raising KeyError with full path on miss."""
    node = cfg
    for k in keys:
        try:
            node = node[k]
        except (KeyError, TypeError) as exc:
            msg = f"Missing key path: {'.'.join(keys)} (failed at '{k}')"
            raise KeyError(msg) from exc
    return node


# ── Test 1: Factor lists must be identical ─────────────────────────────
class TestFactorsIdentical:
    """Each factor list in debug.yaml must exactly match paper_full.yaml."""

    @pytest.mark.parametrize(
        "key_path",
        COMPARED_PATHS,
        ids=[".".join(p) for p in COMPARED_PATHS],
    )
    def test_factors_identical(
        self,
        debug_cfg: dict[str, Any],
        paper_cfg: dict[str, Any],
        key_path: tuple[str, ...],
    ) -> None:
        debug_values = _resolve(debug_cfg, key_path)
        paper_values = _resolve(paper_cfg, key_path)

        # Convert to strings for uniform sorting (aux_calibration has booleans)
        debug_sorted = sorted(str(v) for v in debug_values)
        paper_sorted = sorted(str(v) for v in paper_values)

        assert debug_sorted == paper_sorted, (
            f"Factor mismatch at {'.'.join(key_path)}:\n"
            f"  debug:      {debug_sorted}\n"
            f"  paper_full: {paper_sorted}\n"
            f"Rule #27: debug = production — factors must be IDENTICAL."
        )


# ── Test 2: Model overrides must be identical ──────────────────────────
class TestModelOverridesIdentical:
    """model_overrides dict must match between debug and paper_full."""

    def test_model_overrides_identical(
        self,
        debug_cfg: dict[str, Any],
        paper_cfg: dict[str, Any],
    ) -> None:
        debug_overrides = debug_cfg.get("model_overrides", {})
        paper_overrides = paper_cfg.get("model_overrides", {})

        assert debug_overrides == paper_overrides, (
            f"model_overrides mismatch:\n"
            f"  debug:      {debug_overrides}\n"
            f"  paper_full: {paper_overrides}\n"
            f"Rule #27: VRAM overrides must be identical — same models, same GPUs."
        )


# ── Test 3: Zero-shot baselines identical (ignoring folds) ────────────
class TestZeroShotBaselinesIdentical:
    """Zero-shot baselines must match except for 'folds' (debug reduction)."""

    def test_zero_shot_baselines_identical(
        self,
        debug_cfg: dict[str, Any],
        paper_cfg: dict[str, Any],
    ) -> None:
        debug_baselines = debug_cfg.get("zero_shot_baselines", [])
        paper_baselines = paper_cfg.get("zero_shot_baselines", [])

        assert len(debug_baselines) == len(paper_baselines), (
            f"Different number of zero-shot baselines: "
            f"debug={len(debug_baselines)}, paper_full={len(paper_baselines)}"
        )

        for i, (dbg, ppr) in enumerate(zip(debug_baselines, paper_baselines)):
            # Compare everything except 'folds' (allowed debug reduction per Rule #27)
            dbg_no_folds = {k: v for k, v in dbg.items() if k != "folds"}
            ppr_no_folds = {k: v for k, v in ppr.items() if k != "folds"}

            assert dbg_no_folds == ppr_no_folds, (
                f"Zero-shot baseline [{i}] mismatch (ignoring folds):\n"
                f"  debug:      {dbg_no_folds}\n"
                f"  paper_full: {ppr_no_folds}\n"
                f"Rule #27: same baselines, fewer folds is the only allowed diff."
            )


# ── Test 4: debug flag only in debug.yaml ──────────────────────────────
class TestDebugFlagOnlyInDebug:
    """debug.yaml must have debug: true; paper_full.yaml must NOT have it."""

    def test_debug_flag_only_in_debug(
        self,
        debug_cfg: dict[str, Any],
        paper_cfg: dict[str, Any],
    ) -> None:
        assert debug_cfg.get("debug") is True, (
            "debug.yaml must have 'debug: true' at top level."
        )
        assert "debug" not in paper_cfg, (
            "paper_full.yaml must NOT contain 'debug' key. "
            "Production config is the default — no debug flag needed."
        )


# ── Test 5: Paper has no data limits ───────────────────────────────────
class TestPaperHasNoDataLimits:
    """paper_full.yaml.fixed must NOT have max_train_volumes or max_val_volumes."""

    def test_paper_has_no_data_limits(
        self,
        paper_cfg: dict[str, Any],
    ) -> None:
        fixed = paper_cfg.get("fixed", {})
        banned_keys = {"max_train_volumes", "max_val_volumes"}
        found = banned_keys & set(fixed.keys())

        assert not found, (
            f"paper_full.yaml.fixed contains data-limiting keys: {found}. "
            f"Production runs must use ALL data — no volume caps."
        )


# ── Test 6: Infrastructure skypilot_yaml allowed to differ ─────────────
class TestInfrastructureSkypilotDiffers:
    """debug and paper_full are allowed to use different skypilot_yaml files."""

    def test_infrastructure_skypilot_yaml_differs(
        self,
        debug_cfg: dict[str, Any],
        paper_cfg: dict[str, Any],
    ) -> None:
        debug_sky = debug_cfg.get("infrastructure", {}).get("skypilot_yaml")
        paper_sky = paper_cfg.get("infrastructure", {}).get("skypilot_yaml")

        # Both must be present
        assert debug_sky is not None, "debug.yaml missing infrastructure.skypilot_yaml"
        assert paper_sky is not None, (
            "paper_full.yaml missing infrastructure.skypilot_yaml"
        )

        # They SHOULD differ (debug uses train_factorial.yaml, prod uses train_production.yaml)
        assert debug_sky != paper_sky, (
            f"skypilot_yaml is identical ({debug_sky}) — expected different files "
            f"for debug vs production. Debug uses lighter/cheaper instance config."
        )

        # Verify the expected filenames
        assert "train_factorial" in debug_sky, (
            f"debug skypilot_yaml should reference train_factorial, got: {debug_sky}"
        )
        assert "train_production" in paper_sky, (
            f"paper_full skypilot_yaml should reference train_production, got: {paper_sky}"
        )
