"""Condition name uniqueness guard (Issue #958).

Verifies that the Cartesian product of factorial factors produces unique,
well-formed condition names matching the format used by run_factorial.sh:

    Training:  {model}-{loss}-calib{aux_calib}-f{fold}
    Zero-shot: {model}-zeroshot-{dataset}-f{fold}

Guards against:
  - Duplicate names (would cause SkyPilot job collisions)
  - Substring collisions (would break grep-based status queries)
  - Cross-model prefix collisions (sam3 vs sam3_topolora vs sam3_hybrid)
  - K8s label limit violations (63 char max)
  - Names with spaces or slashes (would break shell quoting / paths)
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
DEBUG_PATH = REPO_ROOT / "configs" / "factorial" / "debug.yaml"
PAPER_FULL_PATH = REPO_ROOT / "configs" / "factorial" / "paper_full.yaml"

K8S_LABEL_MAX = 63


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file with utf-8 encoding."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _generate_condition_names(cfg: dict[str, Any]) -> list[str]:
    """Generate full Cartesian product of condition names from a factorial config.

    Matches the naming convention in scripts/run_factorial.sh:
        Training:  {model}-{loss}-calib{aux_calib}-f{fold}
        Zero-shot: {model}-zeroshot-{dataset}-f{fold}
    """
    factors = cfg["factors"]
    training = factors["training"]
    num_folds = cfg["fixed"]["num_folds"]

    models = training["model_family"]
    losses = training["loss_name"]
    aux_calibrations = training["aux_calibration"]
    folds = list(range(num_folds))

    # Training conditions: full Cartesian product
    names: list[str] = []
    for model, loss, aux_calib, fold in itertools.product(
        models, losses, aux_calibrations, folds
    ):
        # aux_calib is a bool in YAML; run_factorial.sh uses lowercase string
        aux_str = str(aux_calib).lower()
        name = f"{model}-{loss}-calib{aux_str}-f{fold}"
        names.append(name)

    # Zero-shot baselines
    for baseline in cfg.get("zero_shot_baselines", []):
        zs_model = baseline["model"]
        zs_dataset = baseline["dataset"]
        zs_folds = baseline.get("folds", 1)
        for fold in range(zs_folds):
            name = f"{zs_model}-zeroshot-{zs_dataset}-f{fold}"
            names.append(name)

    return names


@pytest.fixture(scope="module")
def debug_cfg() -> dict[str, Any]:
    return _load_yaml(DEBUG_PATH)


@pytest.fixture(scope="module")
def paper_cfg() -> dict[str, Any]:
    return _load_yaml(PAPER_FULL_PATH)


@pytest.fixture(scope="module")
def debug_names(debug_cfg: dict[str, Any]) -> list[str]:
    return _generate_condition_names(debug_cfg)


@pytest.fixture(scope="module")
def paper_names(paper_cfg: dict[str, Any]) -> list[str]:
    return _generate_condition_names(paper_cfg)


# ── Test 1: All condition names unique ──────────────────────────────────


class TestAllConditionNamesUnique:
    """No two conditions may share the same name (SkyPilot job collision)."""

    def test_debug_all_condition_names_unique(
        self, debug_names: list[str]
    ) -> None:
        assert len(debug_names) == len(set(debug_names)), (
            f"Duplicate condition names in debug.yaml: "
            f"{len(debug_names)} total, {len(set(debug_names))} unique. "
            f"Duplicates: {_find_duplicates(debug_names)}"
        )

    def test_paper_all_condition_names_unique(
        self, paper_names: list[str]
    ) -> None:
        assert len(paper_names) == len(set(paper_names)), (
            f"Duplicate condition names in paper_full.yaml: "
            f"{len(paper_names)} total, {len(set(paper_names))} unique. "
            f"Duplicates: {_find_duplicates(paper_names)}"
        )


# ── Test 2: No name is substring of another ────────────────────────────


class TestNoNameIsSubstringOfAnother:
    """If name A is a substring of name B, grep-based status queries break."""

    def test_debug_no_name_is_substring_of_another(
        self, debug_names: list[str]
    ) -> None:
        collisions = _find_substring_collisions(debug_names)
        assert not collisions, (
            f"Substring collisions in debug.yaml ({len(collisions)} pairs): "
            f"{collisions[:10]}"
        )

    def test_paper_no_name_is_substring_of_another(
        self, paper_names: list[str]
    ) -> None:
        collisions = _find_substring_collisions(paper_names)
        assert not collisions, (
            f"Substring collisions in paper_full.yaml ({len(collisions)} pairs): "
            f"{collisions[:10]}"
        )


# ── Test 3: No cross-model prefix collision ─────────────────────────────


class TestNoCrossModelPrefixCollision:
    """Model names must not be prefixes of each other.

    E.g., "sam3" would be a prefix of "sam3_topolora" and "sam3_hybrid",
    causing grep/filter collisions. The current naming (sam3_topolora,
    sam3_hybrid — no bare "sam3") is safe.
    """

    def test_debug_no_cross_model_prefix_collision(
        self, debug_cfg: dict[str, Any]
    ) -> None:
        models = debug_cfg["factors"]["training"]["model_family"]
        # Include zero-shot baseline models
        for baseline in debug_cfg.get("zero_shot_baselines", []):
            models = [*models, baseline["model"]]
        collisions = _find_prefix_collisions(models)
        assert not collisions, (
            f"Model prefix collisions in debug.yaml: {collisions}. "
            f"A model name is a prefix of another, which breaks grep-based filtering."
        )

    def test_paper_no_cross_model_prefix_collision(
        self, paper_cfg: dict[str, Any]
    ) -> None:
        models = paper_cfg["factors"]["training"]["model_family"]
        for baseline in paper_cfg.get("zero_shot_baselines", []):
            models = [*models, baseline["model"]]
        collisions = _find_prefix_collisions(models)
        assert not collisions, (
            f"Model prefix collisions in paper_full.yaml: {collisions}. "
            f"A model name is a prefix of another, which breaks grep-based filtering."
        )


# ── Test 4: Names within K8s label limit (63 chars) ────────────────────


class TestNamesWithinK8sLabelLimit:
    """SkyPilot job names become K8s labels — max 63 characters."""

    def test_debug_names_within_k8s_label_limit(
        self, debug_names: list[str]
    ) -> None:
        violations = [n for n in debug_names if len(n) > K8S_LABEL_MAX]
        assert not violations, (
            f"{len(violations)} condition names in debug.yaml exceed "
            f"K8s label limit ({K8S_LABEL_MAX} chars): "
            f"{[(n, len(n)) for n in violations[:5]]}"
        )

    def test_paper_names_within_k8s_label_limit(
        self, paper_names: list[str]
    ) -> None:
        violations = [n for n in paper_names if len(n) > K8S_LABEL_MAX]
        assert not violations, (
            f"{len(violations)} condition names in paper_full.yaml exceed "
            f"K8s label limit ({K8S_LABEL_MAX} chars): "
            f"{[(n, len(n)) for n in violations[:5]]}"
        )


# ── Test 5: Names have no spaces ────────────────────────────────────────


class TestNamesHaveNoSpaces:
    """Spaces in condition names would break shell quoting in run_factorial.sh."""

    def test_debug_names_have_no_spaces(
        self, debug_names: list[str]
    ) -> None:
        violations = [n for n in debug_names if " " in n]
        assert not violations, (
            f"Condition names with spaces in debug.yaml: {violations[:5]}"
        )

    def test_paper_names_have_no_spaces(
        self, paper_names: list[str]
    ) -> None:
        violations = [n for n in paper_names if " " in n]
        assert not violations, (
            f"Condition names with spaces in paper_full.yaml: {violations[:5]}"
        )


# ── Test 6: Names have no slashes ───────────────────────────────────────


class TestNamesHaveNoSlashes:
    """Slashes in condition names would break file paths and SkyPilot job names."""

    def test_debug_names_have_no_slashes(
        self, debug_names: list[str]
    ) -> None:
        violations = [n for n in debug_names if "/" in n or "\\" in n]
        assert not violations, (
            f"Condition names with slashes in debug.yaml: {violations[:5]}"
        )

    def test_paper_names_have_no_slashes(
        self, paper_names: list[str]
    ) -> None:
        violations = [n for n in paper_names if "/" in n or "\\" in n]
        assert not violations, (
            f"Condition names with slashes in paper_full.yaml: {violations[:5]}"
        )


# ── Helpers (no regex — Rule #16) ───────────────────────────────────────


def _find_duplicates(names: list[str]) -> list[str]:
    """Return names that appear more than once."""
    seen: set[str] = set()
    dupes: list[str] = []
    for n in names:
        if n in seen:
            dupes.append(n)
        seen.add(n)
    return dupes


def _find_substring_collisions(names: list[str]) -> list[tuple[str, str]]:
    """Return pairs (a, b) where a != b and a is a substring of b."""
    collisions: list[tuple[str, str]] = []
    sorted_names = sorted(names, key=len)
    for i, shorter in enumerate(sorted_names):
        for longer in sorted_names[i + 1 :]:
            if shorter != longer and shorter in longer:
                collisions.append((shorter, longer))
    return collisions


def _find_prefix_collisions(model_names: list[str]) -> list[tuple[str, str]]:
    """Return pairs where one model name is a prefix of another.

    E.g., ("sam3", "sam3_topolora") would be a collision.
    Uses str.startswith() — no regex (Rule #16).
    """
    collisions: list[tuple[str, str]] = []
    for a, b in itertools.combinations(model_names, 2):
        if a != b:
            if b.startswith(a) or a.startswith(b):
                shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
                collisions.append((shorter, longer))
    return collisions
