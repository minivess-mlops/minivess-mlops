"""Tests for multi-family champion model registry.

TDD RED phase for Task T-C1 (Issue #765).
Champion model per family: CNN, Foundation, Mamba.
Data-driven selection with human-in-the-loop alerting.
"""

from __future__ import annotations

import dataclasses

import pytest


class TestChampionEntry:
    """Test the ChampionEntry dataclass."""

    def test_champion_entry_is_dataclass(self) -> None:
        from minivess.serving.champion_registry import ChampionEntry

        assert dataclasses.is_dataclass(ChampionEntry)

    def test_champion_entry_fields(self) -> None:
        from minivess.serving.champion_registry import ChampionEntry

        entry = ChampionEntry(
            model_family="cnn",
            model_name="dynunet_quarter",
            checkpoint_path="/checkpoints/dynunet_best.pt",
            dice_score=0.85,
            cldice_score=0.72,
            mlflow_run_id="abc123",
        )
        assert entry.model_family == "cnn"
        assert entry.model_name == "dynunet_quarter"
        assert entry.dice_score == 0.85

    def test_champion_entry_optional_onnx_path(self) -> None:
        from minivess.serving.champion_registry import ChampionEntry

        entry = ChampionEntry(
            model_family="cnn",
            model_name="dynunet",
            checkpoint_path="/ckpt/best.pt",
            dice_score=0.80,
            cldice_score=0.65,
            mlflow_run_id="def456",
            onnx_path="/ckpt/best.onnx",
        )
        assert entry.onnx_path == "/ckpt/best.onnx"

    def test_champion_entry_default_onnx_none(self) -> None:
        from minivess.serving.champion_registry import ChampionEntry

        entry = ChampionEntry(
            model_family="foundation",
            model_name="sam3_vanilla",
            checkpoint_path="/ckpt/sam3.pt",
            dice_score=0.88,
            cldice_score=0.78,
            mlflow_run_id="ghi789",
        )
        assert entry.onnx_path is None


class TestChampionRegistry:
    """Test the multi-family champion registry."""

    def test_registry_starts_empty(self) -> None:
        from minivess.serving.champion_registry import ChampionRegistry

        reg = ChampionRegistry()
        assert len(reg.families) == 0

    def test_register_champion(self) -> None:
        from minivess.serving.champion_registry import (
            ChampionEntry,
            ChampionRegistry,
        )

        reg = ChampionRegistry()
        entry = ChampionEntry(
            model_family="cnn",
            model_name="dynunet",
            checkpoint_path="/ckpt/best.pt",
            dice_score=0.85,
            cldice_score=0.72,
            mlflow_run_id="abc",
        )
        reg.register(entry)
        assert "cnn" in reg.families
        assert reg.get_champion("cnn") == entry

    def test_register_multiple_families(self) -> None:
        from minivess.serving.champion_registry import (
            ChampionEntry,
            ChampionRegistry,
        )

        reg = ChampionRegistry()
        cnn = ChampionEntry(
            model_family="cnn",
            model_name="dynunet",
            checkpoint_path="/a",
            dice_score=0.85,
            cldice_score=0.7,
            mlflow_run_id="1",
        )
        foundation = ChampionEntry(
            model_family="foundation",
            model_name="sam3_vanilla",
            checkpoint_path="/b",
            dice_score=0.88,
            cldice_score=0.78,
            mlflow_run_id="2",
        )
        mamba = ChampionEntry(
            model_family="mamba",
            model_name="mambavesselnet",
            checkpoint_path="/c",
            dice_score=0.82,
            cldice_score=0.68,
            mlflow_run_id="3",
        )
        reg.register(cnn)
        reg.register(foundation)
        reg.register(mamba)
        assert set(reg.families) == {"cnn", "foundation", "mamba"}

    def test_get_champion_unknown_family_raises(self) -> None:
        from minivess.serving.champion_registry import ChampionRegistry

        reg = ChampionRegistry()
        with pytest.raises(KeyError, match="nonexistent"):
            reg.get_champion("nonexistent")

    def test_update_champion_replaces(self) -> None:
        from minivess.serving.champion_registry import (
            ChampionEntry,
            ChampionRegistry,
        )

        reg = ChampionRegistry()
        old = ChampionEntry(
            model_family="cnn",
            model_name="dynunet_v1",
            checkpoint_path="/v1",
            dice_score=0.80,
            cldice_score=0.65,
            mlflow_run_id="old",
        )
        new = ChampionEntry(
            model_family="cnn",
            model_name="dynunet_v2",
            checkpoint_path="/v2",
            dice_score=0.90,
            cldice_score=0.80,
            mlflow_run_id="new",
        )
        reg.register(old)
        reg.register(new)
        assert reg.get_champion("cnn").model_name == "dynunet_v2"

    def test_list_all_champions(self) -> None:
        from minivess.serving.champion_registry import (
            ChampionEntry,
            ChampionRegistry,
        )

        reg = ChampionRegistry()
        reg.register(
            ChampionEntry(
                model_family="cnn",
                model_name="d",
                checkpoint_path="/d",
                dice_score=0.8,
                cldice_score=0.6,
                mlflow_run_id="1",
            )
        )
        reg.register(
            ChampionEntry(
                model_family="foundation",
                model_name="s",
                checkpoint_path="/s",
                dice_score=0.9,
                cldice_score=0.7,
                mlflow_run_id="2",
            )
        )
        champions = reg.list_champions()
        assert len(champions) == 2


class TestChampionSelection:
    """Test data-driven champion selection logic."""

    def test_needs_human_decision_when_close(self) -> None:
        """When scores are within epsilon, flag for human review."""
        from minivess.serving.champion_registry import needs_human_decision

        scores = {"model_a": 0.8500, "model_b": 0.8502, "model_c": 0.8498}
        assert needs_human_decision(scores, epsilon=0.01) is True

    def test_no_human_decision_when_clear_winner(self) -> None:
        from minivess.serving.champion_registry import needs_human_decision

        scores = {"model_a": 0.90, "model_b": 0.70, "model_c": 0.50}
        assert needs_human_decision(scores, epsilon=0.01) is False

    def test_select_best_returns_top_model(self) -> None:
        from minivess.serving.champion_registry import select_best

        scores = {"model_a": 0.90, "model_b": 0.70, "model_c": 0.50}
        best = select_best(scores)
        assert best == "model_a"
