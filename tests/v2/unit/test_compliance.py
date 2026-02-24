from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class TestAuditTrail:
    def test_log_event(self) -> None:
        from minivess.compliance.audit import AuditTrail

        trail = AuditTrail()
        entry = trail.log_event("TEST", "test event", actor="unit_test")
        assert entry.event_type == "TEST"
        assert entry.actor == "unit_test"
        assert len(trail.entries) == 1

    def test_log_data_access(self) -> None:
        from minivess.compliance.audit import AuditTrail

        trail = AuditTrail()
        entry = trail.log_data_access("minivess", ["file1.nii.gz", "file2.nii.gz"])
        assert entry.event_type == "DATA_ACCESS"
        assert entry.data_hash is not None
        assert len(entry.data_hash) == 64  # SHA-256 hex

    def test_log_model_training(self) -> None:
        from minivess.compliance.audit import AuditTrail

        trail = AuditTrail()
        entry = trail.log_model_training("segresnet", {"epochs": 100})
        assert entry.event_type == "MODEL_TRAINING"

    def test_save_and_load(self, tmp_path: Path) -> None:
        from minivess.compliance.audit import AuditTrail

        trail = AuditTrail()
        trail.log_event("E1", "first")
        trail.log_event("E2", "second")

        save_path = tmp_path / "audit.json"
        trail.save(save_path)
        assert save_path.exists()

        loaded = AuditTrail.load(save_path)
        assert len(loaded.entries) == 2
        assert loaded.entries[0].event_type == "E1"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        from minivess.compliance.audit import AuditTrail

        trail = AuditTrail()
        trail.log_event("TEST", "test")
        nested = tmp_path / "a" / "b" / "audit.json"
        trail.save(nested)
        assert nested.exists()


class TestModelCard:
    def test_creation(self) -> None:
        from minivess.compliance.model_card import ModelCard

        card = ModelCard(model_name="SegResNet", model_version="1.0")
        assert card.model_name == "SegResNet"
        assert card.model_type == "3D Segmentation"

    def test_to_markdown(self) -> None:
        from minivess.compliance.model_card import ModelCard

        card = ModelCard(
            model_name="SegResNet",
            model_version="1.0",
            metrics={"dice": 0.85, "f1": 0.82},
            authors=["Researcher A"],
        )
        md = card.to_markdown()
        assert "# Model Card: SegResNet v1.0" in md
        assert "0.8500" in md
        assert "Researcher A" in md

    def test_to_markdown_without_optional(self) -> None:
        from minivess.compliance.model_card import ModelCard

        card = ModelCard(model_name="Test", model_version="0.1")
        md = card.to_markdown()
        assert "Test" in md


class TestAgentGraph:
    def test_training_graph_compiles(self) -> None:
        from minivess.agents.graph import build_training_graph

        graph = build_training_graph()
        assert hasattr(graph, "invoke")

    def test_training_state_has_expected_keys(self) -> None:
        from minivess.agents.graph import TrainingState

        annotations = TrainingState.__annotations__
        assert "model_name" in annotations
        assert "status" in annotations
        assert "results" in annotations


class TestEvalSuite:
    def test_segmentation_suite(self) -> None:
        from minivess.agents.evaluation import build_segmentation_eval_suite

        suite = build_segmentation_eval_suite()
        assert suite.name == "minivess-segmentation-eval"
        assert "dice_score" in suite.scorers

    def test_agent_suite(self) -> None:
        from minivess.agents.evaluation import build_agent_eval_suite

        suite = build_agent_eval_suite()
        assert "task_completion" in suite.scorers

    def test_to_config(self) -> None:
        from minivess.agents.evaluation import EvalSuite

        suite = EvalSuite(name="test")
        suite.add_scorer("metric_a")
        suite.add_scorer("metric_a")  # duplicate
        config = suite.to_config()
        assert config["scorers"] == ["metric_a"]

    def test_eval_result(self) -> None:
        from minivess.agents.evaluation import EvalResult

        result = EvalResult(input_id="sample_01", scores={"dice": 0.9})
        assert result.input_id == "sample_01"
