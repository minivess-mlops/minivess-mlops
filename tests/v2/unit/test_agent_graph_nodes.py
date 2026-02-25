"""Tests for individual agent graph node functions (R5.6).

Unit tests for each node function in isolation, covering state
transformations, routing logic, and edge cases.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# T1: Training graph – prepare_data_node
# ---------------------------------------------------------------------------


class TestPrepareDataNode:
    """Test prepare_data_node in isolation."""

    def test_sets_status_to_data_ready(self) -> None:
        """prepare_data_node should set status to 'data_ready'."""
        from minivess.agents.graph import prepare_data_node

        state: dict[str, Any] = {
            "model_name": "test",
            "dataset": "synthetic",
            "status": "pending",
            "results": {},
            "messages": [],
            "metrics_pass": False,
        }
        result = prepare_data_node(state)
        assert result["status"] == "data_ready"

    def test_appends_message(self) -> None:
        """prepare_data_node should append a message about the dataset."""
        from minivess.agents.graph import prepare_data_node

        state: dict[str, Any] = {
            "model_name": "test",
            "dataset": "minivess",
            "status": "pending",
            "results": {},
            "messages": ["initial"],
            "metrics_pass": False,
        }
        result = prepare_data_node(state)
        assert len(result["messages"]) == 2
        assert "minivess" in result["messages"][-1]

    def test_preserves_existing_messages(self) -> None:
        """prepare_data_node should not drop existing messages."""
        from minivess.agents.graph import prepare_data_node

        state: dict[str, Any] = {
            "model_name": "test",
            "dataset": "ds",
            "status": "pending",
            "results": {},
            "messages": ["msg1", "msg2"],
            "metrics_pass": False,
        }
        result = prepare_data_node(state)
        assert result["messages"][0] == "msg1"
        assert result["messages"][1] == "msg2"


# ---------------------------------------------------------------------------
# T2: Training graph – train_node
# ---------------------------------------------------------------------------


class TestTrainNode:
    """Test train_node in isolation."""

    def test_sets_status_to_trained(self) -> None:
        """train_node should set status to 'trained'."""
        from minivess.agents.graph import train_node

        state: dict[str, Any] = {
            "model_name": "segresnet",
            "dataset": "synthetic",
            "status": "data_ready",
            "results": {},
            "messages": [],
            "metrics_pass": False,
        }
        result = train_node(state)
        assert result["status"] == "trained"

    def test_populates_train_loss(self) -> None:
        """train_node should add train_loss to results."""
        from minivess.agents.graph import train_node

        state: dict[str, Any] = {
            "model_name": "segresnet",
            "dataset": "synthetic",
            "status": "data_ready",
            "results": {},
            "messages": [],
            "metrics_pass": False,
        }
        result = train_node(state)
        assert "train_loss" in result["results"]
        assert isinstance(result["results"]["train_loss"], float)

    def test_populates_val_loss(self) -> None:
        """train_node should add val_loss to results."""
        from minivess.agents.graph import train_node

        state: dict[str, Any] = {
            "model_name": "segresnet",
            "dataset": "synthetic",
            "status": "data_ready",
            "results": {},
            "messages": [],
            "metrics_pass": False,
        }
        result = train_node(state)
        assert "val_loss" in result["results"]

    def test_appends_model_name_to_messages(self) -> None:
        """train_node should mention model name in message."""
        from minivess.agents.graph import train_node

        state: dict[str, Any] = {
            "model_name": "dynunet",
            "dataset": "synthetic",
            "status": "data_ready",
            "results": {},
            "messages": [],
            "metrics_pass": False,
        }
        result = train_node(state)
        assert any("dynunet" in m for m in result["messages"])


# ---------------------------------------------------------------------------
# T3: Training graph – evaluate_node
# ---------------------------------------------------------------------------


class TestEvaluateNode:
    """Test evaluate_node in isolation."""

    def test_sets_status_to_evaluated(self) -> None:
        """evaluate_node should set status to 'evaluated'."""
        from minivess.agents.graph import evaluate_node

        state: dict[str, Any] = {
            "model_name": "segresnet",
            "dataset": "synthetic",
            "status": "trained",
            "results": {"train_loss": 0.3},
            "messages": [],
            "metrics_pass": False,
        }
        result = evaluate_node(state)
        assert result["status"] == "evaluated"

    def test_adds_val_dice_to_results(self) -> None:
        """evaluate_node should add val_dice metric."""
        from minivess.agents.graph import evaluate_node

        state: dict[str, Any] = {
            "model_name": "segresnet",
            "dataset": "synthetic",
            "status": "trained",
            "results": {},
            "messages": [],
            "metrics_pass": False,
        }
        result = evaluate_node(state)
        assert "val_dice" in result["results"]
        assert isinstance(result["results"]["val_dice"], float)

    def test_preserves_existing_results(self) -> None:
        """evaluate_node should not overwrite prior results."""
        from minivess.agents.graph import evaluate_node

        state: dict[str, Any] = {
            "model_name": "segresnet",
            "dataset": "synthetic",
            "status": "trained",
            "results": {"train_loss": 0.3, "val_loss": 0.4},
            "messages": [],
            "metrics_pass": False,
        }
        result = evaluate_node(state)
        assert result["results"]["train_loss"] == 0.3
        assert result["results"]["val_loss"] == 0.4
        assert "val_dice" in result["results"]


# ---------------------------------------------------------------------------
# T4: Training graph – register_node
# ---------------------------------------------------------------------------


class TestRegisterNode:
    """Test register_node in isolation."""

    def test_sets_status_to_completed(self) -> None:
        """register_node should set status to 'completed'."""
        from minivess.agents.graph import register_node

        state: dict[str, Any] = {
            "model_name": "segresnet",
            "dataset": "synthetic",
            "status": "evaluated",
            "results": {},
            "messages": [],
            "metrics_pass": True,
        }
        result = register_node(state)
        assert result["status"] == "completed"

    def test_sets_registered_flag(self) -> None:
        """register_node should set registered=True in results."""
        from minivess.agents.graph import register_node

        state: dict[str, Any] = {
            "model_name": "segresnet",
            "dataset": "synthetic",
            "status": "evaluated",
            "results": {},
            "messages": [],
            "metrics_pass": True,
        }
        result = register_node(state)
        assert result["results"]["registered"] is True

    def test_appends_model_name_to_messages(self) -> None:
        """register_node should mention model name in message."""
        from minivess.agents.graph import register_node

        state: dict[str, Any] = {
            "model_name": "vista3d",
            "dataset": "synthetic",
            "status": "evaluated",
            "results": {},
            "messages": [],
            "metrics_pass": True,
        }
        result = register_node(state)
        assert any("vista3d" in m for m in result["messages"])


# ---------------------------------------------------------------------------
# T5: Training graph – notify_node
# ---------------------------------------------------------------------------


class TestNotifyNode:
    """Test notify_node in isolation."""

    def test_sets_status_to_completed(self) -> None:
        """notify_node should set status to 'completed'."""
        from minivess.agents.graph import notify_node

        state: dict[str, Any] = {
            "model_name": "segresnet",
            "dataset": "synthetic",
            "status": "evaluated",
            "results": {},
            "messages": [],
            "metrics_pass": True,
        }
        result = notify_node(state)
        assert result["status"] == "completed"

    def test_skipped_registration_when_metrics_fail(self) -> None:
        """notify_node should mark skipped_registration when metrics_pass=False."""
        from minivess.agents.graph import notify_node

        state: dict[str, Any] = {
            "model_name": "segresnet",
            "dataset": "synthetic",
            "status": "evaluated",
            "results": {},
            "messages": [],
            "metrics_pass": False,
        }
        result = notify_node(state)
        assert result["results"]["skipped_registration"] is True

    def test_no_skipped_registration_when_metrics_pass(self) -> None:
        """notify_node should NOT set skipped_registration when metrics_pass=True."""
        from minivess.agents.graph import notify_node

        state: dict[str, Any] = {
            "model_name": "segresnet",
            "dataset": "synthetic",
            "status": "evaluated",
            "results": {},
            "messages": [],
            "metrics_pass": True,
        }
        result = notify_node(state)
        assert "skipped_registration" not in result["results"]


# ---------------------------------------------------------------------------
# T6: Routing logic – _route_after_evaluate
# ---------------------------------------------------------------------------


class TestRouteAfterEvaluate:
    """Test conditional routing after evaluation."""

    def test_routes_to_register_when_pass(self) -> None:
        """Should route to 'register' when metrics_pass=True."""
        from minivess.agents.graph import _route_after_evaluate

        state: dict[str, Any] = {
            "model_name": "segresnet",
            "dataset": "synthetic",
            "status": "evaluated",
            "results": {},
            "messages": [],
            "metrics_pass": True,
        }
        assert _route_after_evaluate(state) == "register"

    def test_routes_to_notify_when_fail(self) -> None:
        """Should route to 'notify' when metrics_pass=False."""
        from minivess.agents.graph import _route_after_evaluate

        state: dict[str, Any] = {
            "model_name": "segresnet",
            "dataset": "synthetic",
            "status": "evaluated",
            "results": {},
            "messages": [],
            "metrics_pass": False,
        }
        assert _route_after_evaluate(state) == "notify"

    def test_routes_to_notify_when_missing(self) -> None:
        """Should default to 'notify' when metrics_pass is missing."""
        from minivess.agents.graph import _route_after_evaluate

        state: dict[str, Any] = {
            "model_name": "segresnet",
            "dataset": "synthetic",
            "status": "evaluated",
            "results": {},
            "messages": [],
        }
        assert _route_after_evaluate(state) == "notify"


# ---------------------------------------------------------------------------
# T7: Comparison graph – analyse_runs_node
# ---------------------------------------------------------------------------


class TestAnalyseRunsNode:
    """Test analyse_runs_node in isolation."""

    def test_analyses_run_metrics(self) -> None:
        """analyse_runs_node should compute best Dice from runs."""
        from minivess.agents.comparison import analyse_runs_node

        state: dict[str, Any] = {
            "experiment_name": "test",
            "query": "best model",
            "summary": "",
            "runs_data": [
                {"run_id": "a", "metric_val_dice": 0.85},
                {"run_id": "b", "metric_val_dice": 0.78},
            ],
            "messages": [],
        }
        result = analyse_runs_node(state)
        assert any("0.850" in m for m in result["messages"])

    def test_handles_empty_runs(self) -> None:
        """analyse_runs_node should handle empty runs_data."""
        from minivess.agents.comparison import analyse_runs_node

        state: dict[str, Any] = {
            "experiment_name": "test",
            "query": "best model",
            "summary": "",
            "runs_data": [],
            "messages": [],
        }
        result = analyse_runs_node(state)
        assert any("No runs" in m for m in result["messages"])

    def test_preserves_existing_messages(self) -> None:
        """analyse_runs_node should append to existing messages."""
        from minivess.agents.comparison import analyse_runs_node

        state: dict[str, Any] = {
            "experiment_name": "test",
            "query": "best model",
            "summary": "",
            "runs_data": [{"run_id": "a", "metric_val_dice": 0.9}],
            "messages": ["prior"],
        }
        result = analyse_runs_node(state)
        assert result["messages"][0] == "prior"
        assert len(result["messages"]) == 2


# ---------------------------------------------------------------------------
# T8: Comparison graph – fetch_runs_node with pre-populated data
# ---------------------------------------------------------------------------


class TestFetchRunsNodeEdgeCases:
    """Test fetch_runs_node edge cases."""

    def test_uses_synthetic_data_when_empty(self) -> None:
        """fetch_runs_node should generate synthetic runs when runs_data is empty."""
        from minivess.agents.comparison import fetch_runs_node

        state: dict[str, Any] = {
            "experiment_name": "test",
            "query": "query",
            "summary": "",
            "runs_data": [],
            "messages": [],
        }
        result = fetch_runs_node(state)
        assert len(result["runs_data"]) >= 2

    def test_preserves_existing_runs_data(self) -> None:
        """fetch_runs_node should keep existing runs_data if present."""
        from minivess.agents.comparison import fetch_runs_node

        existing_runs = [{"run_id": "custom", "metric_val_dice": 0.95}]
        state: dict[str, Any] = {
            "experiment_name": "test",
            "query": "query",
            "summary": "",
            "runs_data": existing_runs,
            "messages": [],
        }
        result = fetch_runs_node(state)
        assert result["runs_data"] == existing_runs
