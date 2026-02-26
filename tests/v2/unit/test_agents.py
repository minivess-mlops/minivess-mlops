"""Tests for LangGraph agent orchestration, LLM wrapper, tracing, and evaluation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# T1: Training graph
# ---------------------------------------------------------------------------


class TestTrainingGraph:
    """Test LangGraph training pipeline graph."""

    def test_training_graph_compiles(self) -> None:
        """build_training_graph should return a compiled StateGraph."""
        from minivess.agents.graph import build_training_graph

        graph = build_training_graph()
        # Should be a compiled LangGraph (has invoke method)
        assert hasattr(graph, "invoke")

    def test_training_state_type(self) -> None:
        """TrainingState should be a TypedDict with expected keys."""
        from minivess.agents.graph import TrainingState

        # TypedDict has __annotations__
        annotations = TrainingState.__annotations__
        assert "model_name" in annotations
        assert "dataset" in annotations
        assert "status" in annotations
        assert "results" in annotations

    def test_training_graph_runs_to_completion(self) -> None:
        """Full graph run with mocked pipeline components should complete."""
        from minivess.agents.graph import build_training_graph

        graph = build_training_graph()
        initial_state = {
            "model_name": "test_model",
            "dataset": "synthetic",
            "status": "pending",
            "results": {},
            "messages": [],
            "metrics_pass": True,
        }
        result = graph.invoke(initial_state)
        assert result["status"] == "completed"

    def test_training_graph_skips_register_on_bad_metrics(self) -> None:
        """When metrics fail, graph should skip register and go to notify."""
        from minivess.agents.graph import build_training_graph

        graph = build_training_graph()
        initial_state = {
            "model_name": "test_model",
            "dataset": "synthetic",
            "status": "pending",
            "results": {},
            "messages": [],
            "metrics_pass": False,
        }
        result = graph.invoke(initial_state)
        assert result["status"] == "completed"
        assert "skipped_registration" in result["results"]

    def test_training_graph_nodes_add_messages(self) -> None:
        """Each node should append a message to the state."""
        from minivess.agents.graph import build_training_graph

        graph = build_training_graph()
        initial_state = {
            "model_name": "test_model",
            "dataset": "synthetic",
            "status": "pending",
            "results": {},
            "messages": [],
            "metrics_pass": True,
        }
        result = graph.invoke(initial_state)
        assert len(result["messages"]) >= 3  # At least prepare, train, evaluate


# ---------------------------------------------------------------------------
# T2: Experiment comparison agent
# ---------------------------------------------------------------------------


class TestComparisonGraph:
    """Test experiment comparison agent."""

    def test_comparison_graph_compiles(self) -> None:
        """build_comparison_graph should return a compiled StateGraph."""
        from minivess.agents.comparison import build_comparison_graph

        graph = build_comparison_graph()
        assert hasattr(graph, "invoke")

    def test_comparison_state_type(self) -> None:
        """ComparisonState should have expected keys."""
        from minivess.agents.comparison import ComparisonState

        annotations = ComparisonState.__annotations__
        assert "experiment_name" in annotations
        assert "summary" in annotations

    def test_comparison_fetches_runs(self) -> None:
        """fetch_runs node should populate runs_data in state."""
        from minivess.agents.comparison import fetch_runs_node

        state: dict[str, Any] = {
            "experiment_name": "test_experiment",
            "query": "best models",
            "summary": "",
            "runs_data": [],
            "messages": [],
        }
        result = fetch_runs_node(state)
        assert "runs_data" in result

    @patch("minivess.agents.llm.call_llm")
    def test_comparison_summarise_mocked_llm(self, mock_llm: MagicMock) -> None:
        """Summarise node should call LLM and return summary."""
        from minivess.agents.comparison import summarise_node

        mock_llm.return_value = "Model A outperforms Model B by 5% Dice."
        state: dict[str, Any] = {
            "experiment_name": "test_experiment",
            "query": "compare top models",
            "summary": "",
            "runs_data": [
                {"run_id": "a", "metric_val_dice": 0.85},
                {"run_id": "b", "metric_val_dice": 0.80},
            ],
            "messages": [],
        }
        result = summarise_node(state)
        assert len(result["summary"]) > 0
        mock_llm.assert_called_once()


# ---------------------------------------------------------------------------
# T3: LiteLLM provider wrapper
# ---------------------------------------------------------------------------


class TestLLMWrapper:
    """Test LiteLLM provider abstraction."""

    @patch("litellm.completion")
    def test_call_llm_returns_string(self, mock_completion: MagicMock) -> None:
        from minivess.agents.llm import call_llm

        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Hello from LLM"))]
        )
        result = call_llm("What is 2+2?")
        assert result == "Hello from LLM"

    @patch("litellm.completion")
    def test_call_llm_passes_model(self, mock_completion: MagicMock) -> None:
        from minivess.agents.llm import call_llm

        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )
        call_llm("test", model="anthropic:claude-sonnet-4-6")
        call_args = mock_completion.call_args
        assert call_args.kwargs["model"] == "anthropic:claude-sonnet-4-6"

    @patch("litellm.completion")
    def test_call_llm_structured_returns_dict(self, mock_completion: MagicMock) -> None:
        import json

        from minivess.agents.llm import call_llm_structured

        mock_completion.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content=json.dumps({"answer": 42, "confidence": 0.95})
                    )
                )
            ]
        )
        result = call_llm_structured("structured query")
        assert isinstance(result, dict)
        assert result["answer"] == 42

    @patch("litellm.completion")
    def test_call_llm_default_model(self, mock_completion: MagicMock) -> None:
        from minivess.agents.llm import DEFAULT_MODEL, call_llm

        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )
        call_llm("test")
        call_args = mock_completion.call_args
        assert call_args.kwargs["model"] == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# T4: Langfuse tracing
# ---------------------------------------------------------------------------


class TestTracedGraphRun:
    """Test Langfuse tracing wrapper for graph execution."""

    def test_traced_run_returns_state(self) -> None:
        """traced_graph_run should return the final state."""
        from minivess.agents.tracing import traced_graph_run

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"status": "completed", "messages": []}
        result = traced_graph_run(mock_graph, {"status": "pending"}, trace_name="test")
        assert result["status"] == "completed"

    def test_traced_run_calls_graph_invoke(self) -> None:
        """traced_graph_run should call graph.invoke with the state."""
        from minivess.agents.tracing import traced_graph_run

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"status": "done"}
        state = {"status": "pending", "data": "test"}
        traced_graph_run(mock_graph, state, trace_name="test_trace")
        mock_graph.invoke.assert_called_once_with(state)

    @patch("minivess.agents.tracing._get_langfuse_client")
    def test_traced_run_creates_trace(self, mock_get_client: MagicMock) -> None:
        """Should create a Langfuse trace when client is available."""
        from minivess.agents.tracing import traced_graph_run

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"status": "done"}
        traced_graph_run(mock_graph, {"status": "pending"}, trace_name="traced_test")
        mock_client.trace.assert_called_once()


# ---------------------------------------------------------------------------
# T5: Braintrust evaluation suites
# ---------------------------------------------------------------------------


class TestEvalSuites:
    """Test Braintrust evaluation suite configuration."""

    def test_segmentation_eval_suite_scorers(self) -> None:
        from minivess.agents.evaluation import build_segmentation_eval_suite

        suite = build_segmentation_eval_suite()
        assert "dice_score" in suite.scorers
        assert "surface_dice" in suite.scorers
        assert "calibration_ece" in suite.scorers
        assert len(suite.scorers) >= 5

    def test_agent_eval_suite_scorers(self) -> None:
        from minivess.agents.evaluation import build_agent_eval_suite

        suite = build_agent_eval_suite()
        assert "task_completion" in suite.scorers
        assert "tool_usage_efficiency" in suite.scorers
        assert len(suite.scorers) >= 3

    def test_eval_suite_add_scorer_dedup(self) -> None:
        from minivess.agents.evaluation import EvalSuite

        suite = EvalSuite(name="test")
        suite.add_scorer("dice")
        suite.add_scorer("dice")  # Duplicate
        assert suite.scorers.count("dice") == 1

    def test_eval_suite_to_config(self) -> None:
        from minivess.agents.evaluation import EvalSuite

        suite = EvalSuite(name="test", description="desc", scorers=["a", "b"])
        config = suite.to_config()
        assert config["name"] == "test"
        assert config["description"] == "desc"
        assert config["scorers"] == ["a", "b"]

    def test_eval_result_fields(self) -> None:
        from minivess.agents.evaluation import EvalResult

        result = EvalResult(
            input_id="sample_001",
            scores={"dice": 0.85},
            metadata={"model": "segresnet"},
        )
        assert result.input_id == "sample_001"
        assert result.scores["dice"] == 0.85
