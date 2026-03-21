"""Tests for PostTrainingFlowResult dataclass.

TDD RED phase for T-13 (closes #414): post_training_flow() must return
a typed PostTrainingFlowResult, not a plain dict[str, Any].
"""

from __future__ import annotations

import dataclasses


class TestPostTrainingFlowResult:
    def test_result_is_dataclass(self) -> None:
        from minivess.orchestration.flows.post_training_flow import (
            PostTrainingFlowResult,
        )

        assert dataclasses.is_dataclass(PostTrainingFlowResult)

    def test_result_has_required_fields(self) -> None:
        from minivess.orchestration.flows.post_training_flow import (
            PostTrainingFlowResult,
        )

        field_names = {f.name for f in dataclasses.fields(PostTrainingFlowResult)}
        required = {
            "flow_name",
            "status",
            "mlflow_run_id",
            "upstream_training_run_id",
            "checkpoint_averaging_completed",
            "calibration_completed",
            "conformal_completed",
            "failed_operations",
        }
        missing = required - field_names
        assert not missing, f"PostTrainingFlowResult missing fields: {missing}"

    def test_result_default_values(self) -> None:
        from minivess.orchestration.flows.post_training_flow import (
            PostTrainingFlowResult,
        )

        result = PostTrainingFlowResult()
        assert result.flow_name == "post-training-flow"
        assert result.status == "completed"
        assert result.mlflow_run_id is None
        assert result.upstream_training_run_id is None
        assert result.checkpoint_averaging_completed is False
        assert result.calibration_completed is False
        assert result.conformal_completed is False
        assert result.failed_operations == []

    def test_result_failed_operations_is_mutable_list(self) -> None:
        """Each instance must get its own failed_operations list (no shared default)."""
        from minivess.orchestration.flows.post_training_flow import (
            PostTrainingFlowResult,
        )

        r1 = PostTrainingFlowResult()
        r2 = PostTrainingFlowResult()
        r1.failed_operations.append("test")
        assert r2.failed_operations == [], "Shared default_factory bug"

    def test_post_training_flow_returns_typed_result(self) -> None:
        """post_training_flow() return annotation must be PostTrainingFlowResult."""
        import typing

        from minivess.orchestration.flows.post_training_flow import (
            PostTrainingFlowResult,
            post_training_flow,
        )

        # Check the __wrapped__ annotation if Prefect wraps the function
        fn = getattr(post_training_flow, "__wrapped__", post_training_flow)
        # get_type_hints() resolves PEP 563 string annotations to actual types
        hints = typing.get_type_hints(fn)
        ret = hints.get("return")
        assert ret is PostTrainingFlowResult, (
            f"post_training_flow must return PostTrainingFlowResult, got {ret!r}"
        )
