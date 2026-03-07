"""Tests for ONNX export failure propagation in deploy_flow (T-11, closes #415).

TDD RED phase: ONNX export failures must not be silently swallowed.
DeployResult must expose failed_operations. Plugin failures in
post_training_flow remain best-effort (non-blocking).
"""

from __future__ import annotations

import dataclasses


class TestDeployResultHasFailedOperations:
    def test_deploy_result_has_failed_operations_field(self) -> None:
        """DeployResult must have a failed_operations field."""
        from minivess.orchestration.deploy_flow import DeployResult

        fields = {f.name for f in dataclasses.fields(DeployResult)}
        assert "failed_operations" in fields, (
            "DeployResult missing failed_operations field — "
            "ONNX export failures cannot be communicated to caller"
        )

    def test_failed_operations_default_is_empty_list(self) -> None:
        """failed_operations must default to empty list."""
        from minivess.orchestration.deploy_flow import DeployResult

        # Construct with minimal required fields
        result = DeployResult(
            champions=[],
            onnx_paths={},
            bento_tags={},
            artifacts_dir=__import__("pathlib").Path("/tmp"),
            promotion_results={},
        )
        assert result.failed_operations == []

    def test_failed_operations_is_list_of_str(self) -> None:
        """failed_operations must be a list of strings."""
        from minivess.orchestration.deploy_flow import DeployResult

        fields = {f.name: f for f in dataclasses.fields(DeployResult)}
        fo_field = fields["failed_operations"]
        # Default factory must produce list
        default_val = fo_field.default_factory()  # type: ignore[misc]
        assert isinstance(default_val, list)


class TestPostTrainingPluginBestEffort:
    def test_post_training_result_has_failed_operations_field(self) -> None:
        """PostTrainingFlowResult already has failed_operations (best-effort plugins)."""
        from minivess.orchestration.flows.post_training_flow import (
            PostTrainingFlowResult,
        )

        fields = {f.name for f in dataclasses.fields(PostTrainingFlowResult)}
        assert "failed_operations" in fields, (
            "PostTrainingFlowResult missing failed_operations field"
        )

    def test_post_training_result_failed_operations_default_empty(self) -> None:
        """PostTrainingFlowResult.failed_operations defaults to empty list."""
        from minivess.orchestration.flows.post_training_flow import (
            PostTrainingFlowResult,
        )

        result = PostTrainingFlowResult()
        assert result.failed_operations == []
