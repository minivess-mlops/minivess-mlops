"""Tests for orchestration constants module.

TDD RED phase for T-01 (closes #413): All experiment and flow name strings
must be defined as constants, not scattered inline across flow files.
"""

from __future__ import annotations


class TestExperimentNameConstants:
    def test_experiment_names_are_strings(self) -> None:
        from minivess.orchestration.constants import (
            EXPERIMENT_DASHBOARD,
            EXPERIMENT_DATA,
            EXPERIMENT_DEPLOYMENT,
            EXPERIMENT_EVALUATION,
            EXPERIMENT_HPO,
            EXPERIMENT_POST_TRAINING,
            EXPERIMENT_TRAINING,
        )

        for const in [
            EXPERIMENT_TRAINING,
            EXPERIMENT_DATA,
            EXPERIMENT_EVALUATION,
            EXPERIMENT_POST_TRAINING,
            EXPERIMENT_DEPLOYMENT,
            EXPERIMENT_DASHBOARD,
            EXPERIMENT_HPO,
        ]:
            assert isinstance(const, str), f"Expected str, got {type(const)}"
            assert len(const) > 0, "Constant must be non-empty"

    def test_no_unexpected_duplicate_values(self) -> None:
        """Experiment names must be unique EXCEPT intentional sharing.

        EXPERIMENT_POST_TRAINING == EXPERIMENT_TRAINING per synthesis Part 2.3:
        post-training logs to the SAME experiment so Analysis Flow discovers
        all variants (training + post-training) in one query.
        """
        from minivess.orchestration.constants import (
            EXPERIMENT_DASHBOARD,
            EXPERIMENT_DATA,
            EXPERIMENT_DEPLOYMENT,
            EXPERIMENT_EVALUATION,
            EXPERIMENT_HPO,
            EXPERIMENT_POST_TRAINING,
            EXPERIMENT_TRAINING,
        )

        # Post-training intentionally shares with training (synthesis Part 2.3)
        assert EXPERIMENT_POST_TRAINING == EXPERIMENT_TRAINING

        # All OTHER experiment names must be unique
        non_shared = [
            EXPERIMENT_TRAINING,
            EXPERIMENT_DATA,
            EXPERIMENT_EVALUATION,
            EXPERIMENT_DEPLOYMENT,
            EXPERIMENT_DASHBOARD,
            EXPERIMENT_HPO,
        ]
        assert len(non_shared) == len(set(non_shared)), (
            "Non-shared experiment names must be unique"
        )


class TestFlowNameConstants:
    def test_flow_names_are_lowercase_hyphen(self) -> None:
        from minivess.orchestration.constants import (
            FLOW_NAME_ANALYSIS,
            FLOW_NAME_ANNOTATION,
            FLOW_NAME_DASHBOARD,
            FLOW_NAME_DATA,
            FLOW_NAME_DEPLOY,
            FLOW_NAME_HPO,
            FLOW_NAME_TRAIN,
        )

        for name in [
            FLOW_NAME_TRAIN,
            FLOW_NAME_DATA,
            FLOW_NAME_ANALYSIS,
            FLOW_NAME_DEPLOY,
            FLOW_NAME_DASHBOARD,
            FLOW_NAME_HPO,
            FLOW_NAME_ANNOTATION,
        ]:
            assert isinstance(name, str), f"Expected str, got {type(name)}"
            assert name == name.lower(), f"Flow name must be lowercase: {name!r}"
            assert " " not in name, f"Flow name must not have spaces: {name!r}"
            assert "_" not in name, f"Flow name must not have underscores: {name!r}"
            assert len(name) > 0, "Flow name must be non-empty"

    def test_all_expected_constants_exist(self) -> None:
        import minivess.orchestration.constants as c

        required = [
            "EXPERIMENT_TRAINING",
            "EXPERIMENT_DATA",
            "EXPERIMENT_EVALUATION",
            "EXPERIMENT_POST_TRAINING",
            "EXPERIMENT_DEPLOYMENT",
            "EXPERIMENT_DASHBOARD",
            "EXPERIMENT_HPO",
            "FLOW_NAME_TRAIN",
            "FLOW_NAME_DATA",
            "FLOW_NAME_ANALYSIS",
            "FLOW_NAME_DEPLOY",
            "FLOW_NAME_DASHBOARD",
            "FLOW_NAME_HPO",
            "FLOW_NAME_ANNOTATION",
        ]
        missing = [name for name in required if not hasattr(c, name)]
        assert not missing, f"Missing constants: {missing}"

    def test_constants_importable_from_package(self) -> None:
        from minivess.orchestration import constants  # noqa: F401

        assert hasattr(constants, "EXPERIMENT_TRAINING")
        assert hasattr(constants, "FLOW_NAME_TRAIN")
