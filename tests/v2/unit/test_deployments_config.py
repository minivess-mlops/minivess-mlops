"""Tests for T-03: deployment config completeness.

Verifies that FLOW_WORK_POOL_MAP and FLOW_IMAGE_MAP have identical key sets
and that all required flows (including post_training and annotation) are present.
"""

from __future__ import annotations


class TestFlowWorkPoolMap:
    def test_post_training_in_work_pool_map(self) -> None:
        from minivess.orchestration.deployments import FLOW_WORK_POOL_MAP

        assert "post_training" in FLOW_WORK_POOL_MAP, (
            "post_training flow is missing from FLOW_WORK_POOL_MAP"
        )

    def test_annotation_in_work_pool_map(self) -> None:
        from minivess.orchestration.deployments import FLOW_WORK_POOL_MAP

        assert "annotation" in FLOW_WORK_POOL_MAP, (
            "annotation flow is missing from FLOW_WORK_POOL_MAP"
        )

    def test_post_training_uses_cpu_pool(self) -> None:
        from minivess.orchestration.deployments import FLOW_WORK_POOL_MAP

        assert FLOW_WORK_POOL_MAP["post_training"] == "cpu-pool"

    def test_annotation_uses_cpu_pool(self) -> None:
        from minivess.orchestration.deployments import FLOW_WORK_POOL_MAP

        assert FLOW_WORK_POOL_MAP["annotation"] == "cpu-pool"


class TestFlowImageMap:
    def test_post_training_in_image_map(self) -> None:
        from minivess.orchestration.deployments import FLOW_IMAGE_MAP

        assert "post_training" in FLOW_IMAGE_MAP, (
            "post_training flow is missing from FLOW_IMAGE_MAP"
        )

    def test_annotation_in_image_map(self) -> None:
        from minivess.orchestration.deployments import FLOW_IMAGE_MAP

        assert "annotation" in FLOW_IMAGE_MAP, (
            "annotation flow is missing from FLOW_IMAGE_MAP"
        )

    def test_post_training_image_correct(self) -> None:
        from minivess.orchestration.deployments import FLOW_IMAGE_MAP

        assert FLOW_IMAGE_MAP["post_training"] == "minivess-post-training:latest"

    def test_annotation_image_correct(self) -> None:
        from minivess.orchestration.deployments import FLOW_IMAGE_MAP

        assert FLOW_IMAGE_MAP["annotation"] == "minivess-annotation:latest"


class TestMapConsistency:
    def test_work_pool_map_and_image_map_same_keys(self) -> None:
        from minivess.orchestration.deployments import (
            FLOW_IMAGE_MAP,
            FLOW_WORK_POOL_MAP,
        )

        pool_keys = set(FLOW_WORK_POOL_MAP.keys())
        image_keys = set(FLOW_IMAGE_MAP.keys())
        only_in_pool = pool_keys - image_keys
        only_in_image = image_keys - pool_keys
        assert not only_in_pool, (
            f"Keys in FLOW_WORK_POOL_MAP but not FLOW_IMAGE_MAP: {only_in_pool}"
        )
        assert not only_in_image, (
            f"Keys in FLOW_IMAGE_MAP but not FLOW_WORK_POOL_MAP: {only_in_image}"
        )

    def test_all_core_flows_present(self) -> None:
        from minivess.orchestration.deployments import FLOW_WORK_POOL_MAP

        required = {
            "acquisition",
            "data",
            "train",
            "post_training",
            "analyze",
            "deploy",
            "dashboard",
            "annotation",
        }
        missing = required - set(FLOW_WORK_POOL_MAP.keys())
        assert not missing, f"Missing flows in FLOW_WORK_POOL_MAP: {missing}"


class TestGetFlowDeploymentConfig:
    def test_post_training_config(self) -> None:
        from minivess.orchestration.deployments import get_flow_deployment_config

        config = get_flow_deployment_config("post_training")
        assert config["work_pool"] == "cpu-pool"
        assert config["image"] == "minivess-post-training:latest"
        assert config["flow_name"] == "post_training"

    def test_annotation_config(self) -> None:
        from minivess.orchestration.deployments import get_flow_deployment_config

        config = get_flow_deployment_config("annotation")
        assert config["work_pool"] == "cpu-pool"
        assert config["image"] == "minivess-annotation:latest"

    def test_train_uses_gpu_pool(self) -> None:
        from minivess.orchestration.deployments import get_flow_deployment_config

        config = get_flow_deployment_config("train")
        assert config["work_pool"] == "gpu-pool"
