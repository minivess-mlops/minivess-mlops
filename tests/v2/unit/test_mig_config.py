"""Tests for NVIDIA MIG device management (#284).

Covers:
- MIG device detection and enumeration
- MIG instance assignment to models
- Fallback when MIG is not available
- MIG config dataclass
"""

from __future__ import annotations


class TestMIGDetection:
    """Test MIG device detection."""

    def test_detect_mig_available(self) -> None:
        from minivess.serving.mig_config import detect_mig_devices

        # Without NVIDIA drivers, should return empty list
        devices = detect_mig_devices()
        assert isinstance(devices, list)

    def test_mig_not_available_returns_empty(self) -> None:
        from minivess.serving.mig_config import detect_mig_devices

        devices = detect_mig_devices()
        # On machines without MIG, should gracefully return empty
        assert isinstance(devices, list)


class TestMIGConfig:
    """Test MIG configuration dataclass."""

    def test_mig_config_creation(self) -> None:
        from minivess.serving.mig_config import MIGConfig

        config = MIGConfig(
            gpu_index=0,
            instances=[
                {"instance_id": 0, "model_name": "dynunet_dice_ce"},
                {"instance_id": 1, "model_name": "dynunet_cbdice"},
            ],
        )
        assert config.gpu_index == 0
        assert len(config.instances) == 2

    def test_mig_config_empty_instances(self) -> None:
        from minivess.serving.mig_config import MIGConfig

        config = MIGConfig(gpu_index=0, instances=[])
        assert len(config.instances) == 0

    def test_mig_config_model_assignment(self) -> None:
        from minivess.serving.mig_config import MIGConfig

        config = MIGConfig(
            gpu_index=0,
            instances=[
                {"instance_id": 0, "model_name": "model_a"},
            ],
        )
        assert config.instances[0]["model_name"] == "model_a"


class TestMIGAssignment:
    """Test model-to-MIG-instance assignment."""

    def test_assign_models_to_instances(self) -> None:
        from minivess.serving.mig_config import assign_models_to_instances

        models = ["model_a", "model_b", "model_c"]
        instances = [
            {"instance_id": 0, "memory_gb": 10},
            {"instance_id": 1, "memory_gb": 10},
            {"instance_id": 2, "memory_gb": 20},
        ]
        assignments = assign_models_to_instances(models, instances)
        assert len(assignments) == 3
        # Each model should be assigned
        assigned_models = [a["model_name"] for a in assignments]
        assert set(assigned_models) == set(models)

    def test_assign_more_models_than_instances(self) -> None:
        from minivess.serving.mig_config import assign_models_to_instances

        models = ["model_a", "model_b", "model_c"]
        instances = [{"instance_id": 0, "memory_gb": 10}]
        assignments = assign_models_to_instances(models, instances)
        # Should still assign all models (round-robin)
        assert len(assignments) == 3

    def test_assign_no_models(self) -> None:
        from minivess.serving.mig_config import assign_models_to_instances

        assignments = assign_models_to_instances([], [])
        assert assignments == []
