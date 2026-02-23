from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from minivess.config.models import (
    DataConfig,
    EnsembleConfig,
    EnsembleStrategy,
    ExperimentConfig,
    ModelConfig,
    ModelFamily,
    ServingConfig,
    TrainingConfig,
)

# --- Strategies ---

model_family_st = st.sampled_from(list(ModelFamily))
ensemble_strategy_st = st.sampled_from(list(EnsembleStrategy))

data_config_st = st.builds(
    DataConfig,
    dataset_name=st.text(min_size=1, max_size=50),
    patch_size=st.tuples(
        st.integers(min_value=16, max_value=256),
        st.integers(min_value=16, max_value=256),
        st.integers(min_value=8, max_value=128),
    ),
    num_workers=st.integers(min_value=0, max_value=16),
)

model_config_st = st.builds(
    ModelConfig,
    family=model_family_st,
    name=st.text(min_size=1, max_size=50),
    in_channels=st.integers(min_value=1, max_value=4),
    out_channels=st.integers(min_value=2, max_value=20),
    lora_rank=st.integers(min_value=1, max_value=64),
    lora_alpha=st.floats(min_value=1.0, max_value=128.0, allow_nan=False),
    lora_dropout=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)

training_config_st = st.builds(
    TrainingConfig,
    max_epochs=st.integers(min_value=1, max_value=1000),
    batch_size=st.integers(min_value=1, max_value=64),
    learning_rate=st.floats(min_value=1e-8, max_value=1.0, allow_nan=False, allow_infinity=False),
    weight_decay=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    optimizer=st.sampled_from(["adam", "adamw", "sgd", "lamb"]),
    seed=st.integers(min_value=0, max_value=2**31),
    num_folds=st.integers(min_value=1, max_value=10),
    warmup_epochs=st.integers(min_value=0, max_value=50),
    gradient_clip_val=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    early_stopping_patience=st.integers(min_value=0, max_value=100),
)


# --- Default construction tests ---

class TestDefaultConstruction:
    """All config models should construct with minimal required args."""

    def test_data_config_defaults(self) -> None:
        cfg = DataConfig(dataset_name="minivess")
        assert cfg.dataset_name == "minivess"
        assert cfg.patch_size == (128, 128, 32)
        assert cfg.num_workers == 4

    def test_model_config_defaults(self) -> None:
        cfg = ModelConfig(family=ModelFamily.MONAI_SEGRESNET, name="segresnet")
        assert cfg.in_channels == 1
        assert cfg.out_channels == 2
        assert cfg.pretrained is False

    def test_training_config_defaults(self) -> None:
        cfg = TrainingConfig()
        assert cfg.max_epochs == 100
        assert cfg.learning_rate == 1e-4
        assert cfg.optimizer == "adamw"

    def test_serving_config_defaults(self) -> None:
        cfg = ServingConfig()
        assert cfg.port == 3333
        assert cfg.host == "0.0.0.0"

    def test_ensemble_config_defaults(self) -> None:
        cfg = EnsembleConfig()
        assert cfg.strategy == EnsembleStrategy.MEAN
        assert cfg.weightwatcher_alpha_threshold == 5.0

    def test_experiment_config_minimal(self) -> None:
        cfg = ExperimentConfig(
            experiment_name="test",
            data=DataConfig(dataset_name="minivess"),
            model=ModelConfig(family=ModelFamily.MONAI_SEGRESNET, name="segresnet"),
            training=TrainingConfig(),
        )
        assert cfg.experiment_name == "test"


# --- Validation tests ---

class TestValidation:
    """Invalid values should be rejected."""

    def test_negative_epochs_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(max_epochs=-1)

    def test_zero_batch_size_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=0)

    def test_invalid_optimizer_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(optimizer="invalid_optimizer")

    def test_negative_learning_rate_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=-0.001)

    def test_port_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ServingConfig(port=99999)

    def test_lora_dropout_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(
                family=ModelFamily.SAM3_LORA,
                name="sam3",
                lora_dropout=1.5,
            )

    def test_conformal_alpha_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            EnsembleConfig(conformal_alpha=1.5)


# --- Property-based tests ---

class TestPropertyBased:
    """Hypothesis-driven property-based tests."""

    @given(cfg=data_config_st)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_data_config_roundtrip(self, cfg: DataConfig) -> None:
        """Serialize and deserialize should produce equivalent config."""
        dumped = cfg.model_dump()
        restored = DataConfig.model_validate(dumped)
        assert restored.dataset_name == cfg.dataset_name
        assert restored.patch_size == cfg.patch_size
        assert restored.num_workers == cfg.num_workers

    @given(cfg=model_config_st)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_model_config_roundtrip(self, cfg: ModelConfig) -> None:
        dumped = cfg.model_dump()
        restored = ModelConfig.model_validate(dumped)
        assert restored.family == cfg.family
        assert restored.name == cfg.name

    @given(cfg=training_config_st)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_training_config_roundtrip(self, cfg: TrainingConfig) -> None:
        dumped = cfg.model_dump()
        restored = TrainingConfig.model_validate(dumped)
        assert restored.max_epochs == cfg.max_epochs
        assert restored.optimizer == cfg.optimizer

    @given(
        family=model_family_st,
        strategy=ensemble_strategy_st,
    )
    def test_enum_values_survive_serialization(
        self, family: ModelFamily, strategy: EnsembleStrategy
    ) -> None:
        model_cfg = ModelConfig(family=family, name="test")
        assert ModelConfig.model_validate(model_cfg.model_dump()).family == family

        ensemble_cfg = EnsembleConfig(strategy=strategy)
        assert EnsembleConfig.model_validate(ensemble_cfg.model_dump()).strategy == strategy


# --- JSON serialization ---

class TestJsonSerialization:
    """Config models should serialize to/from JSON."""

    def test_experiment_config_json_roundtrip(self) -> None:
        cfg = ExperimentConfig(
            experiment_name="test_experiment",
            data=DataConfig(dataset_name="minivess"),
            model=ModelConfig(family=ModelFamily.MONAI_VISTA3D, name="vista3d"),
            training=TrainingConfig(max_epochs=50, learning_rate=5e-5),
            tags={"project": "minivess", "phase": "0"},
        )
        json_str = cfg.model_dump_json()
        restored = ExperimentConfig.model_validate_json(json_str)
        assert restored.experiment_name == cfg.experiment_name
        assert restored.model.family == ModelFamily.MONAI_VISTA3D
        assert restored.training.max_epochs == 50
        assert restored.tags["project"] == "minivess"
