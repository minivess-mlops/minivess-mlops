from __future__ import annotations

from hydra_zen import builds, make_config, store

from minivess.config.compute_profiles import _PROFILES, ComputeProfile
from minivess.config.models import (
    DataConfig,
    EnsembleConfig,
    EnsembleStrategy,
    ModelConfig,
    ModelFamily,
    TrainingConfig,
)

# --- Data configs ---
MinivessDataConf = builds(DataConfig, dataset_name="minivess", populate_full_signature=True)

# --- Model configs ---
SegResNetConf = builds(
    ModelConfig,
    family=ModelFamily.MONAI_SEGRESNET,
    name="segresnet",
    in_channels=1,
    out_channels=2,
    populate_full_signature=True,
)

SwinUNETRConf = builds(
    ModelConfig,
    family=ModelFamily.MONAI_SWINUNETR,
    name="swinunetr",
    in_channels=1,
    out_channels=2,
    populate_full_signature=True,
)

Vista3DConf = builds(
    ModelConfig,
    family=ModelFamily.MONAI_VISTA3D,
    name="vista3d",
    in_channels=1,
    out_channels=2,
    pretrained=True,
    populate_full_signature=True,
)

Sam3LoraConf = builds(
    ModelConfig,
    family=ModelFamily.SAM3_LORA,
    name="sam3_lora",
    in_channels=1,
    out_channels=2,
    lora_rank=16,
    lora_alpha=32.0,
    lora_dropout=0.1,
    populate_full_signature=True,
)

# --- Training configs ---
DefaultTrainConf = builds(TrainingConfig, populate_full_signature=True)

FastDebugTrainConf = builds(
    TrainingConfig,
    max_epochs=2,
    batch_size=1,
    num_folds=2,
    early_stopping_patience=1,
    populate_full_signature=True,
)

FullTrainConf = builds(
    TrainingConfig,
    max_epochs=200,
    batch_size=4,
    learning_rate=1e-4,
    num_folds=5,
    mixed_precision=True,
    gradient_checkpointing=True,
    populate_full_signature=True,
)

# --- Ensemble configs ---
MeanEnsembleConf = builds(
    EnsembleConfig,
    strategy=EnsembleStrategy.MEAN,
    populate_full_signature=True,
)

GreedySoupConf = builds(
    EnsembleConfig,
    strategy=EnsembleStrategy.GREEDY_SOUP,
    populate_full_signature=True,
)

ConformalConf = builds(
    EnsembleConfig,
    strategy=EnsembleStrategy.MEAN,
    conformal_alpha=0.1,
    populate_full_signature=True,
)

# --- Top-level experiment config ---
ExperimentConf = make_config(
    experiment_name="minivess_v2",
    data=MinivessDataConf,
    model=SegResNetConf,
    training=DefaultTrainConf,
    ensemble=MeanEnsembleConf,
)

# --- Register in Hydra store ---
cs = store(group="experiment")
cs(ExperimentConf, name="default")

model_store = store(group="experiment/model")
model_store(SegResNetConf, name="segresnet")
model_store(SwinUNETRConf, name="swinunetr")
model_store(Vista3DConf, name="vista3d")
model_store(Sam3LoraConf, name="sam3_lora")

training_store = store(group="experiment/training")
training_store(DefaultTrainConf, name="default")
training_store(FastDebugTrainConf, name="debug")
training_store(FullTrainConf, name="full")

ensemble_store = store(group="experiment/ensemble")
ensemble_store(MeanEnsembleConf, name="mean")
ensemble_store(GreedySoupConf, name="greedy_soup")
ensemble_store(ConformalConf, name="conformal")

# --- Compute profile configs ---
ComputeProfileConf = builds(ComputeProfile, populate_full_signature=True)
compute_store = store(group="experiment/compute")
for _profile_name, _profile in _PROFILES.items():
    compute_store(
        builds(
            ComputeProfile,
            name=_profile.name,
            batch_size=_profile.batch_size,
            patch_size=_profile.patch_size,
            num_workers=_profile.num_workers,
            mixed_precision=_profile.mixed_precision,
            gradient_accumulation_steps=_profile.gradient_accumulation_steps,
        ),
        name=_profile_name,
    )
