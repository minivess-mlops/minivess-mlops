# Training Pipeline Specification

## Overview

The training pipeline runs as a Prefect flow inside a Docker container.
It receives experiment configuration via the EXPERIMENT environment variable,
iterates over folds, trains models, and logs all artifacts to MLflow.

## Scenarios

### Scenario: Training flow receives config via EXPERIMENT env var
- GIVEN the EXPERIMENT environment variable is set to a valid experiment name
- WHEN the training flow starts
- THEN compose_experiment_config() resolves the full configuration
- AND the resolved config is logged as an MLflow artifact

### Scenario: Training flow iterates over folds
- GIVEN an experiment config with N folds defined in configs/splits/
- WHEN the training flow executes
- THEN it trains one model per fold
- AND each fold gets its own MLflow child run

### Scenario: Training flow tags runs with flow_name
- GIVEN the training flow starts an MLflow run
- WHEN the run is created
- THEN the tag flow_name="training-flow" is set
- AND downstream flows can discover this run via find_upstream_run()

### Scenario: Training flow enforces Docker context
- GIVEN the training flow is invoked
- WHEN _require_docker_context() is called
- THEN execution is blocked if not inside a Docker container
- AND the only escape hatch is MINIVESS_ALLOW_HOST=1 (pytest only)

### Scenario: Training flow saves checkpoints
- GIVEN a training fold completes
- WHEN the best validation loss is achieved
- THEN best_val_loss.pth is saved to the checkpoint directory
- AND last.pth is saved at the end of training
- AND epoch_latest.pth is saved at configurable intervals

### Scenario: Training flow logs system info
- GIVEN a training run starts
- WHEN ExperimentTracker.start_run() is called
- THEN Python version, PyTorch version, MONAI version, CUDA version are logged
- AND GPU model, RAM, git hash, branch, dirty state are logged
- AND all params use the sys_ prefix

## Requirements

- Training MUST run inside Docker (STOP protocol S-check)
- Prefect orchestration MUST be active (STOP protocol T-check)
- All artifact paths MUST be volume-mounted (STOP protocol O-check)
- Config MUST be reproducible on another machine (STOP protocol P-check)
- val_interval > max_epochs is the "never validate" sentinel
