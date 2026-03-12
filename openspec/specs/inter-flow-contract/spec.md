# Inter-Flow Contract Specification

## Overview

Flows communicate exclusively through MLflow artifacts and tags.
No shared filesystem, no direct function calls between flows.

## Scenarios

### Scenario: find_upstream_run filters by flow_name tag
- GIVEN a downstream flow needs to find an upstream training run
- WHEN find_upstream_run() is called
- THEN it filters by tags.flow_name="training-flow"
- AND it does NOT simply return the most recent FINISHED run

### Scenario: Training flow sets flow_name tag
- GIVEN the training flow creates an MLflow run
- WHEN tags are set
- THEN flow_name="training-flow" (FLOW_NAME_TRAIN constant) is used
- AND the tag value is a string (None causes TypeError in to_proto())

### Scenario: Downstream flow discovers checkpoints
- GIVEN a downstream flow receives an upstream run_id
- WHEN resolve_checkpoint_paths_from_contract() is called
- THEN it looks for both .ckpt and .pth files
- AND best_val_loss.pth is the preferred checkpoint

### Scenario: MLflow tags are always strings
- GIVEN any flow sets an MLflow tag
- WHEN the tag value could be None
- THEN None MUST be converted to empty string before calling mlflow.set_tag()
- AND failure to do so causes TypeError in to_proto()

## Requirements

- Flows MUST use FLOW_NAME_TRAIN constant, NOT literal "train"
- Tag value is "training-flow", not "train"
- All tags MUST be strings — no None values
- Checkpoint discovery supports both .ckpt and .pth extensions
