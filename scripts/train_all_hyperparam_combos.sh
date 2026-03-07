#!/bin/bash
################################################################################
# train_all_hyperparam_combos.sh  [PLACEHOLDER — see issue #401]
#
# PLACEHOLDER: Full hyperparameter grid script is not yet implemented.
# See GitHub issue #401 for the implementation plan.
#
# Before this script can be created, the following must be decided:
#   1. Which hyperparameters vary per model family (losses, lr, batch sizes)?
#   2. Compute budget (combinations x epochs x folds = hours on GPU)
#   3. Which model families to include in the grid?
#   4. Should this use manual grid or hpo_flow.py (Optuna-based HPO)?
#
# Until then, use:
#   ./scripts/train_all_best.sh    — best-known config per family
#   ./scripts/train_dynunet.sh     — standard DynUNet baseline
#   ./scripts/train_sam3_all_variants.sh  — all SAM3 variants
#   ./scripts/train_mamba_variants.sh     — all Mamba variants
#
# For smarter HPO (Bayesian search), use the Optuna-based flow via Docker:
#   docker compose -f deployment/docker-compose.flows.yml run --rm \
#       -e MODEL_FAMILY=dynunet -e HPO_N_TRIALS=50 train
#
# Closes: #401
################################################################################

set -euo pipefail

cat <<'MSG'
[ERROR] train_all_hyperparam_combos.sh is not yet implemented.

See GitHub issue #401 for the implementation plan:
  https://github.com/minivess-mlops/minivess-mlops/issues/401

The hyperparameter space (which losses, learning rates, batch sizes to sweep)
has not been finalized yet.

In the meantime, use:
  ./scripts/train_all_best.sh           # best config per model family
  ./scripts/train_dynunet.sh            # standard DynUNet baseline
  ./scripts/train_sam3_all_variants.sh  # all SAM3 variants
  ./scripts/train_mamba_variants.sh     # all Mamba variants

For Bayesian HPO (smarter than a manual grid):
  docker compose -f deployment/docker-compose.flows.yml run --rm \
      -e MODEL_FAMILY=dynunet -e HPO_N_TRIALS=50 train
MSG

exit 1
