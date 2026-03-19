#!/usr/bin/env python
"""CLI entry point for training_flow() — the ONLY supported way to run training locally.

Calls training_flow() directly (Prefect compat layer handles graceful degradation
when no Prefect server is running via PREFECT_DISABLED=1).

For production (with a running Prefect server), use:
    prefect deployment run 'training-flow/default' \\
        --params '{"model_family": "dynunet", "loss_name": "cbdice_cldice"}'

Local dev:
    uv run python scripts/run_training_flow.py --model-family dynunet
    uv run python scripts/run_training_flow.py --model-family sam3_vanilla --debug

This replaces all direct calls to scripts/train_monitored.py, which is now
DEPRECATED (exits with code 1 unless ALLOW_STANDALONE_TRAINING=1).
"""

from __future__ import annotations

import argparse
import sys


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the MinIVess training Prefect flow directly.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model-family dynunet
  %(prog)s --model-family sam3_vanilla --compute gpu_low --max-epochs 50
  %(prog)s --model-family vesselfm --debug
  %(prog)s --model-family mambavesselnet --num-folds 2 --loss-name dice_ce
""",
    )
    p.add_argument(
        "--model-family",
        default="dynunet",
        help="Model family string (default: dynunet)",
    )
    p.add_argument(
        "--loss-name",
        default="cbdice_cldice",
        help="Loss function name (default: cbdice_cldice)",
    )
    p.add_argument(
        "--compute",
        default="auto",
        help="Compute profile: auto | gpu_low | gpu_high | cpu (default: auto)",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum training epochs per fold (default: 100)",
    )
    p.add_argument(
        "--num-folds",
        type=int,
        default=3,
        help="Number of cross-validation folds (default: 3)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per GPU (default: 2)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: 1 epoch, reduced data",
    )
    p.add_argument(
        "--experiment-name",
        default="minivess_training",
        help="MLflow experiment name (default: minivess_training)",
    )
    p.add_argument(
        "--trigger-source",
        default="shell_script",
        help="Tag identifying what triggered this run (default: shell_script)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    from minivess.orchestration.flows.train_flow import training_flow

    print(
        f"[run_training_flow] model={args.model_family}  loss={args.loss_name}  "
        f"compute={args.compute}  epochs={args.max_epochs}  folds={args.num_folds}  "
        f"debug={args.debug}"
    )

    result = training_flow(
        model_family=args.model_family,
        loss_name=args.loss_name,
        compute=args.compute,
        max_epochs=args.max_epochs,
        num_folds=args.num_folds,
        batch_size=args.batch_size,
        debug=args.debug,
        experiment_name=args.experiment_name,
        trigger_source=args.trigger_source,
    )

    print(
        f"[run_training_flow] complete — folds={result.n_folds}  "
        f"mlflow_run_id={result.mlflow_run_id}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
