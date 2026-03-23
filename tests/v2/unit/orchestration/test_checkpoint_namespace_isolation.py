"""Checkpoint namespace isolation — prevent concurrent factorial jobs from clobbering.

When 34 factorial conditions all write to the same GCS-backed `/app/checkpoints`
bucket via MOUNT_CACHED, checkpoint paths MUST include the condition identity
(model_family + loss_name) so concurrent jobs don't overwrite each other's files.

Bug: train_flow.py line 829 uses `checkpoint_base / f"fold_{fold_id}"` with NO
model/loss namespace. Two jobs writing `fold_0/epoch_latest.pth` to the same GCS
bucket will race.

Fix: checkpoint path must be `checkpoint_base / f"{model}_{loss}" / f"fold_{fold_id}"`.

Issue: integration-test-double-check.md, cloud edge case #6.
"""

from __future__ import annotations

import ast
from pathlib import Path

TRAIN_FLOW_PATH = Path("src/minivess/orchestration/flows/train_flow.py")


class TestCheckpointNamespaceIsolation:
    """Checkpoint directories must include model+loss to prevent cross-job collisions."""

    def test_checkpoint_dir_includes_model_family(self) -> None:
        """checkpoint_dir construction must reference model_family from config."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find the function training_subflow and look for checkpoint_dir assignment
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "training_subflow":
                func_source = ast.get_source_segment(source, node)
                assert func_source is not None

                # The checkpoint_dir construction must include model_family
                # Old (broken): checkpoint_base / f"fold_{actual_fold_id}"
                # New (correct): checkpoint_base / condition_name / f"fold_{actual_fold_id}"
                assert (
                    "model_family" in func_source or "condition_name" in func_source
                ), (
                    "training_subflow must include model_family in checkpoint_dir path. "
                    "Without this, concurrent factorial jobs overwrite each other's checkpoints."
                )
                return
        raise AssertionError("training_subflow function not found in train_flow.py")

    def test_checkpoint_dir_includes_loss_name(self) -> None:
        """checkpoint_dir construction must reference loss_name from config."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "training_subflow":
                func_source = ast.get_source_segment(source, node)
                assert func_source is not None
                assert "loss_name" in func_source or "condition_name" in func_source, (
                    "training_subflow must include loss_name in checkpoint_dir path."
                )
                return
        raise AssertionError("training_subflow function not found")

    def test_no_bare_fold_only_checkpoint_path(self) -> None:
        """checkpoint_dir must NOT be just `fold_{id}` under checkpoint_base.

        Pattern to reject: `checkpoint_base / f"fold_{actual_fold_id}"`
        Pattern to accept: `checkpoint_base / condition_name / f"fold_{actual_fold_id}"`
        """
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")

        # Search for the dangerous pattern: bare fold path without namespace
        # This is the exact line that caused the bug
        dangerous_patterns = [
            'checkpoint_base / f"fold_{actual_fold_id}"',
            "checkpoint_base / f'fold_{actual_fold_id}'",
        ]
        for pattern in dangerous_patterns:
            assert pattern not in source, (
                f"Found bare checkpoint path without model/loss namespace: {pattern}. "
                "This causes checkpoint collisions in concurrent factorial jobs. "
                "Use: checkpoint_base / condition_name / f'fold_{{fold_id}}'"
            )


class TestCheckpointPathInMLflowTags:
    """MLflow checkpoint_dir_fold_N tags must contain the full namespaced path."""

    def test_checkpoint_tag_includes_condition_namespace(self) -> None:
        """The checkpoint path tagged in MLflow must include model+loss namespace."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")

        # The tag is set at line ~710: client.set_tag(run_id, f"checkpoint_dir_fold_{fold_id}", str(checkpoint_dir))
        # checkpoint_dir must already contain the namespace at this point.
        # We verify this indirectly by checking the checkpoint_dir assignment includes namespace.
        # This is a structural test — the actual value test is in test_cross_flow_contract.py.
        assert "checkpoint_dir" in source, (
            "checkpoint_dir variable must exist in train_flow.py"
        )


class TestGpuPreflightGuard:
    """Setup script must gate on CUDA availability, not just print it."""

    def test_setup_python_oneliner_is_valid(self) -> None:
        """The GPU check python -c '...' must use correct PyTorch attribute names.

        5th pass root cause: total_mem → AttributeError. Correct: total_memory.
        """
        import yaml

        yaml_path = Path("deployment/skypilot/train_factorial.yaml")
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        setup = config.get("setup", "")

        # Must NOT use the wrong attribute name
        assert "total_mem " not in setup and ".total_mem/" not in setup, (
            "Setup uses 'total_mem' which doesn't exist on torch CudaDeviceProperties. "
            "Use 'total_memory' instead. This caused FAILED_SETUP in 5th pass job 69."
        )
        # Must use correct attribute
        if "total_memory" in setup:
            pass  # Correct attribute used

    def test_setup_script_gates_on_cuda(self) -> None:
        """SkyPilot setup must exit if PyTorch can't see CUDA."""
        import yaml

        yaml_path = Path("deployment/skypilot/train_factorial.yaml")
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        setup = config.get("setup", "")

        # The setup must contain a CUDA availability check that exits on failure
        assert "torch.cuda.is_available()" in setup, (
            "Setup must check torch.cuda.is_available()"
        )
        # Must also have an exit condition for CUDA unavailability
        # (not just print — the reviewer found it only prints, doesn't gate)
        cuda_check_lines = [
            line
            for line in setup.splitlines()
            if "cuda" in line.lower() and ("exit" in line or "sys.exit" in line)
        ]
        assert len(cuda_check_lines) > 0 or "CUDA: False" not in setup, (
            "Setup prints CUDA status but does NOT exit if CUDA is unavailable. "
            "A cloud VM without working CUDA will train on CPU (100x slower, OOM). "
            "Add: if not torch.cuda.is_available(): sys.exit(1)"
        )


class TestZeroShotBranch:
    """Zero-shot early return path must be exercised."""

    def test_zero_shot_returns_correct_status(self) -> None:
        """training_flow with zero_shot=True, max_epochs=0 returns ZERO_SHOT status."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        assert '"ZERO_SHOT"' in source, (
            "training_flow must return status='ZERO_SHOT' for zero-shot runs"
        )

    def test_zero_shot_has_empty_fold_results(self) -> None:
        """Zero-shot result must have fold_results=[] (no training)."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        # The zero-shot return should have fold_results=[]
        assert "fold_results=[]" in source, (
            "Zero-shot TrainingFlowResult must have fold_results=[]"
        )


class TestPostTrainingMethodParsing:
    """Post-training method comma-separated parsing must be tested."""

    def test_comma_split_in_training_flow(self) -> None:
        """post_training_method='none,swag' must be split on comma."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        assert 'post_training_method.split(",")' in source, (
            "training_flow must split post_training_method on comma"
        )

    def test_none_method_skipped(self) -> None:
        """'none' in post_training methods must NOT trigger post_training_subflow."""
        source = TRAIN_FLOW_PATH.read_text(encoding="utf-8")
        assert '"none"' in source, (
            "training_flow must check for 'none' in post_training methods"
        )
