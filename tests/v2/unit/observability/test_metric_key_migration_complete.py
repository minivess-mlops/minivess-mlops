"""AST-based integration test verifying slash-prefix migration is complete (T10).

Scans source files for string literals that look like old underscore-prefix
metric keys passed to mlflow.log_metric/mlflow.log_param/mlflow.log_params.

Issue: #790
"""

from __future__ import annotations

import ast
from pathlib import Path

# Source files that should use slash-prefix convention
_SOURCE_FILES = [
    "src/minivess/observability/tracking.py",
    "src/minivess/observability/system_info.py",
    "src/minivess/observability/infrastructure_timing.py",
    "src/minivess/pipeline/trainer.py",
    "src/minivess/orchestration/flows/train_flow.py",
    "src/minivess/data/profiler.py",
    "src/minivess/compute/gpu_profile.py",
]

# Known old underscore-prefix patterns that should have been migrated
_OLD_PREFIXES = (
    "train_loss",
    "val_loss",
    "train_dice",
    "val_dice",
    "train_f1",
    "val_f1",
    "val_cldice",
    "val_masd",
    "val_compound",
    "sys_gpu_",
    "sys_python_",
    "sys_torch_",
    "sys_cuda_",
    "sys_monai_",
    "sys_mlflow_",
    "sys_numpy_",
    "sys_os",
    "sys_hostname",
    "sys_total_ram",
    "sys_cpu_model",
    "sys_git_",
    "sys_dvc_",
    "sys_bench_",
    "data_n_volumes",
    "data_total_size",
    "data_min_shape",
    "data_max_shape",
    "data_median_",
    "data_n_outlier",
    "cost_total_",
    "cost_setup_",
    "cost_training_",
    "cost_effective_",
    "cost_gpu_",
    "cost_epochs_",
    "cost_break_",
    "setup_python_",
    "setup_uv_",
    "setup_dvc_",
    "setup_model_",
    "setup_verification_",
    "setup_total_",
    "prof_first_",
    "prof_steady_",
    "prof_overhead_",
    "prof_trace_",
    "prof_data_to_",
    "prof_forward_",
    "prof_backward_",
    "prof_cfg_",
    "estimated_total_",
    "cost_per_epoch",
    "epoch_seconds",
    "vram_peak_mb",
    "n_folds_completed",
    "cfg_project_",
    "cfg_data_",
    "cfg_dvc_",
    "cfg_mlflow_",
    "arch_",
    "model_family",
    "model_name",
)

# Strings that are allowed even though they match old patterns
# (e.g., in docstrings, comments, MIGRATION_MAP keys, tag names)
_ALLOWED_CONTEXTS = {
    # MIGRATION_MAP keys in metric_keys.py are old by design
    "metric_keys.py",
    # CLAUDE.md references are documentation
    "CLAUDE.md",
}


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _find_old_keys_in_dict_literals(tree: ast.AST) -> list[str]:
    """Find string constants that match old metric key patterns in dict literals."""
    old_keys_found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            for prefix in _OLD_PREFIXES:
                if value.startswith(prefix) or value == prefix.rstrip("_"):
                    old_keys_found.append(value)
                    break
    return old_keys_found


def test_no_old_underscore_keys_in_tracking() -> None:
    """tracking.py should not contain old underscore metric key patterns.

    Note: 'model_family' and 'model_name' are MLflow TAG keys (string
    metadata for identification), not metric/param keys. Tags do not
    benefit from slash-prefix auto-grouping, so they are excluded.
    """
    root = _get_project_root()
    src_path = root / "src" / "minivess" / "observability" / "tracking.py"
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    old_keys = _find_old_keys_in_dict_literals(tree)

    # Tag keys are string metadata — no slash-prefix grouping benefit
    tag_keys = {"model_family", "model_name"}
    filtered = [k for k in old_keys if k not in tag_keys]

    assert not filtered, f"tracking.py still has old underscore keys: {filtered}"


def test_no_old_underscore_keys_in_system_info() -> None:
    """system_info.py should not contain old underscore metric key patterns."""
    root = _get_project_root()
    src_path = root / "src" / "minivess" / "observability" / "system_info.py"
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    old_keys = _find_old_keys_in_dict_literals(tree)

    assert not old_keys, f"system_info.py still has old underscore keys: {old_keys}"


def test_no_old_underscore_keys_in_infrastructure_timing() -> None:
    """infrastructure_timing.py should not contain old underscore keys.

    Note: 'setup_total' is an internal durations dict key from
    parse_setup_timing(), not an MLflow key. It gets transformed to
    'setup/total_seconds' before logging.
    """
    root = _get_project_root()
    src_path = root / "src" / "minivess" / "observability" / "infrastructure_timing.py"
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    old_keys = _find_old_keys_in_dict_literals(tree)

    # Internal dict keys that are not MLflow keys
    internal_keys = {"setup_total"}
    filtered = [k for k in old_keys if k not in internal_keys]

    assert not filtered, (
        f"infrastructure_timing.py still has old underscore keys: {filtered}"
    )


def test_no_old_underscore_keys_in_trainer() -> None:
    """trainer.py should not contain old underscore metric key patterns."""
    root = _get_project_root()
    src_path = root / "src" / "minivess" / "pipeline" / "trainer.py"
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    old_keys = _find_old_keys_in_dict_literals(tree)

    # trainer.py still uses some old-style strings for internal purposes:
    # - "train_loss" / "val_loss": history dict keys (backward compat)
    # - "vram_peak_mb": return dict key (logged as vram/peak_mb in train_flow)
    # - "val_loss_compute": torch.profiler record_function label, not MLflow
    internal_keys = {"train_loss", "val_loss", "vram_peak_mb", "val_loss_compute"}
    external_old_keys = [k for k in old_keys if k not in internal_keys]

    assert not external_old_keys, (
        f"trainer.py still has old underscore MLflow keys: {external_old_keys}"
    )


def test_no_old_underscore_keys_in_profiler() -> None:
    """profiler.py should not contain old underscore metric key patterns."""
    root = _get_project_root()
    src_path = root / "src" / "minivess" / "data" / "profiler.py"
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    old_keys = _find_old_keys_in_dict_literals(tree)

    assert not old_keys, f"profiler.py still has old underscore keys: {old_keys}"


def test_no_old_underscore_keys_in_gpu_profile() -> None:
    """gpu_profile.py should not contain old underscore metric key patterns."""
    root = _get_project_root()
    src_path = root / "src" / "minivess" / "compute" / "gpu_profile.py"
    source = src_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    old_keys = _find_old_keys_in_dict_literals(tree)

    assert not old_keys, f"gpu_profile.py still has old underscore keys: {old_keys}"


def test_metric_keys_migration_map_covers_all_old_prefixes() -> None:
    """MIGRATION_MAP should cover all the common old prefix patterns."""
    from minivess.observability.metric_keys import MIGRATION_MAP

    # Verify key categories are represented
    categories = {
        "train_": False,
        "val_": False,
        "sys_": False,
        "cost_": False,
        "setup_": False,
        "data_": False,
        "prof_": False,
    }
    for old_key in MIGRATION_MAP:
        for prefix in categories:
            if old_key.startswith(prefix):
                categories[prefix] = True

    missing = [p for p, found in categories.items() if not found]
    assert not missing, f"MIGRATION_MAP missing keys for prefixes: {missing}"
