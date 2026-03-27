"""Verification script for the 8th pass backlog fix (Task 3.7).

Programmatically checks that all backlog fixes are in place:
- No remaining hardcoded alpha=0.05 or seed=42 in function signatures
- No MLFLOW_ARTIFACTS_DESTINATION in Pulumi code
- No MLFLOW_TRACKING_USERNAME in SkyPilot YAMLs
- train_production.yaml and train_hpo.yaml use GAR (not GHCR)
- MetricKeys constants used in train_flow.py
- Builder uses MetricKeys constants
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src" / "minivess"
PASSED = 0
FAILED = 0


def _check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED  # noqa: PLW0603
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}: {detail}")


def _find_default_violations(
    target_value: object,
    param_names: set[str],
) -> list[str]:
    """Find function parameters with hardcoded defaults matching target_value.

    Only flags parameters whose NAME matches param_names (e.g., 'alpha',
    'threshold') — ignores coincidental matches (e.g., noise std=0.05).
    """
    violations = []
    for py_file in SRC_DIR.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Build param_name → default_value mapping
                args = node.args
                # Positional defaults (right-aligned)
                pos_args = args.args
                pos_defaults = args.defaults
                offset = len(pos_args) - len(pos_defaults)
                for i, default in enumerate(pos_defaults):
                    if isinstance(default, ast.Constant) and default.value == target_value:
                        param_name = pos_args[offset + i].arg
                        if param_name in param_names:
                            rel = py_file.relative_to(REPO_ROOT)
                            if "config" in str(rel).lower():
                                continue
                            violations.append(f"{rel}:{node.lineno} {node.name}({param_name})")
                # Keyword defaults
                for kwarg, default in zip(args.kwonlyargs, args.kw_defaults, strict=False):
                    if default is not None and isinstance(default, ast.Constant):
                        if default.value == target_value and kwarg.arg in param_names:
                            rel = py_file.relative_to(REPO_ROOT)
                            if "config" in str(rel).lower():
                                continue
                            violations.append(f"{rel}:{node.lineno} {node.name}({kwarg.arg})")
    return violations


def check_no_hardcoded_alpha_in_signatures() -> None:
    """No alpha/threshold: float = 0.05 in src/ function signatures."""
    print("\n[1] No hardcoded alpha=0.05 in function signatures")
    alpha_params = {"alpha", "significance_level", "p_val_threshold"}
    violations = _find_default_violations(0.05, alpha_params)
    _check(
        "alpha=0.05 defaults",
        len(violations) == 0,
        f"{len(violations)} violations: {violations[:5]}",
    )


def check_no_hardcoded_seed_in_signatures() -> None:
    """No seed: int = 42 in src/ function signatures (except config models)."""
    print("\n[2] No hardcoded seed=42 in function signatures")
    seed_params = {"seed", "random_seed", "random_state"}
    violations = _find_default_violations(42, seed_params)
    _check(
        "seed=42 defaults",
        len(violations) == 0,
        f"{len(violations)} violations: {violations[:5]}",
    )


def check_no_mlflow_artifacts_destination_in_pulumi() -> None:
    """No MLFLOW_ARTIFACTS_DESTINATION in Pulumi code."""
    print("\n[3] No MLFLOW_ARTIFACTS_DESTINATION in Pulumi code")
    pulumi_dir = REPO_ROOT / "deployment" / "pulumi"
    if not pulumi_dir.exists():
        _check("pulumi dir exists", False, "deployment/pulumi/ not found")
        return
    found = []
    for f in pulumi_dir.rglob("*.py"):
        content = f.read_text(encoding="utf-8")
        if "MLFLOW_ARTIFACTS_DESTINATION" in content:
            found.append(str(f.relative_to(REPO_ROOT)))
    _check(
        "MLFLOW_ARTIFACTS_DESTINATION removed",
        len(found) == 0,
        f"Found in: {found}",
    )


def check_no_mlflow_tracking_username_in_skypilot() -> None:
    """No MLFLOW_TRACKING_USERNAME in SkyPilot YAMLs."""
    print("\n[4] No MLFLOW_TRACKING_USERNAME in SkyPilot YAMLs")
    found = []
    for yaml_dir in [REPO_ROOT / "configs" / "cloud", REPO_ROOT / "deployment" / "skypilot"]:
      for yaml_file in yaml_dir.rglob("*.yaml"):
        for line in yaml_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            # Skip comments (lines starting with #)
            if stripped.startswith("#"):
                continue
            if "MLFLOW_TRACKING_USERNAME" in stripped:
                found.append(str(yaml_file.relative_to(REPO_ROOT)))
    _check(
        "MLFLOW_TRACKING_USERNAME removed",
        len(found) == 0,
        f"Found in: {found}",
    )


def check_production_yaml_uses_gar() -> None:
    """train_production.yaml and train_hpo.yaml use GAR (not GHCR)."""
    print("\n[5] Production YAMLs use GAR registry")
    for name in ["train_production.yaml", "train_hpo.yaml"]:
        yaml_path = REPO_ROOT / "deployment" / "skypilot" / name
        if not yaml_path.exists():
            _check(f"{name} exists", False, "file not found")
            continue
        content = yaml_path.read_text(encoding="utf-8")
        has_gar = "europe-west4-docker.pkg.dev" in content
        no_ghcr = "ghcr.io" not in content
        _check(f"{name} uses GAR", has_gar, "GAR registry not found")
        _check(f"{name} no GHCR", no_ghcr, "GHCR registry still present")


def check_metric_keys_in_train_flow() -> None:
    """train_flow.py imports and uses MetricKeys."""
    print("\n[6] train_flow.py uses MetricKeys constants")
    tf = SRC_DIR / "orchestration" / "flows" / "train_flow.py"
    source = tf.read_text(encoding="utf-8")
    _check("MetricKeys imported", "from minivess.observability.metric_keys import MetricKeys" in source)
    _check("MetricKeys.VAL_LOSS used", "MetricKeys.VAL_LOSS" in source)
    _check("MetricKeys.VRAM_PEAK_MB used", "MetricKeys.VRAM_PEAK_MB" in source)
    _check("MetricKeys.FOLD_N_COMPLETED used", "MetricKeys.FOLD_N_COMPLETED" in source)


def check_builder_uses_metric_keys() -> None:
    """builder.py uses MetricKeys for tracked metrics."""
    print("\n[7] builder.py uses MetricKeys constants")
    bf = SRC_DIR / "ensemble" / "builder.py"
    source = bf.read_text(encoding="utf-8")
    _check("MetricKeys.VAL_LOSS in _DEFAULT_TRACKED_METRICS", "MetricKeys.VAL_LOSS" in source)
    _check("No hardcoded val_loss in tracked metrics", '"val_loss"' not in source)


def main() -> None:
    print("=" * 60)
    print("8th Pass Backlog Verification Script")
    print("=" * 60)

    check_no_hardcoded_alpha_in_signatures()
    check_no_hardcoded_seed_in_signatures()
    check_no_mlflow_artifacts_destination_in_pulumi()
    check_no_mlflow_tracking_username_in_skypilot()
    check_production_yaml_uses_gar()
    check_metric_keys_in_train_flow()
    check_builder_uses_metric_keys()

    print("\n" + "=" * 60)
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print("=" * 60)

    sys.exit(1 if FAILED > 0 else 0)


if __name__ == "__main__":
    main()
