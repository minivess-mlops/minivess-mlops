"""Guard test: No hardcoded researcher-configurable params in flow/pipeline code.

Extends the test_no_hardcoded_alpha.py pattern (Issue #881) to detect hardcoded
seed, batch_size, max_epochs, learning_rate, n_bootstrap, n_permutations in
orchestration flows and pipeline modules.

Rule 29 (CLAUDE.md): ZERO Hardcoded Parameters — Config Is the Only Source.
Every numeric parameter that a researcher might want to change MUST come from
the Hydra-zen / Dynaconf config chain, NEVER from Python defaults.

Metalearning: .claude/metalearning/2026-03-20-hardcoded-significance-level-antipattern.md
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Directories to scan — ONLY flow and pipeline code (NOT config definitions)
_SCAN_DIRS: list[Path] = [
    _REPO_ROOT / "src" / "minivess" / "orchestration" / "flows",
    _REPO_ROOT / "src" / "minivess" / "pipeline",
]

# Parameter names that should NEVER have hardcoded literal defaults
# in production flow/pipeline code. Maps param name to human-readable config source.
_SUSPICIOUS_PARAMS: dict[str, str] = {
    "seed": "Hydra config (training.seed)",
    "batch_size": "Hydra config (training.batch_size)",
    "max_epochs": "Hydra config (training.max_epochs)",
    "learning_rate": "Hydra config (training.learning_rate)",
    "lr": "Hydra config (training.learning_rate)",
    "n_bootstrap": "BiostatisticsConfig().n_bootstrap",
    "n_permutations": "BiostatisticsConfig().n_permutations",
    "alpha": "BiostatisticsConfig().alpha",
    "rope": "BiostatisticsConfig().rope",
    "num_epochs": "Hydra config (training.max_epochs)",
    "epochs": "Hydra config (training.max_epochs)",
}

# Parameters that are ONLY suspicious in function SIGNATURES (defaults),
# NOT in keyword arguments of function calls. "alpha" is heavily used in
# matplotlib (opacity), loss functions (weighting), and other non-statistical
# contexts as a keyword arg. We only flag alpha=0.05 in function defaults
# (where it's almost always the significance level).
_SIGNATURE_ONLY_PARAMS: set[str] = {"alpha", "rope"}

# Sentinel values that indicate "disabled" or "not set" — not hardcoded config.
_SENTINEL_VALUES: set[int | float] = {0, -1}

# Files that contain visualization code where alpha= means opacity, not significance.
# These are excluded from the keyword argument check for alpha.
_VIZ_FILES: set[str] = {
    "biostatistics_figures.py",
    "external_generalization.py",
    "factorial_analysis.py",
    "loss_comparison.py",
    "observability_plots.py",
    "training_curves.py",
    "fold_heatmap.py",
    "uq_plots.py",
    "metric_correlation.py",
    "figure_export.py",
    "plot_config.py",
    "generate_all_figures.py",
}

# Files where alpha= means loss weighting, not significance level.
_LOSS_FILES: set[str] = {
    "loss_functions.py",
    "calibration_losses.py",
    "multitask_loss.py",
}

# Well-known literal values that are suspicious in function defaults
# Maps value to the likely parameter it represents.
_SUSPICIOUS_VALUES: dict[int | float, str] = {
    42: "seed=42 (hardcoded seed)",
    0.05: "alpha=0.05 (hardcoded significance level)",
    0.001: "learning_rate=0.001 (hardcoded LR)",
    0.0001: "learning_rate=0.0001 (hardcoded LR)",
    10000: "n_bootstrap=10000 (hardcoded bootstrap count)",
    1000: "n_permutations=1000 (hardcoded permutation count)",
}

# Files that are legitimately allowed to contain these patterns.
# Private helpers that receive values from callers, dataclass definitions
# that set sentinel defaults (None), etc.
_ALLOWLIST_FILES: set[str] = {
    # This test file itself
    "test_no_hardcoded_params_in_flows.py",
    # Config definition files re-export defaults
    "__init__.py",
    # Dashboard config is a presentation layer, not a research parameter
    "dashboard_config.py",
    "dashboard_sections.py",
}

# Function names that are allowed to have literal defaults because they
# are private helpers receiving values from config-reading callers, or
# they are type-annotation helpers / constants.
_ALLOWLIST_FUNCTIONS: set[str] = {
    # Prefect task wrappers that receive config from the flow
    "__init__",
    "__post_init__",
    # Private helpers that are always called with explicit args
    "_format_result",
    "_format_table",
    "_log_metric",
    "_log_metrics",
    "_write_jsonl",
    # Visualization helpers where numeric values are layout/display, not research
    "_compute_figsize",
    "_get_color_palette",
}

# Known existing violations that predate this guard test.
# Each entry is "filename:lineno" — the violation is documented as tech debt.
# This list MUST ONLY SHRINK over time. Adding new entries is BANNED.
# To track: these are all Rule 29 violations that need config wiring.
_KNOWN_SIGNATURE_VIOLATIONS: set[str] = {
    # analysis_flow.py — n_permutations default in embedding_drift_task
    "analysis_flow.py:embedding_drift_task:n_permutations",
    # biostatistics_flow.py — n_permutations default in task wrapper
    "biostatistics_flow.py:task_compute_specification_curve:n_permutations",
    # train_flow.py — max_epochs and batch_size in training_flow signature
    "train_flow.py:training_flow:max_epochs",
    "train_flow.py:training_flow:batch_size",
    # biostatistics_statistics.py — n_bootstrap default
    "biostatistics_statistics.py:compute_pairwise_comparisons:n_bootstrap",
    # biostatistics_statistics.py — rope default
    "biostatistics_statistics.py:compute_bayesian_comparisons:rope",
    # topology_comparison.py — n_bootstrap default
    "topology_comparison.py:paired_bootstrap:n_bootstrap",
    # biostatistics_statistics.py — utility functions with sensible defaults
    # (flow always passes config values; defaults are for standalone/test use)
    "biostatistics_statistics.py:stratified_permutation_test:n_permutations",
    "biostatistics_statistics.py:stratified_permutation_test:seed",
    "biostatistics_statistics.py:bootstrap_ci:seed",
    "biostatistics_statistics.py:bootstrap_ci:n_bootstrap",
    "biostatistics_statistics.py:bootstrap_ci:bca_min_n",
}

# Known existing keyword argument violations (hardcoded params in calls).
_KNOWN_CALL_VIOLATIONS: set[str] = {
    # data_flow.py — seed=42 in __main__ block
    "data_flow.py:seed",
}


def _collect_python_files() -> list[Path]:
    """Collect all .py files from scan directories."""
    files: list[Path] = []
    for scan_dir in _SCAN_DIRS:
        if scan_dir.is_dir():
            files.extend(scan_dir.rglob("*.py"))
    return sorted(files)


def _check_function_defaults(
    tree: ast.Module,
    filepath: Path,
) -> tuple[list[str], list[str]]:
    """Check ast.FunctionDef nodes for suspicious default parameter values.

    Flags function definitions where a parameter named in _SUSPICIOUS_PARAMS
    has a literal constant as its default value.

    Returns
    -------
    tuple of (new_violations, known_violations)
        new_violations: violations not in _KNOWN_SIGNATURE_VIOLATIONS
        known_violations: violations matching _KNOWN_SIGNATURE_VIOLATIONS
    """
    new_violations: list[str] = []
    known_violations: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        func_name = node.name

        # Skip allowlisted functions
        if func_name in _ALLOWLIST_FUNCTIONS:
            continue

        # Match defaults to their argument names.
        # In Python AST, defaults are right-aligned with args.
        args = node.args
        all_args = args.args + args.posonlyargs + args.kwonlyargs
        all_defaults = (
            [None] * (len(args.args) - len(args.defaults))
            + list(args.defaults)
            + list(args.kw_defaults)
        )

        for arg, default in zip(all_args, all_defaults, strict=False):
            if default is None:
                continue
            if not isinstance(default, ast.Constant):
                continue

            arg_name = arg.arg
            value = default.value

            # Check 1: Param name is suspicious AND default is a numeric literal
            if arg_name in _SUSPICIOUS_PARAMS and isinstance(value, (int, float)):
                # None defaults are fine (sentinel)
                if value is None:
                    continue
                # Sentinel values (0, -1) mean "disabled" — not hardcoded config
                if value in _SENTINEL_VALUES:
                    continue
                config_source = _SUSPICIOUS_PARAMS[arg_name]
                violation_msg = (
                    f"{filepath.name}:{default.lineno} — "
                    f"function '{func_name}' has hardcoded {arg_name}={value}. "
                    f"Read from {config_source} instead."
                )
                # Check if this is a known pre-existing violation
                violation_key = f"{filepath.name}:{func_name}:{arg_name}"
                if violation_key in _KNOWN_SIGNATURE_VIOLATIONS:
                    known_violations.append(violation_msg)
                else:
                    new_violations.append(violation_msg)

            # Check 2: Value is a well-known suspicious constant
            if isinstance(value, (int, float)) and value in _SUSPICIOUS_VALUES:
                # Only flag if the param name also looks research-relevant
                if arg_name in _SUSPICIOUS_PARAMS:
                    # Already caught by Check 1
                    continue
                # Don't flag port numbers, retry counts, etc.
                if arg_name in {
                    "port",
                    "timeout",
                    "retries",
                    "retry_count",
                    "max_retries",
                    "width",
                    "height",
                    "dpi",
                    "figsize",
                    "n_cols",
                    "n_rows",
                    "fontsize",
                    "linewidth",
                    "padding",
                    "margin",
                    "indent",
                    "delay",
                    "interval",
                    "max_workers",
                    "num_workers",
                    "verbosity",
                    "level",
                }:
                    continue

    return new_violations, known_violations


def _check_keyword_arguments(
    tree: ast.Module,
    filepath: Path,
) -> tuple[list[str], list[str]]:
    """Check ast.keyword nodes in function calls for hardcoded suspicious params.

    Flags calls like `train(seed=42)` or `compute_stats(n_bootstrap=10000)`
    where the argument name is in _SUSPICIOUS_PARAMS and the value is a literal.

    Excludes:
    - Signature-only params (alpha, rope) that have non-statistical meanings
      in keyword argument contexts (matplotlib opacity, loss weighting).
    - Visualization files where alpha= means plot opacity.
    - Loss function files where alpha= means loss weighting.
    - Sentinel values (0, -1) that mean "disabled."

    Returns
    -------
    tuple of (new_violations, known_violations)
    """
    new_violations: list[str] = []
    known_violations: list[str] = []

    filename = filepath.name

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        for keyword in node.keywords:
            if keyword.arg is None:
                # **kwargs expansion, skip
                continue

            if keyword.arg not in _SUSPICIOUS_PARAMS:
                continue

            # Skip params that are only suspicious in function signatures,
            # not in keyword call arguments (alpha=opacity, rope=threshold)
            if keyword.arg in _SIGNATURE_ONLY_PARAMS:
                continue

            # Skip visualization files where seed= in np.random calls is ok
            if filename in _VIZ_FILES:
                continue

            # Skip loss function files where params have loss-specific meaning
            if filename in _LOSS_FILES:
                continue

            if not isinstance(keyword.value, ast.Constant):
                continue

            value = keyword.value.value
            if not isinstance(value, (int, float)):
                continue

            # None sentinel is fine
            if value is None:
                continue

            # Sentinel values (0, -1) mean "disabled"
            if value in _SENTINEL_VALUES:
                continue

            config_source = _SUSPICIOUS_PARAMS[keyword.arg]
            violation_msg = (
                f"{filepath.name}:{keyword.value.lineno} — "
                f"hardcoded {keyword.arg}={value} in function call. "
                f"Pass from {config_source} instead."
            )
            # Check if known violation
            violation_key = f"{filepath.name}:{keyword.arg}"
            if violation_key in _KNOWN_CALL_VIOLATIONS:
                known_violations.append(violation_msg)
            else:
                new_violations.append(violation_msg)

    return new_violations, known_violations


class TestNoHardcodedParamsInFlows:
    """No hardcoded researcher-configurable parameters in flow/pipeline code.

    Rule 29: EVERY numeric parameter that a researcher might want to change
    MUST come from the Hydra-zen / Dynaconf config chain, NEVER from Python defaults.
    """

    def test_no_hardcoded_defaults_in_function_signatures(self) -> None:
        """Scan flow/pipeline function signatures for hardcoded param defaults.

        Checks for patterns like:
        - `def train(seed=42, ...)` — should receive seed from config
        - `def run(max_epochs=100, ...)` — should receive from config
        - `def analyze(alpha=0.05, ...)` — should use BiostatisticsConfig().alpha

        Known pre-existing violations are tracked in _KNOWN_SIGNATURE_VIOLATIONS
        and reported separately. Only NEW violations cause test failure.
        """
        new_violations: list[str] = []
        known_count = 0

        for py_file in _collect_python_files():
            if py_file.name in _ALLOWLIST_FILES:
                continue

            source = py_file.read_text(encoding="utf-8")

            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            new, known = _check_function_defaults(tree, py_file)
            new_violations.extend(new)
            known_count += len(known)

        assert not new_violations, (
            f"NEW hardcoded researcher-configurable defaults in "
            f"{len(new_violations)} location(s):\n"
            + "\n".join(f"  - {v}" for v in new_violations)
            + f"\n\n({known_count} known pre-existing violations tracked "
            f"in _KNOWN_SIGNATURE_VIOLATIONS)"
            + "\n\nFix: Read these values from Hydra config or "
            "BiostatisticsConfig() — never hardcode. See CLAUDE.md Rule 29."
        )

    def test_no_hardcoded_params_in_function_calls(self) -> None:
        """Scan flow/pipeline function calls for hardcoded param values.

        Checks for patterns like:
        - `train_model(seed=42)` — should pass `cfg.seed`
        - `compute_stats(n_bootstrap=10000)` — should pass `cfg.n_bootstrap`
        - `build_loader(batch_size=2)` — should pass `cfg.batch_size`

        Known pre-existing violations are tracked in _KNOWN_CALL_VIOLATIONS
        and reported separately. Only NEW violations cause test failure.
        """
        new_violations: list[str] = []
        known_count = 0

        for py_file in _collect_python_files():
            if py_file.name in _ALLOWLIST_FILES:
                continue

            source = py_file.read_text(encoding="utf-8")

            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            new, known = _check_keyword_arguments(tree, py_file)
            new_violations.extend(new)
            known_count += len(known)

        assert not new_violations, (
            f"NEW hardcoded researcher-configurable params in "
            f"{len(new_violations)} function call(s):\n"
            + "\n".join(f"  - {v}" for v in new_violations)
            + f"\n\n({known_count} known pre-existing violations tracked "
            f"in _KNOWN_CALL_VIOLATIONS)"
            + "\n\nFix: Pass config values (cfg.seed, cfg.batch_size, etc.) "
            "instead of literals. See CLAUDE.md Rule 29."
        )

    def test_known_violations_are_still_present(self) -> None:
        """Verify known violation entries actually match existing violations.

        If a known violation is fixed but not removed from the allowlist,
        the allowlist becomes stale. This test ensures the allowlist ONLY
        shrinks over time by flagging entries that no longer match.
        """
        found_signature_keys: set[str] = set()
        found_call_keys: set[str] = set()

        for py_file in _collect_python_files():
            if py_file.name in _ALLOWLIST_FILES:
                continue

            source = py_file.read_text(encoding="utf-8")

            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            _, known_sigs = _check_function_defaults(tree, py_file)
            _, known_calls = _check_keyword_arguments(tree, py_file)

            # Reconstruct the keys that were matched
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_name = node.name
                    args = node.args
                    all_args = args.args + args.posonlyargs + args.kwonlyargs
                    all_defaults = (
                        [None] * (len(args.args) - len(args.defaults))
                        + list(args.defaults)
                        + list(args.kw_defaults)
                    )
                    for arg, default in zip(all_args, all_defaults, strict=False):
                        if default is None or not isinstance(default, ast.Constant):
                            continue
                        key = f"{py_file.name}:{func_name}:{arg.arg}"
                        if key in _KNOWN_SIGNATURE_VIOLATIONS:
                            found_signature_keys.add(key)

                if isinstance(node, ast.Call):
                    for keyword in node.keywords:
                        if keyword.arg is None:
                            continue
                        key = f"{py_file.name}:{keyword.arg}"
                        if key in _KNOWN_CALL_VIOLATIONS:
                            found_call_keys.add(key)

        stale_sigs = _KNOWN_SIGNATURE_VIOLATIONS - found_signature_keys
        stale_calls = _KNOWN_CALL_VIOLATIONS - found_call_keys
        stale = stale_sigs | stale_calls

        assert not stale, (
            "Stale known-violation entries (violation was fixed but not "
            "removed from allowlist):\n"
            + "\n".join(f"  - {s}" for s in sorted(stale))
            + "\n\nFix: Remove these from _KNOWN_SIGNATURE_VIOLATIONS or "
            "_KNOWN_CALL_VIOLATIONS."
        )

    def test_scan_directories_exist(self) -> None:
        """Verify that the scanned directories actually exist.

        Prevents silent no-op if directory structure changes.
        """
        for scan_dir in _SCAN_DIRS:
            assert scan_dir.is_dir(), (
                f"Scan directory does not exist: {scan_dir}. "
                f"Update _SCAN_DIRS if directory structure changed."
            )

    def test_scanned_files_are_nonempty(self) -> None:
        """Verify that scanning actually finds Python files.

        Prevents silent no-op if glob pattern doesn't match.
        """
        files = _collect_python_files()
        assert len(files) > 0, (
            "No Python files found in scan directories. "
            "Check _SCAN_DIRS paths."
        )
        # Sanity: we expect at least 10 files in flows + pipeline
        assert len(files) >= 10, (
            f"Only {len(files)} Python files found — expected at least 10. "
            f"Scan directories may be incomplete."
        )
