# Metalearning: Hardcoded Parameters (α=0.05, seed=42) Anti-Pattern

**Date**: 2026-03-20
**Severity**: P0 — violates single-source-of-truth principle
**Root Cause**: Claude Code defaults to hardcoding `0.05`, `42`, `500`, etc.
in tests and source code instead of reading from config. This affects ALL
researcher-configurable parameters, not just alpha.

## What Happened

While implementing biostatistics pipeline phases 2-6 (specification curve,
rank stability, calibration metrics, factorial ANOVA K=1 fallback), Claude
Code hardcoded `alpha=0.05` in:

1. **Test assertions**: `assert result.p_values[model_key] < 0.05`
2. **Function defaults**: `alpha: float = 0.05` in function signatures
3. **Spec curve engine**: `alpha: float = 0.05` parameter default

This is a **textbook single-source-of-truth violation**. The significance
level should come from `BiostatisticsConfig.alpha` (Pydantic model in
`src/minivess/config/biostatistics_config.py`), which is configured via
the biostatistics YAML config and Hydra-zen composition.

## Why This Is Dangerous

1. **Silent inconsistency**: If someone changes `alpha` in the YAML config
   to 0.01, tests still assert against 0.05 → false passes or false failures.
2. **Reproducibility**: The resolved config in MLflow is the single source
   of truth (per CLAUDE.md). Hardcoded values bypass this chain.
3. **Platform principle violated**: This is a platform paper — users should
   configure alpha via YAML, not edit Python code.

## The Anti-Pattern

Claude Code's tendency to use hardcoded values stems from:
- Training data where `0.05` is the "obvious" default for significance
- Not reading the existing config class before writing tests
- Treating test files as standalone rather than config-driven
- "Quick path" thinking: hardcoding is faster than wiring config

## How To Fix

1. **Tests must read alpha from BiostatisticsConfig defaults** — never
   hardcode `0.05` in assertions:
   ```python
   # WRONG
   assert result.p_values["model"] < 0.05

   # RIGHT
   from minivess.config.biostatistics_config import BiostatisticsConfig
   cfg = BiostatisticsConfig()
   assert result.p_values["model"] < cfg.alpha
   ```

2. **Function signatures should NOT default alpha** — force caller to
   pass it from config:
   ```python
   # WRONG
   def compute_spec_curve(..., alpha: float = 0.05) -> ...

   # RIGHT (or at minimum, document that default mirrors config)
   def compute_spec_curve(..., alpha: float) -> ...
   ```

3. **Pre-commit check**: Add a test that greps for `< 0.05` in test files
   related to biostatistics and fails if found without a config reference.

## Prevention Rule

**NEVER hardcode statistical thresholds (alpha, ROPE, power).** Always
read from `BiostatisticsConfig` or pass explicitly from the config chain.
If writing a test that checks significance, use `BiostatisticsConfig().alpha`.

## Files Affected

- `tests/v2/unit/test_factorial_anova.py` — multiple `< 0.05` assertions
- `tests/v2/integration/test_biostatistics_factorial_integration.py` — `< 0.05`
- `src/minivess/pipeline/biostatistics_specification_curve.py` — `alpha: float = 0.05`
- `src/minivess/pipeline/biostatistics_statistics.py` — `alpha: float = 0.05`

## Cross-References

- CLAUDE.md Rule #22: Single-source config via `.env.example`
- CLAUDE.md Rule #9: Task-agnostic architecture
- `src/minivess/config/biostatistics_config.py`: `BiostatisticsConfig.alpha`
- See also: `.claude/metalearning/2026-03-06-regex-ban.md` (similar pattern: "it's just a default" → fragility)
