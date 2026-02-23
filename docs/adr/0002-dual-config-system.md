# ADR-0002: Dual Configuration System (Hydra-zen + Dynaconf)

## Status

Accepted

## Date

2026-02-23

## Context

The v0.1-alpha codebase used Hydra with handwritten YAML configuration files. This worked for single-model experiments but had several problems:

1. YAML configs were unvalidated at load time, leading to silent misconfiguration.
2. Experiment sweeps and deployment settings were conflated in the same config tree.
3. Environment-specific overrides (local, staging, production) required manual YAML editing.
4. No Python-level type checking of configuration values.

The v2 platform has two fundamentally different configuration domains:

- **Experiment configuration**: model architecture, hyperparameters, data augmentation, sweep axes. These vary per experiment run and are swept by Optuna.
- **Deployment configuration**: service ports, database URLs, feature flags, secrets references. These vary per environment (dev, staging, production) and must not leak into experiment configs.

## Decision

We adopt a dual configuration system:

**Hydra-zen** (experiment configs in `configs/experiment/`):

- Python-first structured configs with full Pydantic v2 validation.
- `builds()` and `make_config()` generate type-safe Hydra configs from Pydantic models (`ExperimentConfig`, `ModelConfig`, `TrainingConfig`, etc.).
- Optuna integration via `hydra-optuna-sweeper` for hyperparameter search.
- All experiment configs are serializable and logged to MLflow as artifacts.

**Dynaconf** (deployment configs in `configs/deployment/`):

- Layered settings files: `settings.toml` (defaults) + `settings.{env}.toml` (overrides).
- Environment switching via `ENV_FOR_DYNACONF=production`.
- Secrets from environment variables with `@format` token interpolation.
- Docker Compose `.env` file integration.

The two systems share the Pydantic v2 models defined in `src/minivess/config/models.py` as the source of truth for types and validation constraints. Hydra-zen wraps these for experiment composition; Dynaconf reads deployment settings and validates them against the same Pydantic schemas.

## Consequences

**Positive:**

- Experiment configs are fully type-checked at both Python and CLI levels.
- Deployment settings support environment layering without code changes.
- Secrets never appear in experiment config YAML files committed to git.
- Optuna sweeps operate on Hydra-zen structured configs with native multirun support.
- Pydantic v2 models serve as a single source of truth, preventing config drift between the two systems.

**Negative:**

- Two configuration systems add cognitive overhead for new contributors.
- Hydra-zen's `builds()` API requires familiarity with Hydra's override grammar.
- Mapping between Dynaconf TOML keys and Pydantic model fields must be kept in sync manually.

**Neutral:**

- The v0.1-alpha `configs/defaults.yaml` is preserved as a reference but is not loaded by the v2 pipeline.
