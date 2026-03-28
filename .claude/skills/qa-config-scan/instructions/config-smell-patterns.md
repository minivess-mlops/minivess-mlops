# Config Smell Patterns

Taxonomy adapted from Sharma et al. "Does Your Configuration Code Smell?"
(MSR 2016) for the MinIVess MLOps codebase. Each pattern includes detection
heuristics and codebase-specific examples.

## Category 1: Missing Abstraction

### 1.1 Scattered Constant

**Definition**: The same magic number or string literal appears in multiple
files without a shared constant or config entry.

**Detection**: AST scan for `ast.Constant` nodes with identical values across
files. Group by value, flag groups with count > 1 in non-config directories.

**Codebase example**:
- GCS bucket name `minivess-mlops-dvc-data` hardcoded in both SkyPilot YAML
  and Python source instead of reading from `.env.example` `DVC_GCS_BUCKET`.
- MLflow experiment name string duplicated across flow files instead of using
  `orchestration/constants.py`.

**Fix**: Extract to `.env.example` (infrastructure values) or Hydra config
(research parameters). Reference from a single constant module.

### 1.2 Missing Override Point

**Definition**: A value that a researcher would want to change is hardcoded
in Python source rather than exposed through the config chain.

**Detection**: AST scan for function defaults with names in the suspicious
params list (seed, batch_size, max_epochs, learning_rate, alpha, n_bootstrap,
n_permutations). See `test_no_hardcoded_params_in_flows.py`.

**Codebase example**:
- `alpha=0.05` hardcoded in biostatistics comparison functions instead of
  reading from `BiostatisticsConfig().alpha` (Issue #881).
- `seed=42` as a default parameter instead of receiving from `cfg.seed`.

**Fix**: Remove the default. Require callers to pass the value from config.

## Category 2: Unnecessary Complexity

### 2.1 Redundant Config Layer

**Definition**: A config value is defined in one system (Hydra-zen) but also
mirrored or overridden in another (Dynaconf, environment variable) without
clear separation of concerns.

**Detection**: Grep for the same key name appearing in both
`configs/**/*.yaml` (Hydra) and `src/minivess/config/settings.py` (Dynaconf).
Cross-reference with `.env.example`.

**Codebase example**:
- Hydra config defines `training.max_epochs` AND `.env.example` defines
  `MAX_EPOCHS` — which one wins? (Answer: Hydra for training params,
  `.env.example` for infrastructure params. See config CLAUDE.md.)

**Fix**: Each value lives in exactly one config system. Document the boundary.

### 2.2 Dead Config Entry

**Definition**: A config key exists in YAML/TOML/env but no code path reads it.

**Detection**: For each key in `.env.example`, search for `os.environ["KEY"]`,
`os.getenv("KEY")`, or `settings.KEY` in Python source. Keys with zero
references are dead.

**Codebase example**:
- Deprecated env vars from v0.1 that were never cleaned up after the
  greenfield rewrite (Rule 26: no legacy).

**Fix**: Delete the dead entry. This is a greenfield project (Rule 26).

## Category 3: Deficiency

### 3.1 Implicit Dependency

**Definition**: Config A requires config B to be set, but the dependency is
not documented or enforced. Setting A without B causes a silent failure.

**Detection**: Trace `os.environ["A"]` usage sites. Check if the same
function also reads `os.environ["B"]`. If both are needed but only one is
documented in `.env.example`, flag as implicit dependency.

**Codebase example**:
- `MLFLOW_TRACKING_URI` requires `MLFLOW_TRACKING_USERNAME` and
  `MLFLOW_TRACKING_PASSWORD` when pointing to a remote server, but the
  dependency is not enforced at startup.

**Fix**: Add a validation function that checks co-required env vars at
flow startup. Use `resolve_tracking_uri()` pattern.

### 3.2 Type Mismatch Risk

**Definition**: A config value is stored as a string (env var, YAML) but
consumed as a different type (int, float, bool) with no validation.

**Detection**: AST scan for `int(os.environ["KEY"])` or
`float(os.getenv("KEY"))` patterns. These are type-conversion sites where
invalid values cause runtime crashes.

**Codebase example**:
- `MAX_EPOCHS` read from env as string, cast to int at usage site without
  validation. A typo like `MAX_EPOCHS=ten` crashes at runtime.

**Fix**: Use Dynaconf type coercion or Pydantic/dataclass validation at
the config loading boundary. Never cast at usage sites.

### 3.3 Stale Value

**Definition**: A config value was correct at one point but the code has
evolved and the value is now wrong or misleading.

**Detection**: Version pin consistency checks (`.env.example` vs Dockerfile
vs docker-compose). Also: check that default values in config dataclasses
match the documented defaults in YAML comments.

**Codebase example**:
- `MLFLOW_SERVER_VERSION` in `.env.example` set to 2.19.0 but Dockerfile
  uses a different version tag.

**Fix**: Single source of truth in `.env.example`. All consumers reference
via `ARG` / `${VAR}` substitution.

## Category 4: Violation

### 4.1 Environment-Specific Hardcoding

**Definition**: A value that differs between environments (local, dev, staging,
prod) is hardcoded instead of coming from the environment config chain.

**Detection**: AST scan for `os.environ.get("VAR", "fallback")` where the
fallback is a concrete value (not None). The fallback becomes the de-facto
hardcoded value that masks missing config.

**Codebase example**:
- `os.environ.get("MLFLOW_TRACKING_URI", "mlruns")` — the fallback hides
  the fact that the env var is not set, leading to local-only tracking.

**Fix**: Fail loudly (`resolve_tracking_uri()` pattern) or use
`.env.example` + `python-dotenv` at the single entry point.

### 4.2 Cross-File Inconsistency

**Definition**: The same logical value appears in multiple files with
different literal values.

**Detection**: Parse all version pins, bucket names, image tags, port
numbers across YAML, TOML, Dockerfiles, and Python. Group by logical
identity. Flag groups where values differ.

**Codebase example**:
- Docker image tag `minivess-base:latest` in docker-compose but
  `minivess-base:v0.2` in SkyPilot YAML.

**Fix**: Single source in `.env.example`, reference everywhere via
variable substitution.

## Detection Priority

| Priority | Pattern | Automated Test |
|----------|---------|----------------|
| P0 | Missing Override Point (1.2) | `test_no_hardcoded_params_in_flows.py` |
| P0 | Stale Value (3.3) | `test_version_pin_consistency.py` |
| P1 | Cross-File Inconsistency (4.2) | `test_yaml_contract_enforcement.py` |
| P1 | Environment-Specific Hardcoding (4.1) | `test_env_single_source.py` |
| P2 | Scattered Constant (1.1) | Phase 2 Agent B |
| P2 | Implicit Dependency (3.1) | Phase 2 Agent C |
| P3 | Dead Config Entry (2.2) | Phase 2 Agent A |
| P3 | Redundant Config Layer (2.1) | Phase 2 Agent A |
| P3 | Type Mismatch Risk (3.2) | Phase 2 Agent C |
