# Phase 1: COLLECT Protocol

Deterministic data collection. Zero LLM judgment required. All checks use
AST parsing, YAML/TOML parsing, and string splitting (Rule 16: no regex).

## Prerequisites

- Working directory: repository root
- Python 3.12+ with `ast`, `pathlib`, `tomllib`, `yaml` available
- All source files accessible via `pathlib.Path`

## Step 1: Enumerate Sources

Collect all configuration sources into categorized file lists:

```
config_sources = {
    "hydra_yaml":   glob("configs/**/*.yaml"),
    "env_example":  [".env.example"],
    "dockerfiles":  glob("deployment/docker/**/*Dockerfile*"),
    "compose":      glob("deployment/docker-compose*.yml"),
    "skypilot":     glob("deployment/skypilot/**/*.yaml"),
    "dynaconf":     ["src/minivess/config/settings.py"],
    "pulumi":       glob("deployment/pulumi/**/*.py"),
    "flow_code":    glob("src/minivess/orchestration/flows/**/*.py"),
    "pipeline_code": glob("src/minivess/pipeline/**/*.py"),
    "config_defs":  glob("src/minivess/config/**/*.py"),
}
```

## Step 2: Parse .env.example

Extract all key-value pairs from `.env.example`:

```python
from pathlib import Path

env_vars: dict[str, str] = {}
for line in Path(".env.example").read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    key, _, value = stripped.partition("=")
    env_vars[key.strip()] = value.strip().strip('"').strip("'")
```

Categorize by suffix:
- `*_VERSION` entries -> version pin consistency check (Step 4)
- `*_URI`, `*_URL` entries -> cross-file URI check (Step 5)
- `*_BUCKET` entries -> bucket name check (Step 5)

## Step 3: AST Scan Flow/Pipeline Code

For each Python file in `flow_code` and `pipeline_code`:

1. Parse with `ast.parse(source)`
2. Walk AST for `ast.FunctionDef` / `ast.AsyncFunctionDef`:
   - Check `args.defaults` for `ast.Constant` nodes where arg name is suspicious
   - Suspicious params: seed, batch_size, max_epochs, learning_rate, lr,
     n_bootstrap, n_permutations, alpha, rope, num_epochs, epochs
3. Walk AST for `ast.Call`:
   - Check `keywords` for `ast.keyword` where `arg` is suspicious and
     `value` is `ast.Constant`
4. Skip allowlisted files and functions (see test file for allowlists)

Output: list of `(file, line, param_name, value, fix_suggestion)` tuples.

## Step 4: Version Pin Consistency

For each `*_VERSION` key in `.env.example`:

1. Get the canonical value from `.env.example`
2. Search all Dockerfiles for the version string (as FROM tag, ARG, or ENV)
3. Search docker-compose files for the version string
4. Search Pulumi code for the version string
5. Search SkyPilot YAML for the version string

Output: list of `(key, expected, file, actual)` tuples where expected != actual.

## Step 5: Cross-File Value Consistency

For each URI/bucket/image-tag key in `.env.example`:

1. Get the canonical value
2. Search all YAML files for the literal value
3. Search all Python files for the literal value (as string constant in AST)
4. Flag any file that has a DIFFERENT literal for the same logical value

Output: list of `(logical_name, canonical, file, divergent_value)` tuples.

## Step 6: Config Layer Boundary Check

Verify separation of concerns:

1. Hydra configs (`configs/**/*.yaml`) should NOT reference env vars directly
2. Dynaconf settings should NOT duplicate Hydra config keys
3. Flow code should NOT define config values (only read them)
4. Pipeline code should receive config as function parameters, not import env

Detection approach:
- AST scan flow files for `os.environ.get()` with non-None fallback
- AST scan pipeline files for top-level `os.environ` reads (not in functions)
- YAML parse Hydra configs for `${oc.env:VAR}` patterns (Omegaconf env interpolation)

## Step 7: Aggregate Results

Combine all outputs into a structured result:

```python
collect_result = {
    "hardcoded_params": [...],       # Step 3
    "version_pin_drift": [...],      # Step 4
    "cross_file_inconsistency": [...], # Step 5
    "boundary_violations": [...],    # Step 6
    "files_scanned": count,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}
```

Pass `collect_result` to Phase 2 agents.
