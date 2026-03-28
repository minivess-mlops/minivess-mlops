# Phase 2: REVIEW Protocol

Three parallel LLM agents examine the Phase 1 COLLECT output and source files
for patterns requiring judgment. Each agent operates independently and produces
a findings list. Agent outputs are merged in Phase 3.

## Agent A: Config Smell Detector

**Input**: Phase 1 `collect_result` + full `config_defs` file list

**Task**: Apply the Sharma et al. (MSR 2016) adapted taxonomy from
`instructions/config-smell-patterns.md` to identify:

1. **Dead config entries**: Keys in `.env.example` or Hydra YAML that no code reads
2. **Redundant config layers**: Same logical value in both Hydra and Dynaconf
3. **Stale values**: Defaults in config dataclasses that don't match YAML comments
4. **Type mismatch risks**: String-to-numeric casts at usage sites without validation

**Procedure**:

```
FOR each config key in .env.example:
    Search src/ for references to this key
    IF zero references → flag as "Dead Config Entry" (P3)

FOR each config dataclass in src/minivess/config/:
    Read field defaults
    Compare against corresponding Hydra YAML group defaults
    IF mismatch → flag as "Stale Value" (P0)

FOR each os.environ / os.getenv call in src/:
    Check if wrapped in int() / float() / bool()
    IF yes AND no try/except → flag as "Type Mismatch Risk" (P3)
```

**Output**: List of `(smell_type, file, line, description, severity)`

## Agent B: Cross-File Consistency Checker

**Input**: Phase 1 `cross_file_inconsistency` + `version_pin_drift` results

**Task**: Deep verification of cross-file value consistency beyond what
deterministic checks catch:

1. **Semantic equivalence**: Values that LOOK different but mean the same thing
   (e.g., `europe-north1` vs `EU` vs `EUROPE_NORTH1`)
2. **Derived values**: A value in file A is computed from a value in file B
   (e.g., Docker image tag = registry + name + version)
3. **Implicit contracts**: Files that must agree on values but have no
   mechanical link (e.g., SkyPilot disk size must fit the Docker image)

**Procedure**:

```
FOR each logical value group from Phase 1 Step 5:
    Read all files containing this value
    Check if differences are semantic equivalence or true drift
    IF true drift → confirm the Phase 1 finding
    IF semantic equivalence → mark as false positive

FOR each SkyPilot YAML:
    Read resources (GPU type, disk_size, cloud provider)
    Verify against yaml_contract.yaml golden reference
    Check image_id matches a known image tag in docker-compose

FOR each docker-compose service:
    Read image tag, ports, volumes
    Verify ports don't conflict with Dynaconf settings
    Verify volume paths match STOP protocol required mounts
```

**Output**: List of `(check_type, files_involved, description, severity)`

## Agent C: Architecture Boundary Validator

**Input**: Phase 1 `boundary_violations` + all source files

**Task**: Verify that the config architecture boundaries from CLAUDE.md are
respected:

1. **Config definitions stay in config/**: No `@dataclass` config classes
   defined in flow or pipeline files
2. **Flows read config, never define it**: Flow files import from `config/`
   module, never define default values for research params
3. **Pipeline receives config as params**: Pipeline functions take config
   values as explicit parameters, never read env vars directly
4. **No os.environ.get with fallback in flows**: Flows must use
   `resolve_*()` functions or fail loudly (Rule 22)

**Procedure**:

```
FOR each flow file in src/minivess/orchestration/flows/:
    AST parse the file
    Check for dataclass definitions → boundary violation
    Check for os.environ.get() with non-None second arg → Rule 22 violation
    Check for hardcoded param defaults in @flow / @task functions

FOR each pipeline file in src/minivess/pipeline/:
    AST parse the file
    Check for top-level os.environ reads → should be in function params
    Check for config imports from unexpected modules
    Verify function signatures receive config values as params

FOR each config file in src/minivess/config/:
    Verify it is imported ONLY by flow files and other config files
    Pipeline files should not import config directly (receive as params)
```

**Output**: List of `(boundary_type, file, line, description, severity)`

## Merge Protocol

After all three agents complete:

1. Concatenate all findings into a single list
2. Deduplicate: if Agent A and Agent C flag the same file:line, keep the
   higher-severity finding
3. Sort by severity (CRITICAL > WARNING > INFO), then by file path
4. Pass merged findings to Phase 3 for false-positive filtering
