# Protocol: Deep Exploration v3 (During Spot Queuing / Monitoring Downtime)

> Maximize compound learning from cloud execution by finding test gaps
> while jobs queue or run. Every minute of wait time should produce
> actionable findings that make the NEXT pass more robust.
>
> v3 improvements (from 6th pass lessons):
> - Added Agent 5: SkyPilot/infra config validation (caught 6 errors in v2)
> - Added metalearning search BEFORE exploration (prevent re-discovering known issues)
> - Output ranked by "cost of NOT fixing" (GPU-hours wasted × probability)
> - Agents must cite documentation URLs, not assumptions

## When to Activate

- Jobs are PENDING (spot queuing) or RUNNING (training in progress)
- The harness is in Phase 3: EXECUTE, monitoring is READ-ONLY
- There is idle time that can be used productively

## Pre-Exploration: Search Metalearning First

BEFORE launching agents, run `/search-metalearning` with:
- "silent failure" — find known silent failure patterns
- "skypilot" — find known SkyPilot misconfigurations
- "unauthorized" — find known unauthorized changes
- "config drift" — find known config-vs-code mismatches

Pass the metalearning results to ALL agents so they don't re-discover known issues.

## Execution: 5 Parallel Reviewer Agents

Launch ALL agents in a SINGLE message for maximum parallelism. Each agent
gets a DIFFERENT search domain. Agents MUST cite documentation URLs for any
claim about library behavior (no assumptions, no comments-as-truth).

### Agent 1 — CLOUD EXECUTION PATHS (runtime, not structure)

```
Trace every shell command in the SkyPilot setup/run scripts AND every env
var propagation chain from .env → SkyPilot envs → Python code.

For EACH command/chain, answer:
  "Is this tested at the RUNTIME level (not just YAML structure)?"
  "If not, what test should exist and what would it assert?"
  "What is the COST if this fails undetected?" (GPU-hours × probability)

Search domains:
  - deployment/skypilot/train_factorial.yaml (setup + run sections)
  - .env.example → SkyPilot envs → os.environ.get() in Python
  - Docker image content (what files/packages must be present)
  - GCS mount behavior (MOUNT_CACHED write/read/persist)
  - DVC pull path (dvc.yaml + dvc.lock in image, ADC auth chain)
```

### Agent 2 — ERROR HANDLING & SILENT FAILURES

```
Audit every error-swallowing pattern in the cloud execution path.

Search for:
  - try/except Exception: pass (silent swallow)
  - try/except Exception: logger.warning(...) (logged but continues)
  - 2>/dev/null || true in shell scripts (suppresses errors)
  - or {} / or [] / or None (silent fallback to empty)
  - Functions that return empty results on failure without signaling
  - contextlib.suppress(Exception) (explicit but still swallows)

Classify EACH by cost-of-not-fixing:
  CRITICAL — production data loss (metrics, checkpoints, artifacts silently gone)
  HIGH     — training waste (GPU hours spent, results unusable)
  MEDIUM   — degraded output (results exist but quality reduced)
  LOW      — cosmetic (no impact on correctness)

Search domains:
  - src/minivess/orchestration/flows/ (ALL flow files)
  - src/minivess/pipeline/post_training_plugins/
  - src/minivess/observability/tracking.py
  - src/minivess/adapters/model_builder.py
  - scripts/run_factorial.sh (shell error handling)
```

### Agent 3 — CROSS-FLOW CONTRACTS

```
For EACH flow boundary in the 5-flow pipeline:
  train → post_training → analysis → biostatistics → deploy

Verify:
  1. Metric keys logged by upstream MATCH what downstream queries
     (use MetricKeys constants — no raw strings)
  2. Checkpoint file formats saved by upstream MATCH what downstream loads
     (canonical name: CHECKPOINT_BEST_FILENAME in constants.py)
  3. Experiment names consistent (resolve_experiment_name(), no hardcoding)
  4. FlowContract tags written by upstream are discoverable by downstream
  5. Error states from upstream handled by downstream (not silently ignored)

Search domains:
  - src/minivess/orchestration/flow_contract.py
  - src/minivess/orchestration/constants.py
  - src/minivess/observability/metric_keys.py
  - All 5+ flow files in src/minivess/orchestration/flows/
  - src/minivess/ensemble/builder.py
  - src/minivess/pipeline/evaluation_runner.py
```

### Agent 4 — CONFIGURATION CONSISTENCY

```
Verify every config value referenced in Python code exists in its YAML source,
and every YAML key is consumed by code (no orphan configs).

Search for:
  1. config.get("key") calls — does "key" exist in the YAML?
  2. os.environ.get("VAR") calls — is "VAR" in .env.example AND SkyPilot envs?
  3. Default values (config.get("key", DEFAULT)) — does DEFAULT match YAML?
     Flag Rule #22 violations (hardcoded fallbacks for researcher-configurable values)
  4. Hydra config group references — do referenced files exist?
  5. Factorial config params — are all used by run_factorial.sh?

ORPHAN = YAML key exists but no code reads it
PHANTOM = Code reads a key not in any YAML
DRIFT = Default value in code != YAML value

Search domains:
  - configs/cloud/*.yaml vs src/ code
  - configs/factorial/*.yaml vs scripts/run_factorial.sh
  - .env.example vs deployment/skypilot/*.yaml envs
  - configs/splits/*.json vs train_flow.py
```

### Agent 5 — SKYPILOT & INFRASTRUCTURE VALIDATION (NEW in v3)

```
This agent exists because Claude Code made 6 SkyPilot errors in 2 sessions.
It validates ALL SkyPilot-related configurations against ACTUAL documentation.

For EACH SkyPilot config value, VERIFY against docs (cite URL):
  1. .sky.yaml — is every key valid per docs/reference/config.html?
  2. train_factorial.yaml resources — are all fields valid per YAML spec?
  3. job_recovery — is max_restarts_on_errors correctly configured?
  4. file_mounts — is MOUNT_CACHED behavior correct per docs?
  5. accelerators format — string vs dict vs any_of, which is correct?
  6. Controller placement — does infra: format match what SkyPilot expects?
  7. Allowed clouds — does the config match what sky check shows?

ALSO verify:
  8. scripts/sync_sky_config.py — does output match SkyPilot's expected format?
  9. scripts/validate_yaml_contract.py — does it catch all actual violations?
  10. Preflight checks — do they query the RIGHT SkyPilot/GCP APIs?

Search domains:
  - .sky.yaml (project config)
  - ~/.sky/config.yaml (user config)
  - deployment/skypilot/*.yaml (ALL task YAMLs)
  - configs/cloud/yaml_contract.yaml (golden contract)
  - scripts/sync_sky_config.py, scripts/validate_yaml_contract.py
  - scripts/preflight_gcp.py (SkyPilot-related checks)

MANDATORY: Cite SkyPilot documentation URLs for every claim.
Do NOT trust comments in code — verify against actual docs/API.
```

## Output Format

Each agent MUST return a table ranked by cost-of-not-fixing:

```
| Rank | File:Line | Gap Description | Risk | Cost if Unfixed | Test File | Test Name |
|------|-----------|----------------|------|-----------------|-----------|-----------|
```

Where Cost if Unfixed = estimated GPU-hours wasted × probability of occurrence.

## Post-Exploration: Report Update

After ALL agents complete:

1. **Search metalearning** for already-known issues → exclude from "new" findings
2. **Deduplicate** findings across agents (same gap found by multiple agents)
3. **Rank by cost** (highest GPU-hour waste × probability first)
4. **Update the live report** with:
   - Findings table (deduplicated, cost-ranked)
   - New test opportunities (grouped by test file, with test count)
   - Issues to create (one per SYSTEMIC finding, not per individual gap)
   - SkyPilot-specific findings highlighted separately (trust deficit pattern)
5. **Commit** the updated report

## Quality Gate

The exploration is complete when:
- [ ] All 5 agents returned findings (not 4 — Agent 5 is mandatory)
- [ ] Metalearning searched before exploration (no re-discoveries)
- [ ] Findings deduplicated, cost-ranked, and documented
- [ ] Report updated with all findings
- [ ] At least 3 new test files identified
- [ ] At least 1 systemic issue identified for GitHub issue creation
- [ ] ALL SkyPilot claims cite documentation URLs (Agent 5 requirement)
