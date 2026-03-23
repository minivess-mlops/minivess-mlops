# Protocol: Deep Exploration (During Spot Queuing / Monitoring Downtime)

> Maximize compound learning from cloud execution by finding test gaps
> while jobs queue or run. Every minute of wait time should produce
> actionable findings that make the NEXT pass more robust.

## When to Activate

- Jobs are PENDING (spot queuing) or RUNNING (training in progress)
- The harness is in Phase 3: EXECUTE, monitoring is READ-ONLY
- There is idle time that can be used productively

## Execution: 4 Parallel Reviewer Agents

Launch ALL agents in a single message for maximum parallelism. Each agent
gets a DIFFERENT search domain to maximize coverage breadth.

### Agent 1 — CLOUD EXECUTION PATHS

```
Trace every shell command in the SkyPilot setup/run scripts AND every env
var propagation chain from .env → SkyPilot envs → Python code.

For EACH command/chain, answer:
  "Is this tested at the RUNTIME level (not just YAML structure)?"
  "If not, what test should exist and what would it assert?"

Search domains:
  - deployment/skypilot/train_factorial.yaml (setup + run sections)
  - .env.example → SkyPilot envs → os.environ.get() in Python
  - Docker image content (what files/packages must be present)
  - GCS mount behavior (MOUNT_CACHED write/read/persist)
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

Classify EACH as:
  CRITICAL — production data loss (metrics, checkpoints, artifacts silently gone)
  HIGH     — training waste (GPU hours spent, results unusable)
  MEDIUM   — degraded output (results exist but quality reduced)
  LOW      — cosmetic (no impact on correctness)

Search domains:
  - src/minivess/orchestration/flows/train_flow.py
  - src/minivess/orchestration/flows/post_training_flow.py
  - src/minivess/orchestration/flows/analysis_flow.py
  - src/minivess/pipeline/post_training_plugins/swag.py
  - src/minivess/observability/tracking.py
  - src/minivess/adapters/model_builder.py
```

### Agent 3 — CROSS-FLOW CONTRACTS

```
For EACH flow boundary in the 5-flow pipeline:
  train → post_training → analysis → biostatistics → deploy

Verify:
  1. Metric keys logged by upstream MATCH what downstream queries
  2. Checkpoint file formats saved by upstream MATCH what downstream loads
  3. Experiment names are consistent (no hardcoding, all use resolve_experiment_name())
  4. FlowContract tags written by upstream are discoverable by downstream
  5. Error states from upstream are handled by downstream (not silently ignored)

Search domains:
  - src/minivess/orchestration/flow_contract.py
  - src/minivess/orchestration/constants.py
  - src/minivess/observability/metric_keys.py
  - All 5 flow files in src/minivess/orchestration/flows/
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
  3. Default values (config.get("key", DEFAULT)) — is DEFAULT the same as the YAML default?
  4. Hydra config group references — do referenced files exist?
  5. Factorial config params — are all used by run_factorial.sh?

Search domains:
  - configs/cloud/*.yaml vs src/ code that reads them
  - configs/factorial/*.yaml vs scripts/run_factorial.sh
  - .env.example vs deployment/skypilot/train_factorial.yaml envs
  - configs/splits/*.json vs train_flow.py load_fold_splits_task()
```

## Output Format

Each agent MUST return a table:

```
| File:Line | Gap Description | Risk | Test File | Test Name |
|-----------|----------------|------|-----------|-----------|
```

## Post-Exploration: Report Update

After ALL agents complete:

1. **Deduplicate** findings across agents (same gap found by multiple agents)
2. **Sort by risk** (CRITICAL first)
3. **Update the live report** with:
   - Findings table (deduplicated)
   - New test opportunities (grouped by test file, with test count)
   - Issues to create (one per SYSTEMIC finding, not per individual gap)
4. **Commit** the updated report

## Quality Gate

The exploration is complete when:
- [ ] All 4 agents returned findings
- [ ] Findings are deduplicated and risk-ranked
- [ ] Report updated with all findings
- [ ] At least 3 new test files identified
- [ ] At least 1 systemic issue identified for GitHub issue creation
