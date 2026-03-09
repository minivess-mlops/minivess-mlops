# Skill: overnight-runner

**Version:** 1.0.0
**Invocation:** `/overnight-runner`
**Purpose:** Observable, resumable batch execution of XML plan files via Claude headless sessions.

---

## When to Use This Skill

Use `overnight-runner` whenever you need to:
- Execute 2+ XML plan files sequentially as Claude headless sessions
- Run plans unattended (overnight, over lunch, on a server)
- Guarantee observability: live streaming output, heartbeat, stall detection
- Support crash recovery: resume from a specific child after failure

**Do NOT use this skill for:**
- Single plan files (just run `claude -p "Read and execute plan at: /path/to/plan.xml" --dangerously-skip-permissions`)
- Interactive sessions where you want to guide Claude manually
- Plans that have cross-child code dependencies (those need sequential merge — use `--merge-after-each`)

---

## Invocation

### Basic: run all children from a config

```
/overnight-runner --config docs/planning/my-batch.yaml
```

### Resume from a specific child after failure

```
/overnight-runner --config docs/planning/my-batch.yaml --skip-to 2
```

### Override timeout

```
/overnight-runner --config docs/planning/my-batch.yaml --timeout 14400
```

---

## What This Skill Does (responsibilities)

The skill generates and runs a **bash master executor** that:

1. **Validates prerequisites** — claude CLI, gh, uv, git clean state
2. **Streams output in real-time** — uses `--output-format stream-json --verbose --include-partial-messages`
3. **Detects stalls** — watchdog: STALL WARNING if log is 0 bytes after 10 min
4. **Beats heartbeat** — master script logs a timestamp every 60 seconds
5. **Enforces timeout** — `timeout N` hard-kills frozen sessions (default: 7200s / 2h)
6. **Reports TIMEOUT vs FAILED** — exit code 124 = timeout, anything else = failure
7. **Supports resumption** — `SKIP_TO=N` skips already-completed children
8. **Cleans up** — kills heartbeat and watchdog processes when child finishes

### What Claude sessions inside the script do

Each child session:
1. Reads the XML plan file via the Read tool
2. Executes all phases using the self-learning-iterative-coder TDD skill
3. Commits + pushes its branch
4. Creates a PR via `gh pr create`

---

## Batch Config Format

Create a YAML config file in `docs/planning/`:

```yaml
# docs/planning/my-batch.yaml
name: "My Overnight Batch"
closes_issues: [328, 330, 331]
log_dir: /tmp/minivess-overnight
timeout_sec: 7200          # per child; default 2 hours
base_branch: main

children:
  - num: 1
    plan: docs/planning/overnight-child-01-acquisition.xml
    label: "Flow 0: Data Acquisition (#328 #139 #150 #176)"
    issues: [328, 139, 150, 176]
    branch: feat/acquisition-flow

  - num: 2
    plan: docs/planning/overnight-child-02-annotation.xml
    label: "Data Annotation — MONAI Label (#330)"
    issues: [330]
    branch: feat/annotation-flow

  - num: 3
    plan: docs/planning/overnight-child-03-dashboard.xml
    label: "Dashboard + QA (#331 #342)"
    issues: [331, 342]
    branch: feat/dashboard-qa
```

---

## Generated Script Structure

The skill generates `docs/planning/<batch-name>-runner.sh` with this architecture:

```
master script
├── prerequisites()         — verify CLI tools + clean git state
├── run_child(N)
│   ├── verify plan file exists
│   ├── checkout base branch
│   ├── truncate log file (watchdog baseline)
│   ├── start_heartbeat()   — background: log timestamp every 60s
│   ├── start_watchdog()    — background: STALL WARNING if 0 bytes after 10 min
│   ├── timeout N claude -p "Read plan at: PATH" \
│   │     --dangerously-skip-permissions \
│   │     --output-format stream-json \
│   │     --verbose \
│   │     --include-partial-messages \
│   │     2>&1 | tee "${log_file}" | [human-readable jq filter]
│   ├── kill heartbeat + watchdog
│   └── report DONE / TIMEOUT / FAILED
└── post_run_summary()     — PR list + issue states
```

---

## Claude Session Prompt Template

The prompt passed to each child session is deliberately short:

```
You are executing a TDD implementation plan for the MinIVess MLOps repo.

REPO ROOT: /path/to/repo
PLAN FILE: /path/to/plan.xml

Read the plan file above first, then execute every phase in order.

CRITICAL RULES (from CLAUDE.md — non-negotiable):
- TDD: write failing tests first (RED), then minimum implementation (GREEN)
- Library-First: use MONAI, Prefect, existing code before writing custom logic
- No regex for structured data
- All flows via Prefect in Docker
- /tmp FORBIDDEN for artifacts
- .env.example = single source of truth
- GitHub Actions EXPLICITLY DISABLED
- Every test failure must be fixed or reported immediately

After all phases: run pytest, pre-commit, push, gh pr create, report status.
```

**Why short?** Passing the plan file path (not contents) means:
- Claude starts streaming output immediately (no 19KB argument to parse)
- Context budget goes to plan execution, not argument ingestion
- Shell argument size limits are irrelevant

---

## Observability Architecture

```
Terminal output (screen/tmux session)
│
├── [MASTER]  heartbeat every 60s              ← always visible, proves script is alive
├── [MASTER]  STALL WARNING if 0 bytes/10min   ← catches frozen sessions early
├── [CHILD-N] streaming tokens via jq filter   ← see what claude is doing in real-time
│
└── File output
    ├── /tmp/minivess-overnight/master.log       ← master events + heartbeats
    └── /tmp/minivess-overnight/child-N.log      ← full stream-json session transcript
```

The `child-N.log` is stream-json (newline-delimited JSON events). To re-read after the fact:
```bash
# Human-readable replay of what claude said:
jq -rj 'select(.type=="stream_event" and .event.delta.type?=="text_delta") | .event.delta.text' \
  /tmp/minivess-overnight/child-1.log
```

---

## How to Run

```bash
# ALWAYS use screen — NEVER nohup
screen -S overnight
./docs/planning/my-batch-runner.sh

# Detach: Ctrl-A D
# Reattach: screen -r overnight

# Resume after crash (child 2 failed, restart from 2):
SKIP_TO=2 ./docs/planning/my-batch-runner.sh
```

---

## Related

- `.claude/skills/self-learning-iterative-coder/SKILL.md` — TDD skill executed within each child
- `.claude/scripting-research.md` — background research on headless Claude patterns
- `.claude/metalearning/2026-03-09-overnight-script-silent-freeze.md` — why this skill exists
