# kg-sync Skill

A reproducible Skill for keeping the Knowledge Graph synchronized with the live repo state
and exporting a snapshot to sci-llm-writer for manuscript generation.

## When to Use

- After a milestone: new GPU run completed, new model adapter added, new flow implemented
- When `scripts/check_projection_staleness.py` reports stale downstream files
- When preparing to write Introduction/Discussion in sci-llm-writer
- Invocation: `/kg-sync` or `/kg-sync --export-to sci-llm-writer`

## Generic Reusability

This Skill is designed for any repo following the pattern:
```
knowledge-graph/
  code-structure/    ← from AST (any Python repo)
  experiments/       ← from experiment tracker (MLflow, W&B, etc.)
  manuscript/        ← from human narrative + projections.yaml
```
Future repos (foundation-PLR, any research repo) can use the same Skill with their
own scanner + experiment tracker.

## 7-Step Workflow

```
1. SCAN-CODE   →  ast.parse() scan of src/ → code-structure/*.yaml
2. SCAN-EXP    →  DuckDB query over mlruns/ → experiments/*.yaml
2b. PROPAGATE  →  Check _network.yaml propagation: for modified nodes → requires_review flags
3. STAMP       →  Write _generated_at.yaml (git commit hash + timestamp)
4. STALENESS   →  check_projection_staleness.py → stale file report
5. GENERATE    →  Render stale projections via Jinja2 templates/*.j2 → .tex files
6. VALIDATE    →  (a) Schema check (b) Idempotency (c) pdflatex compile
7. EXPORT      →  If --export-to: atomic copy to sci-llm-writer kg-snapshot/
```

## Step-by-Step Reference

### Step 1: SCAN-CODE
```bash
uv run python scripts/scan_code_structure.py \
  --source-dir src/minivess/ \
  --output-dir knowledge-graph/code-structure/
```
**What it does:**
- `ast.parse()` all .py files in src/minivess/ (CLAUDE.md Rule #16 — no regex)
- Extracts: @flow decorated functions, ModelAdapter subclasses, Hydra config nodes, test markers
- Writes: `flows.yaml`, `adapters.yaml`, `config-schema.yaml`, `test-coverage.yaml`
- FALLBACK: if ast.parse() fails on a file → log warning, skip file, continue

### Step 2: SCAN-EXP
```bash
uv run python scripts/scan_experiments.py \
  --mlruns-dir mlruns/ \
  --output-dir knowledge-graph/experiments/
```
**What it does:**
- DuckDB analytics query over mlruns/ (already in src/minivess/observability/analytics.py)
- Extracts FINISHED runs only — FAILED/KILLED runs are excluded
- Writes: one YAML per experiment name + `current_best.yaml`
- MISSING RESULTS: if SAM3/VesselFM absent → insert `{status: "PENDING", reason: "GPU runs not complete"}`
- HARD ERROR: run_id in existing YAML absent from mlruns/ → error, not silent fallback

### Step 2b: PROPAGATE
- Parse `knowledge-graph/_network.yaml propagation:` section
- For each source node with git changes (compare to last `_generated_at.yaml` commit hash):
  - Find target nodes in propagation section
  - For type=hard/soft: add `requires_review: true` + `review_reason: "parent changed: <id>"`
  - For type=signal: log only, no YAML change
- Run `scripts/review_prd_integrity.py` to validate _network.yaml consistency
- Report propagation tree to user before continuing

### Step 3: STAMP
Writes `knowledge-graph/code-structure/_generated_at.yaml` and `knowledge-graph/experiments/_generated_at.yaml`:
```yaml
generated_at: "2026-03-15T10:00:00Z"
git_commit: "<sha>"
git_dirty: false
generator: scripts/scan_code_structure.py
```

### Step 4: STALENESS
```bash
uv run python scripts/check_projection_staleness.py \
  --projections knowledge-graph/manuscript/projections.yaml
```
Reports which downstream .tex files are stale (input mtime > output mtime OR output missing).

### Step 5: GENERATE
For each stale projection:
```bash
uv run python -c "
import jinja2
# Load template + KG data
# Render .tex file
# Write header: %% AUTO-GENERATED from run_id: <id>
"
```
**Key invariants:**
- AUTO-GENERATED .tex files NEVER hand-edited (pre-commit hook checks header)
- HUMAN-AUTHORED sections marked `%% HUMAN-AUTHORED: do not overwrite`
- All numbers from MLflow via DuckDB — never manually typed
- No timestamps/UUIDs in generated .tex — only run_ids (for idempotency)

### Step 6: VALIDATE
```bash
# (a) Schema check
uv run python scripts/review_prd_integrity.py

# (b) Idempotency check
uv run python scripts/check_projection_staleness.py  # must report "all up to date"

# (c) Compile check
cd docs/manuscript/latent-methods-results/
pdflatex latent-methods-results.tex 2>&1 | tail -5
# Must exit 0 with no errors
```
If ANY check fails → BLOCK export, report specific failure.

### Step 7: EXPORT (optional — only with --export-to flag)
```bash
# Atomic write: tmp → mv (never partial state in sci-llm-writer)
python3 -c "
import shutil, pathlib, tempfile, yaml
snapshot = {
    'generated_at': '...',
    'git_commit': '...',
    'status': 'complete' if not pending else 'incomplete',
    'pending_experiments': [...],  # from current_best.yaml
    'decisions': [...],  # from navigator.yaml
    'claims': [...],  # from manuscript/claims.yaml
}
tmp = pathlib.Path(tempfile.mktemp())
tmp.write_text(yaml.dump(snapshot))
dest = pathlib.Path('../sci-llm-writer/manuscripts/vasculature-mlops/kg-snapshot/kg-snapshot.yaml')
dest.parent.mkdir(parents=True, exist_ok=True)
shutil.move(str(tmp), str(dest))
"
```
**Export gates (all must pass):**
1. pdflatex compiles cleanly
2. Schema check passes (review_prd_integrity.py)
3. Idempotency: running Step 5 twice = identical output
4. No orphan run_ids (run_id in snapshot missing from mlruns/ → ERROR)

## Key Files

| File | Purpose |
|------|---------|
| `scripts/scan_code_structure.py` | Step 1 — AST scanner |
| `scripts/scan_experiments.py` | Step 2 — MLflow/DuckDB extractor |
| `scripts/check_projection_staleness.py` | Step 4 — Staleness detector |
| `knowledge-graph/templates/*.j2` | Step 5 — Jinja2 templates |
| `knowledge-graph/manuscript/projections.yaml` | Step 4 input — dependency map |
| `knowledge-graph/code-structure/_generated_at.yaml` | Step 3 output — staleness stamp |
| `knowledge-graph/experiments/_generated_at.yaml` | Step 3 output — staleness stamp |

## Invariants

1. No timestamps/UUIDs in generated .tex (idempotency)
2. All numbers from MLflow via DuckDB — never typed manually
3. HUMAN-AUTHORED sections never overwritten
4. Export is atomic (tmp + mv — no partial state)
5. Orphan run_id = hard ERROR (not silent skip)
6. PENDING experiments = `{status: "PENDING"}` header in kg-snapshot.yaml

## This Skill is Generic

To adapt for another repo:
1. Replace `src/minivess/` with your source directory in Step 1
2. Replace `mlruns/` with your experiment tracker path in Step 2
3. Replace Jinja2 templates with your downstream output format
4. Update `projections.yaml` with your dependency map
Everything else (stamp, staleness, validate, export) works unchanged.
