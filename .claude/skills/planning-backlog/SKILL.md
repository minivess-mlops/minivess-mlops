# Planning & Backlog Skill

## Purpose
Manage the MinIVess MLOps v2 implementation backlog — triaging research into
prioritized GitHub issues, tracking progress across the project board, and
connecting PRD decisions to implementation work.

## Activation
Use this skill when:
- Planning next implementation sprint from the PRD backlog
- Triaging new research papers into implementation issues
- Reviewing and reprioritizing the GitHub project board
- Checking implementation progress against PRD decisions
- Creating implementation plans from PRD option selections

## GitHub Project Reference
- **Repository**: `minivess-mlops/minivess-mlops`
- **Project**: `Minivess MLOps` (ID: `PVT_kwDOCPpnGc4AYSAM`, Number: 1)
- **Priority field ID**: `PVTSSF_lADOCPpnGc4AYSAMzgPhgsk`
  - High: `9ac26196` (P0 — critical, must do next)
  - Medium: `9d51ccf2` (P1 — high priority, should do soon)
  - Low: `17b6c978` (P2 — medium priority, nice to have)

## Labels
| Label | Color | Description |
|-------|-------|-------------|
| `P0-critical` | `#B60205` | Must do next |
| `P1-high` | `#D93F0B` | Should do soon |
| `P2-medium` | `#FBCA04` | Nice to have |
| `enhancement` | `#a2eeef` | New feature or request |
| `models` | `#1D76DB` | Model architecture |
| `monitoring` | `#B60205` | System/model monitoring |
| `training` | `#FBCA04` | Training pipeline |
| `metrics` | `#0E8A16` | Evaluation metrics |
| `uncertainty` | `#5319E7` | Uncertainty quantification |
| `compliance` | `#D93F0B` | Regulatory compliance |
| `annotation` | `#C2E0C6` | Data annotation |
| `data-quality` | `#006B75` | Data quality assessment |
| `validation` | `#BFD4F2` | Model/data validation |
| `observability` | `#E99695` | LLM/system observability |
| `research` | `#F9D0C4` | Research exploration |
| `ci-cd` | `#C5DEF5` | CI/CD pipeline |
| `documentation` | `#0075ca` | Documentation |

## Operations

### 1. Sprint Planning
Review the project board and select issues for the next sprint:
```bash
# List all open issues by priority
gh issue list --label P0-critical --state open
gh issue list --label P1-high --state open
gh issue list --label P2-medium --state open

# View project board
gh project item-list 1 --owner minivess-mlops --format json
```

### 2. Create Issue from PRD
When a PRD decision reveals implementation work:
```bash
gh issue create --title "<title>" \
  --label "enhancement,<domain>,<priority>" \
  --body "$(cat <<'EOF'
## Summary
[Evidence-backed description]

## PRD Context
- **Decision**: `decision_id` (level)
- **Option**: `option_id` (prior: X.XX)
- **Bibliography**: `citation_key`

## Acceptance Criteria
- [ ] [Specific items]
- [ ] Unit tests (TDD mandatory)

## References
- Author et al. (Year). "Title." DOI/arXiv
EOF
)"

# Add to project with priority
ITEM_ID=$(gh project item-add 1 --owner minivess-mlops --url <issue_url> --format json | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
gh project item-edit --project-id PVT_kwDOCPpnGc4AYSAM --id "$ITEM_ID" \
  --field-id PVTSSF_lADOCPpnGc4AYSAMzgPhgsk \
  --single-select-option-id <priority_option_id>
```

### 3. Progress Review
Check implementation status against PRD decisions:
```bash
# Count issues by priority
gh issue list --label P0-critical --state open --json number | python3 -c "import json,sys; print(f'P0 open: {len(json.load(sys.stdin))}')"
gh issue list --label P1-high --state open --json number | python3 -c "import json,sys; print(f'P1 open: {len(json.load(sys.stdin))}')"
gh issue list --label P2-medium --state open --json number | python3 -c "import json,sys; print(f'P2 open: {len(json.load(sys.stdin))}')"

# Recently closed
gh issue list --state closed --limit 10
```

### 4. Reprioritize
When evidence changes priorities:
```bash
# Change issue labels
gh issue edit <number> --remove-label P2-medium --add-label P1-high

# Update project priority field
gh project item-edit --project-id PVT_kwDOCPpnGc4AYSAM --id <item_id> \
  --field-id PVTSSF_lADOCPpnGc4AYSAMzgPhgsk \
  --single-select-option-id <new_priority_option_id>
```

## Priority Assignment Guidelines

| Priority | Criteria |
|----------|----------|
| **P0 (High)** | Strong evidence from multiple papers, directly impacts core pipeline, implementation path is clear, enables other work |
| **P1 (Medium)** | Good evidence, important but not blocking, may require exploration |
| **P2 (Low)** | Exploratory, single-paper evidence, nice-to-have, future consideration |

## Connecting PRD to Implementation

Each issue should reference:
1. **Decision ID** — which `.decision.yaml` file it relates to
2. **Option ID** — which option the implementation addresses
3. **Bibliography keys** — which papers provide the evidence
4. **Prior probability** — the current PRD probability for context

When an issue is implemented:
1. Update the decision file's `implementation_status` field
2. Consider adjusting prior probabilities based on implementation experience
3. Update the PRD update plan if priorities shifted

## Current Backlog Summary (as of 2026-02-23)
- **P0 (High)**: 4 issues — vesselFM adapter, PPRM monitoring, topology losses, CI reporting
- **P1 (Medium)**: 6 issues — conformal prediction, regulatory docs, COMMA/Mamba, VessQC, DATA-CARE, CyclOps
- **P2 (Low)**: 10 issues — nnQC, reporting templates, AtlasSegFM, DiLLS, SynthICL, MC dropout, calibration-shift, EU AI Act, RegOps CI/CD, MedSAM2
