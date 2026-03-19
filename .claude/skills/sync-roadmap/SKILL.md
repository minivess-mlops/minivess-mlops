---
name: sync-roadmap
version: 1.1.0
description: >
  Synchronize GitHub Project Roadmap timeline fields (Start date, Target date, Size,
  Estimate) on project items. Use when new issues need timeline fields, issues are
  closed, or running periodic sync to backfill missing data.
  Do NOT use for: creating issues (use issue-creator) or sprint planning
  (use planning-backlog).
last_updated: 2026-03-19
activation: manual
invocation: /sync-roadmap
metadata:
  category: operations
  tags: [github-projects, roadmap, timeline, automation]
  relations:
    compose_with:
      - planning-backlog
    depend_on: []
    similar_to: []
    belong_to: []
---

# Sync Roadmap Skill

## Purpose
Synchronize the GitHub Project Roadmap view by setting timeline fields
(Start date, Target date, Size, Estimate) on project items. Works for
both backfill (bulk) and incremental (new issues) updates.

## Activation
Use this skill when:
- A new issue is created and needs timeline fields set
- An issue is closed and needs its Target date updated
- Running a periodic sync to catch any items missing timeline data
- The user invokes `/sync-roadmap`

## GitHub Project Reference
- **Repository**: `minivess-mlops/minivess-mlops`
- **Project**: `Minivess MLOps` (ID: `PVT_kwDOCPpnGc4AYSAM`, Number: 1)

### Field IDs
| Field | ID | Type |
|-------|-----|------|
| Status | `PVTSSF_lADOCPpnGc4AYSAMzgPhgrw` | SingleSelect |
| Priority | `PVTSSF_lADOCPpnGc4AYSAMzgPhgsk` | SingleSelect |
| Start date | `PVTF_lADOCPpnGc4AYSAMzgPhgso` | Date |
| Target date | `PVTF_lADOCPpnGc4AYSAMzg-z7gU` | Date |
| Size | `PVTSSF_lADOCPpnGc4AYSAMzg-zBm8` | SingleSelect |
| Estimate | `PVTF_lADOCPpnGc4AYSAMzg-zBns` | Number |
| Iteration | `PVTIF_lADOCPpnGc4AYSAMzg-zB-I` | Iteration |

### Status Options
| Name | ID |
|------|-----|
| Backlog | `14dd5c1c` |
| Ready | `56933ea5` |
| In progress | `f6b5cf49` |
| In review | `782da6c8` |
| Done | `c82e9af3` |

### Priority Options
| Name | ID |
|------|-----|
| P0 | `c128192a` |
| P1 | `b419b06f` |
| P2 | `5bb35602` |
| P3 | `4b1f5dd3` |

### Size Options
| Name | ID | Estimate |
|------|----|----------|
| XS | `62e469d2` | 1 |
| S | `7e3dee78` | 2 |
| M | `184d9e87` | 3 |
| L | `9252603f` | 5 |
| XL | `763b524a` | 8 |

## Operations

### 1. Sync All (backfill mode)
Find all project items missing Start date or Target date and set them.

```bash
python3 scripts/sync_roadmap.py --mode backfill
```

### 2. Sync Single Issue
Set timeline fields for a specific issue number.

```bash
python3 scripts/sync_roadmap.py --issue 343
```

### 3. Sync Recently Closed
Find issues closed in the last N days and update their Target date.

```bash
python3 scripts/sync_roadmap.py --mode recent --days 7
```

## Timeline Rules

### Start date
- **New issue**: Set to `createdAt` date
- **Already set**: Do not overwrite

### Target date
- **Closed issue**: Set to `closedAt` date
- **Open issue**: Set to current iteration end date (or Sprint end)
- **Already set**: Overwrite only if issue was just closed (Target date
  should always reflect actual close date)

### Size Heuristic
| Condition | Size |
|-----------|------|
| Has label `bug` | XS |
| Has label `v0.1-legacy` | S |
| Title contains task ID (SAM-xx, Txx:, Deploy Task) | S |
| Issue #3-#22 (original PRD features) | L |
| Issue #23-#32 (scaffold phases) | M |
| Has label `research` | M |
| Default | S |

### Estimate
Derived from Size: XS=1, S=2, M=3, L=5, XL=8

## Script Location
`scripts/sync_roadmap.py` — standalone script, no dependencies beyond
`gh` CLI and Python 3.12+ stdlib.
