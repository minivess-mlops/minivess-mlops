# Skill: create-literature-report v1.0.0

> Reproducible scientific literature research report with citation-verified,
> hallucination-free synthesis. Composes engaging-review-writer (Markov novelty),
> citation-researcher (gap detection), citation-content-verifier (zero bullshit),
> and iterated-llm-council (convergence loop) into a single executable workflow.

## Trigger Phrases

- "create literature report"
- "research report on [topic]"
- "literature survey for [topic]"
- "write a research synthesis"

## Input Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `topic` | Yes | — | Research topic or question |
| `seed_papers` | Yes | — | List of seed paper citations (author-year + URL) |
| `output_path` | Yes | — | Path for the .md report file |
| `repo_context` | No | Current repo | Which repo/manuscript this informs |
| `target_paper_count` | No | 50 | Target total papers (seeds + discovered) |
| `quality_target` | No | MINOR_REVISION | Convergence target for council review |
| `create_issues` | No | true | Whether to create GitHub issues from findings |
| `update_kg` | No | true | Whether to propose KG decision node updates |
| `max_iterations` | No | 3 | Max council review iterations |

## Output Artifacts

1. **Report** (`{output_path}`) — Markdown with inline academic citations
2. **State** (`state/literature-report-state.json`) — Crash recovery
3. **Git commits** — One per phase, incremental progress
4. **GitHub issues** — P1/P2 from report findings (if `create_issues=true`)
5. **Verification log** — Per-citation VERIFIED/HALLUCINATED/MISMATCH status

## Architecture: 6-Phase Pipeline

```
Phase 0: CAPTURE & ALIGN
    ↓
Phase 1: GATHER (parallel research agents)
    ↓
Phase 2: SYNTHESIZE (Markov novelty-compliant writing)
    ↓
Phase 3: VERIFY (citation verification — zero tolerance)
    ↓
Phase 4: REVIEW (iterated-llm-council convergence loop)
    ↓
Phase 5: DELIVER (issues, KG updates, commit)
```

---

## Phase 0: CAPTURE & ALIGN

**Purpose**: Save prompt, establish scope, prevent misalignment.

**Steps**:
1. Save the user's prompt verbatim to `Appendix A` of the report
2. Save seed papers to `Appendix B` (numbered list with URLs)
3. Ask alignment questions via `AskUserQuestion` (max 4):
   - Framing angle for the report
   - Priority classification for issues (P1 vs P2)
   - Depth (lean / focused / comprehensive)
   - Any topics to explicitly defer or exclude
4. Save alignment decisions to `Appendix C`
5. Initialize `state/literature-report-state.json`:
   ```json
   {
     "phase": "CAPTURE",
     "topic": "...",
     "seed_count": 27,
     "discovered_count": 0,
     "verified_count": 0,
     "hallucinated_count": 0,
     "iteration": 0,
     "quality_verdict": null
   }
   ```
6. **CHECKPOINT**: Git commit — `docs: initialize literature report for {topic}`

**Exit criteria**: State file exists, prompt saved, alignment confirmed.

---

## Phase 1: GATHER (Parallel Research)

**Purpose**: Discover papers beyond the seed set. Target: `target_paper_count`.

**Steps**:
1. **Local inventory agent** (background):
   - Search repo for existing implementations/plans related to topic
   - Search `sci-llm-writer/biblio/` for relevant local papers
   - Report: {implemented, planned, research_only} inventory

2. **Web research agents** (2-3 in parallel, background):
   - Agent A: Search for **real implementations** (not surveys)
   - Agent B: Search for **MLOps/DevOps/automation** papers
   - Agent C (optional): Search for **domain-specific** papers
   - Each agent: 10-15 search queries, WebFetch to verify URLs exist
   - Each agent: Return structured list with citation, summary, relevance

3. **Deduplication**:
   - Remove papers that appear in multiple agent results
   - Remove papers already in seed list
   - Count unique papers discovered

4. **Update state**:
   ```json
   { "phase": "GATHER", "discovered_count": 35 }
   ```

5. **CHECKPOINT**: Git commit — `docs: gather {N} papers for {topic}`

**Exit criteria**: `seed_count + discovered_count >= target_paper_count`

**Anti-patterns**:
- NEVER include papers from LLM memory without URL verification
- NEVER fabricate DOIs or arXiv IDs
- If a paper seems relevant but URL returns 404 → DISCARD immediately

---

## Phase 2: SYNTHESIZE (Markov Novelty Writing)

**Purpose**: Write the report body with cross-domain synthesis.

**Protocol** (from engaging-review-writer):

### Pre-writing checklist
- [ ] Identify ≥3 previously disconnected domains to connect
- [ ] Formulate ≥3 specific research gaps (not "more research needed")
- [ ] Prepare ≥2 bold/controversial claims with supporting evidence
- [ ] Map all findings to the specific repo/manuscript context

### Writing rules (Markov principle)
1. **Synthesize, don't catalogue**: Every paragraph must connect ≥2 papers
2. **No annotated bibliography**: BANNED pattern — "Paper X found Y. Paper Z reported W."
3. **Cross-domain convergence**: The value is in what no single paper addresses
4. **"So What?" test per section**: Convergence? Gap? Implication? Vision? Controversy?
5. **All numbers attributed**: Never derive/calculate numbers as if findings
6. **Inline citations with hyperlinks**: `[Author et al. (Year). "Title." *Journal*.](URL)`

### Report structure
```markdown
# Title
## 1. Introduction (convergence framing)
## 2. The Landscape (what exists)
   ### 2.1-2.N Thematic subsections (NOT per-paper)
## 3. Platform Engineering Perspective
## 4. Implementable Extensions for THIS Repo
   ### 4.1-4.N Concrete features with P1/P2 classification
## 5. Regulatory/Ethical Alignment (if applicable)
## 6. Discussion: Novel Synthesis
   ### 6.1-6.N Cross-domain insights
   ### 6.N+1 Future Work Directions (for manuscript)
## 7. Recommended GitHub Issues
## 8. Academic Reference List
   ### Verified Seed Papers
   ### Web-Discovered Papers
## Appendix A: Original Prompt
## Appendix B: Alignment Decisions
```

### Quality markers (must be present)
- [ ] "We synthesize X and Y to propose Z" (novel framing)
- [ ] "While X addresses part of the problem, it misses..." (critical engagement)
- [ ] "This leads us to propose..." (forward contribution)
- [ ] No section is purely descriptive — all sections argue a position

**CHECKPOINT**: Git commit — `docs: synthesize {topic} report (draft)`

---

## Phase 3: VERIFY (Citation Verification — Zero Tolerance)

**Purpose**: Fetch every URL, verify every title, remove every hallucination.

**Protocol** (from citation-content-verifier + citation-validator):

### Step 1: Batch URL verification
Launch 2-3 parallel verification agents, each handling ~25 citations:
```
For each citation:
1. WebFetch the URL
2. Extract the actual title from the page
3. Extract the actual first author
4. Compare against claimed title/author/year
5. Classify: VERIFIED | TITLE_MISMATCH | AUTHOR_MISMATCH | YEAR_MISMATCH | URL_BROKEN | HALLUCINATED
```

### Step 2: Correction protocol
| Classification | Action |
|---------------|--------|
| VERIFIED | No change needed |
| TITLE_MISMATCH | Replace with actual verified title |
| AUTHOR_MISMATCH | Replace with actual first author |
| YEAR_MISMATCH | Replace with actual publication year |
| URL_BROKEN | Search for correct URL; if not found → REMOVE |
| HALLUCINATED | **REMOVE from report entirely** + remove from body text |

### Step 3: Body text audit
After removing hallucinated references from the reference list:
- Search body text for any inline citations to removed papers
- Remove or replace those citations
- Ensure no orphaned claims (claims that lost their only citation)

### Step 4: Update state
```json
{
  "phase": "VERIFY",
  "verified_count": 64,
  "hallucinated_count": 3,
  "corrections": [
    {"citation": "#33", "type": "AUTHOR_MISMATCH", "old": "Fang Y.", "new": "Fang J."}
  ]
}
```

### Zero-tolerance gates
- **0 HALLUCINATED** citations may remain in the final report
- **0 URL_BROKEN** links may remain
- **All titles must match** the actual paper (shortened acceptable, wrong title not)
- **All first authors must match**

**CHECKPOINT**: Git commit — `fix: citation verification — remove {N} hallucinated, fix {M} errors`

---

## Phase 4: REVIEW (Iterated LLM Council — Optional)

**Purpose**: If `quality_target` requires it, run council review for synthesis quality.

**Protocol** (from iterated-llm-council, simplified for reports):

### Reviewer agents (L3)
Spawn 2-3 domain expert agents to review the report:
1. **Domain expert**: Does the synthesis accurately represent the field?
2. **Methodology reviewer**: Are claims supported by cited evidence?
3. **Novelty assessor**: Does the report pass the Markov novelty test?

### Synthesis (L2)
Aggregate reviewer feedback into a verdict:
- ACCEPT: Report is publication-ready
- MINOR_REVISION: Small fixes needed (typos, missing citations, weak sections)
- MAJOR_REVISION: Structural issues (missing themes, annotated-bibliography sections)

### Convergence
- If verdict >= `quality_target` → DONE
- If verdict < `quality_target` and `iteration < max_iterations` → Fix and re-verify
- If `iteration >= max_iterations` → DONE with warning

### Per-iteration fixes
1. Apply reviewer feedback
2. Re-run Phase 3 (VERIFY) on any new citations
3. Update state with new iteration count

**CHECKPOINT**: Git commit per iteration

---

## Phase 5: DELIVER

**Purpose**: Create all downstream artifacts.

**Steps**:
1. **GitHub issues** (if `create_issues=true`):
   - Parse Section 7 (Recommended Issues) from report
   - Create P1 issues with implementation scope
   - Create P2 issues with research scope
   - Log issue numbers in report

2. **KG updates** (if `update_kg=true`):
   - Identify stale KG decision nodes referenced in report
   - Propose posterior probability updates based on new evidence
   - Create/update YAML files (human review before merge)

3. **Final state update**:
   ```json
   {
     "phase": "COMPLETE",
     "verified_count": 64,
     "hallucinated_count": 3,
     "issues_created": ["#845", "#846", ...],
     "iterations": 1,
     "quality_verdict": "MINOR_REVISION"
   }
   ```

4. **Final commit + push**

---

## State Management

### State file: `state/literature-report-state.json`

```json
{
  "skill_version": "1.0.0",
  "topic": "Biomedical Agentic AI",
  "output_path": "docs/planning/biomedical-agentic-ai-research-report.md",
  "phase": "VERIFY",
  "seed_count": 27,
  "discovered_count": 43,
  "verified_count": 64,
  "hallucinated_count": 3,
  "corrections_applied": 6,
  "iteration": 0,
  "quality_verdict": null,
  "issues_created": [],
  "started_at": "2026-03-18T22:00:00Z",
  "last_checkpoint": "2026-03-18T23:30:00Z"
}
```

### Session recovery
At session start, if state file exists:
1. Read state
2. Resume from `phase`
3. Skip completed phases
4. Continue from last checkpoint

---

## Anti-Patterns (BANNED)

| Pattern | Why Dangerous | Detection |
|---------|--------------|-----------|
| **Memory-based citations** | LLMs hallucinate paper titles/DOIs | Phase 3 catches all |
| **Annotated bibliography** | Violates Markov novelty principle | Phase 4 reviewer detects |
| **Citation spam** | >3 citations per sentence without integration | Engaging-review-writer rules |
| **Serial verification** | Fix one citation, re-run, find another (whac-a-mole) | Phase 3 verifies ALL at once |
| **Skip verification** | "I'm confident these are real" | Phase 3 is MANDATORY, not optional |
| **Unattributed numbers** | Statistics without source | Phase 4 reviewer checks |
| **Generic gaps** | "More research needed" | "So What?" test |

---

## Skill Composition Map

```
create-literature-report
├── Phase 0: CAPTURE
│   └── AskUserQuestion (alignment)
├── Phase 1: GATHER
│   ├── Agent (local inventory — Explore type)
│   ├── Agent (web research A — general-purpose)
│   ├── Agent (web research B — general-purpose)
│   └── Deduplication logic
├── Phase 2: SYNTHESIZE
│   ├── engaging-review-writer (Markov novelty rules)
│   ├── citation-researcher (gap detection, novelty scoring)
│   └── Report structure template
├── Phase 3: VERIFY
│   ├── citation-content-verifier (URL fetch + title match)
│   ├── Parallel verification agents (2-3)
│   └── Hallucination removal protocol
├── Phase 4: REVIEW (optional)
│   ├── iterated-llm-council (L3→L2→L1 cycle)
│   └── Convergence check
└── Phase 5: DELIVER
    ├── GitHub issue creation
    ├── KG update proposals
    └── Final commit
```

---

## Eval Criteria (Skill v2.0 Targets)

### Quantitative
| Metric | Target | Measurement |
|--------|--------|-------------|
| Hallucination rate | 0% | Phase 3 verified / total |
| URL validity | 100% | All links resolve to real pages |
| Title accuracy | 100% | All titles match actual paper |
| Author accuracy | 100% | All first authors match |
| Paper count | >= target_paper_count | Verified count |
| Markov novelty score | >= 2.0 per section | "So What?" test (5 criteria) |
| Citation density | 2-5 per paragraph | No citation deserts or spam |
| Issue creation | All P1/P2 from report | Cross-reference §7 vs GitHub |

### Qualitative
| Criterion | Pass/Fail |
|-----------|-----------|
| No annotated bibliography sections | Pass = no sequential "Paper X found Y" |
| Cross-domain synthesis | Pass = ≥3 domains connected |
| Specific research gaps | Pass = ≥3 specific gaps (not generic) |
| Repo-contextualized | Pass = findings mapped to specific repo features |
| Actionable issues | Pass = issues have scope, dependencies, references |

### Regression test (for v2.0)
Run the skill on the biomedical-agentic-ai topic with the same 27 seeds.
Expected: 0 hallucinations, ≥60 verified papers, ≥3 novel synthesis insights.
Time budget: <45 minutes total (all phases).

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-03-18 | Initial release. Codifies biomedical agentic AI report workflow. |

## References

- `/home/petteri/Dropbox/github-personal/sci-llm-writer/.claude/skills/iterated-llm-council/SKILL.md`
- `/home/petteri/Dropbox/github-personal/sci-llm-writer/.claude/skills/engaging-review-writer/SKILL.md`
- `/home/petteri/Dropbox/github-personal/sci-llm-writer/.claude/skills/citation-content-verifier/SKILL.md`
- `/home/petteri/Dropbox/github-personal/sci-llm-writer/.claude/skills/citation-researcher/SKILL.md`
- `/home/petteri/Dropbox/github-personal/sci-llm-writer/.claude/skills/academic-qa/SKILL.md`
- `/home/petteri/Dropbox/github-personal/minivess-mlops/.claude/metalearning/2026-03-18-whac-a-mole-serial-failure-fixing.md`
