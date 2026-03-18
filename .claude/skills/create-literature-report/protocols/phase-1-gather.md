# Phase 1: GATHER — Parallel Research

**Time budget**: 15 minutes
**Exit criteria**: `seed_count + discovered_count >= target_paper_count`

## Step 1.1: Construct Search Queries

### Query generation algorithm
For each seed paper, extract 2-3 key terms from the title. Cross these with
domain qualifiers to produce search queries. Example:

```
Seed: "Agentic Systems in Radiology" (Bluethgen 2025)
Key terms: "agentic systems", "radiology"
Queries:
  - "agentic systems" medical imaging implementation 2025
  - "radiology AI agent" multi-step reasoning
```

### Domain qualifier sets (choose based on topic)
- **Biomedical**: medical imaging, clinical, pathology, radiology, segmentation
- **MLOps**: pipeline automation, experiment tracking, drift monitoring, deployment
- **Scientific**: self-driving lab, hypothesis generation, experiment design

## Step 1.2: Launch Parallel Agents

Spawn ALL agents simultaneously using the Agent tool with `run_in_background: true`.

### Agent A: Local Inventory (Explore type)
```
Search {repo_context} and {biblio_search_paths} for existing
implementations, plans, and literature related to "{topic}".
Report as: {implemented: [...], planned: [...], research_only: [...]}
```

### Agent B: Web Research — Domain (general-purpose type)
Use prompt template from `prompts/gather-web-research.md` with domain-specific qualifiers.

### Agent C: Web Research — Infrastructure (general-purpose type)
Use prompt template from `prompts/gather-web-research.md` with MLOps/automation qualifiers.

## Step 1.3: Wait for All Agents

**Synchronization barrier**: Do NOT proceed to deduplication until ALL agents complete.
Check completion via output file existence/size. If an agent has not completed after
15 minutes, proceed with available results and note the gap in state.

## Step 1.4: Deduplication

### Algorithm (priority order)
1. **DOI match**: If two papers share a DOI → keep first occurrence
2. **URL match**: If two papers share a URL → keep first occurrence
3. **Title fuzzy match**: If normalized titles are >90% similar (lowercase, strip punctuation) → keep the one with more metadata

### Against seed list
Remove any paper whose first author + year matches a seed paper.

### Output
Write `workspace/literature-report-{topic_slug}/phase-1-gather/deduplicated-papers.json`

## Step 1.5: Update State

```json
{
  "phase": "GATHER",
  "substep": "dedup_complete",
  "discovered_count": 43,
  "agents_completed": {
    "gather_local": true,
    "gather_web_a": true,
    "gather_web_b": true
  }
}
```

## Step 1.6: CHECKPOINT

Git commit: `docs: gather {N} papers for {topic}`

## Fallback: Insufficient Papers

If `seed_count + discovered_count < target_paper_count * 0.7`:
1. Log a warning in state: `"gather_warning": "below 70% of target"`
2. Generate 5 additional cross-domain queries from underrepresented themes
3. Run ONE more web search agent with these queries
4. If still below target: proceed anyway (quality over quantity)
