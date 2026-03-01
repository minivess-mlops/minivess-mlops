# Protocol: Academic Citation Guide

## Purpose
The probabilistic SDD is designed for **academic software engineering projects**.
All references must follow academic citation standards. This guide is the canonical
reference for citation formatting across all SDD files.

## Citation Architecture

### Central Bibliography
`bibliography.yaml` is the single source of truth for all cited works.
Every reference in a `.decision.yaml` file MUST have a corresponding entry here.

### Per-Decision References
Each `.decision.yaml` file has a `references` array linking to bibliography entries
by `citation_key`, with decision-specific relevance notes.

### In-Text Citations
Use author-year format in all prose fields (`description`, `rationale`, `research_notes`):

| Authors | Format | Example |
|---------|--------|---------|
| 1 author | Surname (Year) | Myronenko (2019) |
| 2 authors | Surname & Surname (Year) | Angelopoulos & Bates (2023) |
| 3+ authors | Surname et al. (Year) | Hatamizadeh et al. (2022) |
| Organization | Organization (Year) | IEC (2015) |

Parenthetical form for supporting claims: `(Surname et al., Year)`

## Adding a New Reference

### Step 1: Check for Existing Entry
Search `bibliography.yaml` for the paper. If it exists, use its `citation_key`.

### Step 2: Create Citation Key
Format: `firstauthor_surname + year` in lowercase.
- `isensee2021nnu` (for Isensee et al., 2021, nnU-Net paper)
- `hu2022lora` (for Hu et al., 2022, LoRA paper)
- If ambiguous (same author + year): add suffix `a`, `b`, etc.
- If key word needed for clarity: add it after year

### Step 3: Add to bibliography.yaml
Required fields:
```yaml
- citation_key: surname2024
  inline_citation: "Surname et al. (2024)"
  authors: "Surname, A. B., Coauthor, C. D., & Last, E. F."
  year: 2024
  title: "Full Paper Title as Published"
  venue: "Conference/Journal Name, Volume(Issue), Pages"
  doi: "10.xxxx/xxxxx"
  url: "https://arxiv.org/abs/xxxx.xxxxx"
  evidence_type: benchmark    # benchmark | survey | tool_release | experience | theoretical | regulatory
  topics:                     # decision_ids this reference is relevant to
    - decision_id_1
    - decision_id_2
```

### Step 4: Add to Decision File(s)
In each relevant `.decision.yaml`:
```yaml
references:
  - citation_key: surname2024
    relevance: "Introduces the architecture used as option X"
    sections: ["Table 2", "Section 3.1"]
    supports_options: ["option_x", "option_y"]
```

### Step 5: Add In-Text Citation
Update the decision's `rationale`, `description`, or `research_notes` to include
the author-year citation.

## Sub-Citation Tracking

When ingesting a paper, also extract its key references that are relevant to
the SDD decision network. These are **sub-citations** — papers cited BY the
ingested paper that provide evidence for other decision nodes.

### Process
1. Read the ingested paper's reference list
2. Identify references relevant to SDD decision nodes
3. Add each sub-citation to `bibliography.yaml`
4. Link sub-citations to the appropriate `.decision.yaml` files

## Evidence Types

| Type | Description | Typical Probability Shift |
|------|-------------|--------------------------|
| `benchmark` | Published evaluation on established datasets | +0.05 to +0.15 |
| `survey` | Systematic review or meta-analysis | +0.03 to +0.08 |
| `tool_release` | New software release or major version | +0.03 to +0.10 |
| `experience` | Practical implementation outcome | +0.02 to +0.05 |
| `theoretical` | Analytical or proof-based contribution | +0.01 to +0.05 |
| `regulatory` | Standards body publication | Context-dependent |

## Validation Rules (checked by validate protocol)
1. Every `citation_key` in a `.decision.yaml` MUST exist in `bibliography.yaml`
2. Every decision file SHOULD have at least one reference
3. In-text citations in `rationale` SHOULD match entries in `references`
4. `bibliography.yaml` entries SHOULD have non-empty `doi` or `url`
5. Author-year format MUST be used consistently (no numeric citations)

## Venue Formatting
- **Journals**: "Journal Name, Volume(Issue), StartPage--EndPage"
- **Conferences**: "Conference Name Year" or "Workshop@Conference Year"
- **Preprints**: "arXiv preprint" (with URL)
- **Software**: "Software Release" (with URL)
- **Standards**: "International Standard" (with URL)
- **Books**: "Publisher, Edition"
