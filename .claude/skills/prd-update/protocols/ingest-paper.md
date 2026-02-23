# Protocol: Ingest Research Paper

## When to Use
A new paper has been discovered that may affect technology decisions.

## CRITICAL: Citation Preservation
This protocol ADDS references. It NEVER removes existing references.
All citations extracted here become permanent records for peer review.

## Steps

### 1. Read the Paper — Extract Full Bibliographic Data
Extract and record in author-year format:
- **citation_key**: `firstauthor_year` (e.g., `isensee2021nnu`)
- **inline_citation**: Author-year string (e.g., "Isensee et al. (2021)")
- **authors**: Full author list ("Surname, A. B., Coauthor, C. D., & Last, E. F.")
- **year**: Publication year
- **title**: Full title as published
- **venue**: Journal/conference with volume, issue, pages
- **doi**: Digital Object Identifier (if available)
- **url**: Direct link (arXiv, publisher, or repository)
- **evidence_type**: benchmark | survey | tool_release | experience | theoretical | regulatory

Also extract:
- Core contribution (method, benchmark, framework)
- Evaluated technologies and their performance results
- Relevance to medical image segmentation
- Key tables and figures with specific results

### 2. Extract Sub-Citations
Read the ingested paper's reference list. For each cited work:
1. Is it relevant to any PRD decision node?
2. If YES → extract full bibliographic data (same format as Step 1)
3. Record which decision nodes each sub-citation affects

**This step is mandatory.** A single paper typically yields 3-10 relevant sub-citations
that strengthen the evidence base across multiple decision nodes.

### 3. Add to Central Bibliography
Add the paper AND all relevant sub-citations to `docs/planning/prd/bibliography.yaml`:
```yaml
- citation_key: surname2024
  inline_citation: "Surname et al. (2024)"
  authors: "Surname, A. B., Coauthor, C. D., & Last, E. F."
  year: 2024
  title: "Full Paper Title"
  venue: "Conference Name Year, Pages"
  doi: "10.xxxx/xxxxx"
  url: "https://arxiv.org/abs/xxxx.xxxxx"
  evidence_type: benchmark
  topics:
    - decision_id_1
    - decision_id_2
```

### 4. Map to Decision Nodes
For each technology/method discussed:
1. Does it map to an existing decision? → `update-priors`
2. Does it represent a new option? → `add-option`
3. Does it represent a new decision category? → `add-decision`

### 5. Extract Evidence with Citations
For each relevant finding, document with full citation:
- What option does it support? (link to option_id)
- How strong is the evidence? (benchmark vs. anecdotal)
- Is it domain-specific or general?
- What specific result? (e.g., "0.85 Dice on BraTS, Table 3")

### 6. Apply Updates
Run the appropriate protocol for each update:
- `update-priors` for probability shifts
- `add-option` for new technologies
- `add-decision` for new decision categories

### 7. Document References in Decision Files
For EACH affected `.decision.yaml`:
1. **Read existing references** (MUST preserve all)
2. **Append** new reference(s):
   ```yaml
   references:
     # ... existing references preserved ...
     - citation_key: surname2024
       relevance: "Why this reference matters for this decision"
       sections: ["Table 2", "Fig. 3"]
       supports_options: ["option_a", "option_b"]
   ```
3. **Update rationale** with in-text citation (author-year format):
   ```yaml
   rationale: >
     ... existing text preserved ... Recent work by Surname et al. (2024)
     demonstrates that option_a achieves 0.92 Dice on benchmark X (Table 2),
     supporting a prior increase from 0.25 to 0.30.
   ```

### 8. Validate
Run the `validate` protocol. Specifically check:
- [ ] All new `citation_key` values resolve in `bibliography.yaml`
- [ ] No existing references were removed from any decision file
- [ ] In-text citations match the references array
- [ ] Sub-citations are recorded in `bibliography.yaml`

## Example

```
Paper: "VISTA-3D: Foundation Model for 3D Medical Image Analysis"
       He, Y., Nath, V., Yang, D., Tang, Y., & Myronenko, A. (2024)

citation_key: he2024vista3d
inline_citation: "He et al. (2024)"
venue: arXiv preprint
doi: (none)
url: https://arxiv.org/abs/2406.05285
evidence_type: benchmark

Maps to:
  segmentation_models → add/boost vista3d (if not present)
  foundation_model_integration → boost lora_finetune
  model_strategy → boost foundation_model_first

Evidence:
  "0.85 Dice on vessel seg, 0.92 on BraTS with LoRA fine-tuning"
  (He et al., 2024, Table 3)
  Strength: Strong (published benchmark on relevant datasets)

Sub-citations extracted:
  - Myronenko (2019) → segmentation_models (SegResNet baseline comparison)
  - Hu et al. (2022) → foundation_model_integration (LoRA method used)
  - Isensee et al. (2021) → segmentation_models (nnU-Net comparison)
  - Hatamizadeh et al. (2022) → segmentation_models (SwinUNETR comparison)
```

## Checklist
- [ ] Full bibliographic data extracted (authors, year, title, venue, DOI/URL)
- [ ] citation_key follows `firstauthor_year` format
- [ ] Paper added to `bibliography.yaml`
- [ ] Sub-citations identified and added to `bibliography.yaml`
- [ ] All affected decision files updated (references array + rationale text)
- [ ] No existing references removed from any file
- [ ] In-text citations use author-year format: "Surname et al. (Year)"
- [ ] Validate protocol passes (including citation integrity check)
