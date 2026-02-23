# Protocol: Update Prior Probabilities

## When to Use
New evidence (benchmark results, paper findings, tool release, practical experience)
warrants updating probability distributions in one or more decision nodes.

## CRITICAL: Citation Preservation
When updating a decision file, ALL existing references MUST be preserved.
New evidence MUST be cited with author-year format. Never remove citations.

## Steps

### 1. Identify Evidence with Full Citation
Document the evidence source in author-year format:
- **Paper**: "Surname et al. (Year)" + full bibliographic entry
- **Benchmark**: Dataset name, metric, value + paper citation
- **Tool release**: Tool name, version, URL + citation if published
- **Practical experience**: Implementation context, outcome, date

If the evidence comes from a paper, ensure it is in `bibliography.yaml`.
If not, add it using the `citation-guide` protocol.

### 2. Determine Affected Decisions
Which decision nodes are affected by this evidence?
List each `decision_id` and the relevant `option_id`(s).

### 3. Apply Bayesian Update
For each affected decision:
1. Read current prior probabilities
2. Assess likelihood: How likely is this evidence given each option?
3. Compute posterior (informal Bayesian reasoning):
   - If evidence strongly supports option X: increase X by 0.05-0.15
   - If evidence moderately supports: increase by 0.03-0.08
   - Redistribute probability from other options (total must remain 1.0)
4. Update `prior_probability` values
5. Update `last_updated` date
6. **Update `rationale`** — APPEND the update reason with citation:
   ```
   Recent benchmark by Surname et al. (2024) shows option_a
   achieves 0.92 Dice on dataset X (Table 3), shifting prior
   from 0.25 → 0.30. Probability redistributed from option_b
   and option_c.
   ```
   **PRESERVE all existing rationale text and citations.**

### 4. Update References Array
For each affected decision file:
1. Read existing `references` array (DO NOT modify or remove any entries)
2. Append new reference if not already present:
   ```yaml
   - citation_key: surname2024
     relevance: "Evidence for prior update: 0.92 Dice on X"
     sections: ["Table 3"]
     supports_options: ["option_a"]
   ```

### 5. Update Conditional Tables
If the evidence also changes conditional relationships:
1. Update relevant `conditional_table` entries
2. Ensure all rows still sum to 1.0

### 6. Check Cascading Effects
Does this update affect child decisions? If a parent's prior shifts significantly:
- Review child conditional tables
- Consider if archetype weights need adjustment

### 7. Validate
Run the `validate` protocol. Check:
- [ ] All probabilities sum to 1.0
- [ ] No references were lost
- [ ] New citation_key exists in `bibliography.yaml`
- [ ] Rationale includes author-year citation for the update

## Example
```
Evidence: "VISTA-3D achieves 0.85 Dice on vessel segmentation"
          (He et al., 2024, Table 3)
Source: he2024vista3d in bibliography.yaml

Affected: segmentation_models, foundation_model_integration
Update:
  segmentation_models:
    vista3d prior 0.25 → 0.30
    segresnet 0.25 → 0.22, swinunetr 0.20 → 0.18
    (redistributed -0.07 across lower options)
  foundation_model_integration:
    lora_finetune prior 0.35 → 0.38
    (VISTA-3D uses LoRA, He et al. 2024)

Rationale update (appended):
  "He et al. (2024) report 0.85 Dice on vessel segmentation with
  VISTA-3D using LoRA fine-tuning, supporting a prior increase for
  both vista3d (+0.05) and lora_finetune (+0.03)."
```
