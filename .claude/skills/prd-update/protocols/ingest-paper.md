# Protocol: Ingest Research Paper

## When to Use
A new paper has been discovered that may affect technology decisions.

## Steps

### 1. Read the Paper
Extract key information:
- Title, authors, venue, date
- Core contribution (method, benchmark, framework)
- Evaluated technologies
- Performance results
- Relevance to medical image segmentation

### 2. Map to Decision Nodes
For each technology/method discussed:
1. Does it map to an existing decision? → `update-priors`
2. Does it represent a new option? → `add-option`
3. Does it represent a new decision category? → `add-decision`

### 3. Extract Evidence
For each relevant finding:
- What option does it support?
- How strong is the evidence? (benchmark vs. anecdotal)
- Is it domain-specific or general?

### 4. Apply Updates
Run the appropriate protocol for each update:
- `update-priors` for probability shifts
- `add-option` for new technologies
- `add-decision` for new decision categories

### 5. Document Source
Add paper to `references` in all affected decision files.
Add to `research_notes` if it provides important context.

### 6. Validate
Run the `validate` protocol.

## Example
```
Paper: "VISTA-3D: Foundation Model for 3D Medical Image Analysis"
Maps to: segmentation_models (add vista3d if not present)
         foundation_model_integration (boost lora_finetune)
         model_strategy (boost foundation_model_first)
Evidence: "0.85 Dice on vessel seg, 0.92 on BraTS with LoRA fine-tuning"
Strength: Strong (published benchmark on relevant datasets)
```
