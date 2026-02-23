# Protocol: Update Prior Probabilities

## When to Use
New evidence (benchmark results, paper findings, tool release, practical experience)
warrants updating probability distributions in one or more decision nodes.

## Steps

### 1. Identify Evidence
Document the evidence source:
- Paper citation or URL
- Benchmark results
- Practical experience (implementation outcome)
- Tool release or deprecation

### 2. Determine Affected Decisions
Which decision nodes are affected by this evidence?

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
6. Add note to `rationale` explaining the update

### 4. Update Conditional Tables
If the evidence also changes conditional relationships:
1. Update relevant `conditional_table` entries
2. Ensure all rows still sum to 1.0

### 5. Check Cascading Effects
Does this update affect child decisions? If a parent's prior shifts significantly:
- Review child conditional tables
- Consider if archetype weights need adjustment

### 6. Validate
Run the `validate` protocol.

## Example
```
Evidence: "VISTA-3D achieves 0.85 Dice on vessel segmentation in new benchmark"
Affected: segmentation_models, foundation_model_integration
Update: vista3d prior 0.25 → 0.30, segresnet 0.25 → 0.22, swinunetr 0.20 → 0.18
        lora_finetune prior 0.35 → 0.38 (VISTA-3D works well with LoRA)
```
