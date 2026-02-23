# Protocol: Create New Scenario

## When to Use
A new implementation path needs to be documented — a specific combination
of resolved decisions that represents a coherent project configuration.

## Steps

### 1. Choose Archetype
Which team profile is this scenario for?
- solo_researcher
- lab_group
- clinical_deployment

### 2. Choose Domain
Which domain overlay applies?
- vascular_segmentation
- cardiac_imaging
- neuroimaging
- general_medical

### 3. Resolve All 52 Decisions
For each decision in the network, choose exactly one option.
Use archetype preferences and domain overlays as guides.

### 4. Calculate Joint Probability
Multiply the (archetype-adjusted) probabilities of all chosen options.
This gives a rough joint probability for comparison between scenarios.

### 5. Document Trade-offs
What does this scenario gain vs. sacrifice compared to alternatives?

### 6. Save Scenario File
Use `templates/scenario.yaml` template.
Save to `docs/planning/prd/scenarios/{scenario-id}.scenario.yaml`.

### 7. Validate
Ensure all resolved_decisions reference valid decision_id.option_id pairs.
