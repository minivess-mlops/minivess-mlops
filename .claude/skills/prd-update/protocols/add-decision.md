# Protocol: Add Decision Node

## When to Use
A new technology or architecture decision needs to be captured in the PRD.

## Steps

### 1. Determine Decision Level
- L1_research_goals — Strategic project direction
- L2_architecture — Architectural patterns and approaches
- L3_technology — Specific tools and libraries
- L4_infrastructure — Compute, containers, CI/CD
- L5_operations — Monitoring, governance, workflows

### 2. Identify Parent Decisions
- What upstream decisions influence this one?
- What is the influence strength (strong/moderate/weak)?
- Draft conditional probability tables

### 3. Gather Academic References
Before creating the decision file:
1. Identify key papers/sources for each option (minimum 1 per option recommended)
2. Check if they exist in `bibliography.yaml` — if not, add them
3. Use `citation-guide` protocol for correct format
4. Record which sections/tables/figures provide evidence

### 4. Create Decision File
Use template at `templates/decision-node.yaml`:
1. Choose a unique `decision_id` (snake_case)
2. Define 2-5 options with prior probabilities (MUST sum to 1.0)
3. Write conditional tables for each parent (rows MUST sum to 1.0)
4. Add archetype weights for: solo_researcher, lab_group, clinical_deployment
5. Set volatility classification (stable/shifting/volatile)
6. Set domain applicability scores (0.0-1.0)
7. **Write rationale with author-year citations** — every claim must be backed
8. **Populate references array** — link to `bibliography.yaml` entries

### 5. Update Network
In `docs/planning/prd/decisions/_network.yaml`:
1. Add node entry under appropriate level
2. Add edge entries for parent→child relationships
3. Increment network version

### 6. Validate
Run the `validate` protocol to check:
- No cycles introduced
- All probabilities sum to 1.0
- All cross-references resolve
- File exists for the new node
- All citation_keys resolve in bibliography.yaml

### 7. Update Affected Scenarios
If the new decision affects active scenarios, update their `resolved_decisions`.

## Checklist
- [ ] decision_id is unique and snake_case
- [ ] All prior_probability values sum to 1.0
- [ ] All conditional_table rows sum to 1.0
- [ ] All archetype probability_overrides sum to 1.0
- [ ] Node added to _network.yaml
- [ ] Edges added to _network.yaml
- [ ] File saved as `{decision-id}.decision.yaml` in correct L* directory
- [ ] **References array populated** (at least 1 reference per decision)
- [ ] **All citation_keys exist in bibliography.yaml**
- [ ] **Rationale contains author-year citations** matching references
- [ ] **No existing references removed** (if updating an existing file)
- [ ] Validate protocol passes (including citation integrity)
