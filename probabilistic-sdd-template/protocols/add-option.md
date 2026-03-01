# Protocol: Add Option to Existing Decision

## When to Use
A new technology option has been discovered for an existing decision node.
For example: a new model architecture, a new framework, etc.

## CRITICAL: Citation Preservation
When modifying an existing decision file, ALL existing references MUST be preserved.
The new option SHOULD cite at least one academic reference.

## Steps

### 1. Read Existing Decision
Load the `.decision.yaml` file and understand current options.
**Read and note all existing references — these must be preserved.**

### 2. Gather Academic References
Before defining the new option:
1. Identify the key paper(s) introducing or evaluating this technology
2. Check if they exist in `bibliography.yaml` — if not, add them
3. Use `citation-guide` protocol for correct format

### 3. Define New Option
- `option_id` (snake_case, unique within this decision)
- `title` (human-readable)
- `description` (include author-year citation: "Introduced by Surname et al. (Year), ...")
- `prior_probability` (initial probability)
- `status` (viable/experimental/recommended)
- `implementation_status` (not_started/config_only/partial/implemented)

### 4. Redistribute Probabilities
The new option "steals" probability from existing options:
1. Assign initial prior to new option (typically 0.05-0.15 for new discoveries)
2. Reduce other options proportionally so total remains 1.0
3. More established options lose less; similar options lose more

### 5. Update Conditional Tables
For each parent decision's conditional table:
1. Add the new option_id to every `then_probabilities` row
2. Redistribute within each row to maintain sum = 1.0

### 6. Update Archetype Weights
For each archetype:
1. Add new option to `probability_overrides`
2. Redistribute to maintain sum = 1.0

### 7. Update References
1. **Preserve ALL existing references** (do not modify or remove any)
2. **Append** new reference(s) for the added option:
   ```yaml
   - citation_key: surname2024
     relevance: "Introduces the new option added to this decision"
     supports_options: ["new_option_id"]
   ```
3. **Update rationale** — APPEND explanation with citation:
   ```
   Added new_option based on Surname et al. (2024) who demonstrate
   competitive performance on benchmark X. Initial prior set to 0.10,
   redistributed from existing options.
   ```

### 8. Update Complementary Decisions
If other decisions reference this one in `complements`, update them.

### 9. Validate
Run `validate.py --sdd-root <your-sdd>`. Check:
- [ ] All probabilities sum to 1.0
- [ ] No existing references were removed
- [ ] New citation_keys exist in bibliography.yaml
- [ ] Author-year citations in rationale match references array
