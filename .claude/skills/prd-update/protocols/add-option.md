# Protocol: Add Option to Existing Decision

## When to Use
A new technology option has been discovered for an existing decision node.
For example: a new segmentation model, a new HPO framework, etc.

## Steps

### 1. Read Existing Decision
Load the `.decision.yaml` file and understand current options.

### 2. Define New Option
- `option_id` (snake_case, unique within this decision)
- `title` (human-readable)
- `description` (what it is, when to use it)
- `prior_probability` (initial probability)
- `status` (viable/experimental/recommended)
- `implementation_status` (not_started/config_only/partial/implemented)

### 3. Redistribute Probabilities
The new option "steals" probability from existing options:
1. Assign initial prior to new option (typically 0.05-0.15 for new discoveries)
2. Reduce other options proportionally so total remains 1.0
3. More established options lose less; similar options lose more

### 4. Update Conditional Tables
For each parent decision's conditional table:
1. Add the new option_id to every `then_probabilities` row
2. Redistribute within each row to maintain sum = 1.0

### 5. Update Archetype Weights
For each archetype:
1. Add new option to `probability_overrides`
2. Redistribute to maintain sum = 1.0

### 6. Update Complementary Decisions
If other decisions reference this one in `complements`, update them.

### 7. Validate
Run the `validate` protocol.
