# Protocol: Validate PRD Integrity

## When to Use
After any change to the PRD. This protocol checks all invariants.

## Validation Checks

### 1. DAG Acyclicity
Read `_network.yaml` and verify no cycles exist:
- Build adjacency list from edges
- Run topological sort
- If sort fails → CYCLE DETECTED (list the cycle)

### 2. Probability Sums
For every `.decision.yaml` file:
- [ ] `options[].prior_probability` sums to 1.0 (±0.001 tolerance)
- [ ] Each `conditional_table[].then_probabilities` sums to 1.0
- [ ] Each `archetype_weights[].probability_overrides` sums to 1.0

### 3. Cross-Reference Resolution
- [ ] Every `conditional_on[].parent_decision_id` exists as a node in `_network.yaml`
- [ ] Every edge in `_network.yaml` has corresponding nodes
- [ ] Every node in `_network.yaml` has a corresponding `.decision.yaml` file
- [ ] Every `.decision.yaml` file has a corresponding node in `_network.yaml`

### 4. Option ID Consistency
For each conditional table entry:
- [ ] `given_parent_option` matches an `option_id` in the parent's `.decision.yaml`
- [ ] All keys in `then_probabilities` match `option_id`s in the current `.decision.yaml`

### 5. Scenario Validation
For each `.scenario.yaml`:
- [ ] Every key in `resolved_decisions` matches a `decision_id` in `_network.yaml`
- [ ] Every value matches an `option_id` in the corresponding `.decision.yaml`

### 6. Schema Compliance
- [ ] Every `.decision.yaml` has all required fields per `_schema.yaml`
- [ ] All `decision_level` values match allowed enums
- [ ] All `status` values match allowed enums

### 7. Citation Integrity (Academic Standards)
This check ensures peer-review readiness of all citations.

#### 7a. Bibliography Resolution
- [ ] Every `citation_key` in any `.decision.yaml` `references` array
      MUST exist in `bibliography.yaml`
- [ ] Every `bibliography.yaml` entry has non-empty `authors`, `year`, `title`
- [ ] Every `bibliography.yaml` entry has at least one of: `doi`, `url`

#### 7b. Citation Completeness
- [ ] Every `.decision.yaml` SHOULD have at least one entry in `references`
      (WARNING if empty — all decisions should cite evidence)
- [ ] Every `bibliography.yaml` entry's `topics` array matches the decision
      files that actually cite it (WARNING on mismatch)

#### 7c. In-Text Citation Consistency
- [ ] `rationale` fields SHOULD contain at least one author-year citation
      matching `(Surname, Year)` or `(Surname et al., Year)` pattern
- [ ] Each in-text citation SHOULD correspond to a `citation_key` in the
      file's `references` array (WARNING if unmatched)

#### 7d. No Citation Loss (CRITICAL — blocks commit)
When validating after an update:
- [ ] Compare current references arrays with previous git state
- [ ] If any `citation_key` was REMOVED from any file → **FAIL**
      (removals require explicit user approval)
- [ ] If any `bibliography.yaml` entry was deleted → **FAIL**

### 8. Cross-Topic Consistency
- [ ] Every `bibliography.yaml` entry's `topics` includes all `decision_id`s
      where that `citation_key` appears in `references`
- [ ] `supports_options` in decision references match actual `option_id`s

## Output
Report results as:
```
PRD Validation Report
=====================
DAG Acyclicity:        PASS/FAIL
Probability Sums:      PASS/FAIL (N violations)
Cross-References:      PASS/FAIL (N broken refs)
Option IDs:            PASS/FAIL (N mismatches)
Scenarios:             PASS/FAIL (N issues)
Schema Compliance:     PASS/FAIL (N violations)
Citation Integrity:    PASS/FAIL (N issues)
  - Bibliography:      PASS/FAIL (N unresolved keys)
  - Completeness:      PASS/WARN (N decisions without refs)
  - In-Text Format:    PASS/WARN (N format issues)
  - No Citation Loss:  PASS/FAIL (N removals detected)
Cross-Topic:           PASS/WARN (N mismatches)
---------------------
Overall:               PASS/FAIL
```

**Severity Levels:**
- **FAIL** = blocks commit, must fix
- **WARN** = should fix, reported but doesn't block

## Quick Fix Guide
- **Probability doesn't sum to 1.0**: Adjust the smallest option(s) to compensate
- **Broken cross-reference**: Check for typos in decision_id or option_id
- **Missing file**: Create from template or remove node from network
- **Cycle detected**: Remove the edge that creates the cycle (usually the newest)
- **Unresolved citation_key**: Add entry to `bibliography.yaml` or fix typo
- **Missing references**: Add at least one reference per decision using `citation-guide`
- **Citation removed**: Restore from git history — citation removal needs user approval
- **In-text format**: Use "Surname et al. (Year)" format, never numeric
