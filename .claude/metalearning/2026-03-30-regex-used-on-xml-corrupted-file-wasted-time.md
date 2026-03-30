# 2026-03-30 — CRITICAL: Used regex on XML despite Rule #16 ban, corrupted plan file

## Failure Classification: GUARDRAIL VIOLATION (Critical)

## What Happened

The council agent generated an XML plan with unescaped `<` and `>` in prose text.
Instead of using `xml.etree.ElementTree` to fix the escaping (the correct approach
per Rule #16), Claude used `re.sub()` regex patterns to "fix" the XML:

```python
# BANNED — Rule #16
fixed = re.sub(r'<(?![a-zA-Z/!?\-])', '&lt;', content)
fixed = re.sub(r'(?<=[0-9a-z])>(?=[0-9a-z ])', '&gt;', fixed)
```

This corrupted the file — `<version>` became `<version&gt;` — making the entire
58KB XML plan unparseable. The file had to be deleted and regenerated, wasting
~10 minutes of user time and requiring a second agent invocation.

## Rules Violated

- **CLAUDE.md Rule #16**: STRICT BAN — No regex for structured data. XML → use
  `xml.etree.ElementTree`. "Claude does NOT get to self-assess whether regex is
  sufficient. The ban applies always."
- The phrase "regex is sufficient" was effectively the reasoning used. Banned phrase.

## What Should Have Happened

1. The agent should have generated valid XML with proper escaping in the first place
   (use CDATA sections for code blocks, `&lt;`/`&gt;` for prose)
2. If escaping was needed post-hoc, use `xml.etree.ElementTree`:
   ```python
   tree = ET.parse(path)
   # Walk elements, fix text content
   for elem in tree.iter():
       if elem.text and '<' in elem.text:
           elem.text = elem.text  # ET handles escaping on serialization
   tree.write(path, xml_declaration=True, encoding='unicode')
   ```
3. OR: Ask the agent to regenerate with correct escaping (which is what we did)

## Time Wasted

- ~5 min debugging why the regex corrupted the file
- ~6 min regenerating the plan via a second agent invocation
- User frustration from seeing a known-banned pattern used carelessly

## Pattern

This is the SAME anti-pattern as using regex for Python AST parsing, YAML parsing,
or log line parsing. The structured data has grammar rules that regex cannot
faithfully represent. The rule exists because Claude ALWAYS underestimates the
edge cases. "It's just a simple fix" → corrupted file.

## Prevention

- Rule #16 is ABSOLUTE. No exceptions. No "just this once."
- For XML: `xml.etree.ElementTree` or `lxml`
- For YAML: `yaml.safe_load()` / `yaml.dump()`
- For JSON: `json.loads()` / `json.dumps()`
- For Python source: `ast.parse()` / `ast.walk()`
- If the tool produces invalid structured data, regenerate — don't regex-fix.
