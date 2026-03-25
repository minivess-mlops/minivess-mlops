# Evaluation Checklist: readme-enrichment

## Structural Criteria (machine-verifiable)

1. **Zero broken links**: Every `[text](URL)` in README.md returns HTTP 200 when fetched
2. **Bibliography consistency**: Every paper link in README matches a `bibliography.yaml` entry
3. **No orphan references**: Every tool/paper/standard mentioned in README has a `[Name](URL)` link

## Behavioral Criteria (require judgment)

4. **No hallucinated URLs**: Every URL was verified via WebFetch/WebSearch before insertion
5. **Format consistency**: Links follow R3 convention (tool=homepage, paper=DOI/ArXiv, standard=official)
6. **Planned section enriched**: Science issues linked, tools in Planned have homepage links

## Should-Trigger Prompts

- "Add links to all tools in the README"
- "The README has missing hyperlinks, fix them"
- "Enrich the README with paper citations"

## Should-NOT-Trigger Prompts

- "Write a literature review about vessel segmentation" → use create-literature-report
- "Update the knowledge graph" → use kg-sync
