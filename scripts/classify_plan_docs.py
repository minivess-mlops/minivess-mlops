"""Classify 347 planning docs into 13 themes for the v0.2-beta plan archive.

Usage:
    uv run python scripts/classify_plan_docs.py              # Print classification JSON
    uv run python scripts/classify_plan_docs.py --summary     # Print theme counts
    uv run python scripts/classify_plan_docs.py --unclassified # Show only unclassified docs

No regex (CLAUDE.md Rule 16). Uses filename keywords + YAML frontmatter parsing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ARCHIVE_DIR = Path("docs/planning/v0-2_archive/original_docs")

# Theme classification rules: ordered keyword lists per theme.
# First match wins. More specific patterns come first.
THEME_RULES: dict[str, list[str]] = {
    "models": [
        "sam3",
        "sam-3",
        "mambavesselnet",
        "mamba-vessel",
        "vesselfm",
        "vessel-fm",
        "dynunet",
        "atlassegfm",
        "atlas-seg",
        "foundation-model",
        "comma-mamba",
        "monai-segmentor",
        "model-capacity",
        "adapter",
    ],
    "training": [
        "loss-",
        "losses",
        "topology-aware",
        "topology-loss",
        "graph-topology",
        "boundary-loss",
        "cldice",
        "compound-loss",
        "metrics-reporting",
        "multi-metric",
        "augmentation",
        "graph-constrained",
        "calibration-shift",
        "calibration-and-swag",
        "loss-metric-improvement",
        "novel-loss",
        "loss-and-metrics",
        "segmentation-calibration",
    ],
    "evaluation": [
        "biostatist",
        "factorial",
        "ensemble",
        "conformal-uq",
        "post-training",
        "swa-swag",
        "swag-dataloader",
        "specification-curve",
        "preregistration-statistical",
        "analysis-flow",
        "evaluation-and-ensemble",
        "quasi-e2e",
        "final-methods",
        "multi-validation-metric",
        "experiment-planning",
    ],
    "cloud": [
        "skypilot",
        "sky-pilot",
        "gcp-",
        "runpod",
        "run-pod",
        "pulumi",
        "cloud-architecture",
        "cloud-tutorial",
        "hetzner",
        "lambda",
        "spot-preemption",
        "spot-resume",
        "finops",
        "s3-mounting",
        "gcp-vs-",
        "gcp-phase",
        "gcp-spot",
        "run-debug-factorial",
        "pre-gcp",
        "preflight",
        "remaining-runpod",
        "compute-offloading",
    ],
    "infrastructure": [
        "docker-",
        "prefect-",
        "devex-",
        "ci-reporting",
        "profiler-",
        "overnight-",
        "batch-script",
        "robustifying-flaky",
        "modernize-minivess",
        "infrastructure-timing",
        "prefect-container",
        "warning-logger",
        "vision-enforcement",
        "root-cause-bug",
    ],
    "data": [
        "data-",
        "dvc-",
        "dataset-",
        "drift-",
        "synthetic-",
        "vesselnn-",
        "deepvess",
        "tubenet",
        "test-sets-external",
        "zarr-vs-pt",
    ],
    "observability": [
        "mlflow-",
        "mlruns-",
        "monitoring-research",
        "experiment-run-to-mlflow",
        "gpu-params-logging",
        "tracking-plan",
    ],
    "operations": [
        "regulatory",
        "fda-",
        "eu-ai-act",
        "iec62304",
        "complops",
        "regops",
        "openlineage",
        "ge-validation",
        "medical-mlops-standards",
    ],
    "testing": [
        "test-suite",
        "staging-prod",
        "debug-training-testing",
        "e2e-testing",
        "tdd-skill",
        "double-check-all-wiring",
        "code-review-report",
        "final-verification",
    ],
    "manuscript": [
        "lit-report-",
        "repo-to-manuscript",
        "fuel-for-manuscript",
        "preregistration-tripod",
        "tripod-compliance",
        "ai-cards",
        "cover-letter-to-sci",
        "research-reports-general",
    ],
    "harness": [
        "cold-start-",
        "intermedia-plan-synthesis",
        "claude-agent",
        "claude-harness",
        "context-compounding",
        "context7-vs",
        "agentic-",
        "intent-summary",
        "tdd-skill-upgrade",
        "plan-synthesis",
        "avoid-silent-existing",
        "hierarchical-prd",
        "knowledge-management",
        "prd-",
    ],
    "deployment": [
        "bentoml",
        "model-registry",
        "monai-deploy",
        "deploy-flow",
        "annotation",
        "medsam3-annotation",
        "interactive-segmentation",
        "serving",
    ],
}

# Manual overrides for docs that don't match keywords or need reclassification.
# NO duplicate keys allowed.
MANUAL_OVERRIDES: dict[str, str] = {
    # Architecture (from infrastructure/harness keyword matches)
    "hydra-config-verification-report": "architecture",
    "hydra-double-check": "architecture",
    "script-consolidation": "architecture",
    "prefect-flow-connectivity": "architecture",
    "prefect-flow-connectivity-execution-plan": "architecture",
    "prefect-analysis-dashboard-flow-improvement": "architecture",
    "consolidated-devex-training-evaluation-plan": "architecture",
    "flow-data-acquisition-plan": "architecture",
    "double-check-all-wiring": "architecture",
    "training-and-post-training-into-two-subflows-under-one-flow": "architecture",
    "prefect-and-devex-profiling-optimizations": "architecture",
    "oracle-config-planning": "architecture",
    "prd-kg-openspec-architecture": "architecture",
    # Evaluation (from other theme keyword matches)
    "prefect-flow-evaluation-and-ensemble-planning": "evaluation",
    "advanced-ensembling-bootstrapping-report": "evaluation",
    "local-debug-3flow-execution-plan": "evaluation",
    "optuna-hpo-plan": "evaluation",
    "generative-uq-plan": "evaluation",
    "uq-beyond-temperature-plan": "evaluation",
    "self-reflection-low-confidence-regions": "evaluation",
    "mapie-conformal-plan": "evaluation",
    "ram-issue-mock-data-biostatistics-duckup-report": "evaluation",
    "pr-a-biostatistics-gaps-plan": "evaluation",
    "pr-b-evals-analysis-flow-plan": "evaluation",
    "pr-c-post-training-factorial-plan": "evaluation",
    "hpo-implementation-background-research-report": "evaluation",
    "user-prompt-post-analysis-biostats-local-debug": "evaluation",
    # Models
    "advanced-segmentation-double-check-plan": "models",
    "advanced-segmentation-double-check-prompt": "models",
    "advanced-segmentation-execution-plan": "models",
    "dynunet-ablation-plan": "models",
    "dynunet-evaluation-plan": "models",
    "sam3-implementation-plan": "models",
    "sam3-installation-issues-and-synthesis": "models",
    "sam3-installation-issues-and-synthesis-plan": "models",
    "sam3-real-data-e2e-plan": "models",
    "sam3-stub-removal": "models",
    "sam3-nan-loss-fix": "models",
    "sam3-training-reference": "models",
    "topolora-sam3-planning-report": "models",
    "topolora-sam3-planning-report-diff-parameterized-compound-loss": "models",
    "mamba-model-capacity-matching": "models",
    # Harness
    "biomedical-agentic-ai-research-report": "harness",
    "data-science-agents-report": "harness",
    "failure-metalearning-001-training-launch": "harness",
    "agentic-rag-iac-angle-plan": "harness",
    "pr3-kg-tooling-refresh-plan": "harness",
    "github-projects-migration-and-cleaning-plan": "harness",
    "remaining-issue-data-driven-plan": "harness",
    "three-pr-planning-finops-timing-data-quality-kg-tooling-prompts": "harness",
    "STATUS": "harness",
    "sdd": "harness",
    "issue-340-update-body": "harness",
    # Cloud
    "ralph-loop-for-cloud-monitoring": "cloud",
    "docker-pull-runpod-provisioning-mlflow-cloud-run-multipart-upload-report": "cloud",
    "pr-e-cost-reporting-plan": "cloud",
    "oracle-cloud-free": "cloud",
    # Infrastructure
    "mlops-practices-report": "infrastructure",
    "monai-performance-optimization-plan": "infrastructure",
    "r6-remediation-plan": "infrastructure",
    "run-child1-acq": "infrastructure",
    "docker-improvements-for-debug-training": "infrastructure",
    # Deployment
    "pr-d-deploy-flow-plan": "deployment",
    # Operations
    "pprm-plan": "operations",
    "federated-learning-plan": "operations",
    # Observability
    "dills-diagnostics-plan": "observability",
    "cyclops-plan": "observability",
    "langgraph-agents-plan": "observability",
    "langgraph-agents-plan-advanced": "observability",
    "mlflow-and-weights-and-biases-meta-logger-plan": "observability",
    # Testing
    "pytorch-model-testing-best-practices-report": "testing",
    "pre-debug-qa-verification-plan": "testing",
    "validation-pipeline-plan": "testing",
    "prod-staging-gcp-doublecheck-code-review": "testing",
    # Manuscript
    "reporting-templates-plan": "manuscript",
    "kg-manuscript-execution-plan": "manuscript",
    "batch-literature-research-prompt": "manuscript",
    "batch-literature-research-report": "manuscript",
    # Data
    "segmentation-qc-plan": "data",
    "prompt-574-synthetic-data-drift-detection": "data",
    "synthicl-plan": "data",
    "vessqc-plan": "data",
    # Training
    "topology-approaches-discussion-notes": "training",
    "GRAPH-TOPOLOGY-METRICS-INDEX": "training",
    "graph-topology-metrics-survey": "training",
    "graph-topology-metrics-code-examples": "training",
    "debug-training-all-losses-plan": "training",
}


def _stem(path: Path) -> str:
    """Get filename stem without extension."""
    return path.stem


def classify_doc(path: Path) -> str:
    """Classify a single doc into a theme. Returns theme name."""
    stem = _stem(path)

    # Check manual overrides first
    if stem in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[stem]

    # Keyword matching — first theme with a matching keyword wins
    stem_lower = stem.lower()
    for theme, keywords in THEME_RULES.items():
        for keyword in keywords:
            if keyword in stem_lower:
                return theme

    return "unclassified"


def classify_all() -> dict[str, list[dict]]:
    """Classify all docs and return grouped results."""
    if not ARCHIVE_DIR.exists():
        print(f"ERROR: {ARCHIVE_DIR} does not exist", file=sys.stderr)
        sys.exit(1)

    all_files = sorted(ARCHIVE_DIR.iterdir())
    all_themes = (
        set(THEME_RULES.keys()) | set(MANUAL_OVERRIDES.values()) | {"unclassified"}
    )
    results: dict[str, list[dict]] = {theme: [] for theme in sorted(all_themes)}

    for path in all_files:
        if path.is_dir():
            continue
        if path.suffix not in {".md", ".xml", ".sh", ".yaml", ".yml", ".txt", ".pptx"}:
            continue

        theme = classify_doc(path)
        doc_type = _infer_doc_type(path)

        results[theme].append(
            {
                "filename": path.name,
                "stem": _stem(path),
                "extension": path.suffix,
                "doc_type": doc_type,
                "theme": theme,
            }
        )

    return results


def _infer_doc_type(path: Path) -> str:
    """Infer document type from filename patterns."""
    stem = path.stem.lower()
    if path.suffix == ".xml":
        return "execution_plan"
    if "report" in stem:
        return "research_report"
    if "plan" in stem:
        return "plan"
    if "cold-start" in stem:
        return "cold_start"
    if "prompt" in stem:
        return "prompt"
    if "synthesis" in stem:
        return "synthesis"
    if "lit-report" in stem:
        return "literature_report"
    if path.suffix == ".sh":
        return "script"
    return "document"


def main() -> None:
    results = classify_all()

    if "--summary" in sys.argv:
        print("\n=== Theme Classification Summary ===\n")
        total = 0
        for theme in sorted(results.keys()):
            count = len(results[theme])
            total += count
            if count > 0:
                print(f"  {theme:20s}: {count:3d} docs")
        print(f"\n  {'TOTAL':20s}: {total:3d} docs")

        unclassified = results.get("unclassified", [])
        if unclassified:
            print(f"\n  WARNING: {len(unclassified)} unclassified docs:")
            for doc in unclassified:
                print(f"    - {doc['filename']}")

    elif "--unclassified" in sys.argv:
        unclassified = results.get("unclassified", [])
        if unclassified:
            for doc in unclassified:
                print(doc["filename"])
        else:
            print("All docs classified!")

    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
