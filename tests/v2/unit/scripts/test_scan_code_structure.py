"""Tests for scripts/scan_code_structure.py (T22).

RED phase: tests written before implementation.

Validates:
- Flow discovery via ast.parse() from Python source with @flow decorators
- Adapter discovery via ast.parse() from Python source with ModelAdapter subclasses
- Output YAML is valid (yaml.safe_load() without error)
- No import re usage (CLAUDE.md Rule #16)
- Paths use pathlib.Path throughout
- Idempotency: running twice produces identical output
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures: synthetic Python source files (no real repo deps)
# ---------------------------------------------------------------------------

FLOW_SOURCE = textwrap.dedent("""\
    from __future__ import annotations

    from prefect import flow

    FLOW_NAME_TRAIN = "training-flow"


    @flow(name=FLOW_NAME_TRAIN)
    def run_train_flow(config: dict) -> dict:
        \"\"\"Training flow.\"\"\"
        return {}


    @flow(name="data-flow")
    def run_data_flow() -> None:
        \"\"\"Data engineering flow.\"\"\"
""")

ADAPTER_SOURCE = textwrap.dedent("""\
    from __future__ import annotations

    from minivess.adapters.base import ModelAdapter, SegmentationOutput
    from torch import Tensor


    class DynUNetAdapter(ModelAdapter):
        \"\"\"DynUNet adapter.\"\"\"

        def forward(self, images: Tensor, **kwargs):
            pass


    class SegResNetAdapter(ModelAdapter):
        \"\"\"SegResNet adapter.\"\"\"

        def forward(self, images: Tensor, **kwargs):
            pass
""")

NO_FLOW_SOURCE = textwrap.dedent("""\
    from __future__ import annotations

    def helper() -> None:
        pass
""")


@pytest.fixture
def flow_py(tmp_path: Path) -> Path:
    p = tmp_path / "train_flow.py"
    p.write_text(FLOW_SOURCE, encoding="utf-8")
    return p


@pytest.fixture
def adapter_py(tmp_path: Path) -> Path:
    p = tmp_path / "dynunet.py"
    p.write_text(ADAPTER_SOURCE, encoding="utf-8")
    return p


@pytest.fixture
def empty_py(tmp_path: Path) -> Path:
    p = tmp_path / "no_flows.py"
    p.write_text(NO_FLOW_SOURCE, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Import the scanner (will fail in RED phase — that is expected)
# ---------------------------------------------------------------------------

from scripts.scan_code_structure import (  # noqa: E402
    extract_adapter_classes,
    extract_flow_functions,
    scan_adapters_dir,
    scan_flows_dir,
    write_flows_yaml,
)

# ---------------------------------------------------------------------------
# Unit tests: extract_flow_functions
# ---------------------------------------------------------------------------


class TestExtractFlowFunctions:
    def test_finds_decorated_flows(self, flow_py: Path) -> None:
        tree = ast.parse(flow_py.read_text(encoding="utf-8"))
        flows = extract_flow_functions(tree, source_file=flow_py)
        assert len(flows) == 2

    def test_flow_has_name_and_function_name(self, flow_py: Path) -> None:
        tree = ast.parse(flow_py.read_text(encoding="utf-8"))
        flows = extract_flow_functions(tree, source_file=flow_py)
        func_names = {f["function_name"] for f in flows}
        assert "run_train_flow" in func_names
        assert "run_data_flow" in func_names

    def test_flow_has_file_path(self, flow_py: Path) -> None:
        tree = ast.parse(flow_py.read_text(encoding="utf-8"))
        flows = extract_flow_functions(tree, source_file=flow_py)
        for f in flows:
            assert "file" in f
            assert f["file"] == str(flow_py)

    def test_empty_file_returns_empty(self, empty_py: Path) -> None:
        tree = ast.parse(empty_py.read_text(encoding="utf-8"))
        flows = extract_flow_functions(tree, source_file=empty_py)
        assert flows == []


# ---------------------------------------------------------------------------
# Unit tests: extract_adapter_classes
# ---------------------------------------------------------------------------


class TestExtractAdapterClasses:
    def test_finds_adapter_subclasses(self, adapter_py: Path) -> None:
        tree = ast.parse(adapter_py.read_text(encoding="utf-8"))
        adapters = extract_adapter_classes(tree, source_file=adapter_py)
        assert len(adapters) == 2

    def test_adapter_has_class_name(self, adapter_py: Path) -> None:
        tree = ast.parse(adapter_py.read_text(encoding="utf-8"))
        adapters = extract_adapter_classes(tree, source_file=adapter_py)
        class_names = {a["class_name"] for a in adapters}
        assert "DynUNetAdapter" in class_names
        assert "SegResNetAdapter" in class_names

    def test_adapter_has_file_path(self, adapter_py: Path) -> None:
        tree = ast.parse(adapter_py.read_text(encoding="utf-8"))
        adapters = extract_adapter_classes(tree, source_file=adapter_py)
        for a in adapters:
            assert "file" in a

    def test_no_adapters_in_flow_file(self, flow_py: Path) -> None:
        tree = ast.parse(flow_py.read_text(encoding="utf-8"))
        adapters = extract_adapter_classes(tree, source_file=flow_py)
        assert adapters == []


# ---------------------------------------------------------------------------
# Integration tests: scan directories + write YAML
# ---------------------------------------------------------------------------


class TestScanFlowsDir:
    def test_scan_flows_dir_returns_list(self, tmp_path: Path, flow_py: Path) -> None:
        flows_dir = tmp_path / "flows"
        flows_dir.mkdir()
        (flows_dir / "train_flow.py").write_text(FLOW_SOURCE, encoding="utf-8")
        flows = scan_flows_dir(flows_dir)
        assert isinstance(flows, list)
        assert len(flows) == 2

    def test_scan_flows_dir_empty_dir(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        flows = scan_flows_dir(empty_dir)
        assert flows == []

    def test_scan_flows_dir_skips_init(self, tmp_path: Path) -> None:
        flows_dir = tmp_path / "flows"
        flows_dir.mkdir()
        (flows_dir / "__init__.py").write_text("", encoding="utf-8")
        (flows_dir / "train_flow.py").write_text(FLOW_SOURCE, encoding="utf-8")
        flows = scan_flows_dir(flows_dir)
        # __init__.py has no @flow — only train_flow.py contributes
        assert len(flows) == 2


class TestScanAdaptersDir:
    def test_scan_adapters_dir_returns_list(self, tmp_path: Path) -> None:
        adapters_dir = tmp_path / "adapters"
        adapters_dir.mkdir()
        (adapters_dir / "dynunet.py").write_text(ADAPTER_SOURCE, encoding="utf-8")
        adapters = scan_adapters_dir(adapters_dir)
        assert isinstance(adapters, list)
        assert len(adapters) == 2

    def test_scan_adapters_dir_empty(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        adapters = scan_adapters_dir(empty_dir)
        assert adapters == []


class TestWriteFlowsYaml:
    def test_writes_valid_yaml(self, tmp_path: Path, flow_py: Path) -> None:
        tree = ast.parse(flow_py.read_text(encoding="utf-8"))
        flows = extract_flow_functions(tree, source_file=flow_py)
        out_path = tmp_path / "flows.yaml"
        write_flows_yaml(flows, out_path)
        assert out_path.exists()
        data = yaml.safe_load(out_path.read_text(encoding="utf-8"))
        assert "flows" in data
        assert "_meta" in data

    def test_output_has_generated_by(self, tmp_path: Path, flow_py: Path) -> None:
        tree = ast.parse(flow_py.read_text(encoding="utf-8"))
        flows = extract_flow_functions(tree, source_file=flow_py)
        out_path = tmp_path / "flows.yaml"
        write_flows_yaml(flows, out_path)
        data = yaml.safe_load(out_path.read_text(encoding="utf-8"))
        assert data["_meta"]["generated_by"] == "scripts/scan_code_structure.py"

    def test_idempotent(self, tmp_path: Path, flow_py: Path) -> None:
        tree = ast.parse(flow_py.read_text(encoding="utf-8"))
        flows = extract_flow_functions(tree, source_file=flow_py)
        out_path = tmp_path / "flows.yaml"
        write_flows_yaml(flows, out_path)
        content_1 = out_path.read_text(encoding="utf-8")
        write_flows_yaml(flows, out_path)
        content_2 = out_path.read_text(encoding="utf-8")
        assert content_1 == content_2


# ---------------------------------------------------------------------------
# Static analysis: verify no `import re` in script
# ---------------------------------------------------------------------------


class TestNoBannedImports:
    def test_no_import_re(self) -> None:
        """CLAUDE.md Rule #16: import re is BANNED for structured data parsing."""
        script_path = Path("scripts/scan_code_structure.py")
        if not script_path.exists():
            pytest.skip("Script not yet implemented (RED phase)")
        source = script_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "re", (
                        "import re is BANNED (CLAUDE.md Rule #16)"
                    )
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "re", (
                    "from re import ... is BANNED (CLAUDE.md Rule #16)"
                )

    def test_uses_pathlib(self) -> None:
        """Paths must use pathlib.Path — no string concatenation."""
        script_path = Path("scripts/scan_code_structure.py")
        if not script_path.exists():
            pytest.skip("Script not yet implemented (RED phase)")
        source = script_path.read_text(encoding="utf-8")
        assert "pathlib" in source, "Script must use pathlib.Path for paths"
