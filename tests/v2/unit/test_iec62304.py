"""Tests for IEC 62304 compliance framework (Issue #46)."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# T1: SoftwareSafetyClass
# ---------------------------------------------------------------------------


class TestSoftwareSafetyClass:
    """Test IEC 62304 software safety classification."""

    def test_enum_values(self) -> None:
        """SoftwareSafetyClass should have three safety classes."""
        from minivess.compliance.iec62304 import SoftwareSafetyClass

        assert SoftwareSafetyClass.CLASS_A == "class_a"
        assert SoftwareSafetyClass.CLASS_B == "class_b"
        assert SoftwareSafetyClass.CLASS_C == "class_c"

    def test_enum_membership(self) -> None:
        """All three safety classes should be present."""
        from minivess.compliance.iec62304 import SoftwareSafetyClass

        names = {e.name for e in SoftwareSafetyClass}
        assert names == {"CLASS_A", "CLASS_B", "CLASS_C"}


# ---------------------------------------------------------------------------
# T2: LifecycleStage
# ---------------------------------------------------------------------------


class TestLifecycleStage:
    """Test IEC 62304 lifecycle stages."""

    def test_enum_values(self) -> None:
        """LifecycleStage should have five stages."""
        from minivess.compliance.iec62304 import LifecycleStage

        assert LifecycleStage.DEVELOPMENT == "development"
        assert LifecycleStage.VERIFICATION == "verification"
        assert LifecycleStage.VALIDATION == "validation"
        assert LifecycleStage.RELEASE == "release"
        assert LifecycleStage.MAINTENANCE == "maintenance"

    def test_all_stages(self) -> None:
        """All five lifecycle stages should be present."""
        from minivess.compliance.iec62304 import LifecycleStage

        assert len(LifecycleStage) == 5


# ---------------------------------------------------------------------------
# T3: TraceabilityMatrix
# ---------------------------------------------------------------------------


class TestTraceabilityMatrix:
    """Test requirements traceability matrix."""

    def test_construction(self) -> None:
        """TraceabilityMatrix should be constructible empty."""
        from minivess.compliance.iec62304 import TraceabilityMatrix

        matrix = TraceabilityMatrix()
        assert len(matrix.entries) == 0

    def test_add_entry(self) -> None:
        """add_entry should add a traceability record."""
        from minivess.compliance.iec62304 import TraceabilityMatrix

        matrix = TraceabilityMatrix()
        matrix.add_entry(
            requirement_id="FR-001",
            description="Accept NIfTI input",
            implementation_ref="src/minivess/data/loader.py",
            test_ref="tests/v2/unit/test_loader.py::test_nifti",
        )
        assert len(matrix.entries) == 1
        assert matrix.entries[0].requirement_id == "FR-001"

    def test_to_markdown(self) -> None:
        """to_markdown should produce a traceability table."""
        from minivess.compliance.iec62304 import TraceabilityMatrix

        matrix = TraceabilityMatrix()
        matrix.add_entry(
            requirement_id="FR-001",
            description="Accept NIfTI input",
            implementation_ref="loader.py",
            test_ref="test_loader.py",
        )
        matrix.add_entry(
            requirement_id="FR-002",
            description="Produce segmentation masks",
            implementation_ref="pipeline.py",
            test_ref="test_pipeline.py",
        )
        md = matrix.to_markdown()
        assert "Traceability" in md
        assert "FR-001" in md
        assert "FR-002" in md

    def test_coverage_report(self) -> None:
        """coverage_report should identify untested requirements."""
        from minivess.compliance.iec62304 import TraceabilityMatrix

        matrix = TraceabilityMatrix()
        matrix.add_entry(
            requirement_id="FR-001",
            description="Tested req",
            implementation_ref="impl.py",
            test_ref="test.py",
        )
        matrix.add_entry(
            requirement_id="FR-002",
            description="Untested req",
            implementation_ref="impl.py",
            test_ref="",
        )
        report = matrix.coverage_report()
        assert report["total"] == 2
        assert report["tested"] == 1
        assert report["untested"] == 1
        assert "FR-002" in report["gaps"]


# ---------------------------------------------------------------------------
# T4: PCCPTemplate
# ---------------------------------------------------------------------------


class TestPCCPTemplate:
    """Test predetermined change control plan template."""

    def test_construction(self) -> None:
        """PCCPTemplate should be constructible."""
        from minivess.compliance.iec62304 import PCCPTemplate

        pccp = PCCPTemplate(
            product_name="MiniVess",
            product_version="2.0.0",
        )
        assert pccp.product_name == "MiniVess"

    def test_to_markdown(self) -> None:
        """to_markdown should produce a PCCP document."""
        from minivess.compliance.iec62304 import PCCPTemplate

        pccp = PCCPTemplate(
            product_name="MiniVess",
            product_version="2.0.0",
            permitted_changes=[
                "Model retraining with new data (same architecture)",
                "Hyperparameter tuning within predefined ranges",
            ],
        )
        md = pccp.to_markdown()
        assert "Predetermined Change Control Plan" in md
        assert "MiniVess" in md
        assert "Model retraining" in md

    def test_empty_changes(self) -> None:
        """PCCP with no permitted changes should still render."""
        from minivess.compliance.iec62304 import PCCPTemplate

        pccp = PCCPTemplate(product_name="Test", product_version="1.0")
        md = pccp.to_markdown()
        assert "Predetermined Change Control Plan" in md
