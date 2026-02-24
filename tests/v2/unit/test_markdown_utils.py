"""Tests for shared markdown report utilities (Code Review R1.3)."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# T1: timestamp_utc
# ---------------------------------------------------------------------------


class TestTimestampUtc:
    """Test UTC timestamp formatting."""

    def test_format(self) -> None:
        """timestamp_utc should return 'YYYY-MM-DD HH:MM UTC'."""
        from minivess.utils.markdown import timestamp_utc

        ts = timestamp_utc()
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC", ts)

    def test_contains_utc(self) -> None:
        """Timestamp should always end with 'UTC'."""
        from minivess.utils.markdown import timestamp_utc

        assert timestamp_utc().endswith("UTC")


# ---------------------------------------------------------------------------
# T2: markdown_header
# ---------------------------------------------------------------------------


class TestMarkdownHeader:
    """Test markdown report header generation."""

    def test_title_only(self) -> None:
        """markdown_header with just a title should produce # Title."""
        from minivess.utils.markdown import markdown_header

        lines = markdown_header("My Report")
        header = "\n".join(lines)
        assert "# My Report" in header

    def test_with_fields(self) -> None:
        """markdown_header with fields should include bold key-value pairs."""
        from minivess.utils.markdown import markdown_header

        lines = markdown_header("Report", fields={"Version": "1.0", "Author": "Alice"})
        header = "\n".join(lines)
        assert "**Version:** 1.0" in header
        assert "**Author:** Alice" in header

    def test_includes_timestamp(self) -> None:
        """markdown_header should include a Generated timestamp."""
        from minivess.utils.markdown import markdown_header

        lines = markdown_header("Report", fields={"Scope": "test"})
        header = "\n".join(lines)
        assert "**Generated:**" in header
        assert "UTC" in header


# ---------------------------------------------------------------------------
# T3: markdown_table
# ---------------------------------------------------------------------------


class TestMarkdownTable:
    """Test markdown table generation."""

    def test_basic_table(self) -> None:
        """markdown_table should produce a header, separator, and rows."""
        from minivess.utils.markdown import markdown_table

        result = markdown_table(
            headers=["Name", "Score"],
            rows=[["Alice", "95"], ["Bob", "87"]],
        )
        assert "| Name | Score |" in result
        assert "Alice" in result
        assert "Bob" in result

    def test_separator_row(self) -> None:
        """Table should contain separator dashes."""
        from minivess.utils.markdown import markdown_table

        result = markdown_table(headers=["A", "B"], rows=[["1", "2"]])
        assert "---" in result

    def test_empty_rows(self) -> None:
        """Table with no rows should still have header and separator."""
        from minivess.utils.markdown import markdown_table

        result = markdown_table(headers=["X", "Y"], rows=[])
        assert "| X | Y |" in result

    def test_dict_rows(self) -> None:
        """markdown_table should accept dict rows keyed by header names."""
        from minivess.utils.markdown import markdown_table

        result = markdown_table(
            headers=["Name", "Score"],
            rows=[{"Name": "Alice", "Score": "95"}],
        )
        assert "Alice" in result
        assert "95" in result


# ---------------------------------------------------------------------------
# T4: markdown_section
# ---------------------------------------------------------------------------


class TestMarkdownSection:
    """Test conditional markdown section generation."""

    def test_included(self) -> None:
        """Section should render when include_if is True."""
        from minivess.utils.markdown import markdown_section

        lines = markdown_section("Details", "Some content here.")
        text = "\n".join(lines)
        assert "## Details" in text
        assert "Some content here." in text

    def test_excluded(self) -> None:
        """Section should be empty when include_if is False."""
        from minivess.utils.markdown import markdown_section

        lines = markdown_section("Details", "content", include_if=False)
        assert lines == []

    def test_default_included(self) -> None:
        """Section should be included by default."""
        from minivess.utils.markdown import markdown_section

        lines = markdown_section("Title", "body")
        assert len(lines) > 0
