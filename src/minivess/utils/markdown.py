"""Shared markdown report utilities.

Extracted from 13+ to_markdown() implementations across the codebase
to eliminate ~400 LOC of duplicate timestamp, header, table, and
section generation code (Code Review R1.3).
"""

from __future__ import annotations

from datetime import UTC, datetime


def timestamp_utc() -> str:
    """Return current UTC time as ``'YYYY-MM-DD HH:MM UTC'``."""
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


def markdown_header(
    title: str,
    *,
    fields: dict[str, str] | None = None,
) -> list[str]:
    """Generate a standard markdown report header.

    Parameters
    ----------
    title:
        Report title (rendered as ``# title``).
    fields:
        Optional key-value metadata fields (rendered as bold pairs).
        A ``**Generated:**`` timestamp is prepended automatically.

    Returns
    -------
    List of markdown lines.
    """
    sections: list[str] = [f"# {title}", ""]
    if fields:
        sections.append(f"**Generated:** {timestamp_utc()}")
        for key, value in fields.items():
            sections.append(f"**{key}:** {value}")
        sections.append("")
    return sections


def markdown_table(
    headers: list[str],
    rows: list[list[str]] | list[dict[str, str]],
) -> str:
    """Generate a markdown table string.

    Parameters
    ----------
    headers:
        Column header labels.
    rows:
        Row data — either lists of strings (positional) or dicts
        keyed by header names.

    Returns
    -------
    Complete markdown table as a single string.
    """
    header_row = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join("---" for _ in headers) + "|"
    lines = [header_row, separator]
    for row in rows:
        if isinstance(row, dict):
            values = [str(row.get(h, "")) for h in headers]
        else:
            values = [str(v) for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def markdown_section(
    title: str,
    content: str,
    *,
    include_if: bool = True,
) -> list[str]:
    """Generate a conditional markdown section.

    Parameters
    ----------
    title:
        Section heading (rendered as ``## title``).
    content:
        Section body text.
    include_if:
        When ``False``, returns an empty list (section omitted).

    Returns
    -------
    List of markdown lines, or empty list if excluded.
    """
    if not include_if:
        return []
    return ["", f"## {title}", "", content]
