"""DiLLS-style layered agent diagnostics (Sheng et al., 2026).

Provides structured diagnostics for LLM agent interactions at three
levels: individual interaction, session (conversation), and aggregate
(cross-session statistics).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class AgentInteraction:
    """A single agent step (node execution) in a LangGraph session.

    Parameters
    ----------
    node_name:
        Name of the graph node that executed.
    input_summary:
        Brief description of the input to this step.
    output_summary:
        Brief description of the output from this step.
    latency_ms:
        Wall-clock time for this step in milliseconds.
    token_count:
        Number of LLM tokens consumed (input + output).
    metadata:
        Arbitrary key-value pairs (model, temperature, etc.).
    """

    node_name: str
    input_summary: str
    output_summary: str
    latency_ms: float = 0.0
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionSummary:
    """Conversation-level summary aggregating interactions.

    Parameters
    ----------
    session_id:
        Unique identifier for this agent session.
    interactions:
        List of interactions in chronological order.
    """

    session_id: str
    interactions: list[AgentInteraction] = field(default_factory=list)

    @property
    def total_steps(self) -> int:
        """Total number of agent steps in this session."""
        return len(self.interactions)

    @property
    def total_latency_ms(self) -> float:
        """Sum of latencies across all steps."""
        return sum(i.latency_ms for i in self.interactions)

    @property
    def total_tokens(self) -> int:
        """Sum of token counts across all steps."""
        return sum(i.token_count for i in self.interactions)


class AgentDiagnostics:
    """DiLLS-style layered diagnostics for LLM agent systems.

    Three diagnostic layers:
    1. **Interaction** — individual agent step
    2. **Session** — conversation-level summary
    3. **Aggregate** — cross-session statistics
    """

    def __init__(self) -> None:
        self.sessions: dict[str, list[AgentInteraction]] = defaultdict(list)

    def record_interaction(
        self,
        *,
        session_id: str,
        node_name: str,
        input_summary: str,
        output_summary: str,
        latency_ms: float = 0.0,
        token_count: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentInteraction:
        """Record a single agent interaction.

        Parameters
        ----------
        session_id:
            Session to associate with.
        node_name:
            Graph node name.
        input_summary:
            Input description.
        output_summary:
            Output description.
        """
        interaction = AgentInteraction(
            node_name=node_name,
            input_summary=input_summary,
            output_summary=output_summary,
            latency_ms=latency_ms,
            token_count=token_count,
            metadata=metadata or {},
        )
        self.sessions[session_id].append(interaction)
        return interaction

    def summarize_session(self, session_id: str) -> SessionSummary:
        """Produce a session-level summary.

        Parameters
        ----------
        session_id:
            Session to summarize.
        """
        interactions = list(self.sessions.get(session_id, []))
        return SessionSummary(session_id=session_id, interactions=interactions)

    def summarize_aggregate(self) -> dict[str, Any]:
        """Produce cross-session aggregate statistics.

        Returns
        -------
        Dictionary with total_sessions, total_interactions, total_latency_ms,
        total_tokens, and avg_latency_per_session.
        """
        total_interactions = sum(len(v) for v in self.sessions.values())
        total_latency = sum(
            i.latency_ms for interactions in self.sessions.values() for i in interactions
        )
        total_tokens = sum(
            i.token_count for interactions in self.sessions.values() for i in interactions
        )
        n_sessions = len(self.sessions)
        return {
            "total_sessions": n_sessions,
            "total_interactions": total_interactions,
            "total_latency_ms": total_latency,
            "total_tokens": total_tokens,
            "avg_latency_per_session": total_latency / n_sessions if n_sessions else 0.0,
        }

    def to_markdown(self) -> str:
        """Generate a human-readable diagnostic report."""
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        sections = [
            "# Agent Diagnostics Report",
            "",
            f"**Generated:** {now}",
            "",
        ]

        if not self.sessions:
            sections.append("No sessions recorded.")
            sections.append("")
            return "\n".join(sections)

        # Aggregate summary
        agg = self.summarize_aggregate()
        sections.extend([
            "## Aggregate Summary",
            "",
            f"- **Total Sessions:** {agg['total_sessions']}",
            f"- **Total Interactions:** {agg['total_interactions']}",
            f"- **Total Latency:** {agg['total_latency_ms']:.1f} ms",
            f"- **Total Tokens:** {agg['total_tokens']}",
            f"- **Avg Latency/Session:** {agg['avg_latency_per_session']:.1f} ms",
            "",
        ])

        # Per-session details
        for session_id in sorted(self.sessions):
            summary = self.summarize_session(session_id)
            sections.extend([
                f"## Session: {session_id}",
                "",
                f"- **Steps:** {summary.total_steps}",
                f"- **Total Latency:** {summary.total_latency_ms:.1f} ms",
                f"- **Total Tokens:** {summary.total_tokens}",
                "",
                "| Step | Node | Latency (ms) | Tokens |",
                "|------|------|-------------|--------|",
            ])
            for idx, interaction in enumerate(summary.interactions, 1):
                sections.append(
                    f"| {idx} | {interaction.node_name} "
                    f"| {interaction.latency_ms:.1f} | {interaction.token_count} |"
                )
            sections.append("")

        return "\n".join(sections)
