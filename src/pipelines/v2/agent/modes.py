"""Agent run modes.

Declared in its own module so both ``AgentSession`` and ``src.cli`` can import
without a circular dependency on ``agent_session``.
"""
from __future__ import annotations

from enum import Enum


class AgentMode(str, Enum):
    """How the agent should behave at runtime.

    Inherits from ``str`` so the value round-trips through JSON and argparse
    without a wrapper, and so ``AgentMode.AUTONOMOUS == "autonomous"``.
    """

    COLLABORATIVE = "collaborative"
    """Dashboard / WebSocket use. Agent defers to the user between iterations,
    proposes plans, waits for approval, refines on feedback."""

    AUTONOMOUS = "autonomous"
    """CLI / MCP / batch use. No human in the loop. Agent is a perfectionist
    that self-iterates (generate_fcpxml → self-critique → adjust → repeat)
    and commits to reasonable defaults instead of asking clarifying questions.
    The agent still owns ``finish_turn`` — the mode only changes temperament."""


MAX_TOOL_ROUNDS_BY_MODE: dict[AgentMode, int] = {
    AgentMode.COLLABORATIVE: 40,
    AgentMode.AUTONOMOUS: 120,
}
"""Per-mode cap on agent loop iterations. Autonomous mode gets a much larger
budget because there's no human available to nudge it when it drifts."""
