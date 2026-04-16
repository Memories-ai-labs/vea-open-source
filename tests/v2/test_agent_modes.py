"""Tests for the ``AgentMode`` enum and mode-scoped prompt assembly.

Locks in two invariants that a refactor is easy to break:

- Each mode gets exactly the right prompt block (no cross-contamination of
  collaborative-mode behavior into autonomous mode or vice versa).
- The legacy ``autonomous: bool`` kwarg still works and picks the right mode.
"""
from __future__ import annotations

import pytest

from src.pipelines.v2.agent.modes import AgentMode, MAX_TOOL_ROUNDS_BY_MODE
from src.pipelines.v2.agent.system_prompt import (
    _AUTONOMOUS_ADDENDUM,
    _COLLABORATIVE_ADDENDUM,
    _MODE_ADDENDA,
    build_system_prompt,
)


# ─── Enum shape ──────────────────────────────────────────────────────────────

class TestAgentMode:
    def test_enum_has_exactly_two_values(self):
        # Drop is deliberate: anything other than collaborative / autonomous
        # would need a matching addendum entry + round cap. Triggers a
        # review instead of silent add.
        assert {m.value for m in AgentMode} == {"collaborative", "autonomous"}

    def test_string_coercion_roundtrips(self):
        assert AgentMode("collaborative") is AgentMode.COLLABORATIVE
        assert AgentMode("autonomous") is AgentMode.AUTONOMOUS
        with pytest.raises(ValueError):
            AgentMode("single_shot")   # pruned from the plan

    def test_max_rounds_has_entry_per_mode(self):
        assert set(MAX_TOOL_ROUNDS_BY_MODE.keys()) == set(AgentMode)

    def test_autonomous_gets_a_larger_round_cap_than_collaborative(self):
        # Collaborative has a human to nudge; autonomous needs more runway
        # to self-iterate. If this ever inverts, something is wrong.
        assert (
            MAX_TOOL_ROUNDS_BY_MODE[AgentMode.AUTONOMOUS]
            > MAX_TOOL_ROUNDS_BY_MODE[AgentMode.COLLABORATIVE]
        )


# ─── Addendum registry ───────────────────────────────────────────────────────

class TestModeAddenda:
    def test_collaborative_addendum_is_empty(self):
        assert _MODE_ADDENDA[AgentMode.COLLABORATIVE] == ""
        assert _COLLABORATIVE_ADDENDUM == ""

    def test_autonomous_addendum_mentions_key_rules(self):
        text = _AUTONOMOUS_ADDENDUM.lower()
        # Temperament rules — the three pillars of autonomous mode.
        assert "clarifying question" in text or "ask clarifying" in text
        assert "perfectionist" in text
        assert "final_message" in text

    def test_mode_addenda_covers_every_mode(self):
        # A new mode added without an addendum would blow up at build time.
        for mode in AgentMode:
            assert mode in _MODE_ADDENDA


# ─── build_system_prompt — mode selects addendum ─────────────────────────────

class TestBuildSystemPrompt:
    def _args(self, **overrides):
        return {
            "project_name": "p",
            "video_list": "- v.mp4",
            "scratchpads_text": "",
            "current_edit_decision": "",
            **overrides,
        }

    def test_collaborative_has_no_autonomous_language(self):
        prompt = build_system_prompt(mode=AgentMode.COLLABORATIVE, **self._args())
        # The perfectionist rider should NOT leak into collaborative.
        assert "Autonomous mode" not in prompt
        assert "perfectionist" not in prompt.lower()

    def test_autonomous_includes_addendum(self):
        prompt = build_system_prompt(mode=AgentMode.AUTONOMOUS, **self._args())
        assert "Autonomous mode" in prompt
        assert _AUTONOMOUS_ADDENDUM.strip() in prompt

    def test_collaborative_default_when_no_mode_given(self):
        prompt = build_system_prompt(**self._args())
        assert "Autonomous mode" not in prompt

    def test_autonomous_prompt_is_strict_superset_of_collaborative(self):
        # Every character of the collaborative prompt must appear verbatim in
        # the autonomous prompt — the addendum only APPENDS, it doesn't
        # rewrite the base. A refactor that starts mutating the base per mode
        # would trip this.
        collab = build_system_prompt(mode=AgentMode.COLLABORATIVE, **self._args())
        auto = build_system_prompt(mode=AgentMode.AUTONOMOUS, **self._args())
        assert auto.startswith(collab)
        assert len(auto) > len(collab)


# ─── Backcompat: the old autonomous: bool kwarg ─────────────────────────────

class TestAutonomousKwargBackcompat:
    def _args(self):
        return {
            "project_name": "p",
            "video_list": "- v.mp4",
            "scratchpads_text": "",
            "current_edit_decision": "",
        }

    def test_autonomous_true_matches_mode_autonomous(self):
        a = build_system_prompt(**self._args(), autonomous=True)
        b = build_system_prompt(**self._args(), mode=AgentMode.AUTONOMOUS)
        assert a == b

    def test_autonomous_false_matches_mode_collaborative(self):
        a = build_system_prompt(**self._args(), autonomous=False)
        b = build_system_prompt(**self._args(), mode=AgentMode.COLLABORATIVE)
        assert a == b

    def test_autonomous_kwarg_overrides_mode(self):
        # If a legacy caller passes both, the bool wins so existing behavior
        # doesn't change underneath them.
        prompt = build_system_prompt(
            **self._args(),
            mode=AgentMode.COLLABORATIVE,
            autonomous=True,
        )
        assert "Autonomous mode" in prompt
