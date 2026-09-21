"""run_turn passes the right LLM to each agent when llm_for_agent is provided."""
from __future__ import annotations

from datetime import datetime

import pytest

from observa.agents.base import ReflectiveAgent
from observa.graph.turn_controller import run_turn
from observa.models import CaseState, TurnFindings


def _state() -> CaseState:
    return {
        "case_id": "X",
        "created_at": datetime(2026, 4, 25),
        "dbid": 1,
        "oracle_version": "",
        "topology": "",
        "instance_numbers": [],
        "problem_statement": "p",
        "investigation_scope": "s",
        "known_facts": [],
        "input_files": [],
        "current_turn": 1,
        "total_turns": 1,
        "turn_findings": [],
        "agent_questions": [],
        "chat_log": [],
        "hypotheses": [],
        "contradictions": [],
        "contradiction_resolutions": [],
        "final_summary": None,
        "verification": None,
    }


class _CaptureAgent(ReflectiveAgent):
    """Records which LLM instance was passed to .run()."""
    name = "capture"
    display_name = "C"
    specialty_prompt = "x"

    def __init__(self, name: str = "capture") -> None:
        type(self).name = name  # ugly but lets us reuse one class
        self.name = name
        self.received_llm = None

    async def run(self, state, turn, llm, mcp_tools, time_budget_seconds):  # type: ignore[override]
        self.received_llm = llm
        return TurnFindings(agent=self.name, turn=turn, abstained=True)


async def test_run_turn_uses_llm_for_agent_when_provided():
    """The factory must be called per agent, and the returned object must reach .run()."""
    sentinel_a = object()
    sentinel_b = object()
    factory_calls: list[str] = []

    def factory(name: str):
        factory_calls.append(name)
        return sentinel_a if name == "agent_a" else sentinel_b

    a = _CaptureAgent("agent_a")
    b = _CaptureAgent("agent_b")
    await run_turn(
        _state(), turn=1,
        agents=[a, b],
        llm=None,  # type: ignore[arg-type]
        mcp_tools=[],
        time_budget_seconds=1,
        llm_for_agent=factory,
    )
    assert sorted(factory_calls) == ["agent_a", "agent_b"]
    assert a.received_llm is sentinel_a
    assert b.received_llm is sentinel_b


async def test_run_turn_falls_back_to_shared_llm_without_factory():
    """Backward compat: no factory => every agent gets the shared llm arg."""
    sentinel = object()
    a = _CaptureAgent("only")
    await run_turn(
        _state(), turn=1, agents=[a],
        llm=sentinel,  # type: ignore[arg-type]
        mcp_tools=[],
        time_budget_seconds=1,
    )
    assert a.received_llm is sentinel
