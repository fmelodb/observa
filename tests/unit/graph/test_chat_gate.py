"""End-to-end exercise of the turn/gate/resume protocol with fake LLM + no MCP."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from observa.agents.base import ReflectiveAgent
from observa.graph import master
from observa.models import AgentQuestion, Finding, TurnFindings


class _CannedAgent(ReflectiveAgent):
    """Deterministic agent that emits prebaked results per turn index."""

    name = "canned"
    display_name = "Canned"
    specialty_prompt = "n/a"

    def __init__(self, results_by_turn: dict[int, TurnFindings]) -> None:
        self._results_by_turn = results_by_turn

    async def run(self, state, turn, llm, mcp_tools, time_budget_seconds):  # type: ignore[override]
        tf = self._results_by_turn.get(turn)
        if tf is not None:
            return tf
        return TurnFindings(agent=self.name, turn=turn, abstained=True)


def _base_state() -> dict:
    return {
        "case_id": "CASE-2026-0424-01",
        "created_at": datetime(2026, 4, 24),
        "dbid": 1,
        "oracle_version": "",
        "topology": "single-instance",
        "instance_numbers": [1],
        "problem_statement": "x",
        "investigation_scope": "y",
        "known_facts": [],
        "input_files": [],
        "current_turn": 0,
        "total_turns": 3,
        "turn_findings": [],
        "agent_questions": [],
        "chat_log": [],
        "hypotheses": [],
        "contradictions": [],
        "contradiction_resolutions": [],
        "final_summary": None,
        "verification": None,
    }


class _FakeStructuredLLM:
    def __init__(self, result: Any) -> None:
        self._result = result

    async def ainvoke(self, messages: Any) -> Any:
        return self._result


class _FakeLLM:
    """Stub whose only job is to satisfy consolidate()'s structured call."""

    def __init__(self, summary: Any) -> None:
        self._summary = summary

    def with_structured_output(self, schema: Any) -> _FakeStructuredLLM:
        from observa.models import FinalSummary

        return _FakeStructuredLLM(FinalSummary(
            problem_restated="x",
            top_findings=[],
            root_cause="none",
            confidence=0.1,
        ))


def _patch_graph_for_test(monkeypatch, agent: _CannedAgent, turns: int = 3):
    """Install canned agent + fake LLM into the master module for one graph build."""
    from observa.config import Settings, get_settings

    # Force 3 turns but allow override via arg.
    base_settings = get_settings()
    overrides = {
        "turns_count": turns,
        "turns_time_budget_seconds": 5,
        "agents_enable_hypotheses": False,
        "agents_enable_cove": False,
    }
    fake = base_settings.model_copy(update=overrides)
    monkeypatch.setattr("observa.graph.master.get_settings", lambda: fake)
    monkeypatch.setattr("observa.graph.master._build_active_agents", lambda: [agent])
    monkeypatch.setattr("observa.graph.master._ensure_main_llm", lambda: _FakeLLM(None))


async def test_gate_skipped_when_no_questions(monkeypatch):
    """No questions anywhere → graph runs straight through to `done`."""
    agent = _CannedAgent({
        1: TurnFindings(agent="canned", turn=1, abstained=True),
        2: TurnFindings(agent="canned", turn=2, abstained=True),
        3: TurnFindings(agent="canned", turn=3, abstained=True),
    })
    _patch_graph_for_test(monkeypatch, agent, turns=3)

    graph = master.build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t1", "mcp_client": None}}
    result = await graph.ainvoke(_base_state(), config=config)

    # Final summary should be set, no pending interrupts.
    state = await graph.aget_state(config)
    assert not state.next, f"graph should be complete, got pending: {state.next}"
    assert result.get("final_summary") is not None


async def test_gate_pauses_when_agent_has_questions(monkeypatch):
    """Turn 1 emits a question → graph pauses at gate_1 before turn_2."""
    agent = _CannedAgent({
        1: TurnFindings(
            agent="canned",
            turn=1,
            findings=[
                Finding(
                    finding_id="f1",
                    agent="canned",
                    turn=1,
                    severity="info",
                    code="X",
                    description="need info",
                )
            ],
            questions=[AgentQuestion(agent="canned", turn=1, question="Which window?")],
        ),
    })
    _patch_graph_for_test(monkeypatch, agent, turns=3)

    graph = master.build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t2", "mcp_client": None}}

    await graph.ainvoke(_base_state(), config=config)
    state = await graph.aget_state(config)

    # Pending at gate_1 with an interrupt payload.
    assert state.next == ("gate_1",), f"expected paused at gate_1, got {state.next}"
    interrupts: list = []
    for task in state.tasks:
        interrupts.extend(task.interrupts)
    assert len(interrupts) == 1
    payload = interrupts[0].value
    assert payload.get("type") == "chat_gate"
    assert payload.get("turn") == 1
    questions = payload.get("questions") or []
    assert len(questions) == 1
    assert questions[0].get("question") == "Which window?"


async def test_resume_from_gate_records_chat_and_finishes(monkeypatch):
    """Resuming with a user message routes through the blackboard and graph finishes."""
    agent = _CannedAgent({
        1: TurnFindings(
            agent="canned",
            turn=1,
            questions=[AgentQuestion(agent="canned", turn=1, question="Window?")],
        ),
        2: TurnFindings(agent="canned", turn=2, abstained=True),
        3: TurnFindings(agent="canned", turn=3, abstained=True),
    })
    _patch_graph_for_test(monkeypatch, agent, turns=3)

    graph = master.build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t3", "mcp_client": None}}

    await graph.ainvoke(_base_state(), config=config)
    state = await graph.aget_state(config)
    assert state.next == ("gate_1",)

    # Resume with a canned analyst reply.
    response = {"messages": [{"sender": "user", "text": "snap 100-110"}]}
    await graph.ainvoke(Command(resume=response), config=config)

    state = await graph.aget_state(config)
    assert not state.next, f"graph should be complete, got pending: {state.next}"

    # Chat log should contain the analyst message.
    chat = state.values.get("chat_log") or []
    assert len(chat) == 1
    msg = chat[0]
    assert msg.sender == "user"
    assert msg.text == "snap 100-110"
    assert msg.turn == 1
