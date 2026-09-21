"""Turn controller behavior — parallel execution, timeouts, error handling."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import pytest

from observa.agents.base import ReflectiveAgent
from observa.graph.turn_controller import run_turn
from observa.models import CaseState, Finding, InputFile, TurnFindings


def _state() -> CaseState:
    return {
        "case_id": "X",
        "created_at": datetime(2026, 4, 24),
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


class _FastAgent(ReflectiveAgent):
    name = "fast"
    display_name = "Fast"
    specialty_prompt = "fast"

    async def run(self, state, turn, llm, mcp_tools, time_budget_seconds):  # type: ignore[override]
        return TurnFindings(
            agent=self.name,
            turn=turn,
            findings=[
                Finding(
                    finding_id="f1",
                    agent=self.name,
                    turn=turn,
                    severity="low",
                    code="FAST_OK",
                    description="fast finding",
                )
            ],
        )


class _SlowAgent(ReflectiveAgent):
    name = "slow"
    display_name = "Slow"
    specialty_prompt = "slow"

    async def run(self, state, turn, llm, mcp_tools, time_budget_seconds):  # type: ignore[override]
        await asyncio.sleep(5)
        return TurnFindings(agent=self.name, turn=turn, abstained=True)


class _BrokenAgent(ReflectiveAgent):
    name = "broken"
    display_name = "Broken"
    specialty_prompt = "boom"

    async def run(self, state, turn, llm, mcp_tools, time_budget_seconds):  # type: ignore[override]
        raise RuntimeError("kaboom")


async def test_run_turn_returns_one_result_per_agent():
    results = await run_turn(
        _state(),
        turn=1,
        agents=[_FastAgent(), _FastAgent()],
        llm=None,  # type: ignore[arg-type]
        mcp_tools=[],
        time_budget_seconds=1,
    )
    assert len(results) == 2
    assert {r.agent for r in results} == {"fast"}
    assert all(not r.abstained for r in results)


async def test_run_turn_timeout_marks_abstain():
    # Slow agent exceeds (budget + grace). Use a tiny budget to hit watchdog quickly.
    # Patch the grace to 0 via monkeypatching the module constant.
    import observa.graph.turn_controller as tc

    original = tc._WATCHDOG_GRACE_SECONDS
    tc._WATCHDOG_GRACE_SECONDS = 0
    try:
        results = await run_turn(
            _state(),
            turn=1,
            agents=[_SlowAgent()],
            llm=None,  # type: ignore[arg-type]
            mcp_tools=[],
            time_budget_seconds=1,
        )
    finally:
        tc._WATCHDOG_GRACE_SECONDS = original

    assert len(results) == 1
    r = results[0]
    assert r.agent == "slow"
    assert r.abstained is True
    assert "timeout" in r.abstain_reason.lower()


async def test_run_turn_agent_exception_becomes_abstain():
    results = await run_turn(
        _state(),
        turn=1,
        agents=[_BrokenAgent()],
        llm=None,  # type: ignore[arg-type]
        mcp_tools=[],
        time_budget_seconds=1,
    )
    assert len(results) == 1
    r = results[0]
    assert r.agent == "broken"
    assert r.abstained is True
    assert "RuntimeError" in r.abstain_reason
    assert "kaboom" in r.abstain_reason


async def test_run_turn_empty_agent_list():
    results = await run_turn(
        _state(),
        turn=1,
        agents=[],
        llm=None,  # type: ignore[arg-type]
        mcp_tools=[],
        time_budget_seconds=1,
    )
    assert results == []


async def test_run_turn_fast_agents_run_in_parallel():
    """Two 0.3s agents should finish in ~0.3s, not ~0.6s."""
    import time

    class _SleepAgent(ReflectiveAgent):
        name = "sleep"
        display_name = "S"
        specialty_prompt = "x"

        async def run(self, state, turn, llm, mcp_tools, time_budget_seconds):  # type: ignore[override]
            await asyncio.sleep(0.3)
            return TurnFindings(agent=self.name, turn=turn, abstained=True)

    start = time.monotonic()
    await run_turn(
        _state(),
        turn=1,
        agents=[_SleepAgent(), _SleepAgent(), _SleepAgent()],
        llm=None,  # type: ignore[arg-type]
        mcp_tools=[],
        time_budget_seconds=2,
    )
    elapsed = time.monotonic() - start
    # Allow generous margin for test scheduler jitter — parallel should be < 0.5s.
    assert elapsed < 0.5, f"expected parallel execution (<0.5s), got {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Task 4: validate_evidence_refs + tools_for_agent
# ---------------------------------------------------------------------------

class _FakeLedger:
    def __init__(self, ids: frozenset[str]) -> None:
        self._ids = ids

    def known_ids(self) -> frozenset[str]:
        return self._ids


def test_validate_evidence_refs_drops_unknown_and_flags_uncited():
    from observa.graph.turn_controller import validate_evidence_refs

    tf = TurnFindings(
        agent="wait_time", turn=1,
        findings=[
            Finding(finding_id="a", agent="wait_time", turn=1, severity="high",
                    code="A", description="d",
                    evidence_refs=["Q-001", "Q-999"], evidence_note="n"),
            Finding(finding_id="b", agent="wait_time", turn=1, severity="low",
                    code="B", description="d", evidence_refs=["Q-777"]),
        ],
    )
    out = validate_evidence_refs(tf, _FakeLedger(frozenset({"Q-001"})))
    assert out.findings[0].evidence_refs == ["Q-001"]
    assert out.findings[0].uncited is False
    assert out.findings[1].evidence_refs == []
    assert out.findings[1].uncited is True


def test_validate_evidence_refs_noop_without_ledger():
    from observa.graph.turn_controller import validate_evidence_refs

    tf = TurnFindings(
        agent="a", turn=1,
        findings=[Finding(finding_id="x", agent="a", turn=1, severity="info",
                          code="C", description="d", evidence_refs=["Q-1"])],
    )
    out = validate_evidence_refs(tf, None)
    assert out.findings[0].evidence_refs == ["Q-1"]


@pytest.mark.asyncio
async def test_run_turn_uses_tools_for_agent():
    """Per-agent tools are resolved AND delivered: each agent's .run() must
    receive exactly the tool objects its ``tools_for_agent`` call returned."""
    seen: list[tuple[str, int]] = []

    class _FakeTool:
        def __init__(self, name: str) -> None:
            self.name = name

    sentinel_tools: dict[str, list[Any]] = {
        "wait_time": [_FakeTool("query_awr_for_wait_time")],
        "ash": [_FakeTool("query_awr_for_ash")],
    }
    received_tools: dict[str, list[Any]] = {}

    def _tools_for_agent(name: str, turn: int) -> list[Any]:
        seen.append((name, turn))
        return sentinel_tools[name]

    class _RecordingAgent(ReflectiveAgent):
        display_name = "R"
        specialty_prompt = "x"

        async def run(self, state, turn, llm, mcp_tools, time_budget_seconds):  # type: ignore[override]
            received_tools[self.name] = list(mcp_tools)
            return TurnFindings(agent=self.name, turn=turn, abstained=True)

    class _WaitTimeAgent(_RecordingAgent):
        name = "wait_time"

    class _AshAgent(_RecordingAgent):
        name = "ash"

    agents: list[ReflectiveAgent] = [_WaitTimeAgent(), _AshAgent()]
    await run_turn(
        _state(), 2, agents,
        llm=None,  # type: ignore[arg-type]
        mcp_tools=[],
        time_budget_seconds=1,
        tools_for_agent=_tools_for_agent,
    )
    assert sorted(seen) == [("ash", 2), ("wait_time", 2)]
    # The controller may copy the list, but the tool OBJECTS must be the exact
    # ones the callable returned for that agent (identity, not just names).
    assert received_tools["wait_time"] == sentinel_tools["wait_time"]
    assert received_tools["ash"] == sentinel_tools["ash"]
    assert received_tools["wait_time"][0] is sentinel_tools["wait_time"][0]
    assert received_tools["ash"][0] is sentinel_tools["ash"][0]
