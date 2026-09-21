"""Master consolidator — blackboard digest + LLM plumbing."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from observa.graph.master_consolidator import (
    _blackboard_digest,
    _chat_digest,
    _user_prompt,
    consolidate,
)
from observa.models import (
    CaseState,
    ChatMessage,
    Finding,
    FinalSummary,
    SummaryFinding,
    TurnFindings,
)


def _state(**overrides) -> CaseState:
    base: CaseState = {
        "case_id": "C",
        "created_at": datetime(2026, 4, 24),
        "dbid": 1234,
        "oracle_version": "19.22.0",
        "topology": "rac",
        "instance_numbers": [1, 2],
        "problem_statement": "Batch slow",
        "investigation_scope": "window 01:30-03:00",
        "known_facts": ["batch uses SQL_ID abc"],
        "input_files": [],
        "current_turn": 2,
        "total_turns": 2,
        "turn_findings": [],
        "agent_questions": [],
        "chat_log": [],
        "hypotheses": [],
        "contradictions": [],
        "contradiction_resolutions": [],
        "final_summary": None,
        "verification": None,
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def test_blackboard_digest_empty():
    assert "(blackboard is empty" in _blackboard_digest(_state())


def test_blackboard_digest_groups_by_turn_and_includes_abstain():
    findings = [
        TurnFindings(
            agent="wait_time",
            turn=1,
            findings=[
                Finding(
                    finding_id="f1", agent="wait_time", turn=1,
                    severity="high", code="LFS", description="log file sync",
                    evidence_note="SELECT ... DBA_HIST_SYSTEM_EVENT",
                    related_objects=["SYS.FOO"],
                )
            ],
            notes="ran 2 queries",
        ),
        TurnFindings(agent="memory", turn=1, abstained=True, abstain_reason="PGA stable"),
        TurnFindings(agent="sql", turn=2, abstained=True),
    ]
    digest = _blackboard_digest(_state(turn_findings=findings))
    assert "=== TURN 1 ===" in digest
    assert "=== TURN 2 ===" in digest
    assert "HIGH · LFS" in digest
    assert "log file sync" in digest
    assert "DBA_HIST_SYSTEM_EVENT" in digest
    assert "SYS.FOO" in digest
    assert "[memory] ABSTAINED — PGA stable" in digest
    assert "ran 2 queries" in digest


def test_chat_digest_empty_and_populated():
    assert "(no analyst chat" in _chat_digest(_state())
    chat = [
        ChatMessage(sender="agent", agent="memory", turn=1, text="which window?", timestamp=datetime(2026, 4, 24)),
        ChatMessage(sender="user", turn=1, text="100-110", timestamp=datetime(2026, 4, 24)),
    ]
    digest = _chat_digest(_state(chat_log=chat))
    assert "[agent/memory · turn 1] which window?" in digest
    assert "[user · turn 1] 100-110" in digest


def test_user_prompt_includes_all_sections():
    prompt = _user_prompt(_state(known_facts=["fact a", "fact b"]))
    for section in [
        "PROBLEM STATEMENT",
        "INVESTIGATION SCOPE",
        "KNOWN FACTS",
        "DBID: 1234",
        "Oracle: 19.22.0",
        "Topology: rac",
        "=== BLACKBOARD ===",
        "=== ANALYST CHAT ===",
        "Produce the FinalSummary",
    ]:
        assert section in prompt, f"missing: {section}"
    assert "- fact a" in prompt
    assert "- fact b" in prompt


class _FakeStructuredLLM:
    def __init__(self, result: FinalSummary | Exception) -> None:
        self._result = result

    async def ainvoke(self, messages: Any) -> FinalSummary:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _FakeLLM:
    def __init__(self, result: FinalSummary | Exception) -> None:
        self._result = result
        self.last_schema: Any = None

    def with_structured_output(self, schema: Any, **_kwargs: Any) -> _FakeStructuredLLM:
        self.last_schema = schema
        return _FakeStructuredLLM(self._result)


async def test_consolidate_happy_path_returns_summary():
    expected = FinalSummary(
        problem_restated="Batch is slow after 2am",
        top_findings=[
            SummaryFinding(agent="wait_time", severity="high", description="log file sync")
        ],
        root_cause="redo sync contention",
        confidence=0.72,
    )
    llm = _FakeLLM(expected)
    result = await consolidate(_state(), llm)  # type: ignore[arg-type]
    assert result == expected
    assert llm.last_schema is FinalSummary


async def test_consolidate_llm_error_falls_back_to_safe_summary():
    llm = _FakeLLM(RuntimeError("api dead"))
    result = await consolidate(_state(), llm)  # type: ignore[arg-type]
    assert isinstance(result, FinalSummary)
    assert result.confidence == 0.0
    assert "RuntimeError" in result.root_cause
    assert result.top_findings == []
    assert "Master LLM call failed" in " ".join(result.unknowns)


class _FakeLedgerK:
    def __init__(self, ids):
        self._ids = frozenset(ids)

    def known_ids(self):
        return self._ids


def test_summary_finding_refs_validated_against_ledger():
    from observa.graph.master_consolidator import _validate_summary_refs
    from observa.models import FinalSummary, SummaryFinding

    def _fresh():
        return FinalSummary(
            problem_restated="p",
            top_findings=[SummaryFinding(agent="wait_time", severity="high",
                                         description="d",
                                         evidence_refs=["Q-001", "Q-999"])],
            root_cause="r", confidence=0.5,
        )

    out = _validate_summary_refs(_fresh(), _FakeLedgerK({"Q-001"}))
    assert out.top_findings[0].evidence_refs == ["Q-001"]
    out2 = _validate_summary_refs(_fresh(), None)
    assert out2.top_findings[0].evidence_refs == ["Q-001", "Q-999"]


def test_system_prompt_instructs_ref_carry():
    from observa.graph.master_consolidator import _MASTER_SYSTEM_PROMPT

    assert "evidence_refs" in _MASTER_SYSTEM_PROMPT
