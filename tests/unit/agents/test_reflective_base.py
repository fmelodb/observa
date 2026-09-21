"""Unit tests for ReflectiveAgent — prompt plumbing + conversion.

The tool loop and LLM interaction are exercised in integration tests with a
fake LLM. Here we focus on deterministic pieces.
"""
from __future__ import annotations

from datetime import datetime

from observa.agents.base import ReflectiveAgent, _Finding, _Question, _ReflectiveOutput
from observa.models import (
    AgentQuestion,
    CaseState,
    ChatMessage,
    Finding,
    InputFile,
    SnapWindow,
    TurnFindings,
)


class _FakeSpecialist(ReflectiveAgent):
    name = "fake"
    display_name = "Fake Specialist"
    specialty_prompt = "Analyze fakeness only. Ignore everything else."


def _state(**overrides) -> CaseState:
    base: CaseState = {
        "case_id": "CASE-1",
        "created_at": datetime(2026, 4, 24),
        "dbid": 1234567890,
        "oracle_version": "19.20.0",
        "topology": "single-instance",
        "instance_numbers": [1],
        "problem_statement": "Slow batch job at 02:00.",
        "investigation_scope": "Focus on window 01:30-03:00.",
        "known_facts": ["job uses SQL_ID abc123"],
        "input_files": [InputFile(type="alertlog", path="C:/tmp/a.log")],
        "current_turn": 1,
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
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def test_system_prompt_includes_universal_rules_and_specialty():
    agent = _FakeSpecialist()
    prompt = agent._system_prompt()
    assert "OBJECTIVITY" in prompt
    assert "STRICT SPECIALTY" in prompt
    assert "ABSTAIN EXPLICITLY" in prompt
    assert "Fake Specialist" in prompt
    assert "Analyze fakeness only" in prompt


def test_blackboard_digest_first_turn():
    agent = _FakeSpecialist()
    assert agent._blackboard_digest(_state(), 1).startswith("(no prior turns")


def test_blackboard_digest_includes_prior_findings_and_abstain():
    agent = _FakeSpecialist()
    prior = [
        TurnFindings(
            agent="wait_time",
            turn=1,
            findings=[
                Finding(
                    finding_id="f1",
                    agent="wait_time",
                    turn=1,
                    severity="high",
                    code="LOG_FILE_SYNC",
                    description="log file sync dominant in window",
                    evidence_note="query: ...",
                )
            ],
            notes="ran 3 queries",
        ),
        TurnFindings(
            agent="memory",
            turn=1,
            abstained=True,
            abstain_reason="PGA/SGA stable — nothing relevant",
        ),
    ]
    digest = agent._blackboard_digest(_state(turn_findings=prior), current_turn=2)
    assert "[turn 1 · wait_time]" in digest
    assert "HIGH · LOG_FILE_SYNC" in digest
    assert "[turn 1 · memory] ABSTAINED" in digest
    assert "nothing relevant" in digest


def test_blackboard_digest_filters_future_turns():
    agent = _FakeSpecialist()
    future = TurnFindings(agent="x", turn=3, abstained=True, abstain_reason="future")
    digest = agent._blackboard_digest(_state(turn_findings=[future]), current_turn=2)
    assert "future" not in digest


def test_chat_digest_formats_messages():
    agent = _FakeSpecialist()
    chat = [
        ChatMessage(sender="agent", agent="memory", turn=1, text="What AWR window?", timestamp=datetime(2026, 4, 24)),
        ChatMessage(sender="user", turn=1, text="snap 100-110", timestamp=datetime(2026, 4, 24)),
    ]
    digest = agent._chat_digest(_state(chat_log=chat), current_turn=2)
    assert "[agent/memory · turn 1] What AWR window?" in digest
    assert "[user · turn 1] snap 100-110" in digest


def test_files_digest_lists_inputs():
    agent = _FakeSpecialist()
    files = [
        InputFile(type="alertlog", path="C:/a.log", label="prod1"),
        InputFile(type="trace", path="C:/b.trc"),
    ]
    digest = agent._files_digest(_state(input_files=files))
    assert "[alertlog] C:/a.log (prod1)" in digest
    assert "[trace] C:/b.trc" in digest


def test_files_digest_empty():
    agent = _FakeSpecialist()
    assert "no files provided" in agent._files_digest(_state(input_files=[]))


def test_user_prompt_includes_core_sections():
    agent = _FakeSpecialist()
    prompt = agent._user_prompt(_state(), current_turn=1)
    assert "PROBLEM STATEMENT" in prompt
    assert "Slow batch job" in prompt
    assert "KNOWN FACTS" in prompt
    assert "abc123" in prompt
    assert "AVAILABLE FILES" in prompt
    assert "BLACKBOARD" in prompt


def test_to_turn_findings_preserves_findings_and_questions():
    agent = _FakeSpecialist()
    output = _ReflectiveOutput(
        abstained=False,
        findings=[
            _Finding(
                severity="medium",
                code="FAKE_A",
                description="something fake",
                evidence_refs=["Q-001"],
                related_objects=["SYS.X"],
            )
        ],
        questions=[_Question(question="what window?")],
        notes="ran 1 query",
    )
    tf = agent._to_turn_findings(output, turn=2)
    assert tf.agent == "fake"
    assert tf.turn == 2
    assert not tf.abstained
    assert len(tf.findings) == 1
    f = tf.findings[0]
    assert f.severity == "medium"
    assert f.code == "FAKE_A"
    assert f.evidence_note == ""
    assert f.evidence_refs == ["Q-001"]
    assert f.related_objects == ["SYS.X"]
    assert f.agent == "fake"
    assert f.turn == 2
    assert f.finding_id.startswith("fake-2-")
    assert len(tf.questions) == 1
    assert tf.questions[0].question == "what window?"
    assert tf.notes == "ran 1 query"


def test_to_turn_findings_normalizes_empty_output_to_abstain():
    agent = _FakeSpecialist()
    output = _ReflectiveOutput(abstained=False, findings=[], notes="explored")
    tf = agent._to_turn_findings(output, turn=1)
    assert tf.abstained is True
    assert "No evidence-based findings" in tf.abstain_reason
    assert tf.notes == "explored"


def test_to_turn_findings_respects_explicit_abstain():
    agent = _FakeSpecialist()
    output = _ReflectiveOutput(abstained=True, abstain_reason="no data", notes="n/a")
    tf = agent._to_turn_findings(output, turn=1)
    assert tf.abstained is True
    assert tf.abstain_reason == "no data"
    assert tf.findings == []


def test_force_structured_output_falls_back_to_text_json():
    """xAI Grok-4-fast-non-reasoning emits the structured payload as text
    content instead of a tool call. The fallback must json.loads the
    content and validate it as _ReflectiveOutput before abstaining."""
    import asyncio
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage
    from observa.agents.base import ReflectiveAgent

    agent = _FakeSpecialist()
    raw = AIMessage(
        content='{"abstained": false, "findings": [], "notes": "via text-json"}',
        tool_calls=[],
    )
    structured_runnable = MagicMock()

    async def fake_ainvoke(prompt):
        return {"raw": raw, "parsed": None, "parsing_error": None}

    structured_runnable.ainvoke = fake_ainvoke
    llm = MagicMock()
    llm.with_structured_output.return_value = structured_runnable

    output = asyncio.run(agent._force_structured_output([], llm, "xai.grok-4-fast-non-reasoning"))
    assert output.abstained is False
    assert output.notes == "via text-json"


def test_force_structured_output_falls_back_through_markdown_fence():
    """Some models wrap JSON in a ```json …``` fence; the fallback must strip it."""
    import asyncio
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage

    agent = _FakeSpecialist()
    raw = AIMessage(
        content='```json\n{"abstained": true, "abstain_reason": "no data", "findings": []}\n```',
        tool_calls=[],
    )
    structured_runnable = MagicMock()

    async def fake_ainvoke(prompt):
        return {"raw": raw, "parsed": None, "parsing_error": None}

    structured_runnable.ainvoke = fake_ainvoke
    llm = MagicMock()
    llm.with_structured_output.return_value = structured_runnable

    output = asyncio.run(agent._force_structured_output([], llm, "xai.grok-4-fast-non-reasoning"))
    assert output.abstained is True
    assert output.abstain_reason == "no data"


def test_force_structured_output_abstains_when_text_is_not_json():
    """Garbage content still falls through to the diagnostic abstain."""
    import asyncio
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage

    agent = _FakeSpecialist()
    raw = AIMessage(content="this is just prose, not JSON", tool_calls=[])
    structured_runnable = MagicMock()

    async def fake_ainvoke(prompt):
        return {"raw": raw, "parsed": None, "parsing_error": None}

    structured_runnable.ainvoke = fake_ainvoke
    llm = MagicMock()
    llm.with_structured_output.return_value = structured_runnable

    output = asyncio.run(agent._force_structured_output([], llm, "xai.grok-4-fast-non-reasoning"))
    assert output.abstained is True
    assert "did not produce a parseable" in (output.abstain_reason or "")


def test_force_structured_output_accepts_null_related_objects():
    """Gemini sometimes emits ``related_objects: null`` instead of an empty list.
    The schema must coerce None → [] so the text-JSON fallback validates."""
    import asyncio
    import json
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage

    agent = _FakeSpecialist()
    payload = {
        "abstained": False,
        "findings": [
            {
                "severity": "high",
                "code": "X",
                "description": "y",
                "evidence_refs": ["Q-001"],
                "related_objects": None,
            }
        ],
        "notes": "ran 1 query",
    }
    raw = AIMessage(content=json.dumps(payload), tool_calls=[])
    structured_runnable = MagicMock()

    async def fake_ainvoke(prompt):
        return {"raw": raw, "parsed": None, "parsing_error": None}

    structured_runnable.ainvoke = fake_ainvoke
    llm = MagicMock()
    llm.with_structured_output.return_value = structured_runnable

    output = asyncio.run(agent._force_structured_output([], llm, "google.gemini-2.5-flash"))
    assert output.abstained is False
    assert len(output.findings) == 1
    assert output.findings[0].related_objects == []


def test_tool_loop_appends_response_for_every_call_even_on_unexpected_failure():
    """Gemini rejects subsequent turns with HTTP 400 if function_response count
    diverges from function_call count. The tool loop must append a ToolMessage
    per tool_call even when execution raises unexpectedly."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from langchain_core.messages import AIMessage

    agent = _FakeSpecialist()
    response = AIMessage(
        content="",
        tool_calls=[
            {"id": "c1", "name": "good", "args": {}},
            {"id": "c2", "name": "good", "args": {}},
        ],
    )
    second_response = AIMessage(content="done", tool_calls=[])
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[response, second_response])

    call_count = {"n": 0}

    async def boom(self, call, tools_by_name):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated unexpected failure")
        return "ok"

    import observa.agents.base as base_mod
    original = base_mod.ReflectiveAgent._execute_tool_call
    base_mod.ReflectiveAgent._execute_tool_call = boom  # type: ignore[assignment]
    try:
        import time
        deadline = time.monotonic() + 30
        out = asyncio.run(agent._tool_loop([], llm, {}, deadline))
    finally:
        base_mod.ReflectiveAgent._execute_tool_call = original  # type: ignore[assignment]

    tool_messages = [m for m in out if m.__class__.__name__ == "ToolMessage"]
    assert len(tool_messages) == 2, "must emit one ToolMessage per tool_call"
    assert tool_messages[0].tool_call_id == "c1"
    assert "ERROR" in tool_messages[0].content
    assert tool_messages[1].tool_call_id == "c2"


def test_force_structured_output_abstains_clearly_on_empty_response():
    """Gemini sometimes returns no content + no tool calls (safety filter or
    max_tokens). The abstain message must say so explicitly instead of the
    generic "no parseable output" text — there's literally nothing to parse."""
    import asyncio
    from unittest.mock import MagicMock
    from langchain_core.messages import AIMessage

    agent = _FakeSpecialist()
    raw = AIMessage(content="", tool_calls=[])
    structured_runnable = MagicMock()

    async def fake_ainvoke(prompt):
        return {"raw": raw, "parsed": None, "parsing_error": None}

    structured_runnable.ainvoke = fake_ainvoke
    llm = MagicMock()
    llm.with_structured_output.return_value = structured_runnable

    output = asyncio.run(agent._force_structured_output([], llm, "google.gemini-2.5-pro"))
    assert output.abstained is True
    reason = output.abstain_reason or ""
    assert "empty response" in reason
    assert "google.gemini-2.5-pro" in reason
    assert "did not produce a parseable" not in reason


class _WindowsDummy(ReflectiveAgent):
    name = "dummy_windows"
    display_name = "Dummy Windows"
    specialty_prompt = "test specialty"


def _windows_state() -> dict:
    return {
        "investigation_scope": "scope", "problem_statement": "prob",
        "known_facts": [], "input_files": [], "dbid": 111,
        "problem_window": SnapWindow(
            begin_snap=100, end_snap=102,
            begin_time=datetime(2026, 7, 8, 14, 0), end_time=datetime(2026, 7, 8, 16, 0),
        ),
        "baseline_window": SnapWindow(begin_snap=50, end_snap=52),
        "baseline_diff": {
            "has_baseline": True, "suspect": False, "errors": [],
            "problem_label": "snaps 100→102", "baseline_label": "snaps 50→52",
            "load_profile": [{"metric": "DB time (s/s)", "problem": 4.0, "baseline": 1.0, "ratio": 4.0}],
            "top_waits_problem": [], "top_waits_baseline": [],
            "top_sql_problem": [], "top_sql_baseline": [], "os": [],
        },
    }


def test_static_prompt_includes_windows_and_diff():
    agent = _WindowsDummy()
    state = _windows_state()
    prompt = agent._user_prompt_static(state)
    variable = agent._user_prompt_variable(state, 1)
    assert "TIME WINDOWS" in prompt
    assert "snaps100→102" in prompt.replace(" ", "")
    assert "BASELINE DIFF" in prompt
    assert "DB time (s/s)" in prompt
    # Cache correctness: the digest must live ONLY in the static (cacheable) zone.
    assert "TIME WINDOWS" not in variable
    assert "BASELINE DIFF" not in variable


def test_static_prompt_without_windows_says_so():
    state = _windows_state()
    state["problem_window"] = None
    state["baseline_window"] = None
    state["baseline_diff"] = None
    prompt = _WindowsDummy()._user_prompt_static(state)
    assert "no problem/baseline windows defined" in prompt.lower()


def test_universal_rules_mention_window_restriction():
    from observa.agents.base import _UNIVERSAL_RULES
    assert "problem window" in _UNIVERSAL_RULES.lower()


def test_finding_maps_evidence_refs_and_uncited():
    from observa.agents.base import _Finding, _ReflectiveOutput

    agent = _FakeSpecialist()
    out = _ReflectiveOutput(
        abstained=False,
        findings=[
            _Finding(severity="high", code="A", description="cited",
                     evidence_refs=["Q-001", "F-002"], evidence_note="rows 1-3"),
            _Finding(severity="low", code="B", description="uncited"),
        ],
    )
    tf = agent._to_turn_findings(out, turn=1)
    assert tf.findings[0].evidence_refs == ["Q-001", "F-002"]
    assert tf.findings[0].evidence_note == "rows 1-3"
    assert tf.findings[0].uncited is False
    assert tf.findings[1].evidence_refs == []
    assert tf.findings[1].uncited is True


def test_finding_evidence_refs_none_coerced():
    from observa.agents.base import _Finding

    # None is deliberately invalid per the annotation — the validator coerces it.
    f = _Finding(severity="info", code="X", description="d", evidence_refs=None)  # type: ignore[arg-type]
    assert f.evidence_refs == []


def test_universal_rules_mention_evidence_ids():
    from observa.agents.base import _UNIVERSAL_RULES

    assert "evidence_id" in _UNIVERSAL_RULES
    assert "evidence_refs" in _UNIVERSAL_RULES


_ORA_TIMELINE = {
    "coverage": {"begin_ts": "2024-03-15T00:00:00", "end_ts": "2024-03-15T23:00:00",
                 "covers_problem": True, "covers_baseline": False},
    "counts": {"problem": 1, "baseline": 0, "outside": 0, "incidents": 1},
    "incidents": [{"ts": "2024-03-15T14:30:00", "code": "ORA-00060", "text": "deadlock",
                   "instance": 1, "incident_id": None, "byte_offset": 10,
                   "window": "problem", "source_label": "prod", "evidence_id": "S-001"}],
}


def test_static_prompt_includes_ora_timeline_when_present():
    agent = _FakeSpecialist()
    text = agent._user_prompt_static(_state(ora_timeline=_ORA_TIMELINE))
    assert "ORA-00060" in text
    assert "ORA INCIDENT TIMELINE" in text


def test_static_prompt_omits_ora_timeline_when_absent():
    agent = _FakeSpecialist()
    text = agent._user_prompt_static(_state(ora_timeline=None))
    assert "ORA INCIDENT TIMELINE" not in text
