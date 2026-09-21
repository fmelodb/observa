"""Hypothesis synthesis from the blackboard (Feature 5)."""
from __future__ import annotations

from typing import Any

from observa.graph.hypothesis_synth import (
    render_blackboard_with_ids,
    synthesize_hypotheses,
)
from observa.models import CaseState, Finding, TurnFindings


def _state(findings: list[Finding]) -> CaseState:
    tf = TurnFindings(agent="sql", turn=1, findings=findings)
    return {"turn_findings": [tf], "problem_statement": "slow batch"}  # type: ignore


def _finding(fid: str, desc: str) -> Finding:
    return Finding(finding_id=fid, agent="sql", turn=1, severity="high",
                   code="C", description=desc, evidence_refs=["Q-001"])


class _FakeLLM:
    """Stub whose parse target is fed via the module-level parse_structured."""


def test_render_blackboard_includes_finding_ids():
    state = _state([_finding("sql-1-abcd", "full scan on ORDERS")])
    text = render_blackboard_with_ids(state)
    assert "sql-1-abcd" in text
    assert "full scan on ORDERS" in text


async def test_synthesize_maps_and_assigns_stable_ids(monkeypatch):
    from observa.graph import hypothesis_synth as mod
    from observa.graph.hypothesis_synth import _HypothesisDraft, _HypothesisList

    async def fake_parse(llm, schema, messages):
        return _HypothesisList(hypotheses=[
            _HypothesisDraft(statement="CPU saturation", prior=0.6,
                             supporting_findings=["sql-1-abcd"],
                             contradicting_findings=[]),
            _HypothesisDraft(statement="IO bottleneck", prior=0.3,
                             supporting_findings=[], contradicting_findings=[]),
        ])

    monkeypatch.setattr(mod, "parse_structured", fake_parse)
    hyps = await synthesize_hypotheses(_state([_finding("sql-1-abcd", "d")]), _FakeLLM())
    assert [h.hyp_id for h in hyps] == ["H-1", "H-2"]
    assert hyps[0].statement == "CPU saturation"
    assert hyps[0].posterior == hyps[0].prior == 0.6  # posterior seeded = prior
    assert hyps[0].supporting_findings == ["sql-1-abcd"]


async def test_synthesize_empty_blackboard_returns_empty(monkeypatch):
    hyps = await synthesize_hypotheses({"turn_findings": []}, _FakeLLM())  # type: ignore
    assert hyps == []


async def test_synthesize_parse_error_returns_empty(monkeypatch):
    from observa.graph import hypothesis_synth as mod
    from observa.llm.structured import StructuredOutputError

    async def boom(*_a, **_k):
        raise StructuredOutputError("bad", raw_text="x", raw_tool_calls=None,
                                    parsing_error="e")

    monkeypatch.setattr(mod, "parse_structured", boom)
    hyps = await synthesize_hypotheses(_state([_finding("sql-1-abcd", "d")]), _FakeLLM())
    assert hyps == []
