"""Evidence-grounded contradiction resolution (Feature 5)."""
from __future__ import annotations

from observa.graph.contradiction_resolver import resolve_contradictions
from observa.mcp.evidence import EvidenceLedger
from observa.models import CaseState, Finding, Hypothesis, TurnFindings


def _ledger() -> EvidenceLedger:
    lg = EvidenceLedger()
    lg.record_query("query_awr", "SELECT 1 FROM DBA_HIST_OSSTAT WHERE dbid=1",
                    "infra", result={"rows": [{"BUSY": 99}], "row_count": 1})  # Q-001
    return lg


def _state(**kw) -> CaseState:
    f = Finding(finding_id="infra-1-x", agent="infra", turn=1, severity="high",
                code="CPU", description="host cpu 99%", evidence_refs=["Q-001"])
    base: CaseState = {  # type: ignore
        "turn_findings": [TurnFindings(agent="infra", turn=1, findings=[f])],
        "hypotheses": [
            Hypothesis(hyp_id="H-1", statement="CPU saturation", prior=0.6,
                       posterior=0.6, supporting_findings=["infra-1-x"]),
            Hypothesis(hyp_id="H-2", statement="IO wait", prior=0.4, posterior=0.4),
        ],
        "contradictions": [{"hyp_a": "H-1", "hyp_b": "H-2", "explanation": "x"}],
    }
    base.update(kw)  # type: ignore
    return base


class _FakeLLM:
    pass


async def test_resolve_favors_side_from_evidence(monkeypatch):
    from observa.graph import contradiction_resolver as mod

    async def fake_parse(llm, schema, messages):
        return schema(favored_hyp_id="H-1", inconclusive=False,
                      reasoning="Q-001 shows 99% busy",
                      posterior_a=0.85, posterior_b=0.15)

    monkeypatch.setattr(mod, "parse_structured", fake_parse)
    res = await resolve_contradictions(_state(), _ledger(), _FakeLLM())
    assert len(res) == 1
    assert res[0].favored_hyp_id == "H-1"
    assert res[0].posterior_a == 0.85


async def test_resolve_no_contradictions_returns_empty():
    res = await resolve_contradictions(_state(contradictions=[]), _ledger(), _FakeLLM())
    assert res == []


async def test_resolve_unknown_hyp_id_skipped(monkeypatch):
    from observa.graph import contradiction_resolver as mod

    async def fake_parse(*a, **k):  # pragma: no cover — must not be called
        raise AssertionError("should not resolve an unknown pair")

    monkeypatch.setattr(mod, "parse_structured", fake_parse)
    res = await resolve_contradictions(
        _state(contradictions=[{"hyp_a": "H-1", "hyp_b": "H-9", "explanation": "x"}]),
        _ledger(), _FakeLLM(),
    )
    assert res == []


async def test_resolve_llm_error_marks_inconclusive(monkeypatch):
    from observa.graph import contradiction_resolver as mod

    async def boom(*a, **k):
        raise RuntimeError("llm dead")

    monkeypatch.setattr(mod, "parse_structured", boom)
    res = await resolve_contradictions(_state(), _ledger(), _FakeLLM())
    assert len(res) == 1
    assert res[0].inconclusive is True


async def test_resolve_hallucinated_favored_id_is_cleared(monkeypatch):
    """A verdict favoring a hyp id outside the pair (LLM hallucination) must
    not be stored/rendered — it gets clamped to None (FIX 2)."""
    from observa.graph import contradiction_resolver as mod

    async def fake_parse(llm, schema, messages):
        return schema(favored_hyp_id="H-99", inconclusive=False,
                      reasoning="bogus id", posterior_a=0.5, posterior_b=0.5)

    monkeypatch.setattr(mod, "parse_structured", fake_parse)
    res = await resolve_contradictions(_state(), _ledger(), _FakeLLM())
    assert len(res) == 1
    assert res[0].favored_hyp_id is None
