"""Per-finding evidence replay + confidence calibration (Feature 5)."""
from __future__ import annotations

from observa.graph.replay_verify import replay_verify
from observa.mcp.evidence import EvidenceLedger
from observa.models import CaseState, FinalSummary, SummaryFinding


def _ledger() -> EvidenceLedger:
    lg = EvidenceLedger()
    lg.record_query("query_awr", "SELECT 1 FROM DBA_HIST_SQLSTAT WHERE dbid=1",
                    "sql", result={"rows": [{"BUSY": 99}], "row_count": 1})  # Q-001
    return lg


def _state(top: list[SummaryFinding], confidence: float = 0.8) -> CaseState:
    return {  # type: ignore
        "final_summary": FinalSummary(
            problem_restated="p", top_findings=top, root_cause="rc",
            confidence=confidence),
    }


class _FakeLLM:
    def __init__(self) -> None:
        self.calls = 0


async def test_uncited_finding_is_unsupported_without_llm(monkeypatch):
    from observa.graph import replay_verify as mod

    async def must_not_call(*a, **k):  # pragma: no cover
        raise AssertionError("LLM must not be called for an uncited finding")

    monkeypatch.setattr(mod, "parse_structured", must_not_call)
    top = [SummaryFinding(agent="sql", severity="high", description="d",
                          evidence_refs=[])]  # no refs → uncited
    result = await replay_verify(_state(top), _ledger(), _FakeLLM())
    assert result is not None
    assert result.verifications[0].verdict == "unsupported"
    assert result.support_score == 0.0
    assert result.applied_confidence_factor == 0.0


async def test_supported_finding_keeps_confidence(monkeypatch):
    from observa.graph import replay_verify as mod

    async def fake_parse(llm, schema, messages):
        return schema(verdict="supported", reasoning="rows show 99% busy")

    monkeypatch.setattr(mod, "parse_structured", fake_parse)
    top = [SummaryFinding(agent="sql", severity="high", description="host cpu 99%",
                          evidence_refs=["Q-001"])]
    st = _state(top, confidence=0.8)
    result = await replay_verify(st, _ledger(), _FakeLLM())
    assert result.support_score == 1.0
    assert result.applied_confidence_factor == 1.0
    assert st["final_summary"].confidence == 0.8  # patched in place via node; here unchanged factor


async def test_partial_counts_half(monkeypatch):
    from observa.graph import replay_verify as mod

    verdicts = iter(["supported", "partial"])

    async def fake_parse(llm, schema, messages):
        return schema(verdict=next(verdicts), reasoning="r")

    monkeypatch.setattr(mod, "parse_structured", fake_parse)
    top = [
        SummaryFinding(agent="sql", severity="high", description="a", evidence_refs=["Q-001"]),
        SummaryFinding(agent="io", severity="high", description="b", evidence_refs=["Q-001"]),
    ]
    result = await replay_verify(_state(top), _ledger(), _FakeLLM())
    assert result.support_score == 0.75  # (1.0 + 0.5) / 2


async def test_llm_error_excluded_from_score(monkeypatch):
    from observa.graph import replay_verify as mod

    async def boom(*a, **k):
        raise RuntimeError("dead")

    monkeypatch.setattr(mod, "parse_structured", boom)
    top = [SummaryFinding(agent="sql", severity="high", description="a",
                          evidence_refs=["Q-001"])]
    result = await replay_verify(_state(top), _ledger(), _FakeLLM())
    # Only finding errored → scored_count 0 → no penalty.
    assert result.support_score == 1.0
    assert result.applied_confidence_factor == 1.0


async def test_no_final_summary_returns_none():
    result = await replay_verify({}, _ledger(), _FakeLLM())  # type: ignore
    assert result is None


async def test_evidence_content_containing_not_found_still_calls_llm(monkeypatch):
    """A present ledger record whose row content happens to contain the text
    "NOT FOUND" must NOT be mechanically short-circuited — the LLM is still
    consulted and its verdict wins (regression guard for FIX 1)."""
    from observa.graph import replay_verify as mod

    async def fake_parse(llm, schema, messages):
        return schema(verdict="supported", reasoning="object legitimately not found, as expected")

    monkeypatch.setattr(mod, "parse_structured", fake_parse)
    lg = EvidenceLedger()
    lg.record_query("query_awr", "SELECT 1 FROM DBA_HIST_SQLSTAT WHERE dbid=1",
                    "sql", result={"rows": [{"MSG": "object NOT FOUND"}], "row_count": 1})  # Q-001
    top = [SummaryFinding(agent="sql", severity="high", description="object missing",
                          evidence_refs=["Q-001"])]
    result = await replay_verify(_state(top), lg, _FakeLLM())
    assert result is not None
    assert result.verifications[0].verdict == "supported"
    assert result.support_score == 1.0


async def test_ref_absent_from_ledger_is_unsupported_without_llm(monkeypatch):
    from observa.graph import replay_verify as mod

    async def must_not_call(*a, **k):  # pragma: no cover
        raise AssertionError("LLM must not be called when the cited ref is absent from the ledger")

    monkeypatch.setattr(mod, "parse_structured", must_not_call)
    top = [SummaryFinding(agent="sql", severity="high", description="d",
                          evidence_refs=["Q-404"])]  # never recorded
    result = await replay_verify(_state(top), _ledger(), _FakeLLM())
    assert result is not None
    assert result.verifications[0].verdict == "unsupported"
    assert result.verifications[0].reasoning == "All cited evidence missing from the ledger."
