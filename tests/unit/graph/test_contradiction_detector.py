"""Tests for ContradictionDetector — LLM-backed pairwise mutual exclusivity checker."""
from __future__ import annotations

import json
from typing import Any

from observa.graph.contradiction_detector import ContradictionDetector
from observa.models import Hypothesis


def _hyp(hyp_id: str, posterior: float = 0.8, status: str = "open", statement: str = "some cause") -> Hypothesis:
    return Hypothesis(
        hyp_id=hyp_id,
        statement=statement,
        prior=0.5,
        posterior=posterior,
        status=status,  # type: ignore[arg-type]
    )


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    """Fake chat LLM. Returns canned JSON payload(s) on .ainvoke()."""

    def __init__(self, payloads: list[dict] | dict) -> None:
        if isinstance(payloads, dict):
            payloads = [payloads]
        self._payloads = payloads
        self._i = 0
        self.calls: list[Any] = []

    async def ainvoke(self, messages: Any) -> _FakeResponse:
        self.calls.append(messages)
        payload = self._payloads[min(self._i, len(self._payloads) - 1)]
        self._i += 1
        return _FakeResponse(json.dumps(payload))


async def test_run_empty_returns_empty_list():
    det = ContradictionDetector(_FakeLLM({"mutually_exclusive": True, "explanation": "x"}))
    assert await det.run([]) == []


async def test_run_single_hypothesis_returns_empty():
    det = ContradictionDetector(_FakeLLM({"mutually_exclusive": True, "explanation": "x"}))
    assert await det.run([_hyp("H1")]) == []


async def test_run_both_below_threshold_returns_empty():
    llm = _FakeLLM({"mutually_exclusive": True, "explanation": "x"})
    det = ContradictionDetector(llm, threshold=0.5)
    result = await det.run([_hyp("H1", posterior=0.2), _hyp("H2", posterior=0.1)])
    assert result == []
    # LLM should never be invoked when ineligible.
    assert llm.calls == []


async def test_run_one_below_threshold_returns_empty():
    llm = _FakeLLM({"mutually_exclusive": True, "explanation": "x"})
    det = ContradictionDetector(llm, threshold=0.5)
    result = await det.run([_hyp("H1", posterior=0.9), _hyp("H2", posterior=0.2)])
    assert result == []
    assert llm.calls == []


async def test_run_rejected_hypotheses_excluded():
    llm = _FakeLLM({"mutually_exclusive": True, "explanation": "x"})
    det = ContradictionDetector(llm, threshold=0.3)
    result = await det.run([
        _hyp("H1", posterior=0.9, status="rejected"),
        _hyp("H2", posterior=0.9, status="open"),
    ])
    assert result == []
    assert llm.calls == []


async def test_run_two_contradicting_hypotheses_returns_one_pair():
    llm = _FakeLLM({"mutually_exclusive": True, "explanation": "A excludes B"})
    det = ContradictionDetector(llm, threshold=0.3)
    result = await det.run([
        _hyp("HA", posterior=0.8, statement="CPU saturation"),
        _hyp("HB", posterior=0.7, statement="IO saturation"),
    ])
    assert result == [{"hyp_a": "HA", "hyp_b": "HB", "explanation": "A excludes B"}]
    assert len(llm.calls) == 1


async def test_run_compatible_hypotheses_returns_empty():
    llm = _FakeLLM({"mutually_exclusive": False, "explanation": "both can coexist"})
    det = ContradictionDetector(llm, threshold=0.3)
    result = await det.run([
        _hyp("HA", posterior=0.8),
        _hyp("HB", posterior=0.7),
    ])
    assert result == []
    assert len(llm.calls) == 1


async def test_run_three_hypotheses_checks_all_pairs():
    # Three eligible hypotheses → 3 pairs; first two contradict, last is compatible.
    llm = _FakeLLM([
        {"mutually_exclusive": True, "explanation": "A vs B"},
        {"mutually_exclusive": False, "explanation": "A vs C compatible"},
        {"mutually_exclusive": True, "explanation": "B vs C"},
    ])
    det = ContradictionDetector(llm, threshold=0.3)
    result = await det.run([
        _hyp("HA", posterior=0.9),
        _hyp("HB", posterior=0.8),
        _hyp("HC", posterior=0.7),
    ])
    assert result == [
        {"hyp_a": "HA", "hyp_b": "HB", "explanation": "A vs B"},
        {"hyp_a": "HB", "hyp_b": "HC", "explanation": "B vs C"},
    ]
    assert len(llm.calls) == 3


class _ExplodingLLM:
    async def ainvoke(self, messages: Any) -> _FakeResponse:
        raise RuntimeError("boom")


async def test_run_llm_failure_is_swallowed():
    det = ContradictionDetector(_ExplodingLLM(), threshold=0.3)
    result = await det.run([_hyp("HA", posterior=0.8), _hyp("HB", posterior=0.7)])
    assert result == []
