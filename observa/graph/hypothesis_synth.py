"""Synthesize competing root-cause hypotheses from the blackboard.

Populates ``CaseState.hypotheses`` (a field wired to contradiction detection +
the report but historically never produced). Runs once after consolidation.
Fail-open: empty blackboard / parse error → ``[]``.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from observa.llm.structured import StructuredOutputError, parse_structured
from observa.models import CaseState, Hypothesis, TurnFindings

logger = logging.getLogger(__name__)


class _HypothesisDraft(BaseModel):
    statement: str
    prior: float = Field(ge=0, le=1)
    supporting_findings: list[str] = Field(default_factory=list)
    contradicting_findings: list[str] = Field(default_factory=list)


class _HypothesisList(BaseModel):
    hypotheses: list[_HypothesisDraft] = Field(default_factory=list)


_SYSTEM = """You are an Oracle diagnostic reasoner. Given the specialist findings
from a completed investigation, propose 2–4 COMPETING root-cause hypotheses —
distinct, plausibly mutually-exclusive explanations of the scoped problem. For
each, give a prior probability (0–1, your confidence before cross-checking) and
list the finding_ids that support or contradict it. Reference findings ONLY by
the exact finding_id shown in brackets. Return JSON."""


def render_blackboard_with_ids(state: CaseState) -> str:
    """Digest that exposes each finding's finding_id so hypotheses can cite them."""
    turn_findings: list[TurnFindings] = list(state.get("turn_findings") or [])
    lines: list[str] = []
    for tf in sorted(turn_findings, key=lambda t: (t.turn, t.agent)):
        if tf.abstained:
            continue
        for f in tf.findings:
            refs = f", refs={f.evidence_refs}" if f.evidence_refs else ""
            lines.append(
                f"[{f.finding_id}] ({f.agent}, turn {f.turn}, {f.severity}) "
                f"{f.description}{refs}"
            )
    return "\n".join(lines)


async def synthesize_hypotheses(state: CaseState, llm: Any) -> list[Hypothesis]:
    """Return competing hypotheses with stable ids, or ``[]`` (fail-open)."""
    digest = render_blackboard_with_ids(state)
    if not digest.strip():
        return []
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=(
            f"PROBLEM: {state.get('problem_statement', '')}\n\n"
            f"FINDINGS:\n{digest}"
        )),
    ]
    try:
        parsed = await parse_structured(llm, _HypothesisList, messages)
    except StructuredOutputError as exc:
        logger.warning("hypothesis synth: unparseable output — %r", exc.raw_text)
        return []
    except Exception:
        logger.exception("hypothesis synth failed — no hypotheses")
        return []

    out: list[Hypothesis] = []
    for i, draft in enumerate(parsed.hypotheses, start=1):
        out.append(Hypothesis(
            hyp_id=f"H-{i}",
            statement=draft.statement,
            prior=draft.prior,
            posterior=draft.prior,  # seeded equal to prior
            supporting_findings=list(draft.supporting_findings),
            contradicting_findings=list(draft.contradicting_findings),
            status="open",
        ))
    return out
