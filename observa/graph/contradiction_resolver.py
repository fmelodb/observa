"""Resolve mutually-exclusive hypotheses on the REAL cited evidence.

Replaces the retired LLM advocacy debate. For each contradiction, gathers the
actual ledger rows behind each hypothesis's supporting findings and asks the
LLM which side the data favors. Per-pair isolated + fail-open.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from observa.graph.evidence_replay import gather_finding_evidence
from observa.llm.structured import parse_structured
from observa.models import CaseState, ContradictionResolution, Finding, Hypothesis, TurnFindings

logger = logging.getLogger(__name__)


class _Verdict(BaseModel):
    favored_hyp_id: str | None = None
    inconclusive: bool = False
    reasoning: str = ""
    posterior_a: float = Field(ge=0, le=1, default=0.0)
    posterior_b: float = Field(ge=0, le=1, default=0.0)


_SYSTEM = """You are adjudicating two mutually-exclusive Oracle root-cause
hypotheses using ONLY the real evidence rows shown. Decide which hypothesis the
evidence favors, or mark it inconclusive if the data does not distinguish them.
Assign updated posterior probabilities (0–1). Do NOT invent data beyond the rows
shown. Return JSON."""


def _findings_by_id(state: CaseState) -> dict[str, Finding]:
    out: dict[str, Finding] = {}
    for tf in (state.get("turn_findings") or []):
        assert isinstance(tf, TurnFindings)
        for f in tf.findings:
            out[f.finding_id] = f
    return out


def _evidence_for_hyp(hyp: Hypothesis, by_id: dict[str, Finding], ledger: Any) -> str:
    blocks: list[str] = []
    for fid in hyp.supporting_findings:
        f = by_id.get(fid)
        if f is None:
            continue
        block, _, _ = gather_finding_evidence(f, ledger)
        if block:
            blocks.append(block)
    return "\n\n".join(blocks) or "(no cited evidence)"


async def resolve_contradictions(
    state: CaseState, ledger: Any, llm: Any
) -> list[ContradictionResolution]:
    contradictions = list(state.get("contradictions") or [])
    if not contradictions:
        return []
    hyp_map = {h.hyp_id: h for h in (state.get("hypotheses") or [])}
    by_id = _findings_by_id(state)

    out: list[ContradictionResolution] = []
    for pair in contradictions:
        hyp_a_id = pair.get("hyp_a")
        hyp_b_id = pair.get("hyp_b")
        hyp_a = hyp_map.get(hyp_a_id) if isinstance(hyp_a_id, str) else None
        hyp_b = hyp_map.get(hyp_b_id) if isinstance(hyp_b_id, str) else None
        if hyp_a is None or hyp_b is None:
            logger.warning("resolver: unknown hyp id in %s — skipping", pair)
            continue
        try:
            ev_a = _evidence_for_hyp(hyp_a, by_id, ledger)
            ev_b = _evidence_for_hyp(hyp_b, by_id, ledger)
            messages = [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=(
                    f"Hypothesis A (id={hyp_a.hyp_id}): {hyp_a.statement}\n"
                    f"Evidence for A:\n{ev_a}\n\n"
                    f"Hypothesis B (id={hyp_b.hyp_id}): {hyp_b.statement}\n"
                    f"Evidence for B:\n{ev_b}\n\n"
                    "Which hypothesis does the evidence favor?"
                )),
            ]
            v = await parse_structured(llm, _Verdict, messages)
            favored_hyp_id = v.favored_hyp_id
            if favored_hyp_id not in (hyp_a.hyp_id, hyp_b.hyp_id):
                favored_hyp_id = None
            out.append(ContradictionResolution(
                hyp_a=hyp_a.hyp_id, hyp_b=hyp_b.hyp_id,
                favored_hyp_id=favored_hyp_id, inconclusive=v.inconclusive,
                reasoning=v.reasoning,
                posterior_a=v.posterior_a, posterior_b=v.posterior_b,
            ))
        except Exception as exc:  # noqa: BLE001 — per-pair isolation
            logger.warning("resolver: pair %s/%s failed: %s",
                           hyp_a.hyp_id, hyp_b.hyp_id, exc)
            out.append(ContradictionResolution(
                hyp_a=hyp_a.hyp_id, hyp_b=hyp_b.hyp_id,
                favored_hyp_id=None, inconclusive=True,
                reasoning=f"resolution error: {type(exc).__name__}",
                posterior_a=hyp_a.posterior, posterior_b=hyp_b.posterior,
            ))
    return out
