"""Re-materialize each top finding's cited evidence and verify the claim.

Evolves the old CoVe node: instead of asking yes/no questions against the
agents' self-reported evidence text, it dereferences each top finding's
evidence_refs to the ledger's REAL cached rows and asks whether they support
the claim. Confidence only ever decreases. Fail-open.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, cast

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from observa.graph.evidence_replay import gather_finding_evidence
from observa.llm.structured import parse_structured
from observa.models import (
    CaseState,
    Finding,
    FindingVerification,
    FinalSummary,
    SummaryFinding,
    VerificationResult,
    VerificationVerdict,
)

logger = logging.getLogger(__name__)

_WEIGHT = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0, "contradicted": 0.0}


class _Verdict(BaseModel):
    verdict: str
    reasoning: str = ""


_SYSTEM = """You verify one Oracle root-cause finding against its REAL cited
evidence rows. Answer with a single verdict:
- "supported": the rows clearly back the claim.
- "partial": the rows are consistent but incomplete.
- "unsupported": the rows do not back the claim.
- "contradicted": the rows argue against the claim.
Judge ONLY from the rows shown; never assume unseen data. Return JSON."""


def _as_finding(sf: SummaryFinding, idx: int) -> Finding:
    """Adapt a SummaryFinding to a Finding so gather_finding_evidence applies."""
    return Finding(
        finding_id=f"top-{idx}", agent=sf.agent, turn=0, severity=sf.severity,
        code="TOP", description=sf.description, evidence_refs=list(sf.evidence_refs or []),
    )


def patch_confidence(summary: FinalSummary, factor: float) -> FinalSummary:
    return summary.model_copy(update={"confidence": summary.confidence * factor})


async def replay_verify(
    state: CaseState, ledger: Any, llm: Any, *, support_threshold: float = 0.7
) -> Optional[VerificationResult]:
    summary = state.get("final_summary")
    if summary is None or not summary.top_findings:
        return None

    verifications: list[FindingVerification] = []
    weight_sum = 0.0
    scored = 0
    unsupported: list[str] = []

    for idx, sf in enumerate(summary.top_findings, start=1):
        finding = _as_finding(sf, idx)
        refs = finding.evidence_refs
        # Mechanical short-circuit: nothing to replay → unsupported, no LLM.
        if not refs:
            verifications.append(FindingVerification(
                finding_id=finding.finding_id, claim=sf.description,
                verdict="unsupported", reasoning="No evidence refs to replay (uncited).",
            ))
            unsupported.append(sf.description)
            weight_sum += _WEIGHT["unsupported"]
            scored += 1
            continue

        block, any_trunc, missing_refs = gather_finding_evidence(finding, ledger)
        if len(missing_refs) == len(refs):
            verifications.append(FindingVerification(
                finding_id=finding.finding_id, claim=sf.description,
                verdict="unsupported",
                reasoning="All cited evidence missing from the ledger.",
                evidence_refs=refs,
            ))
            unsupported.append(sf.description)
            weight_sum += _WEIGHT["unsupported"]
            scored += 1
            continue

        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=(
                f"CLAIM: {sf.description}\n\nEVIDENCE:\n{block}"
                + ("\n\n(NOTE: some evidence was truncated at a row/match cap.)"
                   if any_trunc else "")
            )),
        ]
        try:
            v = await parse_structured(llm, _Verdict, messages)
        except Exception as exc:  # noqa: BLE001 — our failure ≠ unsupported
            logger.warning("replay_verify: finding %d LLM error: %s", idx, exc)
            continue  # excluded from scoring
        verdict = cast(VerificationVerdict, v.verdict if v.verdict in _WEIGHT else "partial")
        verifications.append(FindingVerification(
            finding_id=finding.finding_id, claim=sf.description,
            verdict=verdict, reasoning=v.reasoning, evidence_refs=refs,
            evidence_truncated=any_trunc,
        ))
        if verdict in ("unsupported", "contradicted"):
            unsupported.append(sf.description)
        weight_sum += _WEIGHT[verdict]
        scored += 1

    support_score = 1.0 if scored == 0 else weight_sum / scored
    factor = support_score if support_score < support_threshold else 1.0
    return VerificationResult(
        verifications=verifications,
        support_score=support_score,
        applied_confidence_factor=factor,
        unsupported=unsupported,
    )
