"""Master consolidation — reduces the full blackboard into a FinalSummary.

The master LLM reads every turn's findings (plus abstain notices and analyst
chat) and emits structured output matching ``FinalSummary``. It must keep the
top-findings list to ≤3 entries, scoped strictly to the analyst's
INVESTIGATION SCOPE, reusing the severity from the underlying findings.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from observa.llm.structured import StructuredOutputError, parse_structured
from observa.models import CaseState, FinalSummary, TurnFindings

logger = logging.getLogger(__name__)


_MASTER_SYSTEM_PROMPT = """You are the master Oracle diagnostic consolidator.

You have just received the complete blackboard from N turns of specialist
agents. Your job is to produce a FinalSummary with:

1. problem_restated: Re-state the original problem in your own words, grounded
   in what the analyst told you.
2. top_findings: AT MOST 3 findings — only the most relevant ones that
   directly explain the problem described in INVESTIGATION SCOPE. Pulled from
   the specialist outputs. Do NOT invent new findings — every entry must
   correspond to something a specialist reported. Preserve the specialist's
   severity. Ruthlessly discard findings that are real but tangential to the
   scope — they do not belong in the final summary.
3. root_cause: A single sentence or short paragraph naming the most likely
   root cause of the scoped problem, grounded in the selected findings. If
   evidence does not support a confident conclusion, say so explicitly and
   set confidence low.
4. confidence: 0.0-1.0. Be calibrated: confidence > 0.8 only when findings
   converge strongly. Use ≤ 0.5 when specialists disagree or when much of the
   blackboard abstained.
5. unknowns: Explicit list of information you wished you had but didn't.
6. recommended_next_steps: Short, concrete next actions that directly test
   the root cause hypothesis for the scoped problem. No shotgun suggestions.

Hard rules:
- SCOPE DISCIPLINE: INVESTIGATION SCOPE is the filter. A finding unrelated to
   the scoped problem — even if real and severe — MUST NOT appear in
   top_findings. Mention it in unknowns or skip it entirely.
- OBJECTIVITY: Do not speculate beyond what specialists reported.
- CONSISTENCY: If a specialist abstained in their area, do not claim findings
   for that area.
- STRICT CAP: top_findings length <= 3.
- EVIDENCE REFS: each blackboard finding lists evidence refs (IDs like Q-014 /
   F-003). For every top finding you select, copy the source finding's refs
   into `evidence_refs` verbatim. Never invent an ID. Findings marked
   "(uncited)" have no verified evidence — prefer cited findings and lower
   your confidence when you must rely on uncited ones.
"""


def _blackboard_digest(state: CaseState) -> str:
    turn_findings: list[TurnFindings] = list(state.get("turn_findings", []) or [])
    if not turn_findings:
        return "(blackboard is empty — no specialist ran)"
    lines: list[str] = []
    by_turn: dict[int, list[TurnFindings]] = {}
    for tf in turn_findings:
        by_turn.setdefault(tf.turn, []).append(tf)
    for turn in sorted(by_turn):
        lines.append(f"=== TURN {turn} ===")
        for tf in sorted(by_turn[turn], key=lambda t: t.agent):
            if tf.abstained:
                lines.append(
                    f"[{tf.agent}] ABSTAINED — {tf.abstain_reason or 'nothing relevant'}"
                )
                continue
            for f in tf.findings:
                lines.append(
                    f"[{tf.agent}] {f.severity.upper()} · {f.code}: {f.description}"
                )
                if f.evidence_text():
                    lines.append(f"    evidence: {f.evidence_text()}")
                if f.uncited:
                    lines.append("    (uncited — no valid evidence refs)")
                if f.related_objects:
                    lines.append(f"    objects: {', '.join(f.related_objects)}")
            if tf.notes:
                lines.append(f"[{tf.agent}] notes: {tf.notes}")
        lines.append("")
    return "\n".join(lines)


def _chat_digest(state: CaseState) -> str:
    chat = state.get("chat_log", []) or []
    if not chat:
        return "(no analyst chat during the turns)"
    return "\n".join(
        f"[{m.sender}{'/' + m.agent if m.agent else ''} · turn {m.turn}] {m.text}"
        for m in chat
    )


def _user_prompt(state: CaseState) -> str:
    return (
        f"PROBLEM STATEMENT:\n{state.get('problem_statement', '')}\n\n"
        f"INVESTIGATION SCOPE:\n{state.get('investigation_scope', '')}\n\n"
        "KNOWN FACTS:\n"
        + "\n".join(f"- {fact}" for fact in (state.get('known_facts') or [])) + "\n\n"
        f"DBID: {state.get('dbid', 'unknown')} · "
        f"Oracle: {state.get('oracle_version', 'unknown')} · "
        f"Topology: {state.get('topology', 'unknown')}\n\n"
        f"=== BLACKBOARD ===\n{_blackboard_digest(state)}\n\n"
        f"=== ANALYST CHAT ===\n{_chat_digest(state)}\n\n"
        "Produce the FinalSummary now."
    )


def _validate_summary_refs(summary: FinalSummary, ledger: Any) -> FinalSummary:
    """Drop refs the ledger does not know. No-op without a ledger."""
    if ledger is None:
        return summary
    known = ledger.known_ids()
    for f in summary.top_findings:
        dropped = [r for r in f.evidence_refs if r not in known]
        if dropped:
            logger.info("consolidator: dropped hallucinated summary refs %s", dropped)
        f.evidence_refs = [r for r in f.evidence_refs if r in known]
    return summary


async def consolidate(state: CaseState, llm: BaseChatModel, ledger: Any = None) -> FinalSummary:
    """Run the master LLM and return a validated FinalSummary."""
    messages = [
        SystemMessage(content=_MASTER_SYSTEM_PROMPT),
        HumanMessage(content=_user_prompt(state)),
    ]
    try:
        return _validate_summary_refs(await parse_structured(llm, FinalSummary, messages), ledger)
    except StructuredOutputError as exc:
        logger.warning(
            "master consolidation produced no parseable FinalSummary — "
            "raw_text=%r tool_calls=%r parsing_error=%r",
            exc.raw_text, exc.raw_tool_calls, exc.parsing_error,
        )
        return FinalSummary(
            problem_restated=state.get("problem_statement", ""),
            top_findings=[],
            root_cause="Master LLM did not return a parseable FinalSummary — see logs.",
            confidence=0.0,
            unknowns=[f"Raw model output (truncated): {exc.raw_text!r}"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("master consolidation failed")
        return FinalSummary(
            problem_restated=state.get("problem_statement", ""),
            top_findings=[],
            root_cause=f"Consolidation error: {type(exc).__name__}",
            confidence=0.0,
            unknowns=["Master LLM call failed — see logs."],
        )
