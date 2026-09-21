"""Transform a finished ``CaseState`` into a plain view model for the HTML
report template.

All report-shaping logic lives here so the Jinja2 template stays declarative:
findings are grouped by agent, severities are ordered, optional sections
(verification / chat) expose ``has_*`` flags, and human-friendly agent labels
are resolved from the agent ``REGISTRY``. Pydantic model instances are passed
through untouched where convenient — Jinja2's autoescape escapes their string
fields at render time regardless of origin.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from observa.agents import REGISTRY
from observa.graph.diff_probe import render_diff_digest
from observa.models import CaseState, TurnFindings

# Severity ranking + palette. Mirrors ``Severity`` in observa/models.py.
SEVERITY_ORDER: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}

SEVERITY_COLORS: dict[str, str] = {
    "critical": "#b03a2e",
    "high": "#ca6f1e",
    "medium": "#b7950b",
    "low": "#2471a3",
    "info": "#566573",
}


def agent_label(name: str) -> str:
    """Human-friendly agent name, reusing each agent class's ``display_name``."""
    cls = REGISTRY.get(name)
    if cls is not None:
        return cls.display_name
    return name.replace("_", " ").title()


def _severity_key(value: str) -> int:
    return SEVERITY_ORDER.get((value or "").lower(), 99)


def _confidence_band(confidence: float) -> str:
    """Coarse band used to color the headline confidence badge."""
    if confidence >= 0.7:
        return "high"
    if confidence >= 0.4:
        return "medium"
    return "low"


def _group_findings_by_agent(turn_findings: list[TurnFindings]) -> list[dict[str, Any]]:
    """Group per-turn findings into one block per agent, ordered by first
    appearance, with each agent's turns sorted ascending."""
    order: list[str] = []
    by_agent: dict[str, list[TurnFindings]] = {}
    for tf in turn_findings:
        if tf.agent not in by_agent:
            by_agent[tf.agent] = []
            order.append(tf.agent)
        by_agent[tf.agent].append(tf)

    blocks: list[dict[str, Any]] = []
    for name in order:
        turns = sorted(by_agent[name], key=lambda tf: tf.turn)
        total_findings = sum(len(tf.findings) for tf in turns)
        all_abstained = all(tf.abstained for tf in turns)
        blocks.append(
            {
                "name": name,
                "label": agent_label(name),
                "anchor": f"agent-{name}",
                "turns": turns,
                "total_findings": total_findings,
                "all_abstained": all_abstained,
            }
        )
    return blocks


def _input_files_by_type(input_files: list[Any]) -> list[dict[str, Any]]:
    """Group analyst-supplied files by their ``type`` for the inputs section."""
    order: list[str] = []
    grouped: dict[str, list[Any]] = {}
    for f in input_files:
        ftype = getattr(f, "type", "other")
        if ftype not in grouped:
            grouped[ftype] = []
            order.append(ftype)
        grouped[ftype].append(f)
    return [{"type": t, "files": grouped[t]} for t in order]


def _sorted_turn_map(raw: dict | None) -> list[dict[str, Any]]:
    """Normalize ``agents_active_per_turn`` (keys may be int or str) into a
    turn-ordered list with resolved agent labels."""
    if not raw:
        return []
    rows: list[dict[str, Any]] = []
    for turn in sorted(raw, key=lambda k: int(k)):
        names = raw[turn] or []
        rows.append(
            {
                "turn": int(turn),
                "agents": [{"name": n, "label": agent_label(n)} for n in names],
            }
        )
    return rows


def _refinement_rows(raw: dict | None) -> list[dict[str, Any]]:
    """Flatten ``agents_refinement_log`` ({turn: {agent: reason}}) into rows."""
    if not raw:
        return []
    rows: list[dict[str, Any]] = []
    for turn in sorted(raw, key=lambda k: int(k)):
        dropped = raw[turn] or {}
        for name, reason in dropped.items():
            rows.append(
                {
                    "turn": int(turn),
                    "name": name,
                    "label": agent_label(name),
                    "reason": reason,
                }
            )
    return rows


def _skipped_rows(skipped: list | None, reasons: dict | None) -> list[dict[str, Any]]:
    reasons = reasons or {}
    rows: list[dict[str, Any]] = []
    for name in skipped or []:
        rows.append(
            {
                "name": name,
                "label": agent_label(name),
                "reason": reasons.get(name, ""),
            }
        )
    return rows


_EVIDENCE_SAMPLE_CAP = 20
_EVIDENCE_EXCERPT_CHARS = 1000


def _evidence_row(rec: Any, cited_ids: set[str]) -> dict[str, Any]:
    rows = rec.result.get("rows") if isinstance(rec.result, dict) else None
    sample_rows: list[dict] = list(rows[:_EVIDENCE_SAMPLE_CAP]) if isinstance(rows, list) else []
    columns = list(sample_rows[0].keys()) if sample_rows else []
    content = rec.result.get("content") if isinstance(rec.result, dict) else None
    excerpt = ""
    if isinstance(content, str) and content:
        excerpt = content[:_EVIDENCE_EXCERPT_CHARS]
        if len(content) > _EVIDENCE_EXCERPT_CHARS:
            excerpt += f"\n… ({len(content)} chars total)"
    window = ""
    if rec.dbid or rec.snap_range:
        window = f"dbid {rec.dbid or '?'}"
        if rec.snap_range:
            window += f" · snaps {rec.snap_range[0]}–{rec.snap_range[1]}"
    return {
        "id": rec.evidence_id,
        "requester": rec.requester,
        "label": agent_label(rec.requester),
        "turn": rec.turn,
        "kind": rec.kind,
        "window": window,
        "row_count": rec.row_count,
        "duration_ms": rec.duration_ms,
        "hits": rec.hits,
        "status": rec.status,
        "error": rec.error,
        "cited": rec.evidence_id in cited_ids,
        "sql": rec.sql,
        "path": rec.path,
        "offset": rec.offset,
        "sample_rows": sample_rows,
        "sample_truncated": bool(isinstance(rows, list) and len(rows) > _EVIDENCE_SAMPLE_CAP),
        "columns": columns,
        "content_excerpt": excerpt,
    }


def build_view_model(state: CaseState | dict, ledger: Any = None) -> dict[str, Any]:
    """Build the plain dict consumed by ``report.html.j2``."""
    state = dict(state or {})
    summary = state.get("final_summary")
    verification = state.get("verification")
    turn_findings: list[TurnFindings] = list(state.get("turn_findings") or [])
    agent_questions = list(state.get("agent_questions") or [])
    chat_log = list(state.get("chat_log") or [])
    hypotheses = list(state.get("hypotheses") or [])
    contradictions = list(state.get("contradictions") or [])
    contradiction_resolutions = list(state.get("contradiction_resolutions") or [])
    input_files = list(state.get("input_files") or [])
    known_facts = list(state.get("known_facts") or [])

    problem_window = state.get("problem_window")
    baseline_window = state.get("baseline_window")
    baseline_diff = state.get("baseline_diff")
    baseline_diff_text = ""
    if baseline_diff:
        baseline_diff_text = render_diff_digest(baseline_diff)

    ora_timeline = state.get("ora_timeline")
    ora_rows: list[dict[str, Any]] = []
    ora_coverage = ""
    ora_not_covered = False
    if ora_timeline:
        cov = ora_timeline["coverage"]
        ora_coverage = (
            f"Alert log covers {cov.get('begin_ts') or '?'} → {cov.get('end_ts') or '?'} · "
            f"{ora_timeline['counts']['incidents']} ORA lines"
        )
        ora_not_covered = not cov.get("covers_problem")
        for inc in ora_timeline["incidents"]:
            ora_rows.append({
                "ts": inc.get("ts") or "",
                "code": inc.get("code") or "(incident)",
                "text": inc.get("text") or "",
                "instance": inc.get("instance"),
                "incident_id": inc.get("incident_id"),
                "window": inc.get("window"),
                "badge_class": f"ora-badge-{inc.get('window')}",
                "source_label": inc.get("source_label") or "",
                "evidence_id": inc.get("evidence_id") or "",
            })

    confidence = float(summary.confidence) if summary is not None else None

    # Sort each top finding's severity is already curated by the master; we only
    # sort the per-agent drill-down for readability.
    agent_blocks = _group_findings_by_agent(turn_findings)

    meta = {
        "case_id": state.get("case_id") or "unnamed",
        "dbid": state.get("dbid"),
        "oracle_version": state.get("oracle_version"),
        "topology": state.get("topology"),
        "instance_numbers": state.get("instance_numbers") or [],
        "total_turns": state.get("total_turns"),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    cited_ids: set[str] = set()
    for tf in turn_findings:
        for f in tf.findings:
            cited_ids.update(f.evidence_refs)
    if summary is not None:
        for sf in summary.top_findings:
            cited_ids.update(sf.evidence_refs or [])

    evidence_records = [
        _evidence_row(rec, cited_ids)
        for rec in (ledger.records if ledger is not None else [])
    ]

    return {
        "meta": meta,
        "summary": summary,
        "confidence": confidence,
        "confidence_pct": f"{confidence:.0%}" if confidence is not None else None,
        "confidence_band": _confidence_band(confidence) if confidence is not None else None,
        # Inputs
        "problem_statement": state.get("problem_statement") or "",
        "investigation_scope": state.get("investigation_scope") or "",
        "known_facts": known_facts,
        "input_file_groups": _input_files_by_type(input_files),
        # Methodology
        "active_per_turn": _sorted_turn_map(state.get("agents_active_per_turn")),
        "skipped": _skipped_rows(
            state.get("agents_skipped"), state.get("agents_skip_reasons")
        ),
        "refinement": _refinement_rows(state.get("agents_refinement_log")),
        "problem_window": problem_window,
        "baseline_window": baseline_window,
        "baseline_diff_text": baseline_diff_text,
        "baseline_suspect": bool(state.get("baseline_suspect")),
        "ora_rows": ora_rows,
        "ora_coverage": ora_coverage,
        "ora_not_covered": ora_not_covered,
        # Gate on the timeline dict, not on ora_rows: a log that COVERS the
        # window but is clean (zero incidents) still shows its coverage line
        # and any not-covered warning — the "searched, found nothing" signal.
        "has_ora_timeline": bool(ora_timeline),
        # Findings drill-down
        "agent_blocks": agent_blocks,
        # Dialogue
        "agent_questions": agent_questions,
        "chat_log": chat_log,
        # Hypotheses & verification
        "hypotheses": hypotheses,
        "contradictions": contradictions,
        "contradiction_resolutions": contradiction_resolutions,
        "verification": verification,
        # Section flags — template renders a section only when it has content.
        "has_summary": summary is not None,
        "has_inputs": bool(
            state.get("problem_statement")
            or state.get("investigation_scope")
            or known_facts
            or input_files
        ),
        "has_methodology": bool(
            state.get("agents_active_per_turn")
            or state.get("agents_skipped")
            or state.get("agents_refinement_log")
            or baseline_diff
            or ora_timeline
        ),
        "has_agents": bool(agent_blocks),
        "has_dialogue": bool(agent_questions or chat_log),
        "has_verification": bool(
            hypotheses or contradiction_resolutions
            or (verification is not None and verification.verifications)
        ),
        # Evidence ledger appendix
        "evidence_records": evidence_records,
        "has_evidence": bool(evidence_records),
    }
