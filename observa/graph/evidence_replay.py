"""Pure evidence-materialization helpers for the verification tail.

Zero I/O: operates on EvidenceRecord objects handed in. "Materialize" means
re-render the record's ALREADY-CACHED result (rows / matches / content) as a
compact text block — the mechanical substrate the resolver and verifier feed
to the LLM, so judgement is grounded in the real retrieved data rather than an
agent's self-summary. Never re-executes anything.
"""
from __future__ import annotations

from typing import Any

from observa.mcp.evidence import EvidenceRecord
from observa.models import Finding


def _clip(text: str, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def materialize_evidence(
    rec: EvidenceRecord, *, max_rows: int = 15, max_line_chars: int = 200
) -> str:
    """Render one ledger record's cached result as a compact text block."""
    eid = rec.evidence_id
    if rec.status in ("error", "rejected"):
        return f"{eid} · [{rec.status}] {rec.error or '(no detail)'}"

    result: dict = rec.result if isinstance(rec.result, dict) else {}

    if rec.kind in ("query_awr", "query_catalog"):
        header = f"{eid} · {_clip(rec.sql or '(sql?)', 240)}"
        rows = result.get("rows")
        if not isinstance(rows, list) or not rows:
            return f"{header}\n  (0 rows)"
        lines = [header]
        for row in rows[:max_rows]:
            if isinstance(row, dict):
                cells = ", ".join(f"{k}={v}" for k, v in row.items())
            else:
                cells = str(row)
            lines.append(f"  {_clip(cells, max_line_chars)}")
        extra = len(rows) - max_rows
        if extra > 0:
            lines.append(f"  (+{extra} more rows)")
        if rec.truncated or result.get("truncated"):
            lines.append("  (TRUNCATED at row cap — claim may rely on unseen rows)")
        return "\n".join(lines)

    if rec.kind == "search_file":
        header = f"{eid} · \"{rec.pattern}\" @ {rec.path}"
        matches = result.get("matches")
        if not isinstance(matches, list) or not matches:
            return f"{header}\n  (0 matches)"
        lines = [header]
        for m in matches[:max_rows]:
            ln = m.get("line_number") if isinstance(m, dict) else None
            txt = m.get("line") if isinstance(m, dict) else str(m)
            lines.append(f"  L{ln if ln is not None else '?'}: {_clip(txt or '', max_line_chars)}")
        extra = len(matches) - max_rows
        if extra > 0:
            lines.append(f"  (+{extra} more matches)")
        if rec.truncated or result.get("truncated"):
            lines.append("  (TRUNCATED at match cap)")
        return "\n".join(lines)

    if rec.kind == "read_file":
        header = f"{eid} · {rec.path} @byte {rec.offset}"
        content = result.get("content")
        if not isinstance(content, str) or not content:
            return f"{header}\n  (no content)"
        return f"{header}\n{_clip(content, max_rows * max_line_chars)}"

    return f"{eid} · (unrenderable kind {rec.kind})"


def gather_finding_evidence(
    finding: Finding, ledger: Any, *, max_rows: int = 15
) -> tuple[str, bool, list[str]]:
    """Materialize every cited record for a finding.

    Returns ``(rendered_block, any_truncated, missing_refs)``. An unknown ref
    emits an explicit ``[<ref>] NOT FOUND in ledger`` line rather than being
    dropped, and is also collected in ``missing_refs`` so callers can detect
    "all cited evidence missing" structurally instead of substring-matching
    the rendered text (real evidence content can itself contain "NOT FOUND").
    """
    blocks: list[str] = []
    any_truncated = False
    missing_refs: list[str] = []
    for ref in finding.evidence_refs:
        rec = ledger.get(ref) if ledger is not None else None
        if rec is None:
            blocks.append(f"[{ref}] NOT FOUND in ledger")
            missing_refs.append(ref)
            continue
        blocks.append(materialize_evidence(rec, max_rows=max_rows))
        if rec.truncated or (isinstance(rec.result, dict) and rec.result.get("truncated")):
            any_truncated = True
    return "\n\n".join(blocks), any_truncated, missing_refs
