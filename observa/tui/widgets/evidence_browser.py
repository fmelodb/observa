"""Pure row/detail formatters for the Evidence Browser (Ctrl+D panel).

No Textual imports — exercised directly in unit tests. The screen feeds
``EvidenceLedger.records`` in and mounts the results.
"""
from __future__ import annotations

from rich.text import Text

from observa.mcp.evidence import EvidenceRecord

_SAMPLE_ROW_CAP = 20
_CONTENT_EXCERPT_CHARS = 1000

_STATUS_COLOR = {"ok": "#A9DFBF", "error": "#E59866", "rejected": "#E59866"}


def _target(rec: EvidenceRecord) -> str:
    if rec.kind == "read_file":
        return f"{rec.path} @{rec.offset}"
    if rec.kind == "search_file":
        return f"{rec.pattern}  ·  {rec.path}"[:48]
    return " ".join(rec.sql.split())[:48]


def evidence_rows(
    records: list[EvidenceRecord], filter_text: str = ""
) -> list[tuple[str, str, str, str, str, str, str, str]]:
    """(id, requester, turn, kind, target, rows, ms, status) per record.

    ``filter_text`` is a case-insensitive substring match against id,
    requester and kind. Empty filter returns everything, ledger order.
    """
    needle = filter_text.strip().lower()
    out = []
    for rec in records:
        if needle and needle not in f"{rec.evidence_id} {rec.requester} {rec.kind}".lower():
            continue
        status = rec.status if rec.status != "ok" else (f"⚡{rec.hits}" if rec.hits else "ok")
        out.append((
            rec.evidence_id,
            rec.requester,
            "" if rec.turn is None else str(rec.turn),
            rec.kind,
            _target(rec),
            "" if rec.row_count is None else str(rec.row_count),
            str(rec.duration_ms),
            status,
        ))
    return out


def format_record_detail(rec: EvidenceRecord | None) -> Text:
    """Full SQL / path + window metadata + sample rows or content excerpt."""
    text = Text()
    if rec is None:
        text.append("(no record selected)", style="#888888")
        return text
    text.append(f"{rec.evidence_id}", style="bold #7FB3D5")
    text.append(f" · {rec.requester}", style="#d4d4d4")
    if rec.turn is not None:
        text.append(f" · turn {rec.turn}", style="#888888")
    text.append(f" · {rec.kind}", style="#888888")
    if rec.hits:
        text.append(f" · ⚡{rec.hits} cache hits", style="#F5B041")
    text.append("\n")
    if rec.dbid or rec.snap_range:
        window = f"dbid {rec.dbid or '?'}"
        if rec.snap_range:
            window += f" · snaps {rec.snap_range[0]}–{rec.snap_range[1]}"
        text.append(window + "\n", style="#888888")
    if rec.status != "ok":
        text.append(f"{rec.status.upper()}: {rec.error}\n", style=_STATUS_COLOR.get(rec.status, "#E59866"))
    if rec.sql:
        text.append(rec.sql.strip() + "\n", style="#A9DFBF")
    if rec.path:
        text.append(
            f"{rec.path} @{rec.offset} (max_bytes={rec.max_bytes})\n", style="#A9DFBF"
        )
    for w in rec.warnings:
        text.append(f"⚠ {w}\n", style="#F5B041")

    if rec.pattern:
        text.append(f"pattern: {rec.pattern}\n", style="#A9DFBF")
    matches = rec.result.get("matches")
    if isinstance(matches, list) and matches:
        text.append(f"{len(matches)} match(es):\n", style="#888888")
        for m in matches[:_SAMPLE_ROW_CAP]:
            text.append(f"  L{m.get('line_number')} @{m.get('byte_offset')}  "
                        f"{m.get('line', '')}\n", style="#d4d4d4")
    elif rec.match_count is not None:
        text.append(f"{rec.match_count} incident(s)\n", style="#888888")

    rows = rec.result.get("rows")
    if isinstance(rows, list) and rows:
        sample = rows[:_SAMPLE_ROW_CAP]
        if len(rows) > _SAMPLE_ROW_CAP:
            text.append(f"showing {len(sample)} of {len(rows)} rows\n", style="#888888")
        cols = list(sample[0].keys())
        text.append("  ".join(cols) + "\n", style="bold #d4d4d4")
        for row in sample:
            text.append("  ".join(str(row.get(c, "")) for c in cols) + "\n",
                        style="#d4d4d4")
    content = rec.result.get("content")
    if isinstance(content, str) and content:
        excerpt = content[:_CONTENT_EXCERPT_CHARS]
        text.append(excerpt, style="#d4d4d4")
        if len(content) > _CONTENT_EXCERPT_CHARS:
            text.append(f"\n… ({len(content)} chars total)", style="#888888")
    return text
