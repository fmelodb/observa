"""Pure streaming scan engine over registered diagnostic files.

Zero MCP I/O — exercised directly in unit tests. Two public functions share one
block-reading line iterator so neither loads a whole (GB-sized) file into memory:

* ``scan_matches`` — generic regex grep returning line + absolute byte offset,
  bounded by hard caps (match count, scan bytes) rather than a timeout.
* ``extract_ora_incidents`` — alert-log-specialized scan that ties every
  ``ORA-`` / ``incident=`` line to the most recent block timestamp.

``byte_offset`` is the absolute offset of the line start — directly usable as a
``read_file`` offset to fetch surrounding context.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Iterator, Optional

SEARCH_MAX_MATCHES = 200
SEARCH_MAX_SCAN_BYTES = 64 * 1024 * 1024   # 64 MB
LINE_CAP = 300
ORA_MAX_BYTES = 64 * 1024 * 1024

_READ_BLOCK = 256 * 1024  # 256 KB read granularity


def _iter_lines_with_offsets(
    fh, *, max_scan_bytes: int, progress: dict | None = None
) -> Iterator[tuple[int, int, str]]:
    """Yield ``(line_number, byte_offset, line_text)`` up to ``max_scan_bytes``.

    ``line_text`` has its trailing newline stripped and is decoded UTF-8 with
    replacement. ``byte_offset`` is the absolute offset of the line start. Reads
    in blocks; a line split across the scan-byte boundary is still yielded whole
    (we finish the current line before stopping).

    If ``progress`` is supplied, ``progress["consumed"]`` is kept updated to the
    true next-line byte offset (the actual position consumed from the file), so
    callers can compute truncation without re-encoding decoded text.
    """
    line_number = 0
    consumed = 0
    line_start = 0
    buf = bytearray()
    while consumed < max_scan_bytes:
        block = fh.read(_READ_BLOCK)
        if not block:
            break
        consumed += len(block)
        buf.extend(block)
        while True:
            # Precise byte-budget stop: a coarse block read may pull past the
            # budget, so gate line emission on the line-start offset.
            if line_start >= max_scan_bytes:
                if progress is not None:
                    progress["consumed"] = line_start
                return
            nl = buf.find(b"\n")
            if nl == -1:
                break
            raw = bytes(buf[:nl])
            del buf[: nl + 1]
            line_number += 1
            text = raw.decode("utf-8", errors="replace").rstrip("\r")
            yield line_number, line_start, text
            line_start += len(raw) + 1  # + the newline byte
            if progress is not None:
                progress["consumed"] = line_start
    # Trailing line with no final newline.
    if buf and line_start < max_scan_bytes:
        raw = bytes(buf)
        line_number += 1
        text = raw.decode("utf-8", errors="replace").rstrip("\r")
        yield line_number, line_start, text
        line_start += len(raw)
        if progress is not None:
            progress["consumed"] = line_start


def scan_matches(
    path: str,
    pattern: str,
    *,
    max_matches: int,
    max_scan_bytes: int,
    line_cap: int = LINE_CAP,
) -> dict:
    """Regex-grep ``path``; return line + byte-offset matches, bounded by caps."""
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid regex: {exc}") from exc

    total_size = os.path.getsize(path)
    matches: list[dict] = []
    progress: dict = {"consumed": 0}
    hit_cap = False
    with open(path, "rb") as fh:
        for line_number, byte_offset, text in _iter_lines_with_offsets(
            fh, max_scan_bytes=max_scan_bytes, progress=progress
        ):
            if rx.search(text):
                matches.append({
                    "line_number": line_number,
                    "byte_offset": byte_offset,
                    "line": text[:line_cap],
                })
                if len(matches) >= max_matches:
                    hit_cap = True
                    break
    consumed = progress["consumed"]
    truncated = hit_cap or (consumed < total_size)
    return {
        "matches": matches,
        "match_count": len(matches),
        "scanned_bytes": consumed,
        "truncated": truncated,
        "total_size": total_size,
    }


# --- alert-log timestamp parsing -------------------------------------------

_ISO_TS = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:\.\d+)?(?:[+-]\d{2}:?\d{2}|Z)?\s*$"
)
_11G_TS = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"(\d{1,2})\s+(\d{2}:\d{2}:\d{2})\s+(\d{4})\s*$"
)
_ORA = re.compile(r"ORA-(\d{2,5})")
_INCIDENT = re.compile(r"incident=(\d+)")
_INSTANCE = re.compile(r"\b[Ii]nst(?:ance)?\s+(\d+)")

_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _parse_ts(line: str) -> Optional[datetime]:
    """Return a naive datetime if ``line`` is a standalone alert-log timestamp."""
    m = _ISO_TS.match(line)
    if m:
        try:
            return datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}")
        except ValueError:
            return None
    m = _11G_TS.match(line)
    if m:
        month = _MONTHS[m.group(1)]
        day = int(m.group(2))
        hh, mm, ss = (int(x) for x in m.group(3).split(":"))
        year = int(m.group(4))
        try:
            return datetime(year, month, day, hh, mm, ss)
        except ValueError:
            return None
    return None


def extract_ora_incidents(path: str, *, max_bytes: int) -> dict:
    """Extract ORA-/incident lines from an alert log, each tied to its block ts."""
    incidents: list[dict] = []
    current_ts: Optional[datetime] = None
    begin_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None

    total_size = os.path.getsize(path)
    progress: dict = {"consumed": 0}
    with open(path, "rb") as fh:
        for _line_number, byte_offset, text in _iter_lines_with_offsets(
            fh, max_scan_bytes=max_bytes, progress=progress
        ):
            ts = _parse_ts(text)
            if ts is not None:
                current_ts = ts
                begin_ts = begin_ts or ts
                end_ts = ts
                continue
            ora_codes = _ORA.findall(text)          # list of digit-strings
            inc = _INCIDENT.search(text)
            if not ora_codes and not inc:
                continue
            instance = _INSTANCE.search(text)
            inst_val = int(instance.group(1)) if instance else None
            inc_val = int(inc.group(1)) if inc else None
            ts_val = current_ts.isoformat() if current_ts else None
            if ora_codes:
                for digits in ora_codes:
                    incidents.append({
                        "ts": ts_val,
                        "code": f"ORA-{int(digits):05d}",
                        "text": text[:LINE_CAP],
                        "instance": inst_val,
                        "incident_id": inc_val,
                        "byte_offset": byte_offset,
                    })
            else:
                incidents.append({
                    "ts": ts_val,
                    "code": "",
                    "text": text[:LINE_CAP],
                    "instance": inst_val,
                    "incident_id": inc_val,
                    "byte_offset": byte_offset,
                })
    consumed = progress["consumed"]
    return {
        "incidents": incidents,
        "scanned_bytes": consumed,
        "truncated": consumed < total_size,
        "coverage": {
            "begin_ts": begin_ts.isoformat() if begin_ts else None,
            "end_ts": end_ts.isoformat() if end_ts else None,
        },
    }
