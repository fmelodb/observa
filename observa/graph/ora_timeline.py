"""Deterministic ORA-incident timeline — pure labeling + digest.

Mirror of ``diff_probe.py``: the I/O-free assembly and the text digest live
here; the thin graph-node wrapper lives in ``master.py``. ``build_ora_timeline``
takes an injected async ``scan`` callable (the client's ``scan_alertlog``, or a
fake in tests) so it never touches the filesystem itself.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Awaitable, Callable, Optional

from observa.models import InputFile, SnapWindow

logger = logging.getLogger(__name__)

ScanCallable = Callable[[str], Awaitable[dict]]


def _parse(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).replace(tzinfo=None)
    except ValueError:
        return None


def _in_window(ts: Optional[datetime], w: Optional[SnapWindow]) -> bool:
    if ts is None or w is None or w.begin_time is None or w.end_time is None:
        return False
    b = w.begin_time.replace(tzinfo=None)
    e = w.end_time.replace(tzinfo=None)
    return b <= ts <= e


def _label(ts: Optional[datetime], problem: Optional[SnapWindow],
           baseline: Optional[SnapWindow]) -> str:
    if _in_window(ts, problem):
        return "problem"
    if _in_window(ts, baseline):
        return "baseline"
    return "outside"


def _covers(window: Optional[SnapWindow], begin: Optional[datetime],
            end: Optional[datetime]) -> bool:
    if window is None or window.begin_time is None or window.end_time is None:
        return False
    if begin is None or end is None:
        return False
    return begin <= window.begin_time.replace(tzinfo=None) and \
        window.end_time.replace(tzinfo=None) <= end


async def build_ora_timeline(
    alert_logs: list[InputFile],
    problem_window: Optional[SnapWindow],
    baseline_window: Optional[SnapWindow],
    scan: ScanCallable,
) -> Optional[dict]:
    """Scan each alert log, label incidents by window, merge. None if empty."""
    all_incidents: list[dict] = []
    begins: list[datetime] = []
    ends: list[datetime] = []
    for f in alert_logs:
        try:
            res = await scan(f.path)
        except Exception as exc:  # noqa: BLE001 — one bad log doesn't sink the rest
            logger.warning("ora_timeline: scan failed for %s: %s", f.path, exc)
            continue
        evidence_id = res.get("evidence_id")
        cov = res.get("coverage") or {}
        b, e = _parse(cov.get("begin_ts")), _parse(cov.get("end_ts"))
        if b:
            begins.append(b)
        if e:
            ends.append(e)
        for inc in res.get("incidents") or []:
            ts = _parse(inc.get("ts"))
            all_incidents.append({
                **inc,
                "window": _label(ts, problem_window, baseline_window),
                "source_label": f.label or f.path,
                "evidence_id": evidence_id,
            })

    cov_begin = min(begins) if begins else None
    cov_end = max(ends) if ends else None
    # Keep the timeline if EITHER incidents OR any coverage span exists;
    # drop only when both are empty (no alert-log signal at all).
    if not all_incidents and cov_begin is None:
        return None

    counts = {"problem": 0, "baseline": 0, "outside": 0, "incidents": len(all_incidents)}
    for inc in all_incidents:
        counts[inc["window"]] += 1

    return {
        "coverage": {
            "begin_ts": cov_begin.isoformat() if cov_begin else None,
            "end_ts": cov_end.isoformat() if cov_end else None,
            "covers_problem": _covers(problem_window, cov_begin, cov_end),
            "covers_baseline": _covers(baseline_window, cov_begin, cov_end),
        },
        "counts": counts,
        "incidents": all_incidents,
    }


def _hhmmss(ts: Optional[str]) -> str:
    d = _parse(ts)
    return d.strftime("%H:%M:%S") if d else "  ??:??"


def render_ora_timeline_digest(ora_timeline: Optional[dict]) -> str:
    """Text block for the agents' cacheable static prompt zone."""
    if not ora_timeline:
        return ""
    cov = ora_timeline["coverage"]
    counts = ora_timeline["counts"]
    lines: list[str] = []
    lines.append(
        f"ALERT LOG COVERAGE: {cov.get('begin_ts') or '?'} -> {cov.get('end_ts') or '?'}  "
        f"({counts['incidents']} ORA lines)"
    )
    if not cov.get("covers_problem"):
        lines.append(
            "  WARNING: the alert log does not cover the problem window — "
            "its timestamps fall outside the problem snap range; "
            "treat window labels below with care."
        )
    for group, header in (("problem", "PROBLEM"), ("baseline", "BASELINE"),
                          ("outside", "OUTSIDE (context)")):
        rows = [i for i in ora_timeline["incidents"] if i["window"] == group]
        lines.append("")
        lines.append(f"-- {header} ({len(rows)}) --")
        if not rows:
            lines.append("  (none in this window)")
            continue
        for i in rows[:30]:
            inst = f" inst {i['instance']}" if i.get("instance") else ""
            inc = f" incident={i['incident_id']}" if i.get("incident_id") else ""
            code = i.get("code") or "(incident)"
            lines.append(
                f"  {_hhmmss(i.get('ts'))}  {code}{inst}{inc}  "
                f"@byte {i['byte_offset']}  [{i.get('evidence_id') or '?'}]"
            )
        if len(rows) > 30:
            lines.append(f"  ... +{len(rows) - 30} more")
    return "\n".join(lines)
