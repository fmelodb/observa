"""Deterministic problem-vs-baseline diff probe — runs once before turn 1.

Modeled on ``topology_probe``: fixed SQL, QueryRunner injection, fail-open.
Cumulative-counter deltas use value(end_snap) − value(begin_snap); an instance
restart — or a RAC node joining/leaving between the two snaps — makes that
section unreliable (a node present only at end_snap adds its full cumulative
counter). Acceptable for a pre-turn digest (agents verify anything
load-bearing), noted here for readers.

DBA_HIST_SQLSTAT is the exception: it already ships per-snap ``*_delta``
columns, so we sum those over ``begin_snap < snap_id <= end_snap``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from observa.models import SnapWindow

logger = logging.getLogger(__name__)

QueryRunner = Callable[[str], Awaitable[dict]]

# Baseline sanity: problem/baseline rate ratio outside [1/5, 5] on either
# DB time/s or executions/s ⇒ the baseline probably isn't comparable.
_SUSPECT_RATIO = 5.0

_TIME_MODEL_STATS = ("DB time", "DB CPU")
_SYSSTAT_STATS = (
    "execute count", "user calls", "user commits", "redo size",
    "session logical reads", "physical reads", "parse count (hard)",
)
_OSSTAT_STATS = ("BUSY_TIME", "IDLE_TIME", "NUM_CPUS", "LOAD")


@dataclass
class DiffProbeResult:
    digest: Optional[dict]
    suspect: bool = False
    errors: list[str] = field(default_factory=list)


def _in_list(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


def _delta_sql(view: str, stats: tuple[str, ...], dbid: int, w: SnapWindow) -> str:
    return (
        f"SELECT stat_name AS NAME, "
        f"SUM(CASE WHEN snap_id = {w.end_snap} THEN value "
        f"WHEN snap_id = {w.begin_snap} THEN -value END) AS DELTA, "
        f"MAX(CASE WHEN snap_id = {w.end_snap} THEN value END) AS END_VALUE "
        f"FROM {view} WHERE dbid = {int(dbid)} "
        f"AND snap_id IN ({w.begin_snap}, {w.end_snap}) "
        f"AND stat_name IN ({_in_list(stats)}) GROUP BY stat_name"
    )


def _waits_sql(dbid: int, w: SnapWindow) -> str:
    return (
        f"SELECT event_name AS EVENT, MAX(wait_class) AS WAIT_CLASS, "
        f"SUM(CASE WHEN snap_id = {w.end_snap} THEN time_waited_micro "
        f"WHEN snap_id = {w.begin_snap} THEN -time_waited_micro END) AS DELTA_MICRO "
        f"FROM DBA_HIST_SYSTEM_EVENT WHERE dbid = {int(dbid)} "
        f"AND snap_id IN ({w.begin_snap}, {w.end_snap}) AND wait_class <> 'Idle' "
        f"GROUP BY event_name ORDER BY DELTA_MICRO DESC NULLS LAST "
        f"FETCH FIRST 10 ROWS ONLY"
    )


def _sql_sql(dbid: int, w: SnapWindow) -> str:
    return (
        f"SELECT sql_id AS SQL_ID, SUM(elapsed_time_delta) / 1000000 AS ELAPSED_S, "
        f"SUM(executions_delta) AS EXECS "
        f"FROM DBA_HIST_SQLSTAT WHERE dbid = {int(dbid)} "
        f"AND snap_id > {w.begin_snap} AND snap_id <= {w.end_snap} "
        f"GROUP BY sql_id ORDER BY ELAPSED_S DESC NULLS LAST FETCH FIRST 10 ROWS ONLY"
    )


def _val(row: dict, *keys: str, default=None):
    for k in keys:
        if k in row:
            return row[k]
        if k.lower() in row:
            return row[k.lower()]
    return default


def _num(value) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _window_label(w: SnapWindow) -> str:
    times = ""
    if w.begin_time is not None and w.end_time is not None:
        times = f" · {w.begin_time.isoformat(timespec='minutes')} → {w.end_time.isoformat(timespec='minutes')}"
    return f"snaps {w.begin_snap}→{w.end_snap}{times}"


async def _stat_deltas(
    run_query: QueryRunner, view: str, stats: tuple[str, ...],
    dbid: int, w: SnapWindow, errors: list[str], section: str,
    evidence_ids: list[str],
) -> dict[str, dict]:
    try:
        res = await run_query(_delta_sql(view, stats, dbid, w))
    except Exception as exc:  # noqa: BLE001 — section isolation
        errors.append(f"{section}: {exc}")
        return {}
    eid = res.get("evidence_id")
    if eid:
        evidence_ids.append(str(eid))
    out: dict[str, dict] = {}
    for row in res.get("rows") or []:
        name = str(_val(row, "NAME", default="") or "")
        out[name] = {"delta": _num(_val(row, "DELTA")), "end_value": _num(_val(row, "END_VALUE"))}
    return out


async def _top_waits(
    run_query: QueryRunner, dbid: int, w: SnapWindow, errors: list[str], section: str,
    evidence_ids: list[str],
) -> list[dict]:
    try:
        res = await run_query(_waits_sql(dbid, w))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{section} (DBA_HIST_SYSTEM_EVENT): {exc}")
        return []
    eid = res.get("evidence_id")
    if eid:
        evidence_ids.append(str(eid))
    out = []
    for row in res.get("rows") or []:
        micro = _num(_val(row, "DELTA_MICRO"))
        out.append({
            "event": str(_val(row, "EVENT", default="") or ""),
            "wait_class": str(_val(row, "WAIT_CLASS", default="") or ""),
            "seconds": round(micro / 1e6, 1) if micro is not None else None,
        })
    return out


async def _top_sql(
    run_query: QueryRunner, dbid: int, w: SnapWindow, errors: list[str], section: str,
    evidence_ids: list[str],
) -> list[dict]:
    try:
        res = await run_query(_sql_sql(dbid, w))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{section} (DBA_HIST_SQLSTAT): {exc}")
        return []
    eid = res.get("evidence_id")
    if eid:
        evidence_ids.append(str(eid))
    out = []
    for row in res.get("rows") or []:
        out.append({
            "sql_id": str(_val(row, "SQL_ID", default="") or ""),
            "elapsed_s": _num(_val(row, "ELAPSED_S")),
            "execs": _num(_val(row, "EXECS")),
        })
    return out


def _rate(delta: Optional[float], seconds: Optional[float], scale: float = 1.0) -> Optional[float]:
    if delta is None or not seconds:
        return None
    return (delta / scale) / seconds


async def run_diff_probe(
    dbid: int,
    problem: SnapWindow,
    baseline: Optional[SnapWindow],
    run_query: QueryRunner,
) -> DiffProbeResult:
    """Collect deltas for both windows and build the digest. Fail-open."""
    if not dbid or dbid <= 0:
        return DiffProbeResult(digest=None, errors=["dbid not set"])

    errors: list[str] = []
    evidence_ids: list[str] = []
    windows: list[tuple[str, SnapWindow]] = [("problem", problem)]
    if baseline is not None:
        windows.append(("baseline", baseline))

    tm: dict[str, dict[str, dict]] = {}
    sysstat: dict[str, dict[str, dict]] = {}
    osstat: dict[str, dict[str, dict]] = {}
    waits: dict[str, list[dict]] = {}
    topsql: dict[str, list[dict]] = {}

    for label, w in windows:
        tm[label] = await _stat_deltas(
            run_query, "DBA_HIST_SYS_TIME_MODEL", _TIME_MODEL_STATS, dbid, w, errors, f"time_model/{label}",
            evidence_ids)
        sysstat[label] = await _stat_deltas(
            run_query, "DBA_HIST_SYSSTAT", _SYSSTAT_STATS, dbid, w, errors, f"sysstat/{label}",
            evidence_ids)
        osstat[label] = await _stat_deltas(
            run_query, "DBA_HIST_OSSTAT", _OSSTAT_STATS, dbid, w, errors, f"osstat/{label}",
            evidence_ids)
        waits[label] = await _top_waits(run_query, dbid, w, errors, f"top_waits/{label}", evidence_ids)
        topsql[label] = await _top_sql(run_query, dbid, w, errors, f"top_sql/{label}", evidence_ids)

    got_anything = any(
        tm.get(l) or sysstat.get(l) or waits.get(l) or topsql.get(l) or osstat.get(l)
        for l, _ in windows
    )
    if not got_anything:
        logger.warning("diff_probe: all sections empty/failed — skipping digest. errors=%s", errors)
        return DiffProbeResult(digest=None, errors=errors)

    p_secs = problem.duration_seconds()
    b_secs = baseline.duration_seconds() if baseline is not None else None

    def _lp_row(metric: str, source: dict[str, dict[str, dict]], stat: str, scale: float = 1.0) -> dict:
        p = _rate((source.get("problem", {}).get(stat) or {}).get("delta"), p_secs, scale)
        b = _rate((source.get("baseline", {}).get(stat) or {}).get("delta"), b_secs, scale)
        ratio = (p / b) if (p is not None and b not in (None, 0)) else None
        return {"metric": metric, "problem": p, "baseline": b, "ratio": ratio}

    load_profile = [
        _lp_row("DB time (s/s)", tm, "DB time", 1e6),
        _lp_row("DB CPU (s/s)", tm, "DB CPU", 1e6),
        _lp_row("executions/s", sysstat, "execute count"),
        _lp_row("user calls/s", sysstat, "user calls"),
        _lp_row("commits/s", sysstat, "user commits"),
        _lp_row("redo bytes/s", sysstat, "redo size"),
        _lp_row("logical reads/s", sysstat, "session logical reads"),
        _lp_row("physical reads/s", sysstat, "physical reads"),
        _lp_row("hard parses/s", sysstat, "parse count (hard)"),
    ]

    os_rows: list[dict] = []
    for stat in _OSSTAT_STATS:
        p_row = osstat.get("problem", {}).get(stat) or {}
        b_row = osstat.get("baseline", {}).get(stat) or {}
        os_rows.append({
            "metric": stat,
            "problem_delta": p_row.get("delta"), "problem_end": p_row.get("end_value"),
            "baseline_delta": b_row.get("delta"), "baseline_end": b_row.get("end_value"),
        })

    suspect = False
    if baseline is not None:
        for key in ("DB time (s/s)", "executions/s"):
            row = next((r for r in load_profile if r["metric"] == key), None)
            if row is None:
                continue
            ratio = row["ratio"]
            if ratio is not None and (ratio > _SUSPECT_RATIO or ratio < 1.0 / _SUSPECT_RATIO):
                suspect = True
            elif row["problem"] and row["baseline"] == 0:
                # Idle baseline: infinite ratio — the most incomparable case.
                # (baseline None = missing data, deliberately NOT flagged.)
                suspect = True

    digest = {
        "has_baseline": baseline is not None,
        "suspect": suspect,
        "problem_label": _window_label(problem),
        "baseline_label": _window_label(baseline) if baseline is not None else None,
        "load_profile": load_profile,
        "top_waits_problem": waits.get("problem", []),
        "top_waits_baseline": waits.get("baseline", []),
        "top_sql_problem": topsql.get("problem", []),
        "top_sql_baseline": topsql.get("baseline", []),
        "os": os_rows,
        "errors": errors,
        "evidence_ids": evidence_ids,
    }
    return DiffProbeResult(digest=digest, suspect=suspect, errors=errors)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def render_diff_digest(digest: Optional[dict]) -> str:
    """Compact text block for the agents' static prompt zone."""
    if not digest:
        return "(diff probe unavailable — no precomputed deltas; discover windows via DBA_HIST_SNAPSHOT)"
    lines: list[str] = []
    lines.append(f"PROBLEM window: {digest['problem_label']}")
    if digest["has_baseline"]:
        lines.append(f"BASELINE window: {digest['baseline_label']}")
        if digest.get("suspect"):
            lines.append(
                "CAUTION: baseline load profile differs >5x from the problem window — "
                "deltas may not be comparable; verify before leaning on them."
            )
    else:
        lines.append("BASELINE: none (dismissed) — ABSOLUTE PROFILE of the problem window only.")
    if digest.get("evidence_ids"):
        lines.append(
            "Probe evidence IDs (citable in evidence_refs): "
            + ", ".join(digest["evidence_ids"])
        )
    lines.append("")
    lines.append("LOAD PROFILE (per second)      problem      baseline     ratio")
    for r in digest["load_profile"]:
        lines.append(
            f"  {r['metric']:<28} {_fmt(r['problem']):>10}  {_fmt(r['baseline']):>10}  {_fmt(r['ratio']):>6}"
        )

    def _os_line(rows: list[dict]) -> None:
        by = {r["metric"]: r for r in rows}
        busy = by.get("BUSY_TIME") or {}
        idle = by.get("IDLE_TIME") or {}

        def _busy_pct(which: str) -> Optional[float]:
            b = busy.get(f"{which}_delta")
            i = idle.get(f"{which}_delta")
            if b is None or i is None or (b + i) <= 0:
                return None
            # BUSY_TIME/IDLE_TIME are both centiseconds — the ratio is unit-free.
            return 100.0 * b / (b + i)

        p_pct, b_pct = _busy_pct("problem"), _busy_pct("baseline")
        ncpus = (by.get("NUM_CPUS") or {}).get("problem_end")
        load = (by.get("LOAD") or {}).get("problem_end")
        if p_pct is None and b_pct is None and ncpus is None and load is None:
            return
        lines.append("")
        lines.append(
            f"HOST CPU: busy% problem={_fmt(p_pct)} baseline={_fmt(b_pct)} · "
            f"cpus={_fmt(ncpus)} · load(end)={_fmt(load)}"
        )

    _os_line(digest.get("os") or [])

    def _wait_block(title: str, rows: list[dict]) -> None:
        if not rows:
            return
        lines.append("")
        lines.append(title)
        for w in rows[:10]:
            lines.append(f"  {_fmt(w['seconds']):>10}s  [{w['wait_class']}] {w['event']}")

    _wait_block("TOP WAITS — problem window (delta seconds):", digest["top_waits_problem"])
    _wait_block("TOP WAITS — baseline window (delta seconds):", digest["top_waits_baseline"])

    def _sql_block(title: str, rows: list[dict]) -> None:
        if not rows:
            return
        lines.append("")
        lines.append(title)
        for s in rows[:10]:
            lines.append(f"  {s['sql_id']:<16} elapsed={_fmt(s['elapsed_s'])}s execs={_fmt(s['execs'])}")

    _sql_block("TOP SQL by elapsed — problem window:", digest["top_sql_problem"])
    _sql_block("TOP SQL by elapsed — baseline window:", digest["top_sql_baseline"])
    if digest.get("errors"):
        lines.append("")
        lines.append(f"(sections unavailable: {len(digest['errors'])} — {'; '.join(digest['errors'][:3])})")
    return "\n".join(lines)
