"""Repository catalog queries for the intake wizard.

Same QueryRunner-callable pattern as ``observa.graph.topology_probe`` so tests
inject fakes. These queries run BEFORE a case exists — they legitimately scan
all dbids, which is why guard case-predicate enforcement only activates after
``McpClient.set_case_context``. Timestamps arrive as strings because
``McpClient`` normalizes NLS to ``YYYY-MM-DD"T"HH24:MI:SS.FF`` (timestamps)
and ``YYYY-MM-DD HH24:MI:SS`` (dates).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable, Optional

from observa.snap_windows import SnapshotInfo

logger = logging.getLogger(__name__)

QueryRunner = Callable[[str], Awaitable[dict]]


@dataclass(frozen=True)
class DatabaseInfo:
    dbid: int
    db_name: str
    version: str
    instance_count: int
    snap_min_time: Optional[datetime]
    snap_max_time: Optional[datetime]
    snap_count: int


def _val(row: dict, *keys: str, default=None):
    for k in keys:
        if k in row:
            return row[k]
        if k.lower() in row:
            return row[k.lower()]
    return default


def _parse_ts(value) -> Optional[datetime]:
    """Parse the NLS-normalized string formats; pass datetimes through."""
    if value is None or isinstance(value, datetime):
        return value
    text = str(value).strip().replace(" ", "T", 1) if " " in str(value) else str(value).strip()
    # Trim sub-microsecond fractional digits (FF can emit 9).
    if "." in text:
        head, frac = text.split(".", 1)
        text = f"{head}.{frac[:6]}"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        logger.warning("repo_catalog: unparseable timestamp %r", value)
        return None


async def list_databases(run_query: QueryRunner) -> list[DatabaseInfo]:
    """Databases loaded in the repository, with snapshot coverage."""
    inst = await run_query(
        "SELECT dbid AS DBID, MAX(db_name) AS DB_NAME, MAX(version) AS VERSION, "
        "COUNT(DISTINCT instance_number) AS INSTANCE_COUNT "
        "FROM DBA_HIST_DATABASE_INSTANCE GROUP BY dbid ORDER BY dbid"
    )
    coverage: dict[int, dict] = {}
    try:
        cov = await run_query(
            "SELECT dbid AS DBID, MIN(begin_interval_time) AS MIN_T, "
            "MAX(end_interval_time) AS MAX_T, COUNT(DISTINCT snap_id) AS SNAP_COUNT "
            "FROM DBA_HIST_SNAPSHOT GROUP BY dbid"
        )
        for row in cov.get("rows") or []:
            coverage[int(_val(row, "DBID", default=0) or 0)] = row
    except Exception as exc:  # noqa: BLE001 — coverage is decorative
        logger.warning("repo_catalog: snapshot coverage query failed: %s", exc)

    out: list[DatabaseInfo] = []
    for row in inst.get("rows") or []:
        dbid = int(_val(row, "DBID", default=0) or 0)
        cov_row = coverage.get(dbid, {})
        out.append(
            DatabaseInfo(
                dbid=dbid,
                db_name=str(_val(row, "DB_NAME", default="") or ""),
                version=str(_val(row, "VERSION", default="") or ""),
                instance_count=int(_val(row, "INSTANCE_COUNT", default=0) or 0),
                snap_min_time=_parse_ts(_val(cov_row, "MIN_T")),
                snap_max_time=_parse_ts(_val(cov_row, "MAX_T")),
                snap_count=int(_val(cov_row, "SNAP_COUNT", default=0) or 0),
            )
        )
    return out


async def list_snapshots(run_query: QueryRunner, dbid: int, days: int) -> list[SnapshotInfo]:
    """Timeline snapshots for one dbid, newest ``days`` days, with DB-time deltas.

    RAC emits one DBA_HIST_SNAPSHOT row per instance per snap — collapsed via
    GROUP BY. The DB-time metric uses per-instance LAG deltas (negative deltas
    from restarts filtered out); its failure degrades to existence marks.

    The horizon anchors to the NEWEST snapshot of this dbid, not SYSDATE —
    imported repositories (awrload) hold historical windows, so a wall-clock
    horizon would show an empty timeline for exactly the data this tool
    exists to analyze.
    """
    snaps_res = await run_query(
        f"SELECT snap_id AS SNAP_ID, MIN(begin_interval_time) AS BEGIN_TIME, "
        f"MAX(end_interval_time) AS END_TIME "
        f"FROM DBA_HIST_SNAPSHOT "
        f"WHERE dbid = {int(dbid)} AND end_interval_time >= ("
        f"SELECT MAX(end_interval_time) - NUMTODSINTERVAL({int(days)}, 'DAY') "
        f"FROM DBA_HIST_SNAPSHOT WHERE dbid = {int(dbid)}) "
        f"GROUP BY snap_id ORDER BY snap_id"
    )
    rows = snaps_res.get("rows") or []
    if not rows:
        return []
    min_snap = min(int(_val(r, "SNAP_ID", default=0) or 0) for r in rows)

    db_time: dict[int, float] = {}
    try:
        dt_res = await run_query(
            f"SELECT snap_id AS SNAP_ID, SUM(delta) / 1000000 AS DB_TIME_SECONDS FROM ("
            f"SELECT snap_id, value - LAG(value) OVER ("
            f"PARTITION BY instance_number ORDER BY snap_id) AS delta "
            f"FROM DBA_HIST_SYS_TIME_MODEL "
            f"WHERE dbid = {int(dbid)} AND stat_name = 'DB time' AND snap_id >= {min_snap - 1}"
            f") WHERE delta > 0 GROUP BY snap_id"
        )
        for row in dt_res.get("rows") or []:
            sid = int(_val(row, "SNAP_ID", default=0) or 0)
            val = _val(row, "DB_TIME_SECONDS")
            if val is not None:
                db_time[sid] = float(val)
    except Exception as exc:  # noqa: BLE001 — timeline degrades to marks
        logger.warning("repo_catalog: DB-time timeline query failed: %s", exc)

    out: list[SnapshotInfo] = []
    for row in rows:
        sid = int(_val(row, "SNAP_ID", default=0) or 0)
        begin = _parse_ts(_val(row, "BEGIN_TIME"))
        end = _parse_ts(_val(row, "END_TIME"))
        if begin is None or end is None:
            continue
        out.append(
            SnapshotInfo(
                snap_id=sid, begin_time=begin, end_time=end,
                db_time_seconds=db_time.get(sid),
            )
        )
    return out
