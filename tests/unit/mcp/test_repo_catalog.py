from datetime import datetime

import pytest

from observa.mcp.repo_catalog import DatabaseInfo, list_databases, list_snapshots


def _runner(responses: list[dict]):
    """Fake QueryRunner returning canned guarded_query-shaped results in order."""
    calls: list[str] = []

    async def run(sql: str) -> dict:
        calls.append(sql)
        return responses[len(calls) - 1]

    run.calls = calls  # type: ignore[attr-defined]
    return run


@pytest.mark.asyncio
async def test_list_databases_merges_instance_and_coverage():
    run = _runner([
        {"rows": [
            {"DBID": 111, "DB_NAME": "PRODFIN", "VERSION": "19.21.0.0.0", "INSTANCE_COUNT": 2},
            {"DBID": 222, "DB_NAME": "DWCORP", "VERSION": "21.3.0.0.0", "INSTANCE_COUNT": 1},
        ], "truncated": False, "row_count": 2},
        {"rows": [
            {"DBID": 111, "MIN_T": "2026-06-30T00:00:00.000000", "MAX_T": "2026-07-10T23:00:00.000000", "SNAP_COUNT": 240},
        ], "truncated": False, "row_count": 1},
    ])
    dbs = await list_databases(run)
    assert dbs == [
        DatabaseInfo(
            dbid=111, db_name="PRODFIN", version="19.21.0.0.0", instance_count=2,
            snap_min_time=datetime(2026, 6, 30, 0, 0),
            snap_max_time=datetime(2026, 7, 10, 23, 0),
            snap_count=240,
        ),
        DatabaseInfo(
            dbid=222, db_name="DWCORP", version="21.3.0.0.0", instance_count=1,
            snap_min_time=None, snap_max_time=None, snap_count=0,
        ),
    ]


@pytest.mark.asyncio
async def test_list_snapshots_parses_times_and_db_time():
    run = _runner([
        {"rows": [
            {"SNAP_ID": 100, "BEGIN_TIME": "2026-07-08T14:00:00.000000", "END_TIME": "2026-07-08 15:00:00"},
            {"SNAP_ID": 101, "BEGIN_TIME": "2026-07-08T15:00:00.000000", "END_TIME": "2026-07-08 16:00:00"},
        ], "truncated": False, "row_count": 2},
        {"rows": [{"SNAP_ID": 101, "DB_TIME_SECONDS": 42.5}], "truncated": False, "row_count": 1},
    ])
    snaps = await list_snapshots(run, dbid=111, days=15)
    assert [s.snap_id for s in snaps] == [100, 101]
    assert snaps[0].begin_time == datetime(2026, 7, 8, 14, 0)
    assert snaps[0].db_time_seconds is None
    assert snaps[1].db_time_seconds == 42.5


@pytest.mark.asyncio
async def test_list_snapshots_db_time_failure_degrades_to_marks():
    async def run(sql: str) -> dict:
        if "SYS_TIME_MODEL" in sql.upper():
            raise RuntimeError("ORA-00942")
        return {"rows": [
            {"SNAP_ID": 100, "BEGIN_TIME": "2026-07-08T14:00:00.000000", "END_TIME": "2026-07-08T15:00:00.000000"},
        ], "truncated": False, "row_count": 1}

    snaps = await list_snapshots(run, dbid=111, days=15)
    assert len(snaps) == 1 and snaps[0].db_time_seconds is None
