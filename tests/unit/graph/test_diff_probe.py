from datetime import datetime

import pytest

from observa.graph.diff_probe import render_diff_digest, run_diff_probe
from observa.models import SnapWindow

PROBLEM = SnapWindow(
    begin_snap=100, end_snap=102,
    begin_time=datetime(2026, 7, 8, 14, 0), end_time=datetime(2026, 7, 8, 16, 0),
)
BASELINE = SnapWindow(
    begin_snap=50, end_snap=52,
    begin_time=datetime(2026, 7, 1, 14, 0), end_time=datetime(2026, 7, 1, 16, 0),
)


def _fake_runner(rows_by_marker: dict[str, list[dict]]):
    """Route by view name found in the SQL; return guarded_query-shaped dicts."""
    async def run(sql: str) -> dict:
        up = sql.upper()
        for marker, rows in rows_by_marker.items():
            if marker in up:
                return {"rows": rows, "truncated": False, "row_count": len(rows)}
        return {"rows": [], "truncated": False, "row_count": 0}
    return run


def _healthy_rows(db_time_delta_us: float) -> dict[str, list[dict]]:
    return {
        "SYS_TIME_MODEL": [
            {"NAME": "DB time", "DELTA": db_time_delta_us, "END_VALUE": None},
            {"NAME": "DB CPU", "DELTA": db_time_delta_us / 2, "END_VALUE": None},
        ],
        "SYSSTAT": [
            {"NAME": "execute count", "DELTA": 7200, "END_VALUE": None},
            {"NAME": "user commits", "DELTA": 720, "END_VALUE": None},
        ],
        "SYSTEM_EVENT": [
            {"EVENT": "db file sequential read", "WAIT_CLASS": "User I/O", "DELTA_MICRO": 9.9e8},
        ],
        "SQLSTAT": [
            {"SQL_ID": "abc123def4567", "ELAPSED_S": 340.5, "EXECS": 12},
        ],
        "OSSTAT": [
            {"NAME": "BUSY_TIME", "DELTA": 500000, "END_VALUE": None},
            {"NAME": "NUM_CPUS", "DELTA": 0, "END_VALUE": 16},
        ],
    }


@pytest.mark.asyncio
async def test_probe_builds_digest_with_both_windows():
    run = _fake_runner(_healthy_rows(7.2e9))  # same rows for both windows is fine
    result = await run_diff_probe(111, PROBLEM, BASELINE, run)
    assert result.digest is not None
    assert result.digest["has_baseline"] is True
    lp = {m["metric"]: m for m in result.digest["load_profile"]}
    # 7200s of DB time over a 7200s window → 1.0 s/s, ratio 1.0.
    assert lp["DB time (s/s)"]["problem"] == pytest.approx(1.0)
    assert lp["DB time (s/s)"]["ratio"] == pytest.approx(1.0)
    assert result.suspect is False
    assert result.digest["top_sql_problem"][0]["sql_id"] == "abc123def4567"


@pytest.mark.asyncio
async def test_probe_without_baseline_marks_absolute_profile():
    run = _fake_runner(_healthy_rows(7.2e9))
    result = await run_diff_probe(111, PROBLEM, None, run)
    assert result.digest is not None
    assert result.digest["has_baseline"] is False
    assert "ABSOLUTE PROFILE" in render_diff_digest(result.digest).upper()


@pytest.mark.asyncio
async def test_probe_flags_suspect_baseline_over_5x():
    calls = {"n": 0}

    async def run(sql: str) -> dict:
        up = sql.upper()
        if "SYS_TIME_MODEL" in up:
            calls["n"] += 1
            # First call = problem window (7200s), second = baseline (600s → 12x off).
            delta = 7.2e9 if calls["n"] == 1 else 6.0e8
            return {"rows": [{"NAME": "DB time", "DELTA": delta, "END_VALUE": None}], "truncated": False, "row_count": 1}
        return {"rows": [], "truncated": False, "row_count": 0}

    result = await run_diff_probe(111, PROBLEM, BASELINE, run)
    assert result.suspect is True


@pytest.mark.asyncio
async def test_probe_flags_suspect_on_idle_baseline():
    calls = {"n": 0}

    async def run(sql: str) -> dict:
        if "SYS_TIME_MODEL" in sql.upper():
            calls["n"] += 1
            delta = 7.2e9 if calls["n"] == 1 else 0
            return {"rows": [{"NAME": "DB time", "DELTA": delta, "END_VALUE": None}], "truncated": False, "row_count": 1}
        return {"rows": [], "truncated": False, "row_count": 0}

    result = await run_diff_probe(111, PROBLEM, BASELINE, run)
    assert result.suspect is True


@pytest.mark.asyncio
async def test_probe_section_failure_isolated_with_other_data():
    async def run(sql: str) -> dict:
        up = sql.upper()
        if "SYSTEM_EVENT" in up:
            raise RuntimeError("ORA-01555")
        if "SYS_TIME_MODEL" in up:
            return {"rows": [{"NAME": "DB time", "DELTA": 7.2e9, "END_VALUE": None}], "truncated": False, "row_count": 1}
        return {"rows": [], "truncated": False, "row_count": 0}

    result = await run_diff_probe(111, PROBLEM, BASELINE, run)
    assert result.digest is not None
    assert result.digest["top_waits_problem"] == []
    assert any("top_waits" in e for e in result.digest["errors"])


@pytest.mark.asyncio
async def test_probe_total_failure_returns_none_digest():
    async def run(sql: str) -> dict:
        raise RuntimeError("no connection")

    result = await run_diff_probe(111, PROBLEM, BASELINE, run)
    assert result.digest is None and result.suspect is False


def test_render_digest_is_compact_text():
    digest = {
        "has_baseline": True, "suspect": False, "errors": [],
        "problem_label": "snaps 100→102 · 2026-07-08T14:00 → 16:00",
        "baseline_label": "snaps 50→52 · 2026-07-01T14:00 → 16:00",
        "load_profile": [
            {"metric": "DB time (s/s)", "problem": 4.1, "baseline": 1.0, "ratio": 4.1},
        ],
        "top_waits_problem": [
            {"event": "db file sequential read", "wait_class": "User I/O", "seconds": 990.0},
        ],
        "top_waits_baseline": [],
        "top_sql_problem": [{"sql_id": "abc", "elapsed_s": 340.5, "execs": 12}],
        "top_sql_baseline": [],
        "os": [],
    }
    text = render_diff_digest(digest)
    assert "DB time (s/s)" in text and "4.1" in text and "db file sequential read" in text


def test_render_digest_shows_host_cpu_line():
    digest = {
        "has_baseline": True, "suspect": False, "errors": [],
        "problem_label": "snaps 100→102", "baseline_label": "snaps 50→52",
        "load_profile": [], "top_waits_problem": [], "top_waits_baseline": [],
        "top_sql_problem": [], "top_sql_baseline": [],
        "os": [
            {"metric": "BUSY_TIME", "problem_delta": 750000.0, "problem_end": None,
             "baseline_delta": 250000.0, "baseline_end": None},
            {"metric": "IDLE_TIME", "problem_delta": 250000.0, "problem_end": None,
             "baseline_delta": 750000.0, "baseline_end": None},
            {"metric": "NUM_CPUS", "problem_delta": 0.0, "problem_end": 16.0,
             "baseline_delta": 0.0, "baseline_end": 16.0},
            {"metric": "LOAD", "problem_delta": None, "problem_end": 12.4,
             "baseline_delta": None, "baseline_end": 2.1},
        ],
    }
    text = render_diff_digest(digest)
    assert "HOST CPU" in text and "75" in text and "16" in text


def test_render_digest_none_states_unavailable():
    text = render_diff_digest(None)
    assert "unavailable" in text.lower()


@pytest.mark.asyncio
async def test_digest_collects_probe_evidence_ids():
    ids = iter([f"Q-{i:03d}" for i in range(1, 20)])

    async def run_query_with_data(sql: str) -> dict:
        base = {"rows": [], "truncated": False, "row_count": 0, "warnings": [],
                "evidence_id": next(ids)}
        if "DBA_HIST_SYS_TIME_MODEL" in sql:
            base["rows"] = [{"NAME": "DB time", "DELTA": 1000000.0, "END_VALUE": 1.0}]
            base["row_count"] = 1
        return base

    window = SnapWindow(
        begin_snap=100, end_snap=101,
        begin_time=datetime(2026, 7, 8, 14, 0), end_time=datetime(2026, 7, 8, 15, 0),
    )
    result = await run_diff_probe(1, window, None, run_query_with_data)
    assert result.digest is not None
    assert result.digest["evidence_ids"]          # non-empty
    assert all(e.startswith("Q-") for e in result.digest["evidence_ids"])


def test_render_diff_digest_lists_evidence_ids():
    digest = {
        "has_baseline": False, "suspect": False,
        "problem_label": "snaps 100→101", "baseline_label": None,
        "load_profile": [], "top_waits_problem": [], "top_waits_baseline": [],
        "top_sql_problem": [], "top_sql_baseline": [], "os": [], "errors": [],
        "evidence_ids": ["Q-001", "Q-002"],
    }
    text = render_diff_digest(digest)
    assert "Q-001, Q-002" in text
