"""Unit tests for the pure ORA-timeline assembly + digest."""
from __future__ import annotations

import pytest

from observa.graph.ora_timeline import build_ora_timeline, render_ora_timeline_digest
from observa.models import InputFile, SnapWindow


def _win(begin_iso, end_iso):
    from datetime import datetime
    return SnapWindow(begin_snap=1, end_snap=2,
                      begin_time=datetime.fromisoformat(begin_iso),
                      end_time=datetime.fromisoformat(end_iso))


def _scan_returning(incidents, coverage):
    async def _scan(path):
        return {"evidence_id": "S-001", "incidents": incidents, "coverage": coverage}
    return _scan


@pytest.mark.asyncio
async def test_labels_incidents_by_window():
    problem = _win("2024-03-15T14:00:00", "2024-03-15T16:00:00")
    baseline = _win("2024-03-08T14:00:00", "2024-03-08T16:00:00")
    incidents = [
        {"ts": "2024-03-15T14:22:01", "code": "ORA-00060", "text": "deadlock",
         "instance": 2, "incident_id": 48291, "byte_offset": 100},
        {"ts": "2024-03-08T15:00:00", "code": "ORA-00600", "text": "baseline err",
         "instance": 1, "incident_id": None, "byte_offset": 50},
        {"ts": "2024-03-01T02:00:00", "code": "ORA-01555", "text": "old",
         "instance": 1, "incident_id": None, "byte_offset": 10},
    ]
    coverage = {"begin_ts": "2024-03-01T00:00:00", "end_ts": "2024-03-15T18:00:00"}
    files = [InputFile(type="alertlog", path="/a.log", label="prod")]
    tl = await build_ora_timeline(files, problem, baseline, _scan_returning(incidents, coverage))
    assert tl is not None
    by_code = {i["code"]: i for i in tl["incidents"]}
    assert by_code["ORA-00060"]["window"] == "problem"
    assert by_code["ORA-00600"]["window"] == "baseline"
    assert by_code["ORA-01555"]["window"] == "outside"
    assert tl["counts"] == {"problem": 1, "baseline": 1, "outside": 1, "incidents": 3}
    assert tl["coverage"]["covers_problem"] is True
    assert by_code["ORA-00060"]["source_label"] == "prod"
    assert by_code["ORA-00060"]["evidence_id"] == "S-001"


@pytest.mark.asyncio
async def test_coverage_not_covering_problem():
    problem = _win("2024-03-15T14:00:00", "2024-03-15T16:00:00")
    coverage = {"begin_ts": "2024-03-01T00:00:00", "end_ts": "2024-03-10T00:00:00"}
    files = [InputFile(type="alertlog", path="/a.log", label="prod")]
    tl = await build_ora_timeline(files, problem, None, _scan_returning([], coverage))
    assert tl is not None
    assert tl["coverage"]["covers_problem"] is False


@pytest.mark.asyncio
async def test_none_ts_is_outside():
    problem = _win("2024-03-15T14:00:00", "2024-03-15T16:00:00")
    incidents = [{"ts": None, "code": "ORA-04031", "text": "oom",
                  "instance": None, "incident_id": None, "byte_offset": 0}]
    coverage = {"begin_ts": None, "end_ts": None}
    files = [InputFile(type="alertlog", path="/a.log", label="prod")]
    tl = await build_ora_timeline(files, problem, None, _scan_returning(incidents, coverage))
    assert tl is not None
    assert tl["incidents"][0]["window"] == "outside"


@pytest.mark.asyncio
async def test_empty_when_no_incidents_and_no_coverage():
    problem = _win("2024-03-15T14:00:00", "2024-03-15T16:00:00")
    files = [InputFile(type="alertlog", path="/a.log", label="prod")]
    tl = await build_ora_timeline(
        files, problem, None,
        _scan_returning([], {"begin_ts": None, "end_ts": None}),
    )
    assert tl is None


def test_digest_groups_and_warns_on_no_coverage():
    tl = {
        "coverage": {"begin_ts": "2024-03-01T00:00:00", "end_ts": "2024-03-10T00:00:00",
                     "covers_problem": False, "covers_baseline": False},
        "counts": {"problem": 0, "baseline": 0, "outside": 1, "incidents": 1},
        "incidents": [{"ts": "2024-03-01T02:00:00", "code": "ORA-00600", "text": "err",
                       "instance": 1, "incident_id": None, "byte_offset": 10,
                       "window": "outside", "source_label": "prod", "evidence_id": "S-001"}],
    }
    text = render_ora_timeline_digest(tl)
    assert "does not cover the problem window" in text
    assert "ORA-00600" in text
    assert "OUTSIDE" in text


def test_digest_empty_timeline_is_blank():
    assert render_ora_timeline_digest(None) == ""


@pytest.mark.asyncio
async def test_multi_log_union_coverage():
    # Neither log alone spans the problem window, but together they do →
    # covers_problem True (union-of-spans is the intended semantic).
    problem = _win("2024-03-15T10:00:00", "2024-03-15T20:00:00")
    files = [InputFile(type="alertlog", path="/a.log", label="a"),
             InputFile(type="alertlog", path="/b.log", label="b")]

    async def _scan(path):
        if path == "/a.log":
            return {"evidence_id": "S-001", "incidents": [],
                    "coverage": {"begin_ts": "2024-03-15T08:00:00", "end_ts": "2024-03-15T15:00:00"}}
        return {"evidence_id": "S-002", "incidents": [],
                "coverage": {"begin_ts": "2024-03-15T15:00:00", "end_ts": "2024-03-15T22:00:00"}}

    tl = await build_ora_timeline(files, problem, None, _scan)
    assert tl is not None
    assert tl["coverage"]["covers_problem"] is True


@pytest.mark.asyncio
async def test_incident_at_window_end_boundary_is_problem():
    problem = _win("2024-03-15T14:00:00", "2024-03-15T16:00:00")
    incidents = [{"ts": "2024-03-15T16:00:00", "code": "ORA-00060", "text": "edge",
                  "instance": None, "incident_id": None, "byte_offset": 0}]
    coverage = {"begin_ts": "2024-03-15T00:00:00", "end_ts": "2024-03-15T23:00:00"}
    files = [InputFile(type="alertlog", path="/a.log", label="p")]
    tl = await build_ora_timeline(files, problem, None, _scan_returning(incidents, coverage))
    assert tl is not None
    assert tl["incidents"][0]["window"] == "problem"  # inclusive end boundary
