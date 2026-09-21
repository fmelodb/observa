"""Tests for the ora_timeline graph node wrapper."""
from __future__ import annotations

from typing import Any, cast

import pytest

from observa.graph.master import ora_timeline_node
from observa.models import InputFile, SnapWindow


async def _run(state: dict, config: Any) -> dict:
    """Call the node with plain-dict fixtures (cast past the CaseState/RunnableConfig types)."""
    return await ora_timeline_node(cast(Any, state), cast(Any, config))


class _FakeClient:
    def __init__(self, incidents, coverage):
        self._incidents, self._coverage = incidents, coverage
        self.scanned: list[str] = []

    async def scan_alertlog(self, path, requester="ora_timeline"):
        self.scanned.append(path)
        return {"evidence_id": "S-001", "incidents": self._incidents,
                "coverage": self._coverage}


def _win(begin_iso, end_iso):
    from datetime import datetime
    return SnapWindow(begin_snap=1, end_snap=2,
                      begin_time=datetime.fromisoformat(begin_iso),
                      end_time=datetime.fromisoformat(end_iso))


@pytest.mark.asyncio
async def test_node_seeds_ora_timeline():
    client = _FakeClient(
        incidents=[{"ts": "2024-03-15T14:30:00", "code": "ORA-00060", "text": "x",
                    "instance": 1, "incident_id": None, "byte_offset": 10}],
        coverage={"begin_ts": "2024-03-15T00:00:00", "end_ts": "2024-03-15T23:00:00"},
    )
    state = {
        "input_files": [InputFile(type="alertlog", path="/a.log", label="prod")],
        "problem_window": _win("2024-03-15T14:00:00", "2024-03-15T16:00:00"),
        "baseline_window": None,
    }
    out = await _run(state, {"configurable": {"mcp_client": client}})
    assert out["ora_timeline"]["counts"]["problem"] == 1
    assert client.scanned == ["/a.log"]


@pytest.mark.asyncio
async def test_node_noops_without_alertlog():
    state = {"input_files": [InputFile(type="trace", path="/t.trc", label="t")],
             "problem_window": None, "baseline_window": None}
    out = await _run(state, {"configurable": {"mcp_client": object()}})
    assert out == {}


@pytest.mark.asyncio
async def test_node_noops_when_timeline_empty():
    client = _FakeClient(incidents=[], coverage={"begin_ts": None, "end_ts": None})
    state = {
        "input_files": [InputFile(type="alertlog", path="/a.log", label="p")],
        "problem_window": None, "baseline_window": None,
    }
    out = await _run(state, {"configurable": {"mcp_client": client}})
    assert out == {}
    assert client.scanned == ["/a.log"]  # it DID scan, just found nothing


@pytest.mark.asyncio
async def test_node_noops_without_client():
    state = {"input_files": [InputFile(type="alertlog", path="/a.log", label="p")],
             "problem_window": None, "baseline_window": None}
    assert await _run(state, None) == {}
