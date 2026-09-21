from datetime import datetime

import pytest

from observa.graph.master import diff_probe_node
from observa.models import SnapWindow


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def query_catalog(self, sql: str, row_limit: int = 2000, **kwargs) -> dict:
        self.calls.append(sql)
        if "SYS_TIME_MODEL" in sql.upper():
            return {"rows": [{"NAME": "DB time", "DELTA": 7.2e9, "END_VALUE": None}], "truncated": False, "row_count": 1}
        return {"rows": [], "truncated": False, "row_count": 0}


def _state() -> dict:
    return {
        "dbid": 111,
        "problem_window": SnapWindow(
            begin_snap=100, end_snap=102,
            begin_time=datetime(2026, 7, 8, 14, 0), end_time=datetime(2026, 7, 8, 16, 0),
        ),
        "baseline_window": None,
    }


@pytest.mark.asyncio
async def test_node_writes_digest_to_state():
    client = _FakeClient()
    out = await diff_probe_node(_state(), {"configurable": {"mcp_client": client}})
    assert out["baseline_diff"] is not None
    assert out["baseline_diff"]["has_baseline"] is False
    assert out["baseline_suspect"] is False
    assert client.calls  # probe actually queried


@pytest.mark.asyncio
async def test_node_skips_without_window_or_client():
    assert await diff_probe_node({"dbid": 111, "problem_window": None}, {"configurable": {"mcp_client": _FakeClient()}}) == {}
    assert await diff_probe_node(_state(), None) == {}
