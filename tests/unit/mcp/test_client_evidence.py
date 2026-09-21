"""McpClient + EvidenceLedger integration: recording, cache, requester tools."""
from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

import observa.mcp.client as client_mod
from observa.mcp.client import McpClient
from observa.mcp.sqlcl_guard import GuardRejection


class _Settings:
    """Duck-typed stand-in for observa.config.Settings (only what __init__ reads)."""
    mcp_query_row_limit = 500


_SQL = (
    "SELECT event_name FROM DBA_HIST_SYSTEM_EVENT "
    "WHERE dbid = 1234567890 AND snap_id BETWEEN 4210 AND 4215"
)
_GUARD_RESULT = {"rows": [{"EVENT_NAME": "db file sequential read"}],
                 "truncated": False, "row_count": 1, "warnings": []}


def _client(
    monkeypatch, guard_result=None, guard_exc=None
) -> tuple[McpClient, list[str]]:
    """Return an 'entered' client plus the list of SQL texts that reached SQLcl."""
    c = McpClient(_Settings())  # type: ignore[arg-type]
    c._sqlcl = cast(Any, object())  # sentinel: "entered"
    c._sqlcl_lock = asyncio.Lock()
    calls: list[str] = []

    async def _fake_guarded_query(session, sql, config=None):
        calls.append(sql)
        if guard_exc is not None:
            raise guard_exc
        return dict(guard_result or _GUARD_RESULT)

    monkeypatch.setattr(client_mod, "guarded_query", _fake_guarded_query)
    return c, calls


@pytest.mark.asyncio
async def test_query_awr_records_and_returns_evidence_id(monkeypatch):
    c, _ = _client(monkeypatch)
    out = await c.query_awr(_SQL, requester="wait_time", turn=1)
    assert out["evidence_id"] == "Q-001"
    assert out["rows"] == _GUARD_RESULT["rows"]
    rec = c.ledger.get("Q-001")
    assert rec is not None
    assert rec.requester == "wait_time" and rec.turn == 1
    assert rec.kind == "query_awr" and rec.status == "ok"
    assert rec.dbid == 1234567890 and rec.snap_range == (4210, 4215)


@pytest.mark.asyncio
async def test_query_awr_cache_hit_skips_sqlcl(monkeypatch):
    c, calls = _client(monkeypatch)
    await c.query_awr(_SQL, requester="wait_time")
    out2 = await c.query_awr(_SQL.lower(), requester="ash")
    assert out2["evidence_id"] == "Q-001"
    assert len(calls) == 1                  # second call never reached SQLcl
    assert c.ledger.totals() == (2, 1)


@pytest.mark.asyncio
async def test_query_awr_rejection_recorded_and_reraised(monkeypatch):
    c, calls = _client(monkeypatch, guard_exc=GuardRejection("V$SESSION not allowed"))
    with pytest.raises(GuardRejection):
        await c.query_awr("SELECT 1 FROM V$SESSION", requester="wait_time")
    recs = c.ledger.records
    assert len(recs) == 1 and recs[0].status == "rejected"
    assert "V$SESSION" in recs[0].error
    # rejected results are not cacheable — a retry hits the guard again
    with pytest.raises(GuardRejection):
        await c.query_awr("SELECT 1 FROM V$SESSION", requester="wait_time")
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_read_file_records_and_caches(monkeypatch):
    c, _ = _client(monkeypatch)
    c._files = cast(Any, object())
    c._files_lock = asyncio.Lock()
    payload = {"content": "ORA-00600", "offset": 0, "bytes_read": 9,
               "eof": True, "truncated": False, "total_size": 9}
    reads = []

    async def _fake_read(path, offset=0, max_bytes=None):
        reads.append(path)
        return dict(payload)

    monkeypatch.setattr(c, "_read_file_raw", _fake_read)
    out = await c.read_file("alert.log", 0, 65536, requester="file_reader")
    assert out["evidence_id"] == "F-001" and out["content"] == "ORA-00600"
    out2 = await c.read_file("alert.log", 0, 65536, requester="master_chat")
    assert out2["evidence_id"] == "F-001"
    assert len(reads) == 1


@pytest.mark.asyncio
async def test_evidence_id_survives_tool_loop_truncation(monkeypatch):
    import json
    big = {"rows": [{"SQL_TEXT": "x" * 200, "N": i} for i in range(100)],
           "truncated": False, "row_count": 100, "warnings": []}
    c, _ = _client(monkeypatch, guard_result=big)
    out = await c.query_awr(_SQL, requester="sql")
    assert '"evidence_id"' in json.dumps(out, default=str)[:8000]


@pytest.mark.asyncio
async def test_langchain_tools_close_over_requester(monkeypatch):
    c, _ = _client(monkeypatch)
    tools = {t.name: t for t in c.langchain_tools(requester="sql", turn=2)}
    out = await tools["query_awr"].ainvoke({"sql": _SQL})
    rec = c.ledger.get(out["evidence_id"])
    assert rec is not None
    assert rec.requester == "sql" and rec.turn == 2
    # the requester must NOT leak into the LLM-visible tool schema
    schema = tools["query_awr"].args_schema
    assert schema is not None and not isinstance(schema, dict)
    assert "requester" not in schema.model_json_schema()["properties"]


_SEARCH_PAYLOAD = {"matches": [{"line_number": 2, "byte_offset": 11, "line": "ORA-00600"}],
                   "match_count": 1, "scanned_bytes": 20, "truncated": False, "total_size": 20}


@pytest.mark.asyncio
async def test_search_file_records_and_returns_evidence_id(monkeypatch):
    c, _ = _client(monkeypatch)
    c._files = cast(Any, object())
    c._files_lock = asyncio.Lock()

    async def _fake_search(path, pattern, max_matches=200):
        return dict(_SEARCH_PAYLOAD)

    monkeypatch.setattr(c, "_search_file_raw", _fake_search)
    out = await c.search_file("alert.log", r"ORA-\d+", requester="file_reader", turn=1)
    assert out["evidence_id"] == "S-001"
    assert out["match_count"] == 1
    rec = c.ledger.get("S-001")
    assert rec is not None and rec.pattern == r"ORA-\d+" and rec.requester == "file_reader"


@pytest.mark.asyncio
async def test_search_file_cache_hit_skips_files(monkeypatch):
    c, _ = _client(monkeypatch)
    c._files = cast(Any, object())
    c._files_lock = asyncio.Lock()
    calls = []

    async def _fake_search(path, pattern, max_matches=200):
        calls.append(pattern)
        return dict(_SEARCH_PAYLOAD)

    monkeypatch.setattr(c, "_search_file_raw", _fake_search)
    await c.search_file("alert.log", r"ORA-\d+", requester="file_reader")
    out2 = await c.search_file("alert.log", r"ORA-\d+", requester="sql")
    assert out2["evidence_id"] == "S-001"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_scan_alertlog_records_incidents(monkeypatch):
    c, _ = _client(monkeypatch)
    c._files = cast(Any, object())
    c._files_lock = asyncio.Lock()
    incidents_payload = {"incidents": [{"ts": "2024-03-15T14:23:01", "code": "ORA-00060",
                          "text": "x", "instance": 1, "incident_id": None, "byte_offset": 10}],
                         "coverage": {"begin_ts": "2024-03-15T00:00:00", "end_ts": "2024-03-15T23:00:00"}}

    async def _fake_extract(path):
        return dict(incidents_payload)

    monkeypatch.setattr(c, "_extract_ora_raw", _fake_extract)
    out = await c.scan_alertlog("alert.log")
    assert out["evidence_id"] == "S-001"
    assert out["incidents"][0]["code"] == "ORA-00060"
    rec = c.ledger.get("S-001")
    assert rec is not None and rec.requester == "ora_timeline"
    assert rec.match_count == 1  # incident count


@pytest.mark.asyncio
async def test_langchain_tools_includes_search_file(monkeypatch):
    c, _ = _client(monkeypatch)
    c._files = cast(Any, object())
    c._files_lock = asyncio.Lock()

    async def _fake_search(path, pattern, max_matches=200):
        return dict(_SEARCH_PAYLOAD)

    monkeypatch.setattr(c, "_search_file_raw", _fake_search)
    tools = {t.name: t for t in c.langchain_tools(requester="file_reader", turn=1)}
    assert "search_file" in tools
    out = await tools["search_file"].ainvoke({"path": "alert.log", "pattern": r"ORA-\d+"})
    rec = c.ledger.get(out["evidence_id"])
    assert rec is not None and rec.requester == "file_reader"
    schema = tools["search_file"].args_schema
    assert schema is not None and not isinstance(schema, dict)
    assert "requester" not in schema.model_json_schema()["properties"]
