"""Unit tests for sqlcl_guard: parsing and allowlist interception."""
from __future__ import annotations

import pytest

from observa.mcp.allowlist import AllowlistConfig
from observa.mcp.sqlcl_guard import (
    GuardRejection,
    SqlclGuardConfig,
    _parse_tool_result,
    _truncate,
    guarded_query,
    pick_sqlcl_tool,
)


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _ToolResult:
    def __init__(self, texts: list[str]) -> None:
        self.content = [_Block(t) for t in texts]


class _FakeSession:
    """Records the last tool call; returns whatever is put in ``next_result``."""

    def __init__(self, next_result: _ToolResult) -> None:
        self.next_result = next_result
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, args: dict):  # noqa: D401
        self.calls.append((name, args))
        return self.next_result


def _tool(name: str, props: list[str]) -> dict:
    return {"name": name, "inputSchema": {"properties": {p: {} for p in props}}}


def test_pick_sqlcl_tool_modern_names():
    """SQLcl >= 25.3 renamed run-sql -> sql_run (and run-sqlcl -> sqlcl_run).

    sqlcl_run comes FIRST in the server's tool order and its name contains
    'sql' — discovery must still pick sql_run (arg 'sql'), never sqlcl_run.
    """
    tools = [
        _tool("connections_list", ["name", "username"]),
        _tool("connect", ["connection_name", "model"]),
        _tool("disconnect", ["model"]),
        _tool("sqlcl_run", ["sqlcl", "model", "execution_type"]),
        _tool("sql_run", ["sql", "model", "execution_type"]),
        _tool("schema_information", ["schema", "level"]),
    ]
    assert pick_sqlcl_tool(tools) == ("sql_run", "sql")


def test_pick_sqlcl_tool_fallback_prefers_recognized_sql_arg():
    """With no exact-name candidate, prefer the 'sql'-named tool whose schema
    actually exposes a recognized SQL argument over one that merely contains
    'sql' in its name."""
    tools = [
        _tool("foo_sqlcl", ["sqlcl"]),          # name matches, arg unknown
        _tool("bar_sqlquery", ["sql", "model"]),  # name matches, arg known
    ]
    assert pick_sqlcl_tool(tools) == ("bar_sqlquery", "sql")


def test_parse_json_array():
    result = _ToolResult(['[{"a": 1}, {"a": 2}]'])
    assert _parse_tool_result(result) == [{"a": 1}, {"a": 2}]


def test_parse_columns_data_shape():
    result = _ToolResult(['{"columns": ["SNAP_ID", "NAME"], "data": [[1, "foo"], [2, "bar"]]}'])
    assert _parse_tool_result(result) == [
        {"SNAP_ID": 1, "NAME": "foo"},
        {"SNAP_ID": 2, "NAME": "bar"},
    ]


def test_parse_rows_key():
    result = _ToolResult(['{"rows": [{"x": 10}], "meta": "ignored"}'])
    assert _parse_tool_result(result) == [{"x": 10}]


def test_parse_non_json_non_error_returns_empty():
    # Non-JSON, non-CSV-looking text becomes a single-row CSV (header only)
    # or no rows, depending on commas. A one-liner with no commas yields one
    # header and zero data rows.
    result = _ToolResult(["SQL> select * from dual;"])
    assert _parse_tool_result(result) == []


def test_parse_csv_payload_parses_quoted_header_and_coerces_numbers():
    payload = '"SNAP_ID","DBID"\n145,2998621142\n153,2998621142\n'
    result = _ToolResult([payload])
    rows = _parse_tool_result(result)
    assert rows == [
        {"SNAP_ID": 145, "DBID": 2998621142},
        {"SNAP_ID": 153, "DBID": 2998621142},
    ]


def test_parse_csv_payload_handles_null_and_float():
    payload = '"NAME","RATIO"\n"foo",0.75\n"bar",NULL\n'
    result = _ToolResult([payload])
    rows = _parse_tool_result(result)
    assert rows == [
        {"NAME": "foo", "RATIO": 0.75},
        {"NAME": "bar", "RATIO": None},
    ]


def test_parse_sqlcl_error_raises():
    from observa.mcp.sqlcl_guard import SqlclRuntimeError

    result = _ToolResult(["ERRO: null connection not allowed"])
    with pytest.raises(SqlclRuntimeError, match="null connection"):
        _parse_tool_result(result)


def test_parse_ora_error_raises():
    from observa.mcp.sqlcl_guard import SqlclRuntimeError

    result = _ToolResult(["ORA-00942: table or view does not exist"])
    with pytest.raises(SqlclRuntimeError, match="ORA-00942"):
        _parse_tool_result(result)


def test_parse_empty_content():
    result = _ToolResult([])
    assert _parse_tool_result(result) == []


def test_truncate_under_limit():
    rows = [{"i": i} for i in range(3)]
    out, truncated = _truncate(rows, 5)
    assert out == rows
    assert truncated is False


def test_truncate_over_limit():
    rows = [{"i": i} for i in range(10)]
    out, truncated = _truncate(rows, 3)
    assert len(out) == 3
    assert truncated is True


async def test_guarded_query_rejects_before_calling_session():
    session = _FakeSession(_ToolResult([]))
    with pytest.raises(GuardRejection):
        await guarded_query(session, "DROP TABLE X")
    assert session.calls == []


async def test_guarded_query_rejects_non_allowlisted_table():
    session = _FakeSession(_ToolResult([]))
    with pytest.raises(GuardRejection):
        await guarded_query(session, "SELECT * FROM DBA_USERS")
    assert session.calls == []


async def test_guarded_query_passes_allowed_sql():
    session = _FakeSession(_ToolResult(['[{"SNAP_ID": 42}]']))
    out = await guarded_query(session, "SELECT snap_id FROM DBA_HIST_SNAPSHOT")
    assert out == {"rows": [{"SNAP_ID": 42}], "truncated": False, "row_count": 1, "warnings": []}
    assert len(session.calls) == 1
    name, args = session.calls[0]
    assert name == "run-sql"
    assert args == {"sql": "SELECT snap_id FROM DBA_HIST_SNAPSHOT"}


async def test_guarded_query_respects_row_limit():
    rows = [{"i": i} for i in range(25)]
    import json as _json
    session = _FakeSession(_ToolResult([_json.dumps(rows)]))
    cfg = SqlclGuardConfig()
    cfg_limited = SqlclGuardConfig(
        allowlist=type(cfg.allowlist)(row_limit=5),
        tool_name=cfg.tool_name,
        sql_arg=cfg.sql_arg,
    )
    out = await guarded_query(session, "SELECT * FROM DBA_HIST_SNAPSHOT", cfg_limited)
    assert out["row_count"] == 5
    assert out["truncated"] is True


async def test_guarded_query_honors_custom_tool_name():
    session = _FakeSession(_ToolResult(["[]"]))
    cfg = SqlclGuardConfig(tool_name="execute_sql", sql_arg="statement")
    await guarded_query(
        session, "SELECT 1 FROM DBA_HIST_SNAPSHOT WHERE rownum <= 1", cfg
    )
    name, args = session.calls[0]
    assert name == "execute_sql"
    assert args == {"statement": "SELECT 1 FROM DBA_HIST_SNAPSHOT WHERE rownum <= 1"}


# --- pick_sqlcl_tool tests ---


def _fake_tool(name: str, properties: dict | None = None):
    from types import SimpleNamespace

    schema = {"properties": properties} if properties is not None else {}
    return SimpleNamespace(name=name, inputSchema=schema)


def test_pick_sqlcl_tool_exact_match_run_sql():
    tools = [
        _fake_tool("connect"),
        _fake_tool("my-custom-sql-runner", {"sql": {"type": "string"}}),
        _fake_tool("run-sql", {"sql": {"type": "string"}}),
    ]
    name, sql_arg = pick_sqlcl_tool(tools)
    assert name == "run-sql"
    assert sql_arg == "sql"


def test_pick_sqlcl_tool_fallback_to_substring():
    tools = [
        _fake_tool("connect"),
        _fake_tool("my-custom-sql-runner", {"sql": {"type": "string"}}),
    ]
    name, sql_arg = pick_sqlcl_tool(tools)
    assert name == "my-custom-sql-runner"
    assert sql_arg == "sql"


def test_pick_sqlcl_tool_no_match_raises():
    tools = [_fake_tool("connect"), _fake_tool("list-schemas")]
    with pytest.raises(ValueError) as exc_info:
        pick_sqlcl_tool(tools)
    msg = str(exc_info.value)
    assert "connect" in msg
    assert "list-schemas" in msg


def test_pick_sqlcl_tool_sql_arg_from_schema():
    tools = [_fake_tool("run-sql", {"statement": {"type": "string"}})]
    name, sql_arg = pick_sqlcl_tool(tools)
    assert name == "run-sql"
    assert sql_arg == "statement"


def test_pick_sqlcl_tool_sql_arg_default():
    tools = [_fake_tool("run-sql", {"foo": {"type": "string"}})]
    name, sql_arg = pick_sqlcl_tool(tools)
    assert name == "run-sql"
    assert sql_arg == "sql"


# --- Case-predicate integration tests ----------------------------------------


class _FakeCaseSession:
    """Minimal ClientSession stand-in returning a fixed CSV payload."""

    def __init__(self, payload: str = '"C"\n1\n') -> None:
        self._payload = payload
        self.calls: list = []

    async def call_tool(self, name: str, args: dict):
        self.calls.append((name, args))
        payload = self._payload

        class _Block:
            def __init__(self, text: str) -> None:
                self.text = text

        class _Result:
            content = [_Block(payload)]

        return _Result()


async def test_guarded_query_rejects_missing_dbid_when_case_set():
    cfg = SqlclGuardConfig(allowlist=AllowlistConfig(case_dbid=111))
    with pytest.raises(GuardRejection, match="dbid"):
        await guarded_query(_FakeCaseSession(), "SELECT snap_id FROM DBA_HIST_SNAPSHOT", cfg)


async def test_guarded_query_surfaces_window_warning():
    cfg = SqlclGuardConfig(allowlist=AllowlistConfig(case_dbid=111, enforce_window="warn"))
    result = await guarded_query(
        _FakeCaseSession(), "SELECT stat_name FROM DBA_HIST_SYSSTAT WHERE dbid = 111", cfg
    )
    assert result["warnings"] and "snap" in result["warnings"][0].lower()


async def test_guarded_query_clean_query_has_empty_warnings():
    cfg = SqlclGuardConfig(allowlist=AllowlistConfig(case_dbid=111))
    result = await guarded_query(
        _FakeCaseSession(),
        "SELECT stat_name FROM DBA_HIST_SYSSTAT WHERE dbid = 111 AND snap_id BETWEEN 1 AND 2",
        cfg,
    )
    assert result["warnings"] == []
