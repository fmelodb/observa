"""Guarded wrapper around the SQLcl MCP query tool.

The SQLcl MCP server exposes a broad SQL execution tool; we intercept every call,
validate the SQL against our allowlist, and cap row counts before letting it
through. This is the single choke point between agents and the database.
"""
from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession

from observa.mcp.allowlist import AllowlistConfig, AllowlistError, check_case_predicates, validate_sql

logger = logging.getLogger(__name__)


# Name of the tool exposed by SQLcl MCP for running SQL. The Oracle SQLcl MCP
# server is still evolving; override via SqlclGuardConfig.tool_name if needed.
DEFAULT_SQLCL_TOOL_NAME = "run-sql"
DEFAULT_SQLCL_SQL_ARG = "sql"


class GuardRejection(ValueError):
    """SQL rejected by the guard before reaching SQLcl."""


# Candidate tool names, in priority order, for SQL execution on SQLcl MCP.
# SQLcl <= 25.2 shipped "run-sql"; 25.3+ renamed it to "sql_run".
_KNOWN_SQLCL_TOOL_NAMES = ("run-sql", "run_sql", "sql_run", "execute-sql", "execute_sql", "sql")
# Candidate argument names for the SQL text on SQLcl MCP tools.
_KNOWN_SQL_ARG_NAMES = ("sql", "statement", "query")


def _get_attr_or_item(obj: Any, key: str, default: Any = None) -> Any:
    """Access ``key`` on a possibly-object-or-dict value."""
    val = getattr(obj, key, None)
    if val is None and isinstance(obj, dict):
        val = obj.get(key, default)
    return val if val is not None else default


def pick_sqlcl_tool(tool_list: list) -> tuple[str, str]:
    """Given MCP list_tools() result, return (tool_name, sql_arg_name) for the SQL-execution tool.

    Priority:
      1. Exact name match from known candidates: "run-sql", "run_sql", "execute-sql", "execute_sql", "sql"
      2. Any tool whose name contains 'sql' case-insensitive
      3. Raise ValueError with list of available tool names

    sql_arg_name is picked by inspecting the tool's inputSchema. Look for a property named
    one of: "sql", "statement", "query". Default to "sql" if none match.
    """
    by_name: dict[str, Any] = {}
    for tool in tool_list:
        name = _get_attr_or_item(tool, "name")
        if isinstance(name, str):
            by_name[name] = tool

    chosen = None
    for candidate in _KNOWN_SQLCL_TOOL_NAMES:
        if candidate in by_name:
            chosen = by_name[candidate]
            break

    if chosen is None:
        # Prefer a tool whose schema exposes a recognized SQL argument —
        # guards against name drift across SQLcl releases (e.g. 25.3 renamed
        # run-sql to sql_run, while sqlcl_run also contains "sql" but takes a
        # SQLcl command via a 'sqlcl' argument, not SQL text).
        def _has_known_sql_arg(tool: Any) -> bool:
            schema = _get_attr_or_item(tool, "inputSchema", {}) or {}
            properties = _get_attr_or_item(schema, "properties", {}) or {}
            return isinstance(properties, dict) and any(
                arg in properties for arg in _KNOWN_SQL_ARG_NAMES
            )

        for name, tool in by_name.items():
            if "sql" in name.lower() and _has_known_sql_arg(tool):
                chosen = tool
                break

    if chosen is None:
        for name, tool in by_name.items():
            if "sql" in name.lower():
                chosen = tool
                break

    if chosen is None:
        available = sorted(by_name.keys())
        raise ValueError(
            f"No SQL-execution tool found on SQLcl MCP server. Available tools: {available}"
        )

    tool_name = _get_attr_or_item(chosen, "name")
    schema = _get_attr_or_item(chosen, "inputSchema", {}) or {}
    properties = _get_attr_or_item(schema, "properties", {}) or {}
    sql_arg = DEFAULT_SQLCL_SQL_ARG
    if isinstance(properties, dict):
        for candidate_arg in _KNOWN_SQL_ARG_NAMES:
            if candidate_arg in properties:
                sql_arg = candidate_arg
                break
    return tool_name, sql_arg


@dataclass(frozen=True)
class SqlclGuardConfig:
    allowlist: AllowlistConfig = AllowlistConfig()
    tool_name: str = DEFAULT_SQLCL_TOOL_NAME
    sql_arg: str = DEFAULT_SQLCL_SQL_ARG


# Text markers that indicate SQLcl returned an error instead of a result set.
_SQLCL_ERROR_MARKERS = (
    "ERRO:",        # Portuguese SQLcl error prefix seen in Oracle installs
    "ERROR:",
    "ORA-",
    "SP2-",
    "null connection",
    "not connected",
    "no connection",
)


class SqlclRuntimeError(RuntimeError):
    """Raised when SQLcl reports an error instead of returning rows."""


def _parse_tool_result(raw: Any) -> list[dict]:
    """Best-effort parse of SQLcl MCP's tool response into list[dict] of rows.

    Raises ``SqlclRuntimeError`` when the text payload looks like a SQLcl
    error message (``ERRO:``, ``ORA-``, ``SP2-``, "null connection"). This
    prevents agents from mistaking a driver-level failure for a 1-row result.
    """
    texts: list[str] = []
    content = getattr(raw, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            texts.append(text)
    if not texts:
        return []
    payload = "\n".join(texts).strip()
    if not payload:
        return []
    # Detect error responses BEFORE attempting to parse JSON.
    upper_head = payload[:400].upper()
    for marker in _SQLCL_ERROR_MARKERS:
        if marker.upper() in upper_head:
            raise SqlclRuntimeError(payload[:400])
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        # Not JSON. SQLcl >= 25 returns CSV by default: first line is a
        # comma-separated header with quoted column names, subsequent lines
        # are the rows. Parse it.
        return _parse_csv_payload(payload)
    if isinstance(data, list):
        return [row if isinstance(row, dict) else {"value": row} for row in data]
    if isinstance(data, dict):
        if isinstance(data.get("rows"), list):
            return [r if isinstance(r, dict) else {"value": r} for r in data["rows"]]
        cols = data.get("columns")
        rows = data.get("data")
        if isinstance(cols, list) and isinstance(rows, list):
            return [dict(zip(cols, r)) for r in rows]
        return [data]
    return [{"value": data}]


def _parse_csv_payload(payload: str) -> list[dict]:
    """Parse SQLcl's default CSV-ish output into list[dict].

    Example input::

        "SNAP_ID","DBID"
        145,2998621142
        153,2998621142

    Header row is required (first non-empty line). Rows with a mismatched
    column count are skipped. Integer/float-looking cells are coerced so
    downstream LLM prompts show native numbers instead of string-wrapped
    values.
    """
    reader = csv.reader(io.StringIO(payload))
    rows_raw: list[list[str]] = [r for r in reader if r]
    if not rows_raw:
        return []
    header = [c.strip() for c in rows_raw[0]]
    out: list[dict] = []
    for row in rows_raw[1:]:
        if len(row) != len(header):
            continue
        out.append({header[i]: _coerce_cell(row[i]) for i in range(len(header))})
    return out


def _coerce_cell(value: str) -> Any:
    v = value.strip()
    if v == "" or v.upper() == "NULL":
        return None
    # Quick numeric coercion — keep strings with leading zeros and explicit
    # signs alone; csv.reader already strips surrounding quotes.
    try:
        if "." in v or "e" in v or "E" in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _truncate(rows: list[dict], limit: int) -> tuple[list[dict], bool]:
    if len(rows) <= limit:
        return rows, False
    return rows[:limit], True


async def guarded_query(
    session: ClientSession,
    sql: str,
    config: SqlclGuardConfig | None = None,
) -> dict:
    """Validate ``sql`` then call SQLcl MCP. Return ``{rows, truncated, row_count}``.

    Raises:
        GuardRejection: when the SQL violates the allowlist.
    """
    cfg = config or SqlclGuardConfig()
    snippet = " ".join(sql.split())[:200]
    try:
        validate_sql(sql, cfg.allowlist)
        warnings = check_case_predicates(sql, cfg.allowlist)
    except AllowlistError as exc:
        logger.warning("guarded_query REJECTED: %s — SQL=%s", exc, snippet)
        raise GuardRejection(str(exc)) from exc

    logger.info("guarded_query → tool=%r sql=%s", cfg.tool_name, snippet)
    try:
        result = await session.call_tool(cfg.tool_name, {cfg.sql_arg: sql})
    except Exception:
        logger.exception("guarded_query: SQLcl call_tool failed (tool=%r)", cfg.tool_name)
        raise
    rows = _parse_tool_result(result)
    rows, truncated = _truncate(rows, cfg.allowlist.row_limit)
    logger.info(
        "guarded_query ← %d row(s)%s",
        len(rows),
        " (truncated)" if truncated else "",
    )
    return {"rows": rows, "truncated": truncated, "row_count": len(rows), "warnings": warnings}
