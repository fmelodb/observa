"""Aggregated MCP client exposing LangChain tools to Observa agents.

Connects to two MCP servers:
  - SQLcl (Oracle) — guarded via ``observa.mcp.sqlcl_guard``.
  - files_server — our stdio server that exposes ``list_files`` / ``read_file``.

Agents consume the high-level tools returned by ``McpClient.langchain_tools()``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool, BaseTool
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from observa.config import Settings
from observa.mcp.allowlist import AllowlistConfig
from observa.mcp.evidence import EvidenceLedger, QueryKind
from observa.mcp.file_expansion import FileEntry, expand_input_files
from observa.mcp.sqlcl_guard import SqlclGuardConfig, GuardRejection, guarded_query, pick_sqlcl_tool
from observa.mcp.sqlcl_launcher import sqlcl_session as _sqlcl_session_cm

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputFileEntry:
    type: str
    path: str
    label: str = ""


def _files_server_params(input_files: list[InputFileEntry]) -> StdioServerParameters:
    env = {
        "OBSERVA_FILES_JSON": json.dumps(
            [{"type": f.type, "path": f.path, "label": f.label or f.path} for f in input_files]
        ),
    }
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "observa.mcp.files_server"],
        env=env,
    )


class McpClient:
    """Session-scoped MCP client. Must be used via ``async with``."""

    def __init__(
        self,
        settings: Settings,
        input_files: list[InputFileEntry] | None = None,
    ) -> None:
        self._settings = settings
        self._input_files = list(input_files or [])
        self._stack = AsyncExitStack()
        self._sqlcl: ClientSession | None = None
        self._files: ClientSession | None = None
        # Holds extracted-zip contents for the lifetime of the session.
        self._extract_tmp: tempfile.TemporaryDirectory | None = None
        # SQLcl's MCP server (Java/Reactor-based) uses a single-subscription
        # sink per transport — concurrent ``call_tool`` invocations crash it
        # with "Failed to enqueue message" and kill the whole session. The
        # files server is simpler but we serialize both for safety. Agents
        # still run in parallel; only the MCP I/O queues up.
        self._sqlcl_lock: asyncio.Lock | None = None
        self._files_lock: asyncio.Lock | None = None
        self._guard_cfg = SqlclGuardConfig(
            allowlist=AllowlistConfig(row_limit=settings.mcp_query_row_limit),
        )
        # Evidence ledger: provenance + shared result cache for every data
        # access this client performs. One per case session (client lifetime).
        self._ledger = EvidenceLedger()

    @property
    def ledger(self) -> EvidenceLedger:
        return self._ledger

    async def __aenter__(self) -> "McpClient":
        self._sqlcl_lock = asyncio.Lock()
        self._files_lock = asyncio.Lock()
        self._sqlcl = await self._stack.enter_async_context(_sqlcl_session_cm(self._settings))
        try:
            tools_response = await self._sqlcl.list_tools()
            name, sql_arg = pick_sqlcl_tool(list(tools_response.tools))
            self._guard_cfg = SqlclGuardConfig(
                allowlist=AllowlistConfig(row_limit=self._settings.mcp_query_row_limit),
                tool_name=name,
                sql_arg=sql_arg,
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "SQLcl MCP tool auto-discovery failed (%s); using default tool name %r",
                exc,
                self._guard_cfg.tool_name,
            )

        # SQLcl MCP starts with no active connection. Call its `connect` tool
        # with the saved connection name so subsequent run-sql calls actually
        # reach the database instead of returning "ERRO: null connection not
        # allowed".
        conn_name = self._settings.mcp_sqlcl_connection_name
        if not conn_name:
            available = await self._fetch_available_connections()
            raise RuntimeError(
                "mcp.sqlcl_connection_name is empty. Save a SQLcl connection "
                "(sql /nolog; connect user/pwd@dsn; conn -save <name>) and set "
                "mcp.sqlcl_connection_name in config/config.yaml. "
                f"Available SQLcl connections: {available}"
            )
        _logger.info("sqlcl: calling `connect` with saved connection %r", conn_name)
        try:
            connect_result = await self._sqlcl.call_tool(
                "connect", {"connection_name": conn_name, "model": "observa"}
            )
        except Exception as exc:  # noqa: BLE001
            available = await self._fetch_available_connections()
            raise RuntimeError(
                f"SQLcl `connect` failed for connection {conn_name!r}: {exc}. "
                f"Available SQLcl connections: {available}. "
                "Update mcp.sqlcl_connection_name in config/config.yaml."
            ) from exc
        raw_text = " ".join(
            getattr(b, "text", "") for b in (connect_result.content or [])
        )
        if "ERRO" in raw_text.upper() or "FAIL" in raw_text.upper() or "NOT FOUND" in raw_text.upper():
            available = await self._fetch_available_connections()
            raise RuntimeError(
                f"SQLcl `connect` failed for connection {conn_name!r}: {raw_text[:300]}. "
                f"Available SQLcl connections: {available}."
            )
        _logger.info("sqlcl: connected to %r", conn_name)

        # Normalize NLS settings so SQLcl's CSV output is unambiguous.
        # Brazilian (and many European) Oracle installs default to a comma
        # decimal separator and ``DD/MM/RR HH24:MI:SS,FF`` timestamp format —
        # both emit commas inside values that break CSV parsing. These ALTERs
        # bypass the allowlist deliberately (they are session-local, read-safe
        # settings, not data modifications).
        for alter_sql in (
            "ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD HH24:MI:SS'",
            "ALTER SESSION SET NLS_TIMESTAMP_FORMAT = 'YYYY-MM-DD\"T\"HH24:MI:SS.FF'",
            "ALTER SESSION SET NLS_TIMESTAMP_TZ_FORMAT = 'YYYY-MM-DD\"T\"HH24:MI:SS.FF TZR'",
            "ALTER SESSION SET NLS_NUMERIC_CHARACTERS = '.,'",
        ):
            try:
                await self._sqlcl.call_tool(
                    self._guard_cfg.tool_name,
                    {self._guard_cfg.sql_arg: alter_sql},
                )
            except Exception as exc:  # noqa: BLE001
                _logger.warning("sqlcl NLS normalization failed (%s): %s", alter_sql, exc)
        _logger.info("sqlcl: NLS normalized")

        # Files server is attached later (case start) via attach_files() — at
        # boot only SQLcl is needed for the intake catalog queries. Legacy
        # callers that passed input_files to the constructor keep the old
        # single-shot behavior.
        if self._input_files:
            await self.attach_files(self._input_files)
        return self

    async def attach_files(self, input_files: list[InputFileEntry]) -> None:
        """Start the files MCP server with the case's file registry.

        MUST be called from the same task that entered this client (the app's
        MCP lifecycle task) — the stdio_client context uses anyio cancel
        scopes that require same-task enter/exit.
        """
        if self._files is not None:
            raise RuntimeError(
                "attach_files already called; McpClient supports one file "
                "registry per session"
            )
        self._input_files = list(input_files)
        # Same stderr redirection trick as SQLcl: inheriting the parent's
        # stderr blows up on Windows inside a Textual TUI (Bad file descriptor
        # from msvcrt.get_osfhandle).
        try:
            files_errfile = open("observa-files-server.log", "a", encoding="utf-8", buffering=1)
        except OSError:
            import os as _os
            files_errfile = open(_os.devnull, "w")
        self._stack.callback(files_errfile.close)

        # Auto-expand TFA/SQLHC zips and directories into individual files
        # before handing them to the files MCP server. Other types pass through.
        self._extract_tmp = tempfile.TemporaryDirectory(prefix="observa-files-")
        self._stack.callback(self._extract_tmp.cleanup)
        raw_entries = [
            FileEntry(type=f.type, path=f.path, label=f.label) for f in self._input_files
        ]
        expanded_entries = expand_input_files(raw_entries, Path(self._extract_tmp.name))
        if len(expanded_entries) != len(raw_entries):
            _logger.info(
                "files: %d input(s) expanded into %d registered file(s)",
                len(raw_entries), len(expanded_entries),
            )
        expanded_input_files = [
            InputFileEntry(type=e.type, path=e.path, label=e.label)
            for e in expanded_entries
        ]

        read, write = await self._stack.enter_async_context(
            stdio_client(_files_server_params(expanded_input_files), errlog=files_errfile)
        )
        files_session = await self._stack.enter_async_context(ClientSession(read, write))
        await files_session.initialize()
        self._files = files_session

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()
        self._sqlcl = None
        self._files = None

    async def _fetch_available_connections(self) -> str:
        """Return a comma-separated list of saved SQLcl connection names, or a hint."""
        if self._sqlcl is None:
            return "(sqlcl session not open)"
        # SQLcl <= 25.2 named the tool "list-connections"; 25.3+ renamed it
        # to "connections_list". Try newest first.
        last_exc: Exception | None = None
        for tool_name in ("connections_list", "list-connections"):
            try:
                result = await self._sqlcl.call_tool(tool_name, {})
                text = " ".join(
                    getattr(b, "text", "") for b in (result.content or [])
                ).strip()
                return text or "(none)"
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        return f"(could not list: {last_exc})"

    async def query_awr(
        self, sql: str, requester: str = "agent", turn: int | None = None
    ) -> dict:
        assert self._sqlcl is not None and self._sqlcl_lock is not None, "McpClient not entered"
        # Benign race: two agents issuing the same novel SQL concurrently both
        # miss and both execute — acceptable, reads are idempotent against an
        # immutable AWR repository.
        cached = self._ledger.lookup_query("query_awr", sql, requester)
        if cached is not None:
            return cached
        return await self._run_recorded_query("query_awr", sql, requester, turn, self._guard_cfg)

    def set_case_context(self, dbid: int) -> None:
        """Activate dbid/window predicate enforcement for agent queries.

        Must be called before any agent query is dispatched; agent queries
        dispatched before this run with enforcement disabled.
        """
        self._guard_cfg = SqlclGuardConfig(
            allowlist=AllowlistConfig(
                row_limit=self._settings.mcp_query_row_limit,
                case_dbid=int(dbid),
                enforce_window=self._settings.mcp_enforce_window,
            ),
            tool_name=self._guard_cfg.tool_name,
            sql_arg=self._guard_cfg.sql_arg,
        )

    async def query_catalog(
        self,
        sql: str,
        row_limit: int = 2000,
        requester: str = "internal",
        turn: int | None = None,
    ) -> dict:
        """Internal fixed queries (intake pickers, timeline, probes).

        Higher row cap than agent queries and NO case-predicate enforcement —
        these statements are ours, not the LLM's. Cache is namespaced apart
        from query_awr so agent queries never get served un-enforced results.
        """
        assert self._sqlcl is not None and self._sqlcl_lock is not None, "McpClient not entered"
        # Cache key deliberately ignores row_limit (all current callers use the
        # default; fold it into the key if a future caller varies it).
        cached = self._ledger.lookup_query("query_catalog", sql, requester)
        if cached is not None:
            return cached
        cfg = SqlclGuardConfig(
            allowlist=AllowlistConfig(row_limit=row_limit),
            tool_name=self._guard_cfg.tool_name,
            sql_arg=self._guard_cfg.sql_arg,
        )
        return await self._run_recorded_query("query_catalog", sql, requester, turn, cfg)

    async def _run_recorded_query(
        self, kind: QueryKind, sql: str, requester: str, turn: int | None, cfg: SqlclGuardConfig
    ) -> dict:
        start = time.monotonic()

        def _elapsed_ms() -> int:
            return int((time.monotonic() - start) * 1000)

        try:
            async with self._sqlcl_lock:  # type: ignore[union-attr]
                result = await guarded_query(self._sqlcl, sql, cfg)  # type: ignore[arg-type]
        except GuardRejection as exc:
            self._ledger.record_query(kind, sql, requester, turn=turn,
                                      status="rejected", error=str(exc),
                                      duration_ms=_elapsed_ms())
            raise
        except Exception as exc:  # noqa: BLE001 — record, then propagate
            self._ledger.record_query(kind, sql, requester, turn=turn,
                                      status="error", error=str(exc),
                                      duration_ms=_elapsed_ms())
            raise
        evidence_id = self._ledger.record_query(
            kind, sql, requester, turn=turn, result=result,
            duration_ms=_elapsed_ms(),
        )
        if evidence_id is None:  # ledger failure — never block the result
            return result
        # evidence_id FIRST: the agent tool loop truncates serialized results,
        # and the ID must survive even when huge `rows` get sliced away.
        return {"evidence_id": evidence_id, **result}

    async def list_files(self) -> list[dict]:
        assert self._files is not None and self._files_lock is not None, "McpClient not entered"
        async with self._files_lock:
            result = await self._files.call_tool("list_files", {})
        return _extract_json_payload(result, default=[])

    async def _read_file_raw(self, path: str, offset: int = 0, max_bytes: int | None = None) -> dict:
        assert self._files is not None and self._files_lock is not None, "McpClient not entered"
        args: dict[str, Any] = {"path": path, "offset": offset}
        if max_bytes is not None:
            args["max_bytes"] = max_bytes
        async with self._files_lock:
            result = await self._files.call_tool("read_file", args)
        return _extract_json_payload(result, default={})

    async def read_file(
        self,
        path: str,
        offset: int = 0,
        max_bytes: int | None = None,
        requester: str = "agent",
        turn: int | None = None,
    ) -> dict:
        cached = self._ledger.lookup_read(path, offset, max_bytes, requester)
        if cached is not None:
            return cached
        start = time.monotonic()
        try:
            result = await self._read_file_raw(path, offset, max_bytes)
        except Exception as exc:  # noqa: BLE001
            self._ledger.record_read(path, offset, max_bytes, requester, turn=turn,
                                     status="error", error=str(exc),
                                     duration_ms=int((time.monotonic() - start) * 1000))
            raise
        evidence_id = self._ledger.record_read(
            path, offset, max_bytes, requester, turn=turn, result=result,
            duration_ms=int((time.monotonic() - start) * 1000),
        )
        if evidence_id is None:
            return result
        # evidence_id FIRST: survives tool-loop truncation of large content.
        return {"evidence_id": evidence_id, **result}

    async def _search_file_raw(
        self, path: str, pattern: str, max_matches: int = 200
    ) -> dict:
        assert self._files is not None and self._files_lock is not None, "McpClient not entered"
        async with self._files_lock:
            result = await self._files.call_tool(
                "search_file",
                {"path": path, "pattern": pattern, "max_matches": max_matches},
            )
        return _extract_json_payload(result, default={})

    async def search_file(
        self,
        path: str,
        pattern: str,
        max_matches: int = 200,
        requester: str = "agent",
        turn: int | None = None,
    ) -> dict:
        # lookup_search is pattern-first (intentional asymmetry vs record_search).
        cached = self._ledger.lookup_search(pattern, path, requester)
        if cached is not None:
            return cached
        start = time.monotonic()
        try:
            result = await self._search_file_raw(path, pattern, max_matches)
        except Exception as exc:  # noqa: BLE001
            self._ledger.record_search(path, pattern, requester, turn=turn,
                                       status="error", error=str(exc),
                                       duration_ms=int((time.monotonic() - start) * 1000))
            raise
        evidence_id = self._ledger.record_search(
            path, pattern, requester, turn=turn, result=result,
            duration_ms=int((time.monotonic() - start) * 1000),
        )
        if evidence_id is None:
            return result
        # evidence_id FIRST: survives tool-loop truncation.
        return {"evidence_id": evidence_id, **result}

    async def _extract_ora_raw(self, path: str) -> dict:
        assert self._files is not None and self._files_lock is not None, "McpClient not entered"
        async with self._files_lock:
            result = await self._files.call_tool("extract_ora", {"path": path})
        return _extract_json_payload(result, default={})

    async def scan_alertlog(self, path: str, requester: str = "ora_timeline") -> dict:
        """Extract ORA incidents from an alert log, recorded as an S-### access.

        Pattern is the sentinel ``"<ora-extract>"`` so the deterministic scan is
        distinguishable in the ledger from an LLM grep. Returns
        ``{evidence_id, incidents, coverage, ...}``. The probe runs pre-turn
        (diff_probe → ora_timeline → turn_1), so the record carries no turn tag."""
        pattern = "<ora-extract>"
        cached = self._ledger.lookup_search(pattern, path, requester)
        if cached is not None:
            return cached
        start = time.monotonic()
        try:
            result = await self._extract_ora_raw(path)
        except Exception as exc:  # noqa: BLE001
            self._ledger.record_search(path, pattern, requester,
                                       status="error", error=str(exc),
                                       duration_ms=int((time.monotonic() - start) * 1000))
            raise
        # match_count mirrors incident count for the ledger row.
        result_for_record = {**result, "match_count": len(result.get("incidents") or [])}
        evidence_id = self._ledger.record_search(
            path, pattern, requester, result=result_for_record,
            duration_ms=int((time.monotonic() - start) * 1000),
        )
        if evidence_id is None:
            return result
        return {"evidence_id": evidence_id, **result}

    def langchain_tools(
        self, requester: str = "unknown", turn: int | None = None
    ) -> list[BaseTool]:
        """Return the three tools, closed over the caller's identity so every
        access is attributed in the evidence ledger."""

        async def _query_awr(sql: str) -> dict:
            return await self.query_awr(sql, requester=requester, turn=turn)

        async def _list_files() -> list[dict]:
            return await self.list_files()

        async def _read_file(path: str, offset: int = 0, max_bytes: int | None = None) -> dict:
            return await self.read_file(
                path, offset=offset, max_bytes=max_bytes, requester=requester, turn=turn
            )

        async def _search_file(path: str, pattern: str, max_matches: int = 200) -> dict:
            return await self.search_file(
                path, pattern, max_matches=max_matches, requester=requester, turn=turn
            )

        return [
            StructuredTool.from_function(
                name="query_awr",
                description=(
                    "Run a read-only SQL query against Oracle DBA_HIST_* tables and "
                    "a small set of fixed views. Only SELECT/WITH statements are "
                    "allowed; queries outside the allowlist are rejected. Returns "
                    "{evidence_id, rows, truncated, row_count, warnings}. Cite the "
                    "evidence_id in evidence_refs for any finding this result "
                    "supports. Heed any warnings — they flag queries scanning "
                    "outside the case's snap windows."
                ),
                coroutine=_query_awr,
            ),
            StructuredTool.from_function(
                name="list_files",
                description=(
                    "List diagnostic files provided by the user for this case, grouped "
                    "by type (alertlog, trace, tfa, hanganalyze, sqlhc, "
                    "ddl, awr_report)."
                ),
                coroutine=_list_files,
            ),
            StructuredTool.from_function(
                name="read_file",
                description=(
                    "Read a slice of one of the files returned by list_files. Provide "
                    "'path' (from list_files), optional 'offset' (byte offset, default "
                    "0) and 'max_bytes' (default 65536, hard cap 1MB). Returns "
                    "{evidence_id, content, offset, bytes_read, eof, truncated, "
                    "total_size}. Cite the evidence_id in evidence_refs for any "
                    "finding this content supports."
                ),
                coroutine=_read_file,
            ),
            StructuredTool.from_function(
                name="search_file",
                description=(
                    "Regex-search one of the files from list_files. Provide 'path' "
                    "and a Python-regex 'pattern' (optional 'max_matches', default "
                    "200). Returns {evidence_id, matches, match_count, truncated, "
                    "total_size} where each match is {line_number, byte_offset, line}. "
                    "Use a match's 'byte_offset' as the 'offset' to read_file for "
                    "surrounding context. Cite the evidence_id in evidence_refs. "
                    "Ideal for locating ORA- errors / keywords in a large log "
                    "without reading it blindly."
                ),
                coroutine=_search_file,
            ),
        ]


def _extract_json_payload(raw: Any, default: Any) -> Any:
    content = getattr(raw, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue
    return default
