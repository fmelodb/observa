"""Main Textual application for Observa."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from textual.app import App, ComposeResult

from observa.tui.new_case import CaseStartRequested, NewCaseScreen
from observa.tui.summary_chat import SummaryChatScreen

logger = logging.getLogger(__name__)


class ObservaApp(App):
    """Main Textual app."""

    CSS_PATH = None
    BINDINGS = [("q", "quit", "Quit")]
    SCREENS = {
        "new_case": NewCaseScreen,
    }

    def __init__(self) -> None:
        super().__init__()
        self._mcp_client: Any = None
        self._mcp_task: asyncio.Task | None = None
        # The MCP stdio client uses anyio cancel scopes internally — they must
        # be entered AND exited from the same asyncio task. The lifecycle task
        # therefore owns enter, attach_files AND exit; other tasks talk to it
        # through ``_mcp_commands`` (None = shutdown sentinel).
        self._mcp_commands: asyncio.Queue = asyncio.Queue()
        self._mcp_ready: asyncio.Event = asyncio.Event()
        self._mcp_error: str | None = None

    def on_mount(self) -> None:
        from observa.config import get_settings

        try:
            theme = get_settings().ui_theme or "flexoki"
        except Exception:
            theme = "flexoki"
        try:
            self.theme = theme
        except Exception:
            logger.warning("theme %r failed", theme, exc_info=True)
        # Boot the MCP client NOW (fire-and-forget) so the intake wizard can
        # query the repository catalog while the analyst reads the screen.
        # NOTE: intentionally NOT self.run_worker() — Textual would cancel
        # workers at shutdown inside its own cancel scope, violating the anyio
        # rule that cancel scopes must exit on the task that entered them.
        # Shutdown is signalled instead via the None sentinel in on_unmount.
        self._mcp_task = asyncio.create_task(self._mcp_lifecycle())
        self.push_screen("new_case")

    def compose(self) -> ComposeResult:
        return []

    # ------------------------------------------------------------------
    # MCP lifecycle (single owner task)
    # ------------------------------------------------------------------

    # NOTE: the token_tracker ContextVar is set per-case later (in
    # on_case_start_requested). This task's context snapshot is taken at app
    # boot, so it always sees token_tracker=None — do not call
    # current_token_tracker() from inside this task.
    async def _mcp_lifecycle(self) -> None:
        from observa.config import get_settings
        from observa.mcp.client import McpClient

        settings = get_settings()
        logger.info("MCP bootstrap (app boot): sqlcl_path=%r", settings.mcp_sqlcl_path)
        try:
            async with McpClient(settings) as client:
                self._mcp_client = client
                self._mcp_ready.set()
                while True:
                    cmd = await self._mcp_commands.get()
                    if cmd is None:
                        break
                    entries, done = cmd
                    try:
                        await client.attach_files(entries)
                    except Exception as exc:  # noqa: BLE001
                        self._mcp_error = f"attach_files: {type(exc).__name__}: {exc}"
                        logger.error("MCP attach_files FAILED: %s", self._mcp_error, exc_info=True)
                    finally:
                        done.set()
        except Exception as exc:  # noqa: BLE001
            self._mcp_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "MCP lifecycle FAILED — intake falls back to manual entry and "
                "agents will run without tools. %s",
                self._mcp_error, exc_info=True,
            )
        finally:
            self._mcp_client = None
            self._mcp_ready.set()

    async def wait_mcp(self) -> tuple[Any, str | None]:
        """Await MCP readiness. Returns (client_or_None, error_or_None)."""
        await self._mcp_ready.wait()
        return self._mcp_client, self._mcp_error

    async def on_case_start_requested(self, message: CaseStartRequested) -> None:
        """Build initial CaseState, attach case files to MCP, launch graph."""
        from observa.agents import REGISTRY as AGENT_REGISTRY
        from observa.config import get_settings
        from observa.graph.master import build_graph
        from observa.llm import TokenTracker, token_tracker
        from observa.mcp.client import InputFileEntry
        from observa.tui.graph_bridge import GraphBridge
        from observa.tui.live_run import LiveRunScreen

        settings = get_settings()
        data = message.data
        now = datetime.now(timezone.utc)
        case_id = f"CASE-{now.strftime('%Y-%m%d')}-01"

        initial_state: dict = {
            "case_id": case_id,
            "created_at": now,
            "dbid": data.get("dbid", 0),
            "oracle_version": data.get("oracle_version", ""),
            "topology": "",
            "instance_numbers": [],
            "problem_statement": data.get("problem_statement", ""),
            "investigation_scope": data.get("investigation_scope", ""),
            "known_facts": data.get("known_facts", []),
            "input_files": data.get("input_files", []),
            "problem_window": data.get("problem_window"),
            "baseline_window": data.get("baseline_window"),
            "baseline_diff": None,
            "baseline_suspect": False,
            "ora_timeline": None,
            "current_turn": 0,
            "total_turns": settings.turns_count,
            "turn_findings": [],
            "agent_questions": [],
            "chat_log": [],
            "hypotheses": [],
            "contradictions": [],
            "contradiction_resolutions": [],
            "final_summary": None,
            "verification": None,
        }

        # Boot error comes back from wait_mcp; attach_files failures land in
        # self._mcp_error only after done.wait(), so the field is re-read below.
        mcp_client, _ = await self.wait_mcp()
        if mcp_client is not None:
            entries = [
                InputFileEntry(type=f.type, path=f.path, label=f.label)
                for f in initial_state["input_files"]
            ]
            done = asyncio.Event()
            await self._mcp_commands.put((entries, done))
            await done.wait()
            mcp_client.set_case_context(int(initial_state["dbid"] or 0))
            logger.info(
                "MCP case context set: dbid=%s files=%d",
                initial_state["dbid"], len(entries),
            )
        mcp_error = self._mcp_error

        graph = build_graph()
        bridge = GraphBridge(graph, initial_state, case_id=case_id, mcp_client=mcp_client)

        tracker = TokenTracker()
        token_tracker.set(tracker)

        agents_order = [
            name for name, enabled in settings.agents_enabled.items()
            if enabled and name in AGENT_REGISTRY
        ]

        self.push_screen(
            LiveRunScreen(
                bridge=bridge,
                case_id=case_id,
                mcp_error=mcp_error,
                token_tracker=tracker,
                agents_order=agents_order,
                total_turns=settings.turns_count,
                ledger=(mcp_client.ledger if mcp_client is not None else None),
            )
        )

    async def on_unmount(self) -> None:
        """Close MCP client on exit by sending the shutdown sentinel."""
        await self._mcp_commands.put(None)
        if self._mcp_task is not None:
            try:
                await self._mcp_task
            except Exception:  # noqa: BLE001
                logger.warning("mcp teardown failed", exc_info=True)
            self._mcp_task = None
