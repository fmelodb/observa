"""Thin wrapper around ``graph.astream_events`` that feeds a TUI event queue.

Supports the turn-gate ``interrupt()`` protocol: when a gate node pauses the
graph, the bridge emits a ``GraphEvent("interrupt", payload)`` and waits for the
TUI to call :py:meth:`resume` with the analyst's response. The payload is the
value passed to ``interrupt()`` — a dict ``{"type": "chat_gate", "turn": N,
"questions": [...]}``.

The MCP client is constructed by ``ObservaApp`` and passed in. The bridge
forwards it through ``RunnableConfig["configurable"]["mcp_client"]`` so each
node can pull it. ``None`` is acceptable — agents run with no tools.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.types import Command

logger = logging.getLogger(__name__)


class GraphEvent:
    def __init__(self, event_type: str, data: Any) -> None:
        # "node_start" | "node_end" | "custom_event" | "interrupt" | "done" | "error"
        self.event_type = event_type
        self.data = data

    def __repr__(self) -> str:  # pragma: no cover
        return f"GraphEvent({self.event_type!r}, {self.data!r})"


class GraphBridge:
    """Runs the graph in a background asyncio task; exposes events via a queue."""

    def __init__(
        self,
        graph: Any,
        initial_state: dict,
        case_id: str,
        mcp_client: Any = None,
    ) -> None:
        self._graph = graph
        self._initial_state = initial_state
        self._case_id = case_id
        self._config = {
            "configurable": {
                "thread_id": case_id,
                "mcp_client": mcp_client,
            }
        }
        self._queue: asyncio.Queue[GraphEvent] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._final_state: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Kick off graph execution in the background."""
        self._task = asyncio.create_task(self._run(self._initial_state))

    async def resume(self, response: dict) -> None:
        """Resume the graph after an interrupt with the analyst's response.

        ``response`` is passed through to ``langgraph.types.Command`` and
        becomes the return value of the ``interrupt()`` call in the paused
        gate node. Typical shape:
            ``{"messages": [{"sender": "user", "text": "..."}]}``
        """
        self._task = asyncio.create_task(self._run(Command(resume=response)))

    async def get_event(self, timeout: float = 0.1) -> GraphEvent | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def final_state(self) -> dict:
        return self._final_state

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run(self, stream_input: Any) -> None:
        logger.info(
            "GraphBridge: %s graph for case_id=%s",
            "starting" if stream_input is self._initial_state else "resuming",
            self._case_id,
        )
        final_state: dict = dict(self._initial_state)
        try:
            async for event in self._graph.astream_events(
                stream_input, config=self._config, version="v2"
            ):
                event_type = event.get("event", "")
                name = event.get("name", "")
                meta = event.get("metadata") or {}
                node_name = meta.get("langgraph_node")

                if event_type == "on_chain_start" and node_name and name == node_name:
                    await self._queue.put(GraphEvent("node_start", {"node": node_name}))
                elif event_type == "on_chain_end" and node_name and name == node_name:
                    output = event.get("data", {}).get("output") or {}
                    if isinstance(output, dict):
                        final_state.update(output)
                    await self._queue.put(
                        GraphEvent("node_end", {"node": node_name, "output": output})
                    )
                elif event_type == "on_custom_event":
                    await self._queue.put(
                        GraphEvent("custom_event", {"name": name, "data": event.get("data")})
                    )

            # Stream drained — either the graph finished or it's paused.
            state = await self._graph.aget_state(self._config)
            self._final_state = dict(state.values) if state and state.values else final_state

            interrupt_payload = self._extract_interrupt_payload(state)
            if interrupt_payload is not None:
                logger.info("GraphBridge: paused at interrupt (case_id=%s)", self._case_id)
                await self._queue.put(GraphEvent("interrupt", interrupt_payload))
                return

            if state and getattr(state, "next", ()):
                logger.warning(
                    "GraphBridge: graph paused without interrupt payload (next=%s) — "
                    "treating as done", state.next,
                )
            await self._queue.put(GraphEvent("done", {"state": self._final_state}))
            logger.info("GraphBridge: graph complete for case_id=%s", self._case_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("GraphBridge: graph failed")
            await self._queue.put(GraphEvent("error", str(exc)))

    @staticmethod
    def _extract_interrupt_payload(state: Any) -> Any | None:
        """Return the payload of the first pending interrupt, or None."""
        if state is None:
            return None
        tasks = getattr(state, "tasks", None) or ()
        for task in tasks:
            interrupts = getattr(task, "interrupts", None) or ()
            for intr in interrupts:
                value = getattr(intr, "value", None)
                if value is not None:
                    return value
        return None
