"""Screen 2: Live execution — streaming findings, master status, chat gate.

Findings stream in as each specialist finishes (via ``agent_done`` custom
events from ``turn_controller``), instead of batching at the end of the turn.

Layout (top → bottom):

    Header
    ┌──────────── #top_bar (height 6) ───────────────┐
    │ #top_left          │            #top_right     │
    │  - mascot (4 ln)   │           ⏵ MCP status    │
    │  - turn status     │                           │
    │  - node status     │                           │
    └────────────────────────────────────────────────┘
    #findings_log  (height 1fr)
    #telemetry
        #agent_grid  (per-agent spinner + bar + tokens)
        #flow_line   (Turn 1 → … → RCA)
    #gate_panel  (hidden until interrupt)
    Footer
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, RichLog, Static

from observa.llm import TokenTracker
from observa.tui.graph_bridge import GraphBridge
from observa.tui.widgets.agent_grid import AGENT_COLOR, render_agent_grid
from observa.tui.widgets.cache_footer import render_cache_footer
from observa.tui.widgets.findings_table import (
    format_abstain_row,
    format_finding_row,
    format_question_row,
    format_status_row,
)
from observa.tui.widgets.flow_line import node_to_step_index, render_flow_line
from observa.tui.widgets.mascot import build_mascot


# Mockup palette — keep in sync with docs/mockups-cli.html.
_CSS = """
LiveRunScreen {
    background: #151515;
}
LiveRunScreen #top_bar {
    height: 7;
    layout: horizontal;
}
LiveRunScreen #top_left {
    width: 50;
    height: 7;
    padding: 0 1;
}
LiveRunScreen #top_right {
    width: 1fr;
    height: 7;
    padding: 1 4 0 1;
    content-align: right top;
}
LiveRunScreen #mcp_status {
    height: 1;
    text-style: bold;
    color: #F5B041;
    text-align: right;
}
LiveRunScreen #mcp_status.ok {
    color: #A9DFBF;
    text-style: none;
}
LiveRunScreen #flow_spacer {
    height: 1fr;
}
LiveRunScreen #turn_status {
    height: 1;
    margin-top: 1;
    text-style: bold;
    color: #7FB3D5;
}
LiveRunScreen #node_status {
    height: 1;
    text-style: bold;
    color: #FFFFFF;
}
LiveRunScreen #findings_log {
    height: 1fr;
    background: #1a1a1a;
    border-top: solid #3a3a3a;
    border-bottom: solid #3a3a3a;
    padding: 0 1;
    overflow-x: hidden;
    scrollbar-background: #1a1a1a;
    scrollbar-color: #3a3a3a;
}
LiveRunScreen #telemetry {
    height: auto;
    padding: 1 1;
}
LiveRunScreen #agent_grid {
    height: auto;
    color: #d4d4d4;
}
LiveRunScreen #flow_line {
    height: 1;
    text-align: right;
}
LiveRunScreen #cache_footer {
    height: 1;
    text-align: right;
    color: #d4d4d4;
}
LiveRunScreen #gate_panel {
    display: none;
    height: auto;
    border-top: solid #F5B041;
    background: #2a2210;
    padding: 1 1;
}
LiveRunScreen #gate_panel.visible {
    display: block;
}
LiveRunScreen #gate_banner {
    height: 1;
    text-style: bold;
    color: #F5B041;
}
LiveRunScreen #gate_questions {
    height: auto;
    max-height: 10;
    color: #d4d4d4;
}
LiveRunScreen #gate_input {
    height: 3;
}
"""


# Severity colors and a per-agent fallback live next to the grid module so
# both files agree, but we still keep a local view for log line rendering.
_DEFAULT_AGENT_COLOR = "#7FB3D5"
_AGENT_LOG_COLOR: dict[str, str] = {
    **AGENT_COLOR,
    "master": "#F5B041",
    "graph": "#888888",
    "analyst": "#F5B041",
    "cove": "#A9DFBF",
}

_SEVERITY_COLOR: dict[str, str] = {
    "critical": "#E59866",
    "high": "#E59866",
    "medium": "#F5B041",
    "low": "#A9DFBF",
    "info": "#888888",
}


class LiveRunScreen(Screen):
    DEFAULT_CSS = _CSS

    BINDINGS = [
        Binding("ctrl+f", "forward_summary", "Forward to RCA"),
    ]

    _NODE_LABELS = {
        "init": "initializing",
        "topology_probe": "probing database topology…",
        "diff_probe": "computing problem vs. baseline deltas…",
        "ora_timeline": "extracting incident timeline from alert logs…",
        "consolidate": "Master reflecting on findings…",
        "detect_contradictions": "Master checking for contradictions…",
        "synthesize_hypotheses": "Master forming competing hypotheses…",
        "resolve_contradiction": "Master resolving contradictions on evidence…",
        "replay_verify": "Master replaying evidence to verify findings…",
    }

    def __init__(
        self,
        bridge: GraphBridge,
        case_id: str,
        mcp_error: str | None = None,
        token_tracker: TokenTracker | None = None,
        agents_order: list[str] | None = None,
        total_turns: int = 3,
        ledger: Any = None,
    ) -> None:
        self._bridge = bridge
        self._case_id = case_id
        self._mcp_error = mcp_error
        self._token_tracker = token_tracker
        self._agents_order = agents_order or []
        self._total_turns = total_turns
        self._ledger = ledger

        self._gate_pending: dict | None = None
        self._current_turn: int | None = None
        self._completed_agents_in_turn = 0
        self._total_findings = 0
        self._total_abstains = 0

        # Pipeline / spinner state.
        self._current_step_idx: int | None = None
        self._all_done = False
        self._spinner_frame = 0

        # Stored once the run finishes so we can re-push it via Ctrl+F when
        # the analyst comes back here for a visual review.
        self._summary_screen: Screen | None = None
        super().__init__()

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _agent_color(self, agent: str) -> str:
        return _AGENT_LOG_COLOR.get(agent, _DEFAULT_AGENT_COLOR)

    def _write(self, line: Text) -> None:
        self.query_one("#findings_log", RichLog).write(line)

    def _term_width(self) -> int:
        """Width available for findings_log content.

        Subtracts the widget's own padding (1 col each side) and border
        (1 col each side) from the screen width so pre-wrapped lines never
        overflow and trigger a horizontal scrollbar.
        """
        try:
            screen = self.size.width
        except Exception:  # noqa: BLE001
            return 120
        # padding 0 1 → 2 cols; border-top/bottom only, but defensive: -4 total.
        return max(60, screen - 4)

    def _stream_row(
        self, agent: str, message: str, *, severity: str | None = None
    ) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = format_status_row(
            ts, agent, message,
            severity=severity or "",
            term_width=self._term_width(),
        )
        self._write(line)

    def _turn_header(self, turn: int) -> None:
        self._write(Text(""))
        bar = Text()
        bar.append("── ", style="#3a3a3a")
        bar.append(f"Turn {turn}", style="bold #7FB3D5")
        bar.append(" ", style="")
        bar.append("─" * 60, style="#3a3a3a")
        self._write(bar)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        mcp_line = (
            f"⚠  MCP unavailable — {self._mcp_error}"
            if self._mcp_error
            else "● MCP connected"
        )
        yield Horizontal(
            Vertical(
                Static(build_mascot(compact=False), id="hero_compact"),
                Static("Turns: waiting…", id="turn_status"),
                Static("", id="node_status"),
                id="top_left",
            ),
            Vertical(
                Static(
                    mcp_line,
                    id="mcp_status",
                    classes="ok" if not self._mcp_error else "",
                ),
                Static("", id="flow_spacer"),
                Static("", id="flow_line"),
                id="top_right",
            ),
            id="top_bar",
        )
        yield RichLog(
            id="findings_log",
            highlight=False,
            markup=False,
            # wrap=False keeps each row on a single physical line so the
            # 5-column alignment holds. Long descriptions are truncated by
            # findings_table.format_*_row to fit the terminal width.
            wrap=False,
            auto_scroll=True,
        )
        yield Vertical(
            Static("", id="agent_grid"),
            Static("", id="cache_footer"),
            id="telemetry",
        )
        yield Vertical(
            Static("● Agents have questions — answer to continue", id="gate_banner"),
            Static("", id="gate_questions"),
            Input(placeholder="Your response (press Enter to send)…", id="gate_input"),
            id="gate_panel",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._bridge.start()
        self.set_interval(0.1, self._poll_events)
        # Spinner tick — refreshes only the telemetry widgets.
        self.set_interval(0.15, self._tick_spinner)
        # Periodic full refresh — captures token deltas without an event.
        self.set_interval(5.0, self._refresh_telemetry)
        self._refresh_telemetry()

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _tick_spinner(self) -> None:
        self._spinner_frame = (self._spinner_frame + 1) % 1000
        # Always re-render both — keeps the flow line visible from frame 1
        # (otherwise an idle Static can stay blank if its initial query_one
        # raced the compose pass) and keeps spinners animating.
        self._refresh_agent_grid()
        self._refresh_flow_line()

    def _refresh_telemetry(self) -> None:
        self._refresh_agent_grid()
        self._refresh_flow_line()
        self._refresh_cache_footer()

    def _evidence_counters(self) -> dict[str, tuple[int, int]] | None:
        # Grid cells are per-AGENT only; footer totals are case-wide (probes,
        # master chat, etc. included) — the mismatch is by design.
        if self._ledger is None:
            return None
        try:
            return {
                name: (c.accesses, c.cache_hits)
                for name, c in self._ledger.counters().items()
            }
        except Exception:  # noqa: BLE001 — telemetry must never crash the screen
            return None

    def _refresh_agent_grid(self) -> None:
        if not self._agents_order:
            return
        by_agent = (
            self._token_tracker.by_agent if self._token_tracker is not None else {}
        )
        # The evidence segment adds 9 cols/cell; a 3-column row then needs
        # ~137 display cols. Below that, drop the segment instead of letting
        # the row soft-wrap and shred the grid alignment.
        try:
            wide_enough = self.size.width >= 140
        except Exception:  # noqa: BLE001
            wide_enough = False
        counters = self._evidence_counters() if wide_enough else None
        text = render_agent_grid(
            by_agent, self._agents_order, self._spinner_frame, columns=3,
            evidence_counters=counters,
        )
        try:
            self.query_one("#agent_grid", Static).update(text)
        except Exception:
            pass

    def _refresh_flow_line(self) -> None:
        text = render_flow_line(
            total_turns=self._total_turns,
            current_step_idx=self._current_step_idx,
            all_done=self._all_done,
            spinner_frame=self._spinner_frame,
        )
        try:
            self.query_one("#flow_line", Static).update(text)
        except Exception:
            pass

    def _refresh_cache_footer(self) -> None:
        by_agent = (
            self._token_tracker.by_agent if self._token_tracker is not None else {}
        )
        totals = None
        if self._ledger is not None:
            try:
                totals = self._ledger.totals()
            except Exception:  # noqa: BLE001
                totals = None
        text = render_cache_footer(by_agent, evidence_totals=totals)
        try:
            self.query_one("#cache_footer", Static).update(text)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def _label_for_node(self, node: str) -> str:
        if node in self._NODE_LABELS:
            return self._NODE_LABELS[node]
        if node.startswith("turn_"):
            return f"Running {node.replace('_', ' ')}"
        if node.startswith("gate_"):
            return f"Chat gate {node.split('_', 1)[1]}"
        return node

    async def _poll_events(self) -> None:
        event = await self._bridge.get_event()
        if event is None:
            return
        if event.event_type == "node_start":
            node = event.data.get("node", "?")
            label = self._label_for_node(node)
            self.query_one("#node_status", Static).update(f"▶ {label}")
            idx = node_to_step_index(node, self._total_turns)
            if idx is not None:
                self._current_step_idx = idx
            if node.startswith("turn_"):
                turn_num = int(node.split("_", 1)[1])
                self._current_turn = turn_num
                self._completed_agents_in_turn = 0
                self._turn_header(turn_num)
            elif node == "consolidate":
                self._stream_row("master", "reflecting on the blackboard…")
            elif node == "detect_contradictions":
                self._stream_row("master", "checking findings for contradictions…")
            elif node == "synthesize_hypotheses":
                self._stream_row("master", "forming competing hypotheses…")
            elif node == "resolve_contradiction":
                self._stream_row("master", "resolving contradictions on evidence…")
            elif node == "replay_verify":
                self._stream_row("master", "replaying evidence to verify findings…")
            self._refresh_flow_line()
        elif event.event_type == "node_end":
            node = event.data.get("node", "?")
            output = event.data.get("output") or {}
            label = self._label_for_node(node)
            self.query_one("#node_status", Static).update(f"✓ {label}")
            self._render_node_end(node, output)
        elif event.event_type == "interrupt":
            self._open_gate(event.data)
        elif event.event_type == "done":
            state = event.data.get("state") or {}
            self.query_one("#node_status", Static).update("✓ analysis complete")
            self._stream_row("graph", "complete — opening summary")
            self._current_step_idx = self._total_turns + 2  # RCA
            self._all_done = True
            self._refresh_telemetry()
            self.set_timer(0.3, lambda: self._push_summary(state))
        elif event.event_type == "error":
            self._stream_row("graph", f"ERROR: {event.data}", severity="critical")
        elif event.event_type == "custom_event":
            name = event.data.get("name", "?")
            data = event.data.get("data") or {}
            if name == "agent_done" and isinstance(data, dict):
                self._render_agent_done(data)
                self._refresh_agent_grid()
            else:
                self._stream_row(name, str(data)[:200])

    # ------------------------------------------------------------------
    # Agent-stream rendering (called per-agent as they finish)
    # ------------------------------------------------------------------

    def _render_agent_done(self, data: dict) -> None:
        agent = str(data.get("agent", "?"))
        turn = data.get("turn")
        if isinstance(turn, int) and turn != self._current_turn:
            self._current_turn = turn
            self._completed_agents_in_turn = 0
            self._turn_header(turn)
        self._completed_agents_in_turn += 1

        abstained = bool(data.get("abstained"))
        findings = data.get("findings") or []
        questions = data.get("questions") or []

        tw = self._term_width()
        if abstained and not findings:
            reason = data.get("abstain_reason") or "nothing relevant"
            self._write(format_abstain_row(
                datetime.now().strftime("%H:%M:%S"), agent, reason, term_width=tw,
            ))
            self._write(Text(""))   # blank line between blocks
        else:
            for f in findings:
                self._write(format_finding_row(
                    datetime.now().strftime("%H:%M:%S"),
                    agent,
                    str(f.get("severity", "")),
                    f.get("code") or "",
                    f.get("description") or "",
                    term_width=tw,
                ))
                self._write(Text(""))   # blank line between blocks
                self._total_findings += 1
        if abstained:
            self._total_abstains += 1

        for q in questions:
            self._write(format_question_row(
                datetime.now().strftime("%H:%M:%S"),
                agent,
                q.get("question", ""),
                term_width=tw,
            ))
            self._write(Text(""))

        if isinstance(turn, int):
            self.query_one("#turn_status", Static).update(
                f"Turn {turn} · {self._total_findings} findings · "
                f"{self._total_abstains} abstains so far"
            )

    def _render_node_end(self, node: str, output: dict) -> None:
        if node.startswith("turn_"):
            tfs = output.get("turn_findings") or []
            turn_num = tfs[0].turn if tfs else self._current_turn or "?"
            findings_count = sum(len(tf.findings) for tf in tfs)
            abstain_count = sum(1 for tf in tfs if tf.abstained)
            self.query_one("#turn_status", Static).update(
                f"Turn {turn_num} · {findings_count} findings · {abstain_count} abstains"
            )
        elif node == "consolidate":
            summary = output.get("final_summary")
            if summary is not None:
                self._stream_row(
                    "master", f"summary ready — {summary.root_cause[:120]}",
                    severity="low",
                )
        elif node == "replay_verify":
            verification = output.get("verification")
            if verification is not None:
                self._stream_row(
                    "cove",
                    f"support={verification.support_score:.2f} "
                    f"(×{verification.applied_confidence_factor:.2f} confidence)",
                    severity="low",
                )

    # ------------------------------------------------------------------
    # Chat gate
    # ------------------------------------------------------------------

    def _open_gate(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            self._stream_row(
                "graph", f"WARN: unexpected interrupt shape: {type(payload).__name__}",
                severity="medium",
            )
            return
        self._gate_pending = payload
        turn = payload.get("turn", "?")
        questions = payload.get("questions") or []
        lines = Text()
        lines.append(f"Turn {turn} — agents raised {len(questions)} question(s):\n",
                     style="bold #F5B041")
        for q in questions:
            lines.append("  · ", style="#666666")
            lines.append(f"[{q.get('agent', '?')}]",
                         style=f"bold {self._agent_color(q.get('agent', ''))}")
            lines.append(f" {q.get('question', '')}\n", style="#d4d4d4")
        lines.append("\nType one response covering all questions, or leave blank to skip.",
                     style="#888888")
        self.query_one("#gate_questions", Static).update(lines)
        panel = self.query_one("#gate_panel", Vertical)
        panel.add_class("visible")
        self.query_one("#gate_input", Input).focus()

    def _close_gate(self) -> None:
        self._gate_pending = None
        self.query_one("#gate_panel", Vertical).remove_class("visible")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "gate_input" or self._gate_pending is None:
            return
        text = event.value.strip()
        event.input.clear()
        self._close_gate()
        if text:
            self._stream_row("analyst", f"→ {text}")
            response = {"messages": [{"sender": "user", "text": text}]}
        else:
            self._stream_row("analyst", "(no response — resuming)", severity="info")
            response = {"messages": []}
        await self._bridge.resume(response)

    # ------------------------------------------------------------------

    _SUMMARY_SCREEN_NAME = "summary"

    def _push_summary(self, state: dict) -> None:
        """Build the summary screen ONCE, install it under a known name, push it.

        Installing (vs. pushing a bare instance) keeps the same Screen object
        alive across pop/push cycles. ``push_screen(name)`` re-mounts it
        cleanly — pushing the raw instance after a pop tries to re-mount
        already-destroyed widgets and stalls or no-ops.
        """
        from observa.tui.summary_chat import SummaryChatScreen

        self._summary_screen = SummaryChatScreen(state=state, bridge=self._bridge, ledger=self._ledger)
        try:
            self.app.uninstall_screen(self._SUMMARY_SCREEN_NAME)
        except Exception:  # noqa: BLE001 — first call, nothing to uninstall
            pass
        self.app.install_screen(self._summary_screen, name=self._SUMMARY_SCREEN_NAME)
        self.app.push_screen(self._SUMMARY_SCREEN_NAME)

    def action_forward_summary(self) -> None:
        """Re-open the RCA/summary screen after the analyst popped back here."""
        if self._summary_screen is None:
            return
        # Push by NAME — Textual reuses the installed instance and re-mounts
        # it without going through the slow/broken path of remounting an
        # un-installed instance.
        self.app.push_screen(self._SUMMARY_SCREEN_NAME)
