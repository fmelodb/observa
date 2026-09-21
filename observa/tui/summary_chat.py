"""Screen 3: Final Summary + post-analysis chat with the Master.

Left column: the ``FinalSummary`` rendered as a stack of mockup-style colored
panels (root cause, findings, unknowns, next steps). Right column: a chat
where each turn is a colored card — amber-tinted for analyst messages,
blue-tinted for Master replies — with Markdown rendering so tables, fenced
SQL blocks, and lists come through structured rather than as raw text.

A single contextual toggle (``⤢`` / ``⤡``) at the right of the chat header
alternates between balanced and full-width chat. The third resize state
(full-width summary, hiding the chat) is reachable via Ctrl+→.
"""
from __future__ import annotations

import pathlib
from datetime import datetime, timezone
from typing import Any, Literal

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Markdown, Static

from observa.export import render_html
from observa.models import ChatMessage, FinalSummary
from observa.tui.widgets.evidence_browser import evidence_rows, format_record_detail


_CSS = """
SummaryChatScreen {
    background: #151515;
}
SummaryChatScreen #body {
    height: 1fr;
}
SummaryChatScreen #summary_scroll {
    width: 55%;
    padding: 0 1;
}
SummaryChatScreen #chat_col {
    width: 45%;
    padding: 0 1;
    border-left: solid #3a3a3a;
}
SummaryChatScreen .panel {
    border: solid #3a3a3a;
    padding: 0 1;
    margin: 0 0 1 0;
    background: #1a1a1a;
}
SummaryChatScreen .panel-green {
    border: solid #A9DFBF;
    background: #0f1a10;
}
SummaryChatScreen .panel-amber {
    border: solid #F5B041;
    background: #2a2210;
}
SummaryChatScreen .panel-coral {
    border: solid #E59866;
    background: #1a1410;
}
SummaryChatScreen .panel-hdr {
    color: #888888;
    text-style: bold;
    height: 1;
    padding: 0;
}
SummaryChatScreen .panel-hdr.green { color: #A9DFBF; }
SummaryChatScreen .panel-hdr.amber { color: #F5B041; }
SummaryChatScreen .panel-hdr.coral { color: #E59866; }
SummaryChatScreen .panel-hdr.blue  { color: #7FB3D5; }
SummaryChatScreen .panel-body {
    height: auto;
    color: #d4d4d4;
    padding: 0;
}
SummaryChatScreen #summary_scroll {
    height: 1fr;
    scrollbar-background: #1a1a1a;
    scrollbar-color: #3a3a3a;
}
SummaryChatScreen #chat_log {
    height: 1fr;
    padding: 0 1;
    background: #151515;
    scrollbar-background: #1a1a1a;
    scrollbar-color: #3a3a3a;
}
SummaryChatScreen #chat_input {
    height: 3;
    background: #0d0d0d;
}
SummaryChatScreen #chat_header {
    height: 1;
}
SummaryChatScreen #chat_title {
    width: 1fr;
    color: #7FB3D5;
    text-style: bold;
    height: 1;
    padding: 0;
}
SummaryChatScreen #chat_toggle {
    width: 3;
    min-width: 3;
    height: 1;
    border: none;
    background: transparent;
    color: #888888;
}
SummaryChatScreen #chat_toggle:hover {
    color: #d4d4d4;
}
SummaryChatScreen .msg {
    height: auto;
    margin: 0 0 1 0;
    padding: 0 1;
    border-left: thick #3a3a3a;
}
SummaryChatScreen .msg-user {
    background: #1a1810;
    border-left: thick #F5B041;
}
SummaryChatScreen .msg-master {
    background: #101820;
    border-left: thick #7FB3D5;
}
SummaryChatScreen .msg-role {
    height: 1;
    text-style: bold;
}
SummaryChatScreen .msg-user .msg-role { color: #F5B041; }
SummaryChatScreen .msg-master .msg-role { color: #7FB3D5; }
SummaryChatScreen .msg Markdown {
    background: transparent;
    padding: 0;
    margin: 0;
}
SummaryChatScreen .msg MarkdownFence {
    background: #0d0d0d;
    margin: 1 0;
}
SummaryChatScreen #evidence_col {
    display: none;
    width: 45%;
    padding: 0 1;
    border-left: solid #3a3a3a;
}
SummaryChatScreen #evidence_col.visible { display: block; }
SummaryChatScreen #evidence_title { height: 1; color: #7FB3D5; text-style: bold; }
SummaryChatScreen #evidence_filter { height: 3; background: #0d0d0d; }
SummaryChatScreen #evidence_table { height: 1fr; }
SummaryChatScreen #evidence_detail_scroll {
    height: 14;
    border-top: solid #3a3a3a;
    scrollbar-background: #1a1a1a;
    scrollbar-color: #3a3a3a;
}
"""


_SEVERITY_COLOR = {
    "critical": "#E59866",
    "high": "#E59866",
    "medium": "#F5B041",
    "low": "#A9DFBF",
    "info": "#888888",
}


class SummaryChatScreen(Screen):
    DEFAULT_CSS = _CSS

    # Text selection: drag the mouse across summary panels or the chat log to
    # select; ``Ctrl+C`` copies the selection (Textual's app-level
    # ``action_copy_text``). When the chat ``Input`` has focus, ``Ctrl+C``
    # falls through to the input's own copy.
    BINDINGS = [
        Binding("ctrl+b", "back_live", "Back to Live Run"),
        # priority=True: the chat Input (focused by default) binds ctrl+e to
        # "go to end of line" and would otherwise swallow the export key.
        Binding("ctrl+e", "export_html", "Export report", priority=True),
        Binding("ctrl+d", "toggle_evidence", "Evidence", priority=True),
        Binding("ctrl+shift+m", "export_md", "Export markdown"),
        Binding("ctrl+shift+e", "export_chat", "Export chat"),
        Binding("ctrl+left", "chat_wider", "← wider chat", priority=True, show=False),
        Binding("ctrl+right", "chat_narrower", "narrower chat →", priority=True, show=False),
        Binding("ctrl+c", "copy_text", "Copy", priority=False),
        Binding("ctrl+n", "new_case", "New case"),
        Binding("q", "quit", "Quit"),
    ]

    def action_back_live(self) -> None:
        """Return to the (already-finished) LiveRunScreen for visual review."""
        self.app.pop_screen()

    def __init__(self, state: dict, bridge: Any = None, ledger: Any = None) -> None:
        self._state: dict = state or {}
        self._bridge = bridge
        self._ledger = ledger
        self._history: list[ChatMessage] = list(self._state.get("chat_log") or [])
        self._layout_state: Literal["balanced", "chat_max", "summary_max"] = "balanced"
        super().__init__()

    # -----------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            VerticalScroll(
                *self._summary_panels(),
                id="summary_scroll",
            ),
            Vertical(
                Horizontal(
                    Static("▎ Chat with Master", id="chat_title"),
                    Button("⤢", id="chat_toggle"),
                    id="chat_header",
                ),
                VerticalScroll(id="chat_log"),
                Input(placeholder="Ask the Master a follow-up…", id="chat_input"),
                id="chat_col",
            ),
            Vertical(
                Static("▎ Evidence Ledger", id="evidence_title"),
                Input(placeholder="filter by agent / kind / id…", id="evidence_filter"),
                DataTable(id="evidence_table", cursor_type="row", zebra_stripes=True),
                VerticalScroll(Static("", id="evidence_detail"), id="evidence_detail_scroll"),
                id="evidence_col",
            ),
            id="body",
        )
        yield Footer()

    def on_mount(self) -> None:
        for msg in self._history:
            self._mount_message(msg)

    # -----------------------------------------------------------------
    # Summary rendering — stacked panels
    # -----------------------------------------------------------------

    def _summary_panels(self) -> list[Any]:
        summary: FinalSummary | None = self._state.get("final_summary")
        panels: list[Any] = []

        # Header block (problem + context).
        panels.append(self._panel(
            "▎ CASE CONTEXT",
            self._header_body(),
            hdr_class="blue",
        ))

        if summary is None:
            panels.append(self._panel(
                "▎ SUMMARY",
                Static(
                    Text("No summary produced — the analysis may have aborted.",
                         style="#888888"),
                    classes="panel-body",
                ),
            ))
            return panels

        # Primary root cause — green highlight panel.
        rc_body = Text()
        rc_body.append(summary.root_cause, style="#d4d4d4")
        rc_body.append("\n")
        rc_body.append(f"confidence {summary.confidence:.0%}",
                       style="#A9DFBF")
        panels.append(self._panel(
            f"▎ PRIMARY ROOT CAUSE · confidence {summary.confidence:.0%}",
            Static(rc_body, classes="panel-body"),
            border_class="panel-green",
            hdr_class="green",
        ))

        # Top findings — one line per finding, severity-colored.
        body = Text()
        if summary.top_findings:
            for i, f in enumerate(summary.top_findings):
                sev = f.severity.lower()
                sev_color = _SEVERITY_COLOR.get(sev, "#d4d4d4")
                body.append(f"[{i + 1}] ", style="#666666")
                body.append(f.severity.upper(), style=f"bold {sev_color}")
                body.append("  ", style="")
                body.append(f"({f.agent})", style="#7FB3D5")
                body.append(" ", style="")
                body.append(f.description, style="#d4d4d4")
                if f.evidence:
                    body.append("\n      evidence: ", style="#666666")
                    body.append(f.evidence, style="#888888")
                if f.evidence_refs:
                    body.append("\n      refs: ", style="#666666")
                    body.append(" ".join(f"[{r}]" for r in f.evidence_refs),
                                style="#7FB3D5")
                else:
                    body.append("\n      ○ uncited", style="#888888")
                if i < len(summary.top_findings) - 1:
                    body.append("\n")
        else:
            body.append("(no findings)", style="#888888")
        panels.append(self._panel(
            f"▎ TOP FINDINGS ({len(summary.top_findings)})",
            Static(body, classes="panel-body"),
        ))

        # Unknowns — amber panel.
        if summary.unknowns:
            u_body = Text()
            for i, u in enumerate(summary.unknowns):
                u_body.append("· ", style="#F5B041")
                u_body.append(u, style="#d4d4d4")
                if i < len(summary.unknowns) - 1:
                    u_body.append("\n")
            panels.append(self._panel(
                "▎ UNKNOWNS",
                Static(u_body, classes="panel-body"),
                border_class="panel-amber",
                hdr_class="amber",
            ))

        # Recommended next steps.
        if summary.recommended_next_steps:
            s_body = Text()
            for i, s in enumerate(summary.recommended_next_steps):
                s_body.append(f"R{i + 1} ", style="bold #A9DFBF")
                s_body.append(s, style="#d4d4d4")
                if i < len(summary.recommended_next_steps) - 1:
                    s_body.append("\n")
            panels.append(self._panel(
                "▎ RECOMMENDED NEXT STEPS",
                Static(s_body, classes="panel-body"),
                hdr_class="green",
            ))

        return panels

    def _header_body(self) -> Static:
        body = Text()
        body.append("problem ", style="#888888")
        body.append(self._state.get("problem_statement", "") or "—", style="#d4d4d4")
        body.append("\n")
        body.append("scope   ", style="#888888")
        body.append(self._state.get("investigation_scope", "") or "—", style="#d4d4d4")
        body.append("\n")
        body.append("dbid    ", style="#888888")
        body.append(str(self._state.get("dbid", "?")), style="#7FB3D5")
        body.append("   oracle ", style="#888888")
        body.append(str(self._state.get("oracle_version", "?")), style="#d4d4d4")
        body.append("   topo ", style="#888888")
        body.append(str(self._state.get("topology", "?")), style="#d4d4d4")
        return Static(body, classes="panel-body")

    def _panel(
        self,
        title: str,
        body: Any,
        *,
        border_class: str = "",
        hdr_class: str = "",
    ) -> Vertical:
        classes = "panel"
        if border_class:
            classes = f"panel {border_class}"
        hdr_cls = f"panel-hdr {hdr_class}".strip()
        return Vertical(
            Static(title, classes=hdr_cls),
            body,
            classes=classes,
        )

    # -----------------------------------------------------------------
    # Chat plumbing
    # -----------------------------------------------------------------

    def _mount_message(self, msg: ChatMessage) -> None:
        """Mount one chat turn as a colored card in the scroll panel.

        User turns get the amber-tinted card, master turns the blue-tinted one.
        The body is a Markdown widget so tables, fenced code blocks (SQL), and
        lists render structured rather than dumped as raw text.
        """
        role_class = "msg-user" if msg.sender == "user" else "msg-master"
        role_label = "you" if msg.sender == "user" else "master"
        container = Vertical(
            Static(role_label, classes="msg-role"),
            Markdown(msg.text or ""),
            classes=f"msg {role_class}",
        )
        log = self.query_one("#chat_log", VerticalScroll)
        log.mount(container)
        self.call_after_refresh(lambda: log.scroll_end(animate=False))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle a chat-input submission.

        Synchronous on purpose: we render the user message + ``thinking…``
        line immediately, then dispatch the LLM round-trip to a Textual
        worker so this handler returns right away. If we awaited the LLM
        here, Textual would not get a chance to paint the user message
        until the reply arrived — everything would appear at the end.
        """
        if event.input.id != "chat_input":
            return
        text = event.value.strip()
        event.input.clear()
        if not text:
            return
        now = datetime.now(timezone.utc)
        user_msg = ChatMessage(sender="user", turn=None, text=text, timestamp=now)
        self._history.append(user_msg)
        self._mount_message(user_msg)

        # Painted before the worker begins so the user sees an immediate
        # "is anyone home?" signal. Removed in _respond_to once the real
        # reply is ready.
        log = self.query_one("#chat_log", VerticalScroll)
        thinking = Vertical(
            Static("master", classes="msg-role"),
            Markdown("_thinking…_"),
            classes="msg msg-master",
            id="chat-thinking",
        )
        log.mount(thinking)
        self.call_after_refresh(lambda: log.scroll_end(animate=False))

        # Detach the LLM round-trip — handler returns now, paint happens.
        self.run_worker(
            self._respond_to(text), exclusive=True, group="chat",
        )

    async def _respond_to(self, text: str) -> None:
        reply = await self._ask_master(text)
        master_msg = ChatMessage(
            sender="master", turn=None, text=reply,
            timestamp=datetime.now(timezone.utc),
        )
        self._history.append(master_msg)
        try:
            self.query_one("#chat-thinking").remove()
        except Exception:  # noqa: BLE001
            pass
        self._mount_message(master_msg)

    async def _ask_master(self, text: str) -> str:
        """Run one master-chat round-trip; gracefully degrade when plumbing is absent."""
        try:
            from observa.graph.master_chat import master_chat
            from observa.llm import make_llm
            from observa.config import get_settings
        except Exception as exc:  # noqa: BLE001
            return f"(chat unavailable: {exc})"

        settings = get_settings()
        llm = make_llm(settings.model_for_master_chat(), slot="master_chat")
        mcp_client = (
            self._bridge and self._bridge._config.get("configurable", {}).get("mcp_client")
        ) if self._bridge else None
        tools = list(mcp_client.langchain_tools(requester="master_chat")) if mcp_client is not None else []

        try:
            return await master_chat(
                self._state,
                self._history[:-1],  # master_chat re-appends the latest user msg
                text,
                llm,
                tools,
            )
        except Exception as exc:  # noqa: BLE001
            return f"(master error: {type(exc).__name__}: {exc})"

    # -----------------------------------------------------------------
    # Evidence Browser
    # -----------------------------------------------------------------

    _EVIDENCE_COLUMNS = ("id", "who", "t", "tool", "target", "rows", "ms", "st")

    def action_toggle_evidence(self) -> None:
        col = self.query_one("#evidence_col")
        chat = self.query_one("#chat_col")
        showing = col.has_class("visible")
        if showing:
            self._apply_layout()  # restores chat/summary per _layout_state, drops .visible
            if self._layout_state != "summary_max":
                self.query_one("#chat_input", Input).focus()
            return
        summary_col = self.query_one("#summary_scroll")
        summary_col.styles.display = "block"
        summary_col.styles.width = "55%"
        chat.styles.display = "none"
        col.add_class("visible")
        self._populate_evidence()
        self.query_one("#evidence_filter", Input).focus()

    def _populate_evidence(self, filter_text: str = "") -> None:
        table = self.query_one("#evidence_table", DataTable)
        table.clear(columns=True)
        table.add_columns(*self._EVIDENCE_COLUMNS)
        records = self._ledger.records if self._ledger is not None else []
        rows = evidence_rows(records, filter_text)
        for row in rows:
            table.add_row(*row, key=row[0])
        first = self._ledger.get(rows[0][0]) if (rows and self._ledger is not None) else None
        self.query_one("#evidence_detail", Static).update(format_record_detail(first))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if getattr(event.data_table, "id", None) != "evidence_table" or self._ledger is None:
            return
        key = event.row_key.value if event.row_key else None
        if key:
            self.query_one("#evidence_detail", Static).update(
                format_record_detail(self._ledger.get(str(key)))
            )

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "evidence_filter":
            self._populate_evidence(event.value)

    # -----------------------------------------------------------------
    # Footer actions
    # -----------------------------------------------------------------

    def _case_id(self) -> str:
        return self._state.get("case_id") or f"observa-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    def action_export_html(self) -> None:
        if self._state.get("final_summary") is None:
            self.notify("No summary to export.", title="Export failed", severity="error")
            return
        out_path = pathlib.Path(f"{self._case_id()}-report.html")
        out_path.write_text(render_html(self._state, self._ledger), encoding="utf-8")
        self.notify(f"Exported to {out_path}", title="Report exported")

    def action_export_md(self) -> None:
        summary: FinalSummary | None = self._state.get("final_summary")
        if summary is None:
            self.notify("No summary to export.", title="Export failed", severity="error")
            return
        out_path = pathlib.Path(f"{self._case_id()}-summary.md")
        out_path.write_text(_render_markdown(self._state, summary), encoding="utf-8")
        self.notify(f"Exported to {out_path}", title="Export complete")

    def action_export_chat(self) -> None:
        if not self._history:
            self.notify("No chat to export.", title="Export failed", severity="warning")
            return
        case_id = self._state.get("case_id") or f"observa-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        out_path = pathlib.Path(f"{case_id}-chat.txt")
        out_path.write_text(_render_chat_text(self._history), encoding="utf-8")
        self.notify(f"Exported to {out_path}", title="Chat exported")

    def action_chat_wider(self) -> None:
        # ← moves the boundary left, giving the chat more room.
        if self._layout_state == "summary_max":
            self._layout_state = "balanced"
        elif self._layout_state == "balanced":
            self._layout_state = "chat_max"
        else:
            return  # already maxed
        self._apply_layout()
        self._update_toggle_icon()

    def action_chat_narrower(self) -> None:
        # → moves the boundary right, giving the summary more room.
        if self._layout_state == "chat_max":
            self._layout_state = "balanced"
        elif self._layout_state == "balanced":
            self._layout_state = "summary_max"
        else:
            return  # already maxed
        self._apply_layout()
        self._update_toggle_icon()

    def action_chat_toggle(self) -> None:
        """Click target: alternate balanced ↔ chat_max only.

        The summary_max state is intentionally not reachable from this button —
        it stays accessible via Ctrl+→ for keyboard users who want to hide the
        chat entirely. One button, one clear action.
        """
        self._layout_state = "balanced" if self._layout_state == "chat_max" else "chat_max"
        self._apply_layout()
        self._update_toggle_icon()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "chat_toggle":
            self.action_chat_toggle()

    def _update_toggle_icon(self) -> None:
        try:
            btn = self.query_one("#chat_toggle", Button)
        except Exception:  # noqa: BLE001
            return
        btn.label = "⤡" if self._layout_state == "chat_max" else "⤢"

    def _apply_layout(self) -> None:
        self.query_one("#evidence_col").remove_class("visible")
        summary_col = self.query_one("#summary_scroll")
        chat_col = self.query_one("#chat_col")
        if self._layout_state == "balanced":
            summary_col.styles.display = "block"
            chat_col.styles.display = "block"
            summary_col.styles.width = "55%"
            chat_col.styles.width = "45%"
        elif self._layout_state == "chat_max":
            summary_col.styles.display = "none"
            chat_col.styles.display = "block"
            chat_col.styles.width = "100%"
        else:  # summary_max
            chat_col.styles.display = "none"
            summary_col.styles.display = "block"
            summary_col.styles.width = "100%"

    def action_new_case(self) -> None:
        self.app.pop_screen()
        self.app.push_screen("new_case")

    def action_quit(self) -> None:
        self.app.exit()


def _render_markdown(state: dict, summary: FinalSummary) -> str:
    lines = [
        f"# Observa Case Summary — {state.get('case_id', 'unnamed')}",
        "",
        f"**Problem statement:** {state.get('problem_statement', '')}",
        "",
        f"**Investigation scope:** {state.get('investigation_scope', '')}",
        "",
        f"**DBID:** {state.get('dbid', '?')} · **Oracle:** {state.get('oracle_version', '?')} · "
        f"**Topology:** {state.get('topology', '?')}",
        "",
        "## Root cause",
        "",
        summary.root_cause,
        "",
        f"**Confidence:** {summary.confidence:.0%}",
        "",
        f"## Top findings ({len(summary.top_findings)})",
        "",
    ]
    for f in summary.top_findings:
        lines.append(f"- **[{f.severity.upper()}]** ({f.agent}) {f.description}")
        if f.evidence:
            lines.append(f"  - *evidence:* {f.evidence}")
        if f.evidence_refs:
            lines.append(f"  - *refs:* {', '.join(f.evidence_refs)}")
    if summary.unknowns:
        lines.append("")
        lines.append("## Unknowns")
        lines.append("")
        lines.extend(f"- {u}" for u in summary.unknowns)
    if summary.recommended_next_steps:
        lines.append("")
        lines.append("## Recommended next steps")
        lines.append("")
        lines.extend(f"- {s}" for s in summary.recommended_next_steps)
    return "\n".join(lines) + "\n"


def _render_chat_text(history: list[ChatMessage]) -> str:
    blocks: list[str] = []
    for msg in history:
        ts = msg.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        speaker = "you" if msg.sender == "user" else msg.sender
        body = "\n".join(f"    {line}" for line in msg.text.splitlines() or [""])
        blocks.append(f"[{ts}] {speaker}\n{body}")
    return "\n\n".join(blocks) + "\n"
