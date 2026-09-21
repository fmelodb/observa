"""Interaction test: evidence panel open/close respects the layout state.

Regression pin for the Ctrl+D toggle — closing the evidence panel must route
through ``_apply_layout`` so a ``summary_max`` layout (chat hidden via
Ctrl+→ ×2) is preserved instead of force-showing the chat with a stale width.
"""
from __future__ import annotations

from textual.app import App

from observa.mcp.evidence import EvidenceLedger
from observa.tui.summary_chat import SummaryChatScreen

_SQL = ("SELECT event_name FROM DBA_HIST_SYSTEM_EVENT "
        "WHERE dbid = 1234567890 AND snap_id BETWEEN 4210 AND 4215")


def _ledger() -> EvidenceLedger:
    led = EvidenceLedger()
    led.record_query("query_awr", _SQL, "wait_time", turn=1, result={
        "rows": [{"EVENT_NAME": "db file sequential read", "SECONDS": 182.4}],
        "truncated": False, "row_count": 1, "warnings": []}, duration_ms=1180)
    return led


class _App(App):
    def __init__(self, screen: SummaryChatScreen) -> None:
        self._screen = screen
        super().__init__()

    def on_mount(self) -> None:
        self.push_screen(self._screen)


async def test_evidence_toggle_preserves_summary_max_layout():
    screen = SummaryChatScreen(state={}, ledger=_ledger())
    app = _App(screen)
    async with app.run_test() as pilot:
        # (a) Ctrl+→ twice: balanced → summary_max (chat hidden).
        await pilot.press("ctrl+right")
        await pilot.press("ctrl+right")
        assert screen._layout_state == "summary_max"
        assert str(screen.query_one("#chat_col").styles.display) == "none"

        # (b) Ctrl+D opens the evidence panel.
        await pilot.press("ctrl+d")
        assert screen.query_one("#evidence_col").has_class("visible")

        # (c) Ctrl+D closes it — summary_max must be preserved: chat stays
        # hidden, layout state untouched.
        await pilot.press("ctrl+d")
        assert not screen.query_one("#evidence_col").has_class("visible")
        assert str(screen.query_one("#chat_col").styles.display) == "none"
        assert screen._layout_state == "summary_max"


def test_uncited_finding_shows_marker():
    """A finding with no evidence_refs must render '○ uncited' in the summary panel.

    Directly calls _summary_panels() — no Textual event loop needed — and
    inspects the content stored inside the Static widgets.  The test MUST fail
    when the else branch is absent (evidence_refs empty, no fallback appended)
    and pass once the branch is present.
    """
    from rich.text import Text
    from textual.widgets import Static

    from observa.models import FinalSummary, SummaryFinding

    summary = FinalSummary(
        problem_restated="slow queries",
        root_cause="index fragmentation",
        confidence=0.7,
        top_findings=[
            SummaryFinding(
                agent="sql",
                severity="high",
                description="high logical reads on T_ORDERS",
                evidence_refs=[],   # no refs → must render ○ uncited
            ),
        ],
    )
    screen = SummaryChatScreen.__new__(SummaryChatScreen)
    screen._state = {"final_summary": summary}

    panels = screen._summary_panels()

    def _collect_text(node) -> str:
        """Walk a widget tree, accumulating plain text from Static widgets.

        Before mount, children live in ``_pending_children``; after mount in
        ``_nodes``.  We check both so the test works without a running app.
        Static stores its content in the name-mangled ``_Static__content``.
        """
        result = ""
        content = getattr(node, "_Static__content", None)
        if isinstance(content, Text):
            result += content.plain
        children = list(getattr(node, "_nodes", None) or []) + list(
            getattr(node, "_pending_children", None) or []
        )
        for child in children:
            result += _collect_text(child)
        return result

    combined = "".join(_collect_text(p) for p in panels)
    assert "○ uncited" in combined, (
        f"Expected '○ uncited' in findings panel for refs-less finding; got: {combined!r}"
    )


async def test_evidence_open_from_chat_max_shows_summary():
    """Opening evidence while in chat_max must not leave dead space — the
    summary column is forced visible next to the evidence panel."""
    screen = SummaryChatScreen(state={}, ledger=_ledger())
    app = _App(screen)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+left")  # balanced → chat_max (summary hidden)
        assert screen._layout_state == "chat_max"
        assert str(screen.query_one("#summary_scroll").styles.display) == "none"

        await pilot.press("ctrl+d")
        assert screen.query_one("#evidence_col").has_class("visible")
        assert str(screen.query_one("#summary_scroll").styles.display) == "block"
        assert str(screen.query_one("#chat_col").styles.display) == "none"
