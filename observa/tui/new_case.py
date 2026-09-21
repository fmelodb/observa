"""Screen 1: New case wizard — two-step intake flow.

Step 1 — "where and when": database picker from the AWR repository catalog,
interactive snapshot timeline for marking the problem and baseline windows, and
a manual-fallback panel when MCP is offline (DBID + snap-range text entry).

Step 2 — "what happened": the classic problem-statement / investigation-scope /
known-facts / diagnostic-files form, now augmented with a context header
summarising the DB and windows chosen in step 1.

Both steps live in ONE screen (two containers, display toggled) so state
survives forward/back navigation. The seven diagnostic-file types collapse into
a single textarea where each non-empty line is formatted as
``<type>: <absolute path>`` — a parsed drop-in replacement for seven separate
TextAreas.
"""
from __future__ import annotations

from typing import get_args

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Static, TextArea

from observa.models import FileType, InputFile
from observa.tui.widgets.mascot import build_mascot


FILE_TYPES: tuple[FileType, ...] = get_args(FileType)

FILE_TYPE_LABELS: dict[FileType, str] = {
    "alertlog": "Alert Log",
    "trace": "Trace File",
    "tfa": "TFA Bundle",
    "hanganalyze": "Hanganalyze Dump",
    "sqlhc": "SQL Health Check",
    "ddl": "DDL Script",
    "awr_report": "AWR Report",
}


_VALID_TYPES_HINT = "alertlog, trace, tfa, hanganalyze, sqlhc, ddl, awr_report"


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _parse_unified_files(text: str) -> list[InputFile]:
    """Parse ``<type>: <path>`` lines into ``InputFile`` records.

    - Blank lines and lines beginning with ``#`` are ignored.
    - Lines without a colon are skipped silently.
    - Types outside ``FILE_TYPES`` are skipped (the screen will have
      rendered a hint showing the valid set).
    """
    valid = set(FILE_TYPES)
    out: list[InputFile] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        file_type, _, path = line.partition(":")
        file_type = file_type.strip().lower()
        path = path.strip()
        if not path or file_type not in valid:
            continue
        out.append(InputFile(type=file_type, path=path))  # type: ignore[arg-type]
    return out


class CaseStartRequested(Message):
    """Posted when the user submits the new-case form."""

    def __init__(self, data: dict) -> None:
        self.data = data
        super().__init__()


_CSS = """
NewCaseScreen {
    background: #151515;
}
NewCaseScreen #hero {
    height: 6;
    padding: 1 1 0 1;
    background: #151515;
}
NewCaseScreen #step1, NewCaseScreen #step2 {
    padding: 0 1 1 1;
    scrollbar-background: #151515;
    scrollbar-color: #3a3a3a;
}
NewCaseScreen .section {
    height: auto;
    margin: 0 0 1 0;
}
NewCaseScreen .section-title {
    color: #7FB3D5;
    text-style: bold;
    height: 1;
}
NewCaseScreen .hint {
    color: #666666;
    height: auto;
}
NewCaseScreen #context_header {
    color: #7FB3D5;
    height: 2;
    margin: 0 0 1 0;
}
NewCaseScreen Input {
    background: #0d0d0d;
    color: #d4d4d4;
    border: solid #3a3a3a;
    height: 3;
    padding: 0 1;
}
NewCaseScreen Input:focus {
    border: solid #F5B041;
}
NewCaseScreen TextArea {
    background: #0d0d0d;
    border: solid #3a3a3a;
    padding: 0 1;
}
NewCaseScreen TextArea:focus {
    border: solid #F5B041;
}
NewCaseScreen OptionList {
    background: #0d0d0d;
    border: solid #3a3a3a;
    height: 8;
}
NewCaseScreen OptionList:focus {
    border: solid #F5B041;
}
NewCaseScreen #problem_statement { height: 5; }
NewCaseScreen #investigation_scope { height: 5; }
NewCaseScreen #known_facts { height: 6; }
NewCaseScreen #files_block { height: 9; }
NewCaseScreen #manual_panel { display: none; height: auto; }
NewCaseScreen #manual_panel.visible { display: block; }
"""


def _parse_snap_range(text: str):
    """Parse manual 'begin-end' snap range input. None on any problem."""
    from observa.models import SnapWindow

    raw = (text or "").strip()
    if not raw or "-" not in raw:
        return None
    left, _, right = raw.partition("-")
    try:
        begin, end = int(left.strip()), int(right.strip())
    except ValueError:
        return None
    if end <= begin:
        return None
    return SnapWindow(begin_snap=begin, end_snap=end)


class NewCaseScreen(Screen):
    """Screen 1 — two-step wizard: (1) database + windows, (2) problem + evidence."""

    BINDINGS = [
        Binding("ctrl+s", "start", "Start", show=True),
        Binding("escape", "back", "Back/Cancel"),
        Binding("enter", "advance", "Next", show=False, priority=False),
        Binding("left", "cursor_left", show=False),
        Binding("right", "cursor_right", show=False),
        Binding("home", "cursor_home", show=False),
        Binding("end", "cursor_end", show=False),
        Binding("p", "mark_problem", "Mark problem", show=False),
        Binding("b", "mark_baseline", "Mark baseline", show=False),
        Binding("x", "dismiss_baseline", "No baseline", show=False),
    ]

    DEFAULT_CSS = _CSS

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step: int = 1
        self._databases: list = []          # list[DatabaseInfo]
        self._selected_db = None            # DatabaseInfo | None
        self._catalog_failed: bool = False

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        from textual.widgets import OptionList

        from observa.tui.widgets.snap_timeline import SnapTimeline

        yield Header(show_clock=True)
        yield Static(build_mascot(compact=False), id="hero")

        # ---- Step 1: database + windows -------------------------------
        yield VerticalScroll(
            Vertical(
                Static("▎ Step 1/2 — Database and windows", classes="section-title"),
                Static("Select the database loaded in the AWR repository.", classes="hint"),
                Static("connecting to the repository…", id="catalog_status", classes="hint"),
                OptionList(id="db_list"),
                classes="section",
            ),
            Vertical(
                Static("▎ Timeline — DB time per snapshot", classes="section-title"),
                Static(
                    "←/→ move · Home/End ends · P×2 mark problem · B×2 mark baseline · "
                    "X dismiss baseline · Enter next",
                    classes="hint",
                ),
                SnapTimeline(id="timeline"),
                classes="section",
            ),
            Vertical(
                Static("▎ Manual entry (fallback)", classes="section-title"),
                Static(
                    "Repository unreachable — enter DBID and snap ranges by hand.",
                    classes="hint",
                ),
                Input(placeholder="DBID, e.g. 2998621142", id="dbid"),
                Input(placeholder="problem window: begin-end (e.g. 18421-18429)", id="manual_problem"),
                Input(placeholder="baseline window: begin-end (optional)", id="manual_baseline"),
                classes="section",
                id="manual_panel",
            ),
            id="step1",
        )

        # ---- Step 2: problem + evidence (the pre-wizard form minus DBID) --
        step2 = VerticalScroll(
            Static("", id="context_header"),
            Vertical(
                Static("▎ Problem statement", classes="section-title"),
                Static("What does the analyst observe? One short paragraph.", classes="hint"),
                TextArea(id="problem_statement"),
                classes="section",
            ),
            Vertical(
                Static("▎ Investigation scope", classes="section-title"),
                Static(
                    "What, specifically, must Observa investigate? "
                    "Name the SQL, session, object or time window.",
                    classes="hint",
                ),
                TextArea(id="investigation_scope"),
                classes="section",
            ),
            Vertical(
                Static("▎ Known facts", classes="section-title"),
                Static("one per line — optional", classes="hint"),
                TextArea(id="known_facts"),
                classes="section",
            ),
            Vertical(
                Static("▎ Diagnostic files", classes="section-title"),
                Static(
                    Text.from_markup(
                        "[#666666]one per line, format[/#666666] "
                        "[#A9DFBF]<type>: <absolute path>[/#A9DFBF]\n"
                        "[#666666]types: [/#666666]"
                        f"[#7FB3D5]{_VALID_TYPES_HINT}[/#7FB3D5]\n"
                        "[#666666]tfa/sqlhc paths may point to a [b].zip[/b] or directory — "
                        "auto-expanded into individual files[/#666666]"
                    ),
                    classes="hint",
                ),
                TextArea(id="files_block"),
                classes="section last",
            ),
            id="step2",
        )
        step2.display = False
        yield step2

        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self._load_catalog(), exclusive=True)

    # ------------------------------------------------------------------
    # Catalog loading (async, degrades to manual fallback)
    # ------------------------------------------------------------------

    async def _load_catalog(self) -> None:
        from textual.widgets import OptionList

        from observa.mcp.repo_catalog import list_databases

        status = self.query_one("#catalog_status", Static)
        client, error = await self.app.wait_mcp()  # type: ignore[attr-defined]
        if client is None:
            self._catalog_failed = True
            status.update(f"repository unavailable ({error or 'no connection'}) — use manual entry")
            self.query_one("#manual_panel").add_class("visible")
            return
        async def _intake_query(sql: str) -> dict:
            return await client.query_catalog(sql, requester="intake")

        try:
            self._databases = await list_databases(_intake_query)
        except Exception as exc:  # noqa: BLE001
            self._catalog_failed = True
            status.update(f"database listing failed ({exc}) — use manual entry")
            self.query_one("#manual_panel").add_class("visible")
            return
        if not self._databases:
            self._catalog_failed = True
            status.update("repository has no databases loaded — use manual entry")
            self.query_one("#manual_panel").add_class("visible")
            return
        option_list = self.query_one("#db_list", OptionList)
        option_list.clear_options()
        for db in self._databases:
            coverage = ""
            if db.snap_min_time and db.snap_max_time:
                coverage = f"   {db.snap_min_time:%d/%m} → {db.snap_max_time:%d/%m} ({db.snap_count} snaps)"
            topo = f"RAC {db.instance_count} nodes" if db.instance_count > 1 else "single"
            option_list.add_option(
                f"{db.db_name:<10} dbid {db.dbid}   {db.version}   {topo}{coverage}"
            )
        status.update(f"{len(self._databases)} database(s) in the repository")
        if self._databases:
            option_list.highlighted = 0

    async def _load_timeline(self, db) -> None:
        from observa.config import get_settings

        from observa.mcp.repo_catalog import list_snapshots
        from observa.tui.widgets.snap_timeline import SnapTimeline

        client, _ = await self.app.wait_mcp()  # type: ignore[attr-defined]
        if client is None:
            return
        timeline = self.query_one("#timeline", SnapTimeline)

        async def _intake_query(sql: str) -> dict:
            return await client.query_catalog(sql, requester="intake")

        try:
            snaps = await list_snapshots(
                _intake_query, dbid=db.dbid, days=get_settings().case_timeline_days
            )
        except Exception as exc:  # noqa: BLE001
            self.query_one("#catalog_status", Static).update(f"timeline failed: {exc}")
            return
        timeline.set_snapshots(snaps)

    def on_option_list_option_highlighted(self, event) -> None:
        # event.option_index is the correct attribute in Textual 8.x
        idx = event.option_index
        if idx is None or idx >= len(self._databases):
            return
        self._selected_db = self._databases[idx]
        self.run_worker(self._load_timeline(self._selected_db), group="timeline", exclusive=True)

    def on_option_list_option_selected(self, event) -> None:
        # Also advance to step 2 when the user presses Enter on the OptionList,
        # because the "enter" Binding has priority=False and the OptionList
        # consumes Enter internally to fire OptionSelected before the screen sees it.
        self.action_advance()

    # ------------------------------------------------------------------
    # Timeline key actions (step 1 only)
    # ------------------------------------------------------------------

    def _timeline(self):
        from observa.tui.widgets.snap_timeline import SnapTimeline
        return self.query_one("#timeline", SnapTimeline)

    def action_cursor_left(self) -> None:
        if self.step != 1:
            return
        t = self._timeline()
        t.state.move_cursor(-1)
        t.refresh_view()

    def action_cursor_right(self) -> None:
        if self.step != 1:
            return
        t = self._timeline()
        t.state.move_cursor(1)
        t.refresh_view()

    def action_cursor_home(self) -> None:
        if self.step != 1:
            return
        t = self._timeline()
        t.state.move_cursor(-len(t.state.snapshots))
        t.refresh_view()

    def action_cursor_end(self) -> None:
        if self.step != 1:
            return
        t = self._timeline()
        t.state.move_cursor(len(t.state.snapshots))
        t.refresh_view()

    def action_mark_problem(self) -> None:
        if self.step != 1:
            return
        from observa.config import get_settings
        t = self._timeline()
        t.state.toggle_mark("problem")
        t.state.suggest_baseline(get_settings().case_baseline_offset_days)
        t.refresh_view()

    def action_mark_baseline(self) -> None:
        if self.step != 1:
            return
        t = self._timeline()
        t.state.toggle_mark("baseline")
        t.refresh_view()

    def action_dismiss_baseline(self) -> None:
        if self.step != 1:
            return
        t = self._timeline()
        t.state.dismiss_baseline()
        t.refresh_view()

    # ------------------------------------------------------------------
    # Step navigation
    # ------------------------------------------------------------------

    def action_advance(self) -> None:
        if self.step != 1:
            return
        windows = self._resolve_windows()
        if windows is None:
            return  # _resolve_windows already notified
        problem, baseline = windows
        self.step = 2
        self.query_one("#step1").display = False
        self.query_one("#step2").display = True
        self._update_context_header(problem, baseline)
        self.query_one("#problem_statement", TextArea).focus()

    def action_back(self) -> None:
        if self.step == 2:
            self.step = 1
            self.query_one("#step2").display = False
            self.query_one("#step1").display = True
            # Focus must leave the now-hidden TextArea, or it keeps consuming
            # printable keys and the timeline marks (P/B/X) go dead.
            from textual.widgets import OptionList
            if self._catalog_failed:
                self.query_one("#dbid", Input).focus()
            else:
                self.query_one("#db_list", OptionList).focus()
            return
        self.app.exit()

    def _update_context_header(self, problem, baseline) -> None:
        db = self._selected_db
        db_txt = f"{db.db_name} · {db.version} · dbid {db.dbid}" if db else f"dbid {self._manual_dbid()}"
        p_txt = f"problem {problem.begin_snap}→{problem.end_snap}" if problem else "no problem window"
        b_txt = f"baseline {baseline.begin_snap}→{baseline.end_snap}" if baseline else "no baseline (absolute profile)"
        self.query_one("#context_header", Static).update(f"{db_txt}\n{p_txt} · {b_txt}")

    def _manual_dbid(self) -> int:
        return _safe_int(self.query_one("#dbid", Input).value)

    def _resolve_windows(self):
        """Return (problem, baseline) SnapWindows or None when invalid.

        Picker mode requires a closed problem range. Manual mode accepts empty
        windows (legacy degradation — diff probe then skips).
        """
        if self._catalog_failed:
            problem = _parse_snap_range(self.query_one("#manual_problem", Input).value)
            baseline = _parse_snap_range(self.query_one("#manual_baseline", Input).value)
            if self._manual_dbid() <= 0:
                self.notify("Enter the DBID (manual entry).", severity="error")
                return None
            return (problem, baseline)
        state = self._timeline().state
        if self._selected_db is None:
            self.notify("Select a database.", severity="error")
            return None
        problem = state.problem_window()
        if problem is None:
            self.notify("Mark the problem window on the timeline (press P).", severity="error")
            return None
        return (problem, state.baseline_window())

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def action_start(self) -> None:
        if self.step != 2:
            self.action_advance()
            if self.step != 2:
                return
        data = self.collect_case_data()
        if data is None:
            return
        self.app.post_message(CaseStartRequested(data=data))

    def collect_case_data(self) -> dict | None:
        """Extract form data as a dict. Called before switching screens."""
        windows = self._resolve_windows()
        if windows is None:
            return None
        problem, baseline = windows
        files_text = self.query_one("#files_block", TextArea).text
        input_files = _parse_unified_files(files_text)
        known_facts = [
            line.strip()
            for line in self.query_one("#known_facts", TextArea).text.split("\n")
            if line.strip()
        ]
        dbid = self._selected_db.dbid if self._selected_db else self._manual_dbid()
        version = self._selected_db.version if self._selected_db else ""
        return {
            "problem_statement": self.query_one("#problem_statement", TextArea).text.strip(),
            "investigation_scope": self.query_one("#investigation_scope", TextArea).text.strip(),
            "known_facts": known_facts,
            "dbid": dbid,
            "input_files": input_files,
            "problem_window": problem,
            "baseline_window": baseline,
            "oracle_version": version,
        }
