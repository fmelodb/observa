"""Snapshot timeline widget for the intake wizard (step 1).

Rendering and selection logic live in ``TimelineState`` (pure, unit-testable);
the ``SnapTimeline`` widget is a thin Textual wrapper that renders the state
and exposes key-driven mutations. Marking model (from the approved mockups):
press P at the cursor to anchor the problem range, move, press P again to
close it; same with B for baseline; X dismisses the baseline.

The timeline renders as a fixed-height column histogram (DB time per snapshot)
plus a "brush" selection track below it. The horizon is far wider than any
terminal, so the display is a scrolling viewport of one column per snapshot
that follows the cursor; ``TimelineState`` tracks ``viewport_start`` and the
widget reconciles it to the cursor on every refresh.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from rich.text import Text
from textual.widgets import Static

from observa.models import SnapWindow
from observa.snap_windows import SnapshotInfo, suggest_baseline, window_from_range

_BLOCKS = "▁▂▃▄▅▆▇█"

_PROBLEM_COLOR = "#d14d41"   # Flexoki red
_BASELINE_COLOR = "#4c8a5f"  # Flexoki green
_CURSOR_COLOR = "#d0a215"    # Flexoki yellow
_BAR_COLOR = "#5b7089"       # muted slate for unselected bars
_MUTED = "#8a8a83"

__all__ = ["SnapTimeline", "TimelineState", "sparkline_chars", "bar_columns"]


def sparkline_chars(values: list[Optional[float]]) -> list[str]:
    """Map values to block characters; None → existence mark."""
    numeric = [v for v in values if v is not None]
    peak = max(numeric) if numeric else 0.0
    out: list[str] = []
    for v in values:
        if v is None:
            out.append("·")
        elif peak <= 0:
            out.append(_BLOCKS[0])
        else:
            idx = min(len(_BLOCKS) - 1, int((v / peak) * (len(_BLOCKS) - 1) + 0.5))
            out.append(_BLOCKS[idx])
    return out


def bar_columns(values: list[Optional[float]], height: int) -> list[str]:
    """Column histogram: ``height`` rows (top→bottom), one char per value.

    Each cell holds an eighth-resolution slice of a column so bars have
    sub-row precision (``_BLOCKS`` fills a partial cell from the bottom).
    None → an existence dot ``·`` on the base row only. Normalized to the
    peak of the numeric values.
    """
    height = max(1, height)
    numeric = [v for v in values if v is not None]
    peak = max(numeric) if numeric else 0.0
    eighths: list[Optional[int]] = []
    for v in values:
        if v is None:
            eighths.append(None)
        elif peak <= 0:
            eighths.append(0)
        else:
            eighths.append(int((v / peak) * height * 8 + 0.5))
    rows: list[str] = []
    for r in range(height):
        level = height - 1 - r  # 0 == bottom row
        row_chars: list[str] = []
        for e in eighths:
            if e is None:
                row_chars.append("·" if level == 0 else " ")
                continue
            cell = min(8, max(0, e - level * 8))
            row_chars.append(" " if cell == 0 else _BLOCKS[cell - 1])
        rows.append("".join(row_chars))
    return rows


@dataclass
class TimelineState:
    snapshots: list[SnapshotInfo] = field(default_factory=list)
    cursor: int = 0                                   # index into snapshots
    viewport_start: int = 0                           # first visible index
    marking: Optional[Literal["problem", "baseline"]] = None
    problem_range: Optional[tuple[int, int]] = None   # (begin_snap, end_snap)
    baseline_range: Optional[tuple[int, int]] = None
    baseline_dismissed: bool = False

    def __post_init__(self) -> None:
        # snap_id of an open (anchored, not yet closed) mark — transient
        # internal state, deliberately not a constructor field.
        self._anchor: Optional[int] = None

    def move_cursor(self, delta: int) -> None:
        if not self.snapshots:
            return
        self.cursor = max(0, min(len(self.snapshots) - 1, self.cursor + delta))

    def toggle_mark(self, kind: Literal["problem", "baseline"]) -> None:
        if not self.snapshots:
            return
        snap_id = self.snapshots[self.cursor].snap_id
        if self.marking == kind and self._anchor is not None:
            begin, end = sorted((self._anchor, snap_id))
            if begin == end:
                return  # a window needs at least two snapshots
            if kind == "problem":
                self.problem_range = (begin, end)
            else:
                self.baseline_range = (begin, end)
                self.baseline_dismissed = False
            self.marking = None
            self._anchor = None
        else:
            self.marking = kind
            self._anchor = snap_id

    def dismiss_baseline(self) -> None:
        self.baseline_range = None
        self.baseline_dismissed = True
        if self.marking == "baseline":
            self.marking = None
            self._anchor = None

    def suggest_baseline(self, offset_days: int) -> None:
        """Auto-suggest unless the analyst already marked or dismissed one."""
        if self.problem_range is None or self.baseline_dismissed or self.baseline_range is not None:
            return
        problem = self.problem_window()
        if problem is None:
            return
        suggestion = suggest_baseline(self.snapshots, problem, offset_days)
        if suggestion is not None:
            self.baseline_range = (suggestion.begin_snap, suggestion.end_snap)

    def problem_window(self) -> Optional[SnapWindow]:
        if self.problem_range is None:
            return None
        return window_from_range(self.snapshots, *self.problem_range)

    def baseline_window(self) -> Optional[SnapWindow]:
        if self.baseline_range is None:
            return None
        return window_from_range(self.snapshots, *self.baseline_range)

    # -- viewport (scrolling display) -----------------------------------

    def visible_range(self, width: int) -> tuple[int, int]:
        """(start, end) index slice currently visible for a chart ``width``."""
        n = len(self.snapshots)
        width = max(1, width)
        if width >= n:
            return (0, n)
        start = max(0, min(self.viewport_start, n - width))
        return (start, start + width)

    def ensure_visible(self, width: int) -> None:
        """Scroll ``viewport_start`` so the cursor stays inside the viewport."""
        n = len(self.snapshots)
        if n == 0:
            self.viewport_start = 0
            return
        width = max(1, width)
        if width >= n:
            self.viewport_start = 0
            return
        margin = min(3, width // 4)
        start = self.viewport_start
        if self.cursor < start + margin:
            start = self.cursor - margin
        elif self.cursor > start + width - 1 - margin:
            start = self.cursor - width + 1 + margin
        self.viewport_start = max(0, min(start, n - width))

    def live_range(self) -> Optional[tuple[int, int]]:
        """In-progress marking band (anchor↔cursor) while a mark is open."""
        if self.marking is None or self._anchor is None or not self.snapshots:
            return None
        cur = self.snapshots[self.cursor].snap_id
        lo, hi = sorted((self._anchor, cur))
        return (lo, hi)


def _in_range(sid: int, rng: Optional[tuple[int, int]]) -> bool:
    return rng is not None and rng[0] <= sid <= rng[1]


class SnapTimeline(Static):
    """Renders a TimelineState as a column histogram + brush selection track."""

    CHART_HEIGHT = 6
    # header + chart + rule + brush + ticks + status = CHART_HEIGHT + 5 content
    # rows; the round border adds 2 → fixed widget height stops the wrapping
    # the old single-row sparkline suffered from.
    DEFAULT_CSS = """
    SnapTimeline {
        height: 13;
        background: #0d0d0d;
        border: round #3a3a3a;
        padding: 0 1;
        color: #d4d4d4;
    }
    """

    _STYLES = {
        "cursor": f"bold {_CURSOR_COLOR}",
        "problem": _PROBLEM_COLOR,
        "baseline": _BASELINE_COLOR,
        "live-problem": f"dim {_PROBLEM_COLOR}",
        "live-baseline": f"dim {_BASELINE_COLOR}",
        "none": _BAR_COLOR,
    }
    _BRUSH = {
        "cursor": "▲",
        "problem": "█",
        "baseline": "▓",
        "live-problem": "█",
        "live-baseline": "▓",
        "none": "░",
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = TimelineState()

    def set_snapshots(self, snapshots: list[SnapshotInfo]) -> None:
        self.state = TimelineState(snapshots=list(snapshots))
        if snapshots:
            self.state.cursor = len(snapshots) - 1
        self.refresh_view()

    def on_mount(self) -> None:
        self.refresh_view()

    def on_resize(self, event) -> None:  # noqa: ARG002 - width may have changed
        self.refresh_view()

    def refresh_view(self) -> None:
        self.update(self._render_text())

    # ------------------------------------------------------------------

    def _chart_width(self) -> int:
        w = getattr(self.content_size, "width", 0)
        if w <= 0:
            w = 72  # fallback before first layout; on_mount/on_resize re-render
        return max(10, w)

    def _kind(self, gi: int) -> str:
        st = self.state
        if gi == st.cursor:
            return "cursor"
        sid = st.snapshots[gi].snap_id
        if _in_range(sid, st.problem_range):
            return "problem"
        if _in_range(sid, st.baseline_range):
            return "baseline"
        live = st.live_range()
        if live and live[0] <= sid <= live[1]:
            return "live-problem" if st.marking == "problem" else "live-baseline"
        return "none"

    def _render_text(self) -> Text:
        st = self.state
        text = Text()
        text.no_wrap = True
        if not st.snapshots:
            text.append("(no snapshots in the selected horizon)", style=_MUTED)
            return text

        width = self._chart_width()
        st.ensure_visible(width)
        start, end = st.visible_range(width)
        vis = st.snapshots[start:end]
        span = len(vis)
        kinds = [self._kind(start + i) for i in range(span)]

        # -- header: title + peak DB time -------------------------------
        peak = max(
            (s.db_time_seconds for s in st.snapshots if s.db_time_seconds is not None),
            default=0.0,
        )
        title = "DB time / snapshot"
        peak_label = f"peak {peak:,.0f}s" if peak else "peak —"
        text.append(title, style="bold #7FB3D5")
        text.append(" " * max(1, width - len(title) - len(peak_label)))
        text.append(peak_label, style=_MUTED)
        text.append("\n")

        # -- histogram --------------------------------------------------
        rows = bar_columns([s.db_time_seconds for s in vis], self.CHART_HEIGHT)
        for row in rows:
            for ci, ch in enumerate(row):
                text.append(ch, style=self._STYLES[kinds[ci]])
            text.append("\n")

        # -- base rule --------------------------------------------------
        text.append("─" * span, style=_MUTED)
        text.append("\n")

        # -- brush selection track --------------------------------------
        for ci in range(span):
            kind = kinds[ci]
            style = self._STYLES[kind] if kind != "none" else _MUTED
            text.append(self._BRUSH[kind], style=style)
        text.append("\n")

        # -- time-axis ticks --------------------------------------------
        text.append(self._tick_row(vis), style=_MUTED)
        text.append("\n")

        # -- status line ------------------------------------------------
        self._append_status(text, start, end)
        return text

    def _tick_row(self, vis: list[SnapshotInfo]) -> str:
        n = len(vis)
        if n == 0:
            return ""
        row = [" "] * n
        count = max(2, min(5, n // 15 + 2))
        positions = sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})
        # Hour-aware ticks when the visible span is short, else day/month, so
        # adjacent ticks never collapse to the same label.
        span_hours = (vis[-1].begin_time - vis[0].begin_time).total_seconds() / 3600
        fmt = "%d/%m %Hh" if span_hours < 48 else "%d/%m"
        for pos in positions:
            label = vis[pos].begin_time.strftime(fmt)
            s = max(0, min(pos - len(label) // 2, n - len(label)))
            for k, ch in enumerate(label):
                if 0 <= s + k < n:
                    row[s + k] = ch
        return "".join(row)

    @staticmethod
    def _duration(w: Optional[SnapWindow]) -> str:
        if w is None or w.begin_time is None or w.end_time is None:
            return "?"
        secs = (w.end_time - w.begin_time).total_seconds()
        return f"{secs / 3600:.0f}h" if secs >= 3600 else f"{secs / 60:.0f}m"

    def _append_status(self, text: Text, start: int, end: int) -> None:
        st = self.state
        if start > 0:
            text.append("◄ ", style=_CURSOR_COLOR)
        if st.problem_range:
            dur = self._duration(st.problem_window())
            text.append(
                f"problem {st.problem_range[0]}→{st.problem_range[1]} · {dur}   ",
                style=_PROBLEM_COLOR,
            )
        elif st.marking == "problem":
            text.append("marking problem… press P to close   ", style=_PROBLEM_COLOR)
        if st.baseline_range:
            dur = self._duration(st.baseline_window())
            text.append(
                f"baseline {st.baseline_range[0]}→{st.baseline_range[1]} · {dur}   ",
                style=_BASELINE_COLOR,
            )
        elif st.baseline_dismissed:
            text.append("baseline dismissed — absolute profile   ", style=_MUTED)
        elif st.marking == "baseline":
            text.append("marking baseline… press B to close   ", style=_BASELINE_COLOR)
        cur = st.snapshots[st.cursor]
        text.append(
            f"snap {cur.snap_id} · {cur.begin_time:%d/%m %H:%M} "
            f"({st.cursor + 1}/{len(st.snapshots)})",
            style=_CURSOR_COLOR,
        )
        if end < len(st.snapshots):
            text.append(" ►", style=_CURSOR_COLOR)
