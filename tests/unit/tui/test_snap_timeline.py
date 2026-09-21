from datetime import datetime, timedelta

from observa.snap_windows import SnapshotInfo
from observa.tui.widgets.snap_timeline import (
    SnapTimeline,
    TimelineState,
    bar_columns,
    sparkline_chars,
)


def _snaps(n: int, with_db_time: bool = True) -> list[SnapshotInfo]:
    t0 = datetime(2026, 7, 8, 0, 0)
    return [
        SnapshotInfo(
            snap_id=100 + i,
            begin_time=t0 + timedelta(hours=i),
            end_time=t0 + timedelta(hours=i + 1),
            db_time_seconds=float(i) if with_db_time else None,
        )
        for i in range(n)
    ]


def test_sparkline_scales_to_max():
    chars = sparkline_chars([0.0, 4.0, 8.0])
    assert len(chars) == 3
    assert chars[0] == "▁" and chars[2] == "█"


def test_sparkline_none_values_render_marks():
    assert sparkline_chars([None, None]) == ["·", "·"]


def test_marking_problem_range():
    st = TimelineState(snapshots=_snaps(10))
    st.cursor = 4
    st.toggle_mark("problem")   # anchor at 104
    st.cursor = 7
    st.toggle_mark("problem")   # close at 107
    assert st.problem_range == (104, 107)
    assert st.marking is None


def test_marking_single_snap_does_not_close():
    st = TimelineState(snapshots=_snaps(10))
    st.cursor = 4
    st.toggle_mark("problem")
    st.toggle_mark("problem")   # same position — a window needs two snaps
    assert st.problem_range is None
    assert st.marking == "problem"


def test_auto_baseline_suggested_after_problem_closed():
    # 9 days hourly → suggestion lands 7 days before the problem.
    st = TimelineState(snapshots=_snaps(24 * 9))
    ids = [s.snap_id for s in st.snapshots]
    st.cursor = ids.index(ids[-10])
    st.toggle_mark("problem")
    st.cursor = ids.index(ids[-8])
    st.toggle_mark("problem")
    st.suggest_baseline(offset_days=7)
    assert st.baseline_range is not None
    assert st.problem_range is not None
    assert st.baseline_range[1] < st.problem_range[0]


def test_suggest_does_not_override_manual_or_dismissed():
    st = TimelineState(snapshots=_snaps(24 * 9))
    st.cursor = 2
    st.toggle_mark("baseline")
    st.cursor = 4
    st.toggle_mark("baseline")
    manual = st.baseline_range
    st.cursor = 100
    st.toggle_mark("problem")
    st.cursor = 110
    st.toggle_mark("problem")
    st.suggest_baseline(offset_days=7)
    assert st.baseline_range == manual  # manual mark preserved

    st2 = TimelineState(snapshots=_snaps(24 * 9))
    st2.dismiss_baseline()
    st2.cursor = 100
    st2.toggle_mark("problem")
    st2.cursor = 110
    st2.toggle_mark("problem")
    st2.suggest_baseline(offset_days=7)
    assert st2.baseline_range is None  # dismissal respected


def test_dismiss_baseline():
    st = TimelineState(snapshots=_snaps(10))
    st.cursor = 2
    st.toggle_mark("baseline")
    st.cursor = 4
    st.toggle_mark("baseline")
    assert st.baseline_range == (102, 104)
    st.dismiss_baseline()
    assert st.baseline_range is None and st.baseline_dismissed is True


def test_windows_built_from_ranges():
    st = TimelineState(snapshots=_snaps(10))
    st.cursor = 4
    st.toggle_mark("problem")
    st.cursor = 7
    st.toggle_mark("problem")
    w = st.problem_window()
    assert w is not None and (w.begin_snap, w.end_snap) == (104, 107)
    assert st.baseline_window() is None


def test_cursor_clamped():
    st = TimelineState(snapshots=_snaps(3))
    st.move_cursor(-99)
    assert st.cursor == 0
    st.move_cursor(99)
    assert st.cursor == 2


def test_sparkline_mixed_none_and_values():
    chars = sparkline_chars([None, 4.0, None, 8.0])
    assert chars == ["·", "▅", "·", "█"]


# -- bar_columns histogram -------------------------------------------------

def test_bar_columns_row_count_matches_height():
    rows = bar_columns([1.0, 2.0, 3.0], height=4)
    assert len(rows) == 4
    assert all(len(r) == 3 for r in rows)


def test_bar_columns_peak_fills_full_column():
    rows = bar_columns([0.0, 8.0], height=2)
    # column 1 is the peak → both rows are full blocks
    assert rows[0][1] == "█" and rows[1][1] == "█"
    # column 0 is zero → empty in every row
    assert rows[0][0] == " " and rows[1][0] == " "


def test_bar_columns_none_is_dot_on_base_row_only():
    rows = bar_columns([None, 4.0], height=3)
    assert rows[-1][0] == "·"           # base row shows existence mark
    assert all(r[0] == " " for r in rows[:-1])


# -- viewport scrolling ----------------------------------------------------

def test_visible_range_smaller_than_width_shows_all():
    st = TimelineState(snapshots=_snaps(5))
    assert st.visible_range(72) == (0, 5)


def test_ensure_visible_scrolls_to_follow_cursor_right():
    st = TimelineState(snapshots=_snaps(100))
    st.cursor = 90
    st.ensure_visible(20)
    start, end = st.visible_range(20)
    assert start <= 90 < end                 # cursor is inside the viewport
    assert end - start == 20


def test_ensure_visible_scrolls_back_left():
    st = TimelineState(snapshots=_snaps(100))
    st.cursor = 90
    st.ensure_visible(20)
    st.cursor = 3
    st.ensure_visible(20)
    start, end = st.visible_range(20)
    assert start <= 3 < end
    assert start == 0                        # clamped at the left edge


def test_ensure_visible_wide_viewport_resets_start():
    st = TimelineState(snapshots=_snaps(10))
    st.viewport_start = 5
    st.ensure_visible(72)
    assert st.viewport_start == 0


# -- live marking band -----------------------------------------------------

def test_live_range_grows_while_marking():
    st = TimelineState(snapshots=_snaps(10))
    assert st.live_range() is None           # nothing open yet
    st.cursor = 3
    st.toggle_mark("problem")                # anchor at snap 103
    st.cursor = 6
    assert st.live_range() == (103, 106)     # band spans anchor→cursor
    st.toggle_mark("problem")                # close it
    assert st.live_range() is None


# -- render smoke ----------------------------------------------------------

def test_render_produces_fixed_row_layout():
    tl = SnapTimeline()
    tl.state = TimelineState(snapshots=_snaps(40))
    tl.state.cursor = 39
    tl.state.problem_range = (130, 135)
    lines = tl._render_text().plain.split("\n")
    # header + CHART_HEIGHT + rule + brush + ticks + status
    assert len(lines) == SnapTimeline.CHART_HEIGHT + 5
    assert "▲" in tl._render_text().plain      # cursor caret drawn on the brush
    assert "█" in tl._render_text().plain      # problem band drawn


def test_render_empty_state():
    tl = SnapTimeline()
    assert "no snapshots" in tl._render_text().plain
