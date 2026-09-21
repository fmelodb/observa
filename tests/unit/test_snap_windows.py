from datetime import datetime, timedelta

import pytest

from observa.models import SnapWindow
from observa.snap_windows import SnapshotInfo, suggest_baseline, window_from_range


def _snaps(start: datetime, count: int, first_id: int = 100) -> list[SnapshotInfo]:
    """Hourly snapshots starting at `start`."""
    out = []
    for i in range(count):
        out.append(
            SnapshotInfo(
                snap_id=first_id + i,
                begin_time=start + timedelta(hours=i),
                end_time=start + timedelta(hours=i + 1),
                db_time_seconds=float(i),
            )
        )
    return out


def test_window_from_range_builds_window_with_times():
    snaps = _snaps(datetime(2026, 7, 8, 0, 0), 24)
    w = window_from_range(snaps, 114, 116)  # 14:00 → 17:00
    assert w == SnapWindow(
        begin_snap=114, end_snap=116,
        begin_time=datetime(2026, 7, 8, 14, 0),
        end_time=datetime(2026, 7, 8, 17, 0),
    )


def test_window_from_range_rejects_unknown_snap():
    snaps = _snaps(datetime(2026, 7, 8, 0, 0), 24)
    with pytest.raises(ValueError, match="not found"):
        window_from_range(snaps, 114, 999)


def test_suggest_baseline_same_time_of_day_7_days_earlier():
    # 8 days of hourly snapshots: 2026-07-01 00:00 → 2026-07-09 00:00.
    snaps = _snaps(datetime(2026, 7, 1, 0, 0), 24 * 8)
    problem = window_from_range(
        snaps,
        snaps[-10].snap_id,   # 2026-07-08 14:00
        snaps[-8].snap_id,    # → 2026-07-08 17:00
    )
    baseline = suggest_baseline(snaps, problem, offset_days=7)
    assert baseline is not None
    assert baseline.begin_time == datetime(2026, 7, 1, 14, 0)
    assert baseline.end_time == datetime(2026, 7, 1, 17, 0)


def test_suggest_baseline_falls_back_to_adjacent_window():
    # Only 1 day of history — nothing 7 days earlier.
    snaps = _snaps(datetime(2026, 7, 8, 0, 0), 24)
    problem = window_from_range(snaps, 114, 116)  # 14:00 → 17:00
    baseline = suggest_baseline(snaps, problem, offset_days=7)
    assert baseline is not None
    # Adjacent: same span (2 snaps) ending exactly at the problem's begin.
    assert baseline.end_snap == 114
    assert baseline.begin_snap == 112


def test_suggest_baseline_none_when_no_room():
    snaps = _snaps(datetime(2026, 7, 8, 0, 0), 3)  # ids 100..102
    problem = window_from_range(snaps, 100, 102)
    assert suggest_baseline(snaps, problem, offset_days=7) is None


def test_suggest_baseline_fallback_is_gap_safe():
    # Non-contiguous ids (awrload gap): fallback must pick the snaps
    # immediately preceding the problem, never reach across the gap.
    t0 = datetime(2026, 7, 8, 0, 0)
    ids = [100, 101, 102, 200, 201, 202]
    snaps = [
        SnapshotInfo(
            snap_id=sid,
            begin_time=t0 + timedelta(hours=i),
            end_time=t0 + timedelta(hours=i + 1),
            db_time_seconds=1.0,
        )
        for i, sid in enumerate(ids)
    ]
    problem = window_from_range(snaps, 201, 202)
    baseline = suggest_baseline(snaps, problem, offset_days=7)
    assert baseline is not None
    assert (baseline.begin_snap, baseline.end_snap) == (200, 201)


def test_suggest_baseline_without_times_uses_fallback():
    # Manual-mode problem window (no datetimes) skips the preferred branch.
    snaps = _snaps(datetime(2026, 7, 8, 0, 0), 24)
    problem = SnapWindow(begin_snap=114, end_snap=116)
    baseline = suggest_baseline(snaps, problem, offset_days=7)
    assert baseline is not None
    assert baseline.end_snap == 114 and baseline.begin_snap == 112
