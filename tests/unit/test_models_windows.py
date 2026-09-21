from datetime import datetime

import pytest
from pydantic import ValidationError

from observa.models import SnapWindow


def test_snap_window_valid():
    w = SnapWindow(
        begin_snap=100,
        end_snap=108,
        begin_time=datetime(2026, 7, 8, 14, 0),
        end_time=datetime(2026, 7, 8, 16, 0),
    )
    assert w.duration_seconds() == 7200.0


def test_snap_window_times_optional_manual_mode():
    w = SnapWindow(begin_snap=100, end_snap=108)
    assert w.begin_time is None
    assert w.duration_seconds() is None


def test_snap_window_rejects_inverted_range():
    with pytest.raises(ValidationError):
        SnapWindow(begin_snap=108, end_snap=100)


def test_snap_window_rejects_equal_snaps():
    with pytest.raises(ValidationError):
        SnapWindow(begin_snap=100, end_snap=100)


def test_snap_window_rejects_inverted_times():
    with pytest.raises(ValidationError):
        SnapWindow(
            begin_snap=100, end_snap=108,
            begin_time=datetime(2026, 7, 8, 16, 0),
            end_time=datetime(2026, 7, 8, 14, 0),
        )
