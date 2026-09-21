"""Pure window math for the intake wizard and diff probe.

Zero I/O — operates on ``SnapshotInfo`` lists produced by
``observa.mcp.repo_catalog``. The auto-baseline policy (validated in the
2026-07-11 design session): same time-of-day range ``offset_days`` earlier;
if no snapshots exist there, the range immediately preceding the problem
window with the same snap span; if not even that, ``None``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from observa.models import SnapWindow


@dataclass(frozen=True)
class SnapshotInfo:
    """One snapshot on the repository timeline (RAC rows already collapsed)."""

    snap_id: int
    begin_time: datetime
    end_time: datetime
    # None when the DB-time metric query failed — timeline degrades to
    # existence marks.
    db_time_seconds: Optional[float] = None


def window_from_range(
    snapshots: list[SnapshotInfo], begin_snap: int, end_snap: int
) -> SnapWindow:
    """Build a SnapWindow from two snap ids that must exist in ``snapshots``."""
    by_id = {s.snap_id: s for s in snapshots}
    begin = by_id.get(begin_snap)
    end = by_id.get(end_snap)
    if begin is None or end is None:
        missing = [i for i in (begin_snap, end_snap) if i not in by_id]
        raise ValueError(f"snap_id(s) not found in repository timeline: {missing}")
    return SnapWindow(
        begin_snap=begin_snap,
        end_snap=end_snap,
        begin_time=begin.begin_time,
        end_time=end.end_time,
    )


def _covering_range(
    snapshots: list[SnapshotInfo], target_begin: datetime, target_end: datetime
) -> Optional[tuple[int, int]]:
    """Snap ids of the snapshots overlapping [target_begin, target_end].

    Caller must pass a list sorted by snap_id.
    """
    hits = [
        s for s in snapshots
        if s.end_time > target_begin and s.begin_time < target_end
    ]
    if len(hits) < 2:
        return None
    return hits[0].snap_id, hits[-1].snap_id


def suggest_baseline(
    snapshots: list[SnapshotInfo],
    problem: SnapWindow,
    offset_days: int,
) -> Optional[SnapWindow]:
    """Auto-suggest a baseline window for ``problem``. Editable by the analyst."""
    ordered = sorted(snapshots, key=lambda s: s.snap_id)

    # Preferred: same time-of-day range offset_days earlier.
    if problem.begin_time is not None and problem.end_time is not None:
        target_begin = problem.begin_time - timedelta(days=offset_days)
        target_end = problem.end_time - timedelta(days=offset_days)
        rng = _covering_range(ordered, target_begin, target_end)
        if rng is not None and rng[1] < problem.begin_snap:
            return window_from_range(ordered, rng[0], rng[1])

    # Fallback: same number of ACTUAL intervals immediately preceding the
    # problem window. Positional indexing on the sorted list — snap ids are
    # not contiguous in real repositories (awrload gaps), so ID-delta
    # arithmetic would reach across gaps.
    problem_snaps = [
        s for s in ordered if problem.begin_snap <= s.snap_id <= problem.end_snap
    ]
    span_count = len(problem_snaps) - 1
    if span_count < 1:
        return None
    prior_with_boundary = [s for s in ordered if s.snap_id <= problem.begin_snap]
    if len(prior_with_boundary) < span_count + 1:
        return None
    baseline_snaps = prior_with_boundary[-(span_count + 1):]
    if baseline_snaps[0].snap_id == baseline_snaps[-1].snap_id:
        return None
    return window_from_range(
        ordered, baseline_snaps[0].snap_id, baseline_snaps[-1].snap_id
    )
