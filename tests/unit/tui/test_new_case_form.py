"""Unit tests for NewCaseScreen form plumbing.

These tests intentionally do NOT run the Textual event loop. They exercise:
  * the pure `_parse_unified_files` helper, and
  * the module-level shape of NewCaseScreen / CaseStartRequested.
"""
from __future__ import annotations

from observa.models import InputFile
from observa.tui.new_case import (
    FILE_TYPES,
    CaseStartRequested,
    NewCaseScreen,
    _parse_unified_files,
)


def test_module_surface_exports_expected_symbols() -> None:
    assert isinstance(NewCaseScreen, type)
    assert hasattr(NewCaseScreen, "collect_case_data")
    assert callable(NewCaseScreen.collect_case_data)


def test_case_start_requested_carries_data() -> None:
    payload = {
        "problem_statement": "slow",
        "investigation_scope": "peak hours",
        "known_facts": [],
        "dbid": 0,
        "input_files": [],
        "problem_window": None,
        "baseline_window": None,
        "oracle_version": "",
    }
    msg = CaseStartRequested(data=payload)
    assert msg.data == payload


def test_screen_declares_two_steps() -> None:
    # The wizard exposes step switching without running the event loop.
    assert hasattr(NewCaseScreen, "action_advance")
    assert hasattr(NewCaseScreen, "action_back")


def test_parse_manual_snap_range() -> None:
    from observa.models import SnapWindow
    from observa.tui.new_case import _parse_snap_range
    assert _parse_snap_range("100-108") == SnapWindow(begin_snap=100, end_snap=108)
    assert _parse_snap_range(" 100 - 108 ") == SnapWindow(begin_snap=100, end_snap=108)
    assert _parse_snap_range("") is None
    assert _parse_snap_range("garbage") is None
    assert _parse_snap_range("108-100") is None


def test_parse_unified_files_empty_returns_empty() -> None:
    assert _parse_unified_files("") == []
    assert _parse_unified_files("   \n\n  \n") == []


def test_parse_unified_files_parses_type_path_lines() -> None:
    block = (
        "alertlog: /var/log/alert.log\n"
        "trace:/tmp/other.trc\n"
        "\n"
        "  awr_report:  /data/awr.html  \n"
    )
    out = _parse_unified_files(block)
    assert out == [
        InputFile(type="alertlog", path="/var/log/alert.log"),
        InputFile(type="trace", path="/tmp/other.trc"),
        InputFile(type="awr_report", path="/data/awr.html"),
    ]


def test_parse_unified_files_skips_unknown_types() -> None:
    block = "bogus: /x\nalertlog: /y\n"
    out = _parse_unified_files(block)
    assert out == [InputFile(type="alertlog", path="/y")]


def test_parse_unified_files_skips_comment_and_missing_colon() -> None:
    block = "# ignore me\n/only/a/path\nalertlog: /x\n"
    out = _parse_unified_files(block)
    assert out == [InputFile(type="alertlog", path="/x")]


def test_parse_unified_files_accepts_every_known_type() -> None:
    block = "\n".join(f"{t}: /p/{t}" for t in FILE_TYPES)
    out = _parse_unified_files(block)
    assert [f.type for f in out] == list(FILE_TYPES)
    assert [f.path for f in out] == [f"/p/{t}" for t in FILE_TYPES]


def test_parse_unified_files_type_is_case_insensitive() -> None:
    out = _parse_unified_files("ALERTLOG: /x\nAwr_Report: /y")
    assert out == [
        InputFile(type="alertlog", path="/x"),
        InputFile(type="awr_report", path="/y"),
    ]
