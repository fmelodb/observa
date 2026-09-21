"""Pure formatters for the Evidence Browser panel."""
from __future__ import annotations

from observa.mcp.evidence import EvidenceLedger
from observa.tui.widgets.evidence_browser import evidence_rows, format_record_detail

_SQL = ("SELECT event_name FROM DBA_HIST_SYSTEM_EVENT "
        "WHERE dbid = 1234567890 AND snap_id BETWEEN 4210 AND 4215")


def _ledger() -> EvidenceLedger:
    led = EvidenceLedger()
    led.record_query("query_awr", _SQL, "wait_time", turn=1, result={
        "rows": [{"EVENT_NAME": "db file sequential read", "SECONDS": 182.4}],
        "truncated": False, "row_count": 1, "warnings": []}, duration_ms=1180)
    led.record_read("alert.log", 0, 65536, "file_reader", turn=2, result={
        "content": "ORA-00600: internal error", "offset": 0, "bytes_read": 25,
        "eof": True, "truncated": False, "total_size": 25}, duration_ms=12)
    led.record_query("query_awr", "SELECT 1 FROM V$SESSION", "sql",
                     status="rejected", error="not in allowlist")
    return led


def test_rows_shape_and_order():
    rows = evidence_rows(_ledger().records)
    assert [r[0] for r in rows] == ["Q-001", "F-001", "Q-002"]
    q = rows[0]
    # (id, requester, turn, kind, target, rows, ms, status)
    assert q[1] == "wait_time" and q[2] == "1" and q[3] == "query_awr"
    assert q[5] == "1" and q[6] == "1180"
    assert rows[2][7] == "rejected"


def test_rows_filter_matches_requester_kind_and_id():
    recs = _ledger().records
    assert len(evidence_rows(recs, "wait_time")) == 1
    assert len(evidence_rows(recs, "read_file")) == 1
    assert len(evidence_rows(recs, "q-002")) == 1
    assert len(evidence_rows(recs, "")) == 3


def test_detail_shows_sql_window_and_sample():
    led = _ledger()
    rec = led.get("Q-001")
    assert rec is not None
    text = format_record_detail(rec).plain
    assert "DBA_HIST_SYSTEM_EVENT" in text
    assert "1234567890" in text and "4210" in text
    assert "db file sequential read" in text


def test_detail_file_and_rejected():
    led = _ledger()
    f_rec = led.get("F-001")
    q_rec = led.get("Q-002")
    assert f_rec is not None and q_rec is not None
    assert "ORA-00600" in format_record_detail(f_rec).plain
    assert "not in allowlist" in format_record_detail(q_rec).plain


def test_detail_handles_none():
    assert "no record selected" in format_record_detail(None).plain


def test_detail_caps_sample_rows_at_20():
    led = EvidenceLedger()
    rows = [{"N": i} for i in range(50)]
    led.record_query("query_awr", _SQL, "a", result={
        "rows": rows, "truncated": False, "row_count": 50, "warnings": []})
    rec = led.get("Q-001")
    assert rec is not None
    text = format_record_detail(rec).plain
    assert "showing 20 of 50" in text


def test_search_record_target_and_detail():
    from observa.mcp.evidence import EvidenceLedger
    from observa.tui.widgets.evidence_browser import evidence_rows, format_record_detail

    led = EvidenceLedger()
    led.record_search("alert.log", r"ORA-\d+", "file_reader", turn=1, result={
        "matches": [{"line_number": 2, "byte_offset": 11, "line": "ORA-00600 boom"}],
        "match_count": 1, "truncated": False, "total_size": 20})
    rows = evidence_rows(led.records)
    r = rows[0]
    assert r[0] == "S-001" and r[3] == "search_file"
    assert "ORA-" in r[4]  # target shows the pattern
    rec = led.get("S-001")
    assert rec is not None
    detail = format_record_detail(rec).plain
    assert r"ORA-\d+" in detail          # the pattern
    assert "ORA-00600 boom" in detail    # a sample match line


def test_ora_extract_sentinel_record_shows_incident_count():
    from observa.mcp.evidence import EvidenceLedger
    from observa.tui.widgets.evidence_browser import format_record_detail

    led = EvidenceLedger()
    led.record_search("alert.log", "<ora-extract>", "ora_timeline", result={
        "incidents": [{"code": "ORA-00060"}], "coverage": {}, "match_count": 1})
    rec = led.get("S-001")
    assert rec is not None
    detail = format_record_detail(rec).plain
    assert "<ora-extract>" in detail
    assert "1 incident(s)" in detail
