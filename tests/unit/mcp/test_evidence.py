"""Unit tests for the evidence ledger: IDs, canonicalization, cache, counters."""
from __future__ import annotations

from observa.mcp.evidence import (
    EvidenceLedger,
    canonical_sql,
    extract_case_predicates,
)

_SQL = (
    "SELECT event_name FROM DBA_HIST_SYSTEM_EVENT "
    "WHERE dbid = 1234567890 AND snap_id BETWEEN 4210 AND 4215"
)
_RESULT = {"rows": [{"EVENT_NAME": "db file sequential read"}], "truncated": False,
           "row_count": 1, "warnings": []}


def test_id_sequences_q_and_f():
    led = EvidenceLedger()
    q1 = led.record_query("query_awr", _SQL, "wait_time", result=_RESULT)
    f1 = led.record_read("alert.log", 0, 65536, "file_reader",
                         result={"content": "x", "offset": 0, "bytes_read": 1,
                                 "eof": True, "truncated": False, "total_size": 1})
    q2 = led.record_query("query_catalog", "SELECT dbid FROM DBA_HIST_SNAPSHOT WHERE dbid = 1 AND snap_id BETWEEN 1 AND 2", "diff_probe", result=_RESULT)
    assert q1 == "Q-001"
    assert f1 == "F-001"
    assert q2 == "Q-002"
    assert led.known_ids() == frozenset({"Q-001", "F-001", "Q-002"})


def test_canonical_sql_ignores_comments_case_whitespace():
    a = canonical_sql("select 1 from dba_hist_snapshot where dbid=5 -- x")
    b = canonical_sql("SELECT 1\n  FROM DBA_HIST_SNAPSHOT\nWHERE dbid = 5")
    assert a == b


def test_canonical_sql_fallback_on_unparseable():
    # sqlglot cannot parse this; fallback is whitespace-collapsed raw string.
    assert canonical_sql("garbage (((") == "garbage ((("


def test_extract_case_predicates():
    dbid, snaps = extract_case_predicates(_SQL)
    assert dbid == 1234567890
    assert snaps == (4210, 4215)
    dbid2, snaps2 = extract_case_predicates("SELECT 1 FROM DBA_HIST_SNAPSHOT")
    assert dbid2 is None and snaps2 is None


def test_cache_hit_returns_original_record_and_counts():
    led = EvidenceLedger()
    led.record_query("query_awr", _SQL, "wait_time", result=_RESULT)
    hit = led.lookup_query("query_awr", "select event_name from dba_hist_system_event "
                           "where dbid = 1234567890 and snap_id between 4210 and 4215",
                           "ash")
    assert hit is not None
    assert hit["evidence_id"] == "Q-001"
    assert hit["rows"] == _RESULT["rows"]
    assert len(led.records) == 1          # no duplicate record
    assert led.records[0].hits == 1
    c = led.counters()
    assert c["wait_time"].queries == 1 and c["wait_time"].cache_hits == 0
    assert c["ash"].queries == 1 and c["ash"].cache_hits == 1


def test_cache_namespaced_by_kind():
    led = EvidenceLedger()
    led.record_query("query_catalog", _SQL, "diff_probe", result=_RESULT)
    # An agent query must NOT be served from a catalog-path result (different
    # guard config / row caps).
    assert led.lookup_query("query_awr", _SQL, "wait_time") is None


def test_error_and_rejected_records_not_cacheable():
    led = EvidenceLedger()
    led.record_query("query_awr", _SQL, "wait_time", status="rejected",
                     error="not in allowlist")
    assert led.lookup_query("query_awr", _SQL, "ash") is None
    assert led.records[0].status == "rejected"


def test_read_cache_key_includes_offset_and_max_bytes():
    led = EvidenceLedger()
    res = {"content": "abc", "offset": 0, "bytes_read": 3, "eof": True,
           "truncated": False, "total_size": 3}
    led.record_read("alert.log", 0, 65536, "file_reader", result=res)
    assert led.lookup_read("alert.log", 0, 65536, "file_reader") is not None
    assert led.lookup_read("alert.log", 100, 65536, "file_reader") is None
    assert led.lookup_read("alert.log", 0, 1024, "file_reader") is None


def test_result_payload_is_isolated_from_caller_and_between_hits():
    led = EvidenceLedger()
    src = {"rows": [{"N": 1}], "truncated": False, "row_count": 1, "warnings": []}
    led.record_query("query_awr", _SQL, "wait_time", result=src)
    src["rows"].append({"N": 2})                      # caller mutates after ingest
    hit1 = led.lookup_query("query_awr", _SQL, "ash")
    assert hit1 is not None
    hit1["rows"].append({"N": 3})                     # consumer mutates a served payload
    hit2 = led.lookup_query("query_awr", _SQL, "sql")
    assert hit2 is not None
    assert led.records[0].result["rows"] == [{"N": 1}]
    assert hit2["rows"] == [{"N": 1}]


def test_totals():
    led = EvidenceLedger()
    led.record_query("query_awr", _SQL, "wait_time", result=_RESULT)
    led.lookup_query("query_awr", _SQL, "ash")
    accesses, hits = led.totals()
    assert accesses == 2 and hits == 1


_SEARCH_RESULT = {"matches": [{"line_number": 2, "byte_offset": 11, "line": "ORA-00600"}],
                  "match_count": 1, "scanned_bytes": 20, "truncated": False, "total_size": 20}


def test_search_id_sequence_is_s():
    led = EvidenceLedger()
    s1 = led.record_search("alert.log", r"ORA-\d+", "file_reader", result=_SEARCH_RESULT)
    assert s1 == "S-001"
    rec = led.get("S-001")
    assert rec is not None
    assert rec.kind == "search_file"
    assert rec.pattern == r"ORA-\d+"
    assert rec.match_count == 1


def test_search_cache_hit_by_pattern_and_path():
    led = EvidenceLedger()
    led.record_search("alert.log", r"ORA-\d+", "file_reader", result=_SEARCH_RESULT)
    hit = led.lookup_search(r"ORA-\d+", "alert.log", "sql")
    assert hit is not None and hit["evidence_id"] == "S-001"
    assert len(led.records) == 1 and led.records[0].hits == 1
    # different pattern or path is a miss
    assert led.lookup_search(r"ORA-600", "alert.log", "sql") is None
    assert led.lookup_search(r"ORA-\d+", "other.log", "sql") is None


def test_search_error_not_cached():
    led = EvidenceLedger()
    led.record_search("alert.log", r"ORA-(", "file_reader",
                      status="error", error="invalid regex")
    assert led.lookup_search(r"ORA-(", "alert.log", "sql") is None


def test_search_counts_as_read_in_counters():
    led = EvidenceLedger()
    led.record_search("alert.log", r"ORA-\d+", "file_reader", result=_SEARCH_RESULT)
    c = led.counters()["file_reader"]
    assert c.reads == 1 and c.queries == 0
