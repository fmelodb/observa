"""Pure evidence-materialization helpers (Feature 5)."""
from __future__ import annotations

from observa.graph.evidence_replay import gather_finding_evidence, materialize_evidence
from observa.mcp.evidence import EvidenceLedger, EvidenceRecord
from observa.models import Finding


def _rec(**kw) -> EvidenceRecord:
    base = dict(evidence_id="Q-001", kind="query_awr", requester="sql")
    base.update(kw)
    return EvidenceRecord(**base)


def test_materialize_query_rows_and_row_cap():
    rows = [{"SQL_ID": f"s{i}", "ELAPSED_S": i} for i in range(20)]
    rec = _rec(sql="SELECT sql_id FROM DBA_HIST_SQLSTAT WHERE dbid=1",
               result={"rows": rows, "row_count": 20})
    out = materialize_evidence(rec, max_rows=15)
    assert "Q-001" in out
    assert "DBA_HIST_SQLSTAT" in out
    assert "SQL_ID=s0" in out
    assert "+5 more rows" in out  # 20 rows, cap 15


def test_materialize_flags_truncated():
    rec = _rec(result={"rows": [{"A": 1}], "row_count": 1, "truncated": True},
               truncated=True)
    out = materialize_evidence(rec)
    assert "TRUNCATED" in out


def test_materialize_search_matches():
    rec = _rec(evidence_id="S-002", kind="search_file", requester="file_reader",
               path="/logs/alert.log", pattern="ORA-00600",
               result={"matches": [{"line_number": 42, "byte_offset": 900,
                                     "line": "ORA-00600 internal error"}],
                       "match_count": 1})
    out = materialize_evidence(rec)
    assert "S-002" in out
    assert "ORA-00600" in out
    assert "L42" in out


def test_materialize_read_file_content():
    rec = _rec(evidence_id="F-003", kind="read_file", requester="file_reader",
               path="/logs/alert.log", offset=100,
               result={"content": "processes exceeded\nresource limit"})
    out = materialize_evidence(rec)
    assert "F-003" in out
    assert "processes exceeded" in out


def test_materialize_error_status_is_explicit():
    rec = _rec(status="rejected", error="not in allowlist", result={})
    out = materialize_evidence(rec)
    assert "[rejected]" in out
    assert "not in allowlist" in out


def test_gather_across_refs_and_missing():
    ledger = EvidenceLedger()
    ledger.record_query("query_awr", "SELECT 1 FROM DBA_HIST_SQLSTAT WHERE dbid=1",
                        "sql", result={"rows": [{"N": 1}], "row_count": 1})
    # First recorded id is Q-001.
    finding = Finding(
        finding_id="sql-1-aaaa", agent="sql", turn=1, severity="high",
        code="X", description="d", evidence_refs=["Q-001", "Q-999"],
    )
    block, any_trunc, missing_refs = gather_finding_evidence(finding, ledger)
    assert "Q-001" in block
    assert "Q-999" in block and "NOT FOUND" in block
    assert any_trunc is False
    assert missing_refs == ["Q-999"]
