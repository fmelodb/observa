"""Tests for the HTML report renderer."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from observa.export import render_html
from observa.models import (
    AgentQuestion,
    ChatMessage,
    ContradictionResolution,
    Finding,
    FinalSummary,
    FindingVerification,
    Hypothesis,
    InputFile,
    SummaryFinding,
    TurnFindings,
    VerificationResult,
)


def _full_state() -> dict:
    """A case state exercising every report section."""
    return {
        "case_id": "CASE-2026-0529-01",
        "dbid": 12345,
        "oracle_version": "19.20",
        "topology": "RAC 2-node",
        "instance_numbers": [1, 2],
        "total_turns": 3,
        "problem_statement": "Batch job slow since Monday.",
        "investigation_scope": "SQL_ID abc123 between 22:00-23:00.",
        "known_facts": ["No recent deploy", "Stats gathered nightly"],
        "input_files": [
            InputFile(type="alertlog", path="/logs/alert.log", label="prod"),
            InputFile(type="awr_report", path="/logs/awr.html"),
        ],
        "agents_active_per_turn": {1: ["sql", "wait_time"], 2: ["sql"]},
        "agents_skipped": ["exadata"],
        "agents_skip_reasons": {"exadata": "Not an Exadata system"},
        "agents_refinement_log": {2: {"wait_time": "No relevant waits in turn 1"}},
        "turn_findings": [
            TurnFindings(
                agent="sql",
                turn=1,
                findings=[
                    Finding(
                        finding_id="F1",
                        agent="sql",
                        turn=1,
                        severity="high",
                        code="PLAN_FLIP",
                        description="Plan changed to nested loops.",
                        evidence_note="DBA_HIST_SQLSTAT shows plan_hash change",
                        related_objects=["SCHEMA.ORDERS"],
                    )
                ],
            ),
            TurnFindings(
                agent="wait_time",
                turn=1,
                abstained=True,
                abstain_reason="Watchdog timeout after 45s",
            ),
        ],
        "agent_questions": [
            AgentQuestion(agent="sql", turn=1, question="Was the index rebuilt?")
        ],
        "chat_log": [
            ChatMessage(
                sender="user",
                turn=1,
                text="No index changes.",
                timestamp=datetime(2026, 5, 29, 22, 30, tzinfo=timezone.utc),
            )
        ],
        "hypotheses": [
            Hypothesis(
                hyp_id="H1",
                statement="Bad plan from stale stats",
                prior=0.3,
                posterior=0.75,
                status="validated",
            )
        ],
        "contradictions": [
            {"hyp_a": "H1", "hyp_b": "H2", "explanation": "Cannot both be true"}
        ],
        "contradiction_resolutions": [
            ContradictionResolution(
                hyp_a="H1",
                hyp_b="H2",
                favored_hyp_id="H1",
                reasoning="Stronger evidence for plan flip.",
                posterior_a=0.8,
                posterior_b=0.2,
            )
        ],
        "final_summary": FinalSummary(
            problem_restated="Batch slowed due to a plan change.",
            top_findings=[
                SummaryFinding(
                    agent="sql",
                    severity="high",
                    description="Plan flip to nested loops.",
                    evidence="plan_hash changed",
                )
            ],
            root_cause="Stale statistics caused a poor execution plan.",
            confidence=0.78,
            unknowns=["Whether stats job ran"],
            recommended_next_steps=["Gather stats and re-test"],
        ),
        "verification": VerificationResult(
            verifications=[
                FindingVerification(
                    finding_id="top-1",
                    claim="Plan flip to nested loops.",
                    verdict="supported",
                    reasoning="plan_hash differs",
                    evidence_refs=["Q-001"],
                )
            ],
            support_score=1.0,
            applied_confidence_factor=1.0,
            unsupported=[],
        ),
    }


def test_full_report_contains_all_sections() -> None:
    html = render_html(_full_state())
    assert html.startswith("<!DOCTYPE html>")
    # Section anchors present
    for anchor in (
        'id="summary"',
        'id="inputs"',
        'id="methodology"',
        'id="findings"',
        'id="dialogue"',
        'id="verification"',
    ):
        assert anchor in html, anchor
    # Key content
    assert "Stale statistics caused a poor execution plan." in html
    assert "78%" in html  # confidence
    assert "SQL &amp; Plans" in html  # agent display label (escaped &)
    assert "Watchdog timeout after 45s" in html  # abstention surfaced
    assert "Stronger evidence for plan flip." in html  # contradiction resolution
    assert "Was the index rebuilt?" in html  # agent question
    assert 'class="verdict-supported"' in html  # verification badge markup


def test_autoescape_blocks_injected_html() -> None:
    state = _full_state()
    state["problem_statement"] = "<script>alert(1)</script>"
    html = render_html(state)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_minimal_state_omits_optional_sections() -> None:
    state = {
        "case_id": "CASE-MIN",
        "final_summary": FinalSummary(
            problem_restated="x",
            top_findings=[],
            root_cause="Unknown root cause.",
            confidence=0.2,
        ),
    }
    html = render_html(state)
    assert 'id="summary"' in html
    assert "Unknown root cause." in html
    # No data for these — sections must be omitted.
    assert 'id="verification"' not in html
    assert 'id="dialogue"' not in html
    assert 'id="methodology"' not in html


def test_empty_state_renders_without_error() -> None:
    html = render_html({})
    assert "<!DOCTYPE html>" in html
    assert "unnamed" in html  # fallback case id


def test_view_model_carries_windows_and_diff():
    from datetime import datetime

    from observa.export.view_model import build_view_model
    from observa.models import SnapWindow

    state = {
        "case_id": "CASE-T",
        "problem_window": SnapWindow(
            begin_snap=100, end_snap=102,
            begin_time=datetime(2026, 7, 8, 14, 0), end_time=datetime(2026, 7, 8, 16, 0),
        ),
        "baseline_window": None,
        "baseline_diff": {
            "has_baseline": False, "suspect": False, "errors": [],
            "problem_label": "snaps 100→102", "baseline_label": None,
            "load_profile": [], "top_waits_problem": [], "top_waits_baseline": [],
            "top_sql_problem": [], "top_sql_baseline": [], "os": [],
        },
        "baseline_suspect": False,
    }
    vm = build_view_model(state)
    assert vm["problem_window"] is not None
    assert "snaps 100" in vm["baseline_diff_text"]
    assert vm["has_methodology"] is True


def test_render_html_shows_windows_section():
    from datetime import datetime

    from observa.export.html_report import render_html as render_html_direct
    from observa.models import SnapWindow

    state = {
        "case_id": "CASE-T",
        "problem_window": SnapWindow(
            begin_snap=100, end_snap=102,
            begin_time=datetime(2026, 7, 8, 14, 0), end_time=datetime(2026, 7, 8, 16, 0),
        ),
        "baseline_window": SnapWindow(begin_snap=50, end_snap=52),
        "baseline_diff": {
            "has_baseline": True, "suspect": True, "errors": [],
            "problem_label": "snaps 100→102", "baseline_label": "snaps 50→52",
            "load_profile": [], "top_waits_problem": [], "top_waits_baseline": [],
            "top_sql_problem": [], "top_sql_baseline": [], "os": [],
        },
        "baseline_suspect": True,
    }
    html = render_html_direct(state)
    assert "Analysis windows" in html
    assert "100" in html and "102" in html
    assert "Baseline diff" in html
    assert "differs" in html  # suspect banner


def _ledger_with_records():
    from observa.mcp.evidence import EvidenceLedger

    led = EvidenceLedger()
    led.record_query(
        "query_awr",
        "SELECT event_name FROM DBA_HIST_SYSTEM_EVENT "
        "WHERE dbid = 1234567890 AND snap_id BETWEEN 4210 AND 4215",
        "wait_time", turn=1,
        result={"rows": [{"EVENT_NAME": "db file sequential read"}],
                "truncated": False, "row_count": 1, "warnings": []},
        duration_ms=1180,
    )
    led.record_query("query_awr", "SELECT 1 FROM V$SESSION", "sql",
                     status="rejected", error="not in allowlist")
    return led


def test_report_renders_evidence_appendix_and_chips():
    from observa.export import render_html

    state = _full_state()
    # give the first agent finding a citation
    state["turn_findings"][0].findings[0].evidence_refs = ["Q-001"]
    html = render_html(state, _ledger_with_records())
    assert 'id="evidence"' in html                       # appendix section
    assert 'id="evidence-Q-001"' in html                 # anchor target
    assert 'href="#evidence-Q-001"' in html              # chip link
    assert "not in allowlist" in html                    # rejected recorded
    assert "not cited" in html                           # Q-002 uncited marker
    assert "DBA_HIST_SYSTEM_EVENT" in html               # SQL text present


def test_report_without_ledger_has_no_appendix():
    from observa.export import render_html

    html = render_html(_full_state())
    assert 'id="evidence"' not in html


def test_appendix_sample_rows_capped_at_20():
    from observa.export.view_model import build_view_model
    from observa.mcp.evidence import EvidenceLedger

    led = EvidenceLedger()
    led.record_query(
        "query_awr",
        "SELECT n FROM DBA_HIST_SYSSTAT WHERE dbid = 1 AND snap_id BETWEEN 1 AND 2",
        "a",
        result={"rows": [{"N": i} for i in range(50)], "truncated": False,
                "row_count": 50, "warnings": []},
    )
    vm = build_view_model(_full_state(), led)
    rec = vm["evidence_records"][0]
    assert len(rec["sample_rows"]) == 20 and rec["sample_truncated"] is True


def test_report_renders_ora_timeline_table():
    from observa.export import render_html

    state = _full_state()
    state["ora_timeline"] = {
        "coverage": {"begin_ts": "2024-03-15T00:00:00", "end_ts": "2024-03-15T23:00:00",
                     "covers_problem": True, "covers_baseline": False},
        "counts": {"problem": 1, "baseline": 0, "outside": 0, "incidents": 1},
        "incidents": [{"ts": "2024-03-15T14:30:00", "code": "ORA-00060", "text": "deadlock",
                       "instance": 2, "incident_id": 48291, "byte_offset": 10,
                       "window": "problem", "source_label": "prod", "evidence_id": "S-001"}],
    }
    html = render_html(state)
    assert "ORA incidents" in html
    assert "ORA-00060" in html
    assert "ora-badge-problem" in html   # per-window badge class


def test_report_ora_not_covered_warning():
    from observa.export import render_html

    state = _full_state()
    state["ora_timeline"] = {
        "coverage": {"begin_ts": "2024-03-01T00:00:00", "end_ts": "2024-03-10T00:00:00",
                     "covers_problem": False, "covers_baseline": False},
        "counts": {"problem": 0, "baseline": 0, "outside": 1, "incidents": 1},
        "incidents": [{"ts": "2024-03-01T02:00:00", "code": "ORA-00600", "text": "err",
                       "instance": 1, "incident_id": None, "byte_offset": 10,
                       "window": "outside", "source_label": "prod", "evidence_id": "S-001"}],
    }
    html = render_html(state)
    assert "does not cover the problem window" in html


def test_report_covered_but_clean_shows_coverage_line():
    # A log that COVERS the window but has zero incidents still shows its
    # coverage line — the "searched the window, found nothing" signal.
    from observa.export import render_html

    state = _full_state()
    state["ora_timeline"] = {
        "coverage": {"begin_ts": "2024-03-15T00:00:00", "end_ts": "2024-03-15T23:00:00",
                     "covers_problem": True, "covers_baseline": False},
        "counts": {"problem": 0, "baseline": 0, "outside": 0, "incidents": 0},
        "incidents": [],
    }
    html = render_html(state)
    assert "ORA incidents" in html
    assert "Alert log covers" in html
    assert "does not cover the problem window" not in html


def test_report_without_ora_timeline_has_no_table():
    from observa.export import render_html
    assert "ORA incidents" not in render_html(_full_state())


def test_ora_row_with_null_instance_renders_blank_cell():
    from observa.export import render_html
    state = _full_state()
    state["ora_timeline"] = {
        "coverage": {"begin_ts": "2024-03-15T00:00:00", "end_ts": "2024-03-15T23:00:00",
                     "covers_problem": True, "covers_baseline": False},
        "counts": {"problem": 1, "baseline": 0, "outside": 0, "incidents": 1},
        "incidents": [{"ts": "2024-03-15T14:30:00", "code": "ORA-00060", "text": "x",
                       "instance": None, "incident_id": None, "byte_offset": 10,
                       "window": "problem", "source_label": "prod", "evidence_id": "S-001"}],
    }
    html = render_html(state)
    assert "ORA-00060" in html          # row renders
    assert 'href="#evidence-S-001"' in html  # evidence_id hyperlinked


def test_verification_section_renders_badges():
    from observa.export.view_model import build_view_model
    from observa.models import (
        FindingVerification, VerificationResult, FinalSummary, SummaryFinding,
        Hypothesis, ContradictionResolution,
    )
    state = {
        "final_summary": FinalSummary(
            problem_restated="p", root_cause="rc", confidence=0.6,
            top_findings=[SummaryFinding(agent="sql", severity="high",
                                         description="full scan", evidence_refs=["Q-001"])],
        ),
        "verification": VerificationResult(
            verifications=[FindingVerification(
                finding_id="top-1", claim="full scan", verdict="supported",
                reasoning="rows confirm", evidence_refs=["Q-001"])],
            support_score=1.0, applied_confidence_factor=1.0, unsupported=[],
        ),
        "hypotheses": [Hypothesis(hyp_id="H-1", statement="cpu", prior=0.6, posterior=0.6)],
        "contradiction_resolutions": [ContradictionResolution(
            hyp_a="H-1", hyp_b="H-2", favored_hyp_id="H-1",
            reasoning="evidence favors H-1", posterior_a=0.8, posterior_b=0.2)],
    }
    vm = build_view_model(state)
    assert vm["has_verification"] is True
    assert vm["verification"].verifications[0].verdict == "supported"
    assert vm["contradiction_resolutions"][0].favored_hyp_id == "H-1"
