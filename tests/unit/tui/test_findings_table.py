"""Multi-line finding-block renderer + single-line status row."""
from __future__ import annotations

from observa.tui.widgets.findings_table import (
    AGENT_W,
    DESC_INDENT,
    SEVERITY_W,
    TIME_W,
    format_abstain_row,
    format_finding_block,
    format_finding_row,
    format_question_row,
    format_status_row,
)


# ---------------------------------------------------------------------------
# Block layout — headline + indented description
# ---------------------------------------------------------------------------

def test_finding_block_has_headline_then_description_lines():
    block = format_finding_block(
        "12:34:56", "wait_time", "high", "LOG_FILE_SYNC_HIGH",
        "Commit waits dominate average wait class",
    )
    lines = block.plain.split("\n")
    # At least 2 lines: headline + 1 description line.
    assert len(lines) >= 2
    assert "12:34:56" in lines[0]
    assert "[Wait]" in lines[0]   # bracketed agent label
    assert "HIGH" in lines[0]
    assert "LOG_FILE_SYNC_HIGH" in lines[0]
    # Description starts on line 2.
    assert "Commit waits" in lines[1]


def test_agent_label_is_bracketed():
    block = format_finding_block("12:00:00", "ash", "low", "C", "x")
    assert "[ASH]" in block.plain


def test_continuation_lines_are_indented_under_code_column():
    block = format_finding_block(
        "12:00:00", "ash", "low", "C",
        "this is a very long description that will definitely wrap to "
        "multiple lines so we can verify the continuation indent is consistent",
        term_width=80,
    )
    lines = block.plain.split("\n")
    assert len(lines) >= 3   # headline + at least 2 wrapped lines
    indent = " " * DESC_INDENT
    for cont in lines[1:]:
        assert cont.startswith(indent), f"continuation not indented: {cont!r}"


def test_long_description_wraps_without_truncation():
    """No ellipsis, no horizontal scroll — every word survives."""
    desc = " ".join(["palavra"] * 40)
    block = format_finding_block(
        "12:00:00", "ash", "high", "C", desc, term_width=80,
    )
    rendered = block.plain
    assert "…" not in rendered, "block must not truncate descriptions"
    # All 40 'palavra' tokens still present.
    assert rendered.count("palavra") == 40


def test_no_visual_line_exceeds_term_width():
    """Every wrapped line stays within the terminal — no horizontal scroll."""
    block = format_finding_block(
        "12:00:00", "ash", "high", "AAAAAAAAAAAAAAAAAA",
        "lorem ipsum " * 50, term_width=80,
    )
    for line in block.plain.split("\n"):
        assert len(line) <= 80, f"line too wide ({len(line)}): {line!r}"


def test_short_description_produces_two_lines():
    block = format_finding_block("12:00:00", "ash", "low", "C", "short note")
    lines = block.plain.split("\n")
    assert len(lines) == 2
    assert lines[1].endswith("short note")


def test_empty_description_still_renders_headline():
    block = format_finding_block("12:00:00", "ash", "info", "CODE", "")
    lines = block.plain.split("\n")
    assert "CODE" in lines[0]


# ---------------------------------------------------------------------------
# Headline alignment
# ---------------------------------------------------------------------------

def test_headlines_align_across_blocks():
    a = format_finding_block("12:00:00", "wait_time", "high", "X", "a")
    b = format_finding_block("12:00:01", "ash", "low", "Y_LONGER_CODE", "b")
    a_h = a.plain.split("\n")[0]
    b_h = b.plain.split("\n")[0]
    agent_start = TIME_W + 1
    assert a_h[agent_start:agent_start + AGENT_W].rstrip() == "[Wait]"
    assert b_h[agent_start:agent_start + AGENT_W].rstrip() == "[ASH]"
    sev_start = TIME_W + AGENT_W + 2
    assert a_h[sev_start:sev_start + SEVERITY_W].startswith("HIGH")
    assert b_h[sev_start:sev_start + SEVERITY_W].startswith("LOW")


def test_continuation_indent_aligns_exactly_with_code_column():
    """The description indent must match where the code column starts."""
    block = format_finding_block(
        "12:00:00", "ash", "high", "MYCODE",
        "first line " * 20, term_width=120,
    )
    lines = block.plain.split("\n")
    # Find where MYCODE starts on the headline.
    code_col = lines[0].index("MYCODE")
    # And where the first non-space char of the continuation starts.
    cont = lines[1]
    cont_col = len(cont) - len(cont.lstrip())
    assert code_col == cont_col == DESC_INDENT, (
        f"code starts at {code_col}, continuation indent {cont_col}, "
        f"DESC_INDENT={DESC_INDENT}"
    )


# ---------------------------------------------------------------------------
# Severity badges
# ---------------------------------------------------------------------------

def test_each_known_severity_renders_with_badge():
    for sev, badge in [
        ("critical", "CRIT"), ("high", "HIGH"), ("medium", "MED"),
        ("low", "LOW"), ("info", "INFO"), ("abstained", "ABST"),
        ("question", "?"),
    ]:
        block = format_finding_block("12:00:00", "ash", sev, "C", "d")
        assert badge in block.plain.split("\n")[0]


def test_unknown_severity_falls_back_uppercased():
    block = format_finding_block("12:00:00", "ash", "weird", "C", "d")
    assert "WEIR" in block.plain.split("\n")[0]


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def test_format_abstain_row_uses_abst_badge():
    block = format_abstain_row("12:00:00", "memory", "no PGA pressure observed")
    assert "ABST" in block.plain
    assert "no PGA pressure" in block.plain


def test_format_question_row_uses_question_badge():
    block = format_question_row("12:00:00", "rac", "is this a 4-node cluster?")
    plain = block.plain
    assert "?" in plain
    assert "4-node" in plain


def test_format_finding_row_alias_works():
    """Backward-compatible alias still functions."""
    a = format_finding_row("12:00:00", "ash", "low", "C", "x")
    b = format_finding_block("12:00:00", "ash", "low", "C", "x")
    assert a.plain == b.plain


# ---------------------------------------------------------------------------
# Status row (single-line)
# ---------------------------------------------------------------------------

def test_status_row_short_message_is_single_line():
    line = format_status_row("12:00:00", "master", "consolidating findings…")
    assert "\n" not in line.plain
    assert "12:00:00" in line.plain
    assert "consolidating" in line.plain


def test_status_row_long_message_wraps_with_indent():
    line = format_status_row("12:00:00", "master", "x " * 100, term_width=80)
    parts = line.plain.split("\n")
    assert len(parts) > 1
    indent = " " * DESC_INDENT
    for cont in parts[1:]:
        assert cont.startswith(indent)
