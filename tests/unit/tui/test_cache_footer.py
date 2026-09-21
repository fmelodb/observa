"""Aggregate cache-stats footer renderer."""
from __future__ import annotations

from observa.llm import AgentRuntimeState
from observa.tui.widgets.cache_footer import (
    aggregate_cache_stats,
    render_cache_footer,
)


def _state(tokens: int = 0, cached: int = 0, created: int = 0) -> AgentRuntimeState:
    return AgentRuntimeState(
        tokens=tokens, cached_tokens=cached, cache_creation_tokens=created,
    )


# ---------------------------------------------------------------------------
# aggregate_cache_stats
# ---------------------------------------------------------------------------

def test_aggregate_empty_dict_returns_zeros():
    assert aggregate_cache_stats({}) == (0, 0, 0)


def test_aggregate_sums_across_agents():
    by_agent = {
        "ash": _state(tokens=100, cached=500, created=10),
        "sql": _state(tokens=200, cached=300, created=20),
        "memory": _state(tokens=50, cached=0, created=5),
    }
    billable, read, created = aggregate_cache_stats(by_agent)
    assert billable == 350
    assert read == 800
    assert created == 35


# ---------------------------------------------------------------------------
# render_cache_footer
# ---------------------------------------------------------------------------

def test_render_always_visible_even_with_no_cache_activity():
    """Footer must always render with the standard structure — no invisible
    placeholder — so the analyst can confirm the widget is alive."""
    by_agent = {"ash": _state(tokens=1000)}
    text = render_cache_footer(by_agent).plain
    assert "cache" in text
    assert "read" in text
    assert "written" in text
    assert "hit" in text
    assert "0" in text  # numbers shown explicitly when zero


def test_render_with_cache_activity_shows_read_written_hit():
    by_agent = {
        "ash": _state(tokens=100, cached=900, created=50),
    }
    text = render_cache_footer(by_agent).plain
    # Contains all three sections.
    assert "read" in text
    assert "written" in text
    assert "hit" in text
    assert "900" in text   # cache_read short form
    assert "50" in text


def test_hit_rate_high_when_cache_dominates():
    """cached=900, billable=100 → 90% hit (900/(900+100))."""
    by_agent = {"ash": _state(tokens=100, cached=900)}
    text = render_cache_footer(by_agent).plain
    assert "90% hit" in text


def test_hit_rate_zero_when_only_fresh_input():
    by_agent = {"ash": _state(tokens=1000, cached=0, created=100)}
    text = render_cache_footer(by_agent).plain
    # cache_read=0 → 0% hit (but cache_creation > 0 so it still renders).
    assert "0% hit" in text
    assert "100" in text  # cache_creation


def test_short_format_for_large_numbers():
    by_agent = {"ash": _state(tokens=10_000, cached=2_500_000)}
    text = render_cache_footer(by_agent).plain
    assert "2.5M" in text


def test_short_format_thousands():
    by_agent = {"ash": _state(tokens=100, cached=12_345)}
    text = render_cache_footer(by_agent).plain
    assert "12.3k" in text


def test_footer_renders_evidence_totals():
    from observa.tui.widgets.cache_footer import render_cache_footer

    text = render_cache_footer({}, evidence_totals=(37, 9))
    plain = text.plain
    assert "37" in plain and "⚡9" in plain and "evidence" in plain


def test_footer_without_evidence_totals_unchanged():
    from observa.tui.widgets.cache_footer import render_cache_footer

    assert "evidence" not in render_cache_footer({}).plain
