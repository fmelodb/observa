"""Tests for the per-agent telemetry grid renderer."""
from __future__ import annotations

from observa.llm import AgentRuntimeState
from observa.tui.widgets.agent_grid import (
    _bar,
    _format_tokens,
    _icon,
    render_agent_grid,
)


def test_format_tokens() -> None:
    assert _format_tokens(0) == "—"
    assert _format_tokens(847) == "847"
    assert _format_tokens(1200) == "1.2k"
    assert _format_tokens(15500) == "15.5k"


def test_bar_zero_tokens_is_all_empty() -> None:
    filled, empty = _bar(0, peak=100)
    assert filled == ""
    assert empty == "░" * 10


def test_bar_full_tokens_caps_at_width() -> None:
    filled, empty = _bar(150, peak=100)
    assert filled == "█" * 10
    assert empty == ""


def test_bar_relative_to_peak() -> None:
    filled, empty = _bar(50, peak=100, width=10)
    assert len(filled) == 5
    assert len(empty) == 5


def test_bar_minimum_one_when_nonzero() -> None:
    # Tiny tokens vs huge peak should still show at least one █.
    filled, _ = _bar(1, peak=10_000)
    assert filled == "█"


def test_icon_animates_when_thinking() -> None:
    char_a, _ = _icon("thinking", spinner_frame=0)
    char_b, _ = _icon("thinking", spinner_frame=1)
    assert char_a != char_b


def test_icon_static_for_terminal_states() -> None:
    assert _icon("done", 0)[0] == "✓"
    assert _icon("abstained", 0)[0] == "✗"
    assert _icon("error", 0)[0] == "⚠"
    assert _icon("rate_limited", 0)[0] == "⏳"
    assert _icon("idle", 0)[0] == "⏸"


def test_render_grid_layout_two_columns() -> None:
    by_agent = {
        "wait_time": AgentRuntimeState(tokens=1000, status="thinking"),
        "ash": AgentRuntimeState(tokens=400, status="done"),
        "sql": AgentRuntimeState(tokens=200, status="abstained"),
        "rac": AgentRuntimeState(tokens=0, status="idle"),
    }
    order = ["wait_time", "ash", "sql", "rac"]
    text = render_agent_grid(by_agent, order, spinner_frame=3)
    rendered = text.plain
    # Two columns × two rows.
    assert rendered.count("\n") == 1
    # Each agent label appears.
    for label in ("Wait", "ASH", "SQL", "RAC"):
        assert label in rendered


def test_render_grid_handles_missing_agents() -> None:
    by_agent = {"wait_time": AgentRuntimeState(tokens=10, status="done")}
    order = ["wait_time", "ash", "sql"]
    text = render_agent_grid(by_agent, order, spinner_frame=0)
    rendered = text.plain
    # Missing agents render as idle with em-dash token count.
    assert "ASH" in rendered
    assert "SQL" in rendered
    assert "—" in rendered


def test_render_grid_relative_bar_against_peak() -> None:
    by_agent = {
        "wait_time": AgentRuntimeState(tokens=500, status="done"),
        "ash": AgentRuntimeState(tokens=1000, status="done"),
    }
    text = render_agent_grid(by_agent, ["wait_time", "ash"], spinner_frame=0)
    rendered = text.plain
    # Peak agent should have a longer or equal █ run.
    wait_section = rendered.split("    ASH")[0]
    ash_section = rendered.split("ASH")[1] if "ASH" in rendered else ""
    assert wait_section.count("█") <= ash_section.count("█")


def test_render_grid_eleven_agents_in_three_columns() -> None:
    """The full registry has 11 agents → 4 rows × 3 cols (last cell empty)."""
    order = [
        "wait_time", "ash", "infra", "sql", "memory", "io_storage",
        "concurrency", "segment_object", "rac", "exadata", "dg",
    ]
    by_agent = {a: AgentRuntimeState(tokens=10, status="idle") for a in order}
    text = render_agent_grid(by_agent, order, spinner_frame=0, columns=3)
    # 11 agents at 3 cols → 4 rows → 3 newlines.
    assert text.plain.count("\n") == 3


def test_grid_renders_evidence_counters():
    from observa.llm import AgentRuntimeState
    from observa.tui.widgets.agent_grid import render_agent_grid

    text = render_agent_grid(
        {"wait_time": AgentRuntimeState()},
        ["wait_time", "ash"],
        spinner_frame=0,
        evidence_counters={"wait_time": (12, 4)},
    )
    plain = text.plain
    assert "12q" in plain and "⚡4" in plain


def test_grid_without_evidence_counters_unchanged():
    from observa.llm import AgentRuntimeState
    from observa.tui.widgets.agent_grid import render_agent_grid

    text = render_agent_grid(
        {"wait_time": AgentRuntimeState()}, ["wait_time"], spinner_frame=0,
    )
    assert "⚡" not in text.plain


def test_grid_idle_agent_renders_aligned_placeholder():
    """Agents with zero accesses show blank padding, not '0q ⚡0'."""
    from observa.llm import AgentRuntimeState
    from observa.tui.widgets.agent_grid import render_agent_grid

    text = render_agent_grid(
        {"wait_time": AgentRuntimeState()},
        ["wait_time", "ash"],
        spinner_frame=0,
        evidence_counters={"wait_time": (12, 4)},
    )
    plain = text.plain
    assert "0q" not in plain          # idle ash cell has no zero counter
    assert "ASH" in plain             # ash cell still rendered


def test_active_and_idle_evidence_segments_same_cell_width():
    """Active and idle evidence segments must be the same cell width.

    ⚡ (U+26A1) is a wide character — 2 terminal cells — so the active
    segment ``" {n:>3}q⚡{h:<2}"`` is 9 cells, not 8. The idle placeholder
    must match so rows mixing active and idle agents stay aligned.
    """
    from rich.cells import cell_len
    import inspect
    from observa.tui.widgets import agent_grid

    active = f" {12:>3}q⚡{4:<2}"   # U+26A1 = ⚡
    assert cell_len(active) == 9, (
        f"active evidence segment is {cell_len(active)} cells, expected 9"
    )
    src = inspect.getsource(agent_grid.render_agent_grid)
    assert '" " * 9' in src, (
        "idle placeholder must be ' ' * 9 to match the 9-cell active segment"
    )
