"""Tests for the pipeline flow line renderer."""
from __future__ import annotations

import pytest

from observa.tui.widgets.flow_line import (
    node_to_step_index,
    render_flow_line,
    steps_for,
)


def test_steps_for_three_turns() -> None:
    assert steps_for(3) == [
        "Turn 1", "Turn 2", "Turn 3", "Consolidate", "Verify", "RCA",
    ]


def test_steps_for_two_turns() -> None:
    assert steps_for(2) == ["Turn 1", "Turn 2", "Consolidate", "Verify", "RCA"]


@pytest.mark.parametrize(
    "node,total_turns,expected",
    [
        ("turn_1", 3, 0),
        ("turn_2", 3, 1),
        ("turn_3", 3, 2),
        ("consolidate", 3, 3),
        ("synthesize_hypotheses", 3, 3),
        ("detect_contradictions", 3, 3),
        ("resolve_contradiction", 3, 3),
        ("replay_verify", 3, 4),
        ("turn_5", 3, None),  # outside total_turns
        ("init", 3, None),
        ("unknown", 3, None),
    ],
)
def test_node_to_step_index(node: str, total_turns: int, expected: int | None) -> None:
    assert node_to_step_index(node, total_turns) == expected


def test_render_flow_line_idle_all_dim() -> None:
    text = render_flow_line(total_turns=3, current_step_idx=None)
    rendered = text.plain
    assert "Turn 1" in rendered and "RCA" in rendered
    # Five separators expected: 6 steps - 1.
    assert rendered.count("→") == 5


def test_render_flow_line_active_step_shows_spinner() -> None:
    text = render_flow_line(total_turns=3, current_step_idx=1, spinner_frame=0)
    rendered = text.plain
    # Some braille char must appear after "Turn 2".
    assert "Turn 2" in rendered
    braille = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    assert any(c in rendered for c in braille)


def test_render_flow_line_all_done_no_spinner() -> None:
    text = render_flow_line(total_turns=3, current_step_idx=5, all_done=True)
    rendered = text.plain
    braille = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    assert not any(c in rendered for c in braille)
    assert "RCA" in rendered


def test_render_flow_line_dynamic_turn_count() -> None:
    text = render_flow_line(total_turns=2, current_step_idx=2)
    rendered = text.plain
    assert "Turn 1" in rendered
    assert "Turn 2" in rendered
    assert "Turn 3" not in rendered
    assert "Consolidate" in rendered
