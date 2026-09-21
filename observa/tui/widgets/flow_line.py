"""Pipeline flow renderer.

Single-line summary of the LangGraph pipeline phases:

    Turn 1 → Turn 2 → … → Turn N → Consolidate → Verify → RCA

Etapas concluídas em verde, etapa atual em âmbar com spinner braille,
futuras em cinza fraco. The renderer is a pure function so it is
trivially unit-testable.
"""
from __future__ import annotations

from rich.text import Text

_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

_DONE_COLOR = "#A9DFBF"
_ACTIVE_COLOR = "#F5B041"
_FUTURE_COLOR = "#3a3a3a"
_SEP_COLOR = "#666666"


def steps_for(total_turns: int) -> list[str]:
    """Return the ordered list of pipeline steps for a given turn count."""
    return [f"Turn {i + 1}" for i in range(total_turns)] + [
        "Consolidate",
        "Verify",
        "RCA",
    ]


def node_to_step_index(node: str, total_turns: int) -> int | None:
    """Map a graph node name to its step index, or None if unknown."""
    if node.startswith("turn_"):
        try:
            n = int(node.split("_", 1)[1])
            if 1 <= n <= total_turns:
                return n - 1
        except ValueError:
            return None
    if node in ("consolidate", "synthesize_hypotheses", "detect_contradictions",
                "resolve_contradiction"):
        return total_turns
    if node == "replay_verify":
        return total_turns + 1
    return None


def render_flow_line(
    total_turns: int,
    current_step_idx: int | None,
    *,
    all_done: bool = False,
    spinner_frame: int = 0,
) -> Text:
    """Render the flow line.

    ``current_step_idx`` is the index of the active step. When ``all_done``
    is True every step is shown as completed (RCA inclusive) and no spinner
    is drawn.
    """
    steps = steps_for(total_turns)
    line = Text()
    for i, label in enumerate(steps):
        if i > 0:
            line.append(" → ", style=_SEP_COLOR)
        if all_done:
            line.append(label, style=f"bold {_DONE_COLOR}")
            continue
        if current_step_idx is None:
            line.append(label, style=_FUTURE_COLOR)
            continue
        if i < current_step_idx:
            line.append(label, style=_DONE_COLOR)
        elif i == current_step_idx:
            line.append(label, style=f"bold {_ACTIVE_COLOR}")
            line.append(" ")
            line.append(
                _BRAILLE[spinner_frame % len(_BRAILLE)],
                style=f"bold {_ACTIVE_COLOR}",
            )
        else:
            line.append(label, style=_FUTURE_COLOR)
    return line
