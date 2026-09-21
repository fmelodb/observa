"""Per-agent telemetry grid renderer.

A single ``Text`` covering N agents in a 2-column layout. Each cell shows
``{label:<5} {state_icon} {status:<10} {bar:<10} {tokens:>5}`` so the analyst
sees at a glance:

* state icon — animated braille spinner while the agent is alive,
  static glyph otherwise (✓ done, ✗ abstained, ⚠ error, ⏳ rate-limited,
  ⏸ idle).
* bar — 10 cells of █/░ proportional to the highest-tokens agent in the
  current frame; the bar color matches the agent's accent color.
* tokens — short form ("1.2k", "847", "—" when zero).
* evidence (optional) — ``{accesses:>3}q ⚡{hits:<2}`` shown when
  ``evidence_counters`` is passed; amber when hits > 0, dim otherwise.

Pure function — no Textual imports — so it can be exercised in unit tests.
"""
from __future__ import annotations

from rich.text import Text

from observa.llm import AgentRuntimeState

_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

_STATE_ICON: dict[str, tuple[str, str]] = {
    "done": ("✓", "#A9DFBF"),
    "abstained": ("✗", "#888888"),
    "error": ("⚠", "#E59866"),
    "rate_limited": ("⏳", "#F5B041"),
    "idle": ("⏸", "#3a3a3a"),
}

_STATUS_TEXT: dict[str, str] = {
    "thinking": "thinking",
    "tool": "tool",
    "done": "done",
    "abstained": "abstained",
    "error": "error",
    "rate_limited": "rate-limit",
    "idle": "waiting",
}

# 5-char short labels for compact 2-column layout.
AGENT_SHORT_LABEL: dict[str, str] = {
    "wait_time": "Wait",
    "ash": "ASH",
    "infra": "Infra",
    "sql": "SQL",
    "memory": "Mem",
    "io_storage": "IO",
    "concurrency": "Conc",
    "segment_object": "Segm",
    "rac": "RAC",
    "exadata": "Exa",
    "dg": "DG",
    "file_reader": "FileR",
}

# 11 distinct pastel hues — every agent gets its own color so log lines and
# grid cells can be told apart at a glance. Hues are spaced ~30° around the
# wheel; saturation/lightness held in the soft pastel range to match the
# rest of the palette.
AGENT_COLOR: dict[str, str] = {
    "wait_time": "#7FB3D5",       # steel blue
    "ash": "#BB8FCE",             # lavender
    "infra": "#A9DFBF",           # mint
    "sql": "#F5B7B1",             # salmon
    "memory": "#F9E79F",          # buttercream
    "io_storage": "#E59866",      # coral
    "concurrency": "#F5B041",     # amber
    "segment_object": "#F8BBD0",  # pink
    "rac": "#85C1E9",             # sky blue
    "exadata": "#D7BDE2",         # plum
    "dg": "#76D7C4",              # turquoise
    "file_reader": "#C5E1A5",     # sage
}
_DEFAULT_AGENT_COLOR = "#7FB3D5"


#: Width reserved for the token column. Sized for "999.9k" so the column
#: stays aligned even when one agent burns far more tokens than the others.
TOKENS_FIELD_WIDTH = 6


def _format_tokens(n: int) -> str:
    if n <= 0:
        return "—"
    if n < 1000:
        return f"{n}"
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.1f}M"


def _bar(tokens: int, peak: int, *, width: int = 10) -> tuple[str, str]:
    """Return (filled_part, empty_part) strings of length width total."""
    if peak <= 0 or tokens <= 0:
        return "", "░" * width
    filled = max(1, round(width * tokens / peak))
    filled = min(width, filled)
    return "█" * filled, "░" * (width - filled)


def _icon(status: str, spinner_frame: int) -> tuple[str, str]:
    if status in ("thinking", "tool"):
        return _BRAILLE[spinner_frame % len(_BRAILLE)], "#F5B041"
    return _STATE_ICON.get(status, _STATE_ICON["idle"])


def render_agent_grid(
    by_agent: dict[str, AgentRuntimeState],
    agents_order: list[str],
    spinner_frame: int,
    *,
    columns: int = 2,
    evidence_counters: dict[str, tuple[int, int]] | None = None,
) -> Text:
    """Render the agent grid as a single Rich Text block.

    ``agents_order`` is the canonical order of cells (e.g. registry order).
    Empty cells are produced by passing more agents than there is data for —
    they render with status ``idle`` and a ``—`` token count.
    """
    # Compute peak tokens for relative bar scaling.
    peak = max(
        (by_agent[a].tokens for a in agents_order if a in by_agent),
        default=0,
    )

    # Build per-agent cells.
    cells: list[Text] = []
    for agent in agents_order:
        state = by_agent.get(agent) or AgentRuntimeState()
        label = AGENT_SHORT_LABEL.get(agent, agent[:5])
        agent_color = AGENT_COLOR.get(agent, _DEFAULT_AGENT_COLOR)
        icon, icon_color = _icon(state.status, spinner_frame)
        status_text = _STATUS_TEXT.get(state.status, state.status)[:10]
        filled, empty = _bar(state.tokens, peak)
        tokens_text = _format_tokens(state.tokens)

        cell = Text()
        cell.append(f"{label:<5}", style=f"bold {agent_color}")
        cell.append(" ")
        cell.append(icon, style=icon_color)
        cell.append(" ")
        cell.append(f"{status_text:<10}", style="#888888")
        cell.append(" ")
        if filled:
            cell.append(filled, style=agent_color)
        if empty:
            cell.append(empty, style="#3a3a3a")
        cell.append(" ")
        cell.append(f"{tokens_text:>{TOKENS_FIELD_WIDTH}}", style="#d4d4d4")
        if evidence_counters is not None:
            accesses, hits = evidence_counters.get(agent, (0, 0))
            if accesses:
                cell.append(f" {accesses:>3}q", style="#888888")
                cell.append(f"⚡{hits:<2}", style="#F5B041" if hits else "#3a3a3a")
            else:
                cell.append(" " * 9)  # keep cells aligned when idle (9 cells: 5 + ⚡(2) + 2)
        cells.append(cell)

    # Lay out into rows.
    sep = Text("    ")  # 4 spaces between columns
    out = Text()
    rows = (len(cells) + columns - 1) // columns
    for r in range(rows):
        if r > 0:
            out.append("\n")
        for c in range(columns):
            idx = r * columns + c
            if idx >= len(cells):
                break
            if c > 0:
                out.append_text(sep)
            out.append_text(cells[idx])
    return out
