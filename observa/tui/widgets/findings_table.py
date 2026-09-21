"""Multi-line finding-block renderer for the LiveRun stream.

Each finding renders as a small block:

    HH:MM:SS  Agent   SEV    CODE
                             description text wraps to the available
                             terminal width, indented under the code
                             column for visual grouping.

The first line is the "headline": time / agent / severity badge / code.
The continuation lines hold the full description, soft-wrapped on word
boundaries — never truncated, never causing horizontal scroll. This keeps
the analyst's description data fully readable while the headline stays
scannable across rows.

Status rows (master / graph / analyst messages) keep the single-line
5-column form because their messages are short.

Pure functions — no Textual imports.
"""
from __future__ import annotations

import textwrap

from rich.text import Text

from observa.tui.widgets.agent_grid import AGENT_COLOR, AGENT_SHORT_LABEL


# --- Column widths (defaults; renderers accept overrides) ------------------

TIME_W = 8       # "HH:MM:SS"
AGENT_W = 8      # "[Wait]", "[ASH]", "[Infra]" — bracketed labels ≤ 7 chars + 1 slack
SEVERITY_W = 6   # "HIGH", "CRIT", "MED", "LOW", "INFO", "ABST", "?"   (4-char text padded)
CODE_W = 28      # "LOG_FILE_SYNC_HIGH" — soft cap with ellipsis on overflow
GUTTER = 1       # spaces between columns

DEFAULT_TERM_WIDTH = 120

# Indent the description block sits at on continuation lines. Aligns
# *exactly* with the start of the code column (time + agent + severity + 3 gutters).
DESC_INDENT = TIME_W + AGENT_W + SEVERITY_W + 3 * GUTTER  # = 25

_ELLIPSIS = "…"

# Severity → (4-char badge label, color). Aligned to fit in SEVERITY_W.
_SEVERITY_BADGE: dict[str, tuple[str, str]] = {
    "critical": ("CRIT", "#E59866"),
    "high":     ("HIGH", "#F5B041"),
    "medium":   ("MED ", "#F5E5B7"),
    "low":      ("LOW ", "#B7D5F5"),
    "info":     ("INFO", "#888888"),
    "abstained": ("ABST", "#666666"),
    "question": ("?   ", "#BB8FCE"),
}

_DEFAULT_SEVERITY_COLOR = "#d4d4d4"
_TIME_COLOR = "#666666"
_CODE_COLOR = "#F5B7B1"
_DESC_COLOR = "#d4d4d4"
_DEFAULT_AGENT_COLOR = "#7FB3D5"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width == 1:
        return _ELLIPSIS
    return text[: width - 1] + _ELLIPSIS


def _pad(text: str, width: int) -> str:
    return _truncate(text, width).ljust(width)


def _agent_label(agent: str) -> str:
    """Bracketed short label, e.g. ``[Wait]`` / ``[ASH]`` / ``[Infra]``."""
    short = AGENT_SHORT_LABEL.get(agent, agent[:AGENT_W - 2])
    return f"[{short}]"


def _agent_color(agent: str) -> str:
    return AGENT_COLOR.get(agent, _DEFAULT_AGENT_COLOR)


def _severity_badge(severity: str) -> tuple[str, str]:
    sev = (severity or "").lower()
    if sev in _SEVERITY_BADGE:
        return _SEVERITY_BADGE[sev]
    if not sev:
        return ("    ", _DEFAULT_SEVERITY_COLOR)
    return (sev.upper()[:4].ljust(4), _DEFAULT_SEVERITY_COLOR)


def _wrap_description(description: str, width: int) -> list[str]:
    """Soft-wrap a description on word boundaries.

    ``width`` is the available width for the description text (after indent).
    Returns at least one element — empty string when description is blank,
    so callers always get a "description line" to render.
    """
    text = (description or "").strip()
    if not text:
        return [""]
    width = max(20, width)  # never less than 20 cols
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=False,    # don't split mid-word
        break_on_hyphens=False,    # avoid splitting things like SQL_ID
    ) or [""]


# ---------------------------------------------------------------------------
# Block renderer (used for findings, abstains, questions)
# ---------------------------------------------------------------------------

def format_finding_block(
    time_str: str,
    agent: str,
    severity: str,
    code: str,
    description: str,
    *,
    term_width: int = DEFAULT_TERM_WIDTH,
) -> Text:
    """Render one finding as a multi-line block:

    Line 1: ``time  agent  SEV  CODE``
    Line 2..N: indented, word-wrapped description.

    The block returns a single ``rich.text.Text`` containing newlines —
    each visual line stays within ``term_width`` so RichLog never needs
    horizontal scrolling.
    """
    label = _agent_label(agent)
    badge_text, badge_color = _severity_badge(severity)
    code_text = (code or "—").strip() or "—"
    code_padded = _truncate(code_text, CODE_W)

    # Description column = whatever's left after the indent (at least 30 cols).
    desc_width = max(30, term_width - DESC_INDENT - 1)
    desc_lines = _wrap_description(description, desc_width)

    out = Text()
    # ── Headline ──────────────────────────────────────────────────────────
    out.append(_pad(time_str, TIME_W), style=_TIME_COLOR)
    out.append(" " * GUTTER)
    out.append(_pad(label, AGENT_W), style=f"bold {_agent_color(agent)}")
    out.append(" " * GUTTER)
    out.append(_pad(badge_text, SEVERITY_W), style=f"bold {badge_color}")
    out.append(" " * GUTTER)
    out.append(code_padded, style=_CODE_COLOR)

    # ── Description (indented, word-wrapped) ──────────────────────────────
    indent_str = " " * DESC_INDENT
    for desc_line in desc_lines:
        out.append("\n")
        out.append(indent_str)
        out.append(desc_line, style=_DESC_COLOR)
    return out


def format_finding_row(
    time_str: str,
    agent: str,
    severity: str,
    code: str,
    description: str,
    *,
    term_width: int = DEFAULT_TERM_WIDTH,
) -> Text:
    """Backward-compatible alias for ``format_finding_block``."""
    return format_finding_block(
        time_str=time_str, agent=agent, severity=severity,
        code=code, description=description, term_width=term_width,
    )


def format_abstain_row(
    time_str: str,
    agent: str,
    reason: str,
    *,
    term_width: int = DEFAULT_TERM_WIDTH,
) -> Text:
    return format_finding_block(
        time_str=time_str, agent=agent, severity="abstained",
        code="—", description=reason or "nothing relevant",
        term_width=term_width,
    )


def format_question_row(
    time_str: str,
    agent: str,
    question: str,
    *,
    term_width: int = DEFAULT_TERM_WIDTH,
) -> Text:
    return format_finding_block(
        time_str=time_str, agent=agent, severity="question",
        code="—", description=question or "",
        term_width=term_width,
    )


# ---------------------------------------------------------------------------
# Single-line status rows (master / graph / analyst messages)
# ---------------------------------------------------------------------------

def format_status_row(
    time_str: str,
    agent: str,
    message: str,
    *,
    severity: str = "",
    term_width: int = DEFAULT_TERM_WIDTH,
) -> Text:
    """Single-line variant for short orchestration messages.

    These are not findings — usually one short string ("consolidating
    findings…", "complete — opening summary"). Keeping them on one line
    avoids visual noise. Description still wraps if the message is long.
    """
    label = _agent_label(agent)
    badge_text, badge_color = _severity_badge(severity)

    desc_width = max(30, term_width - DESC_INDENT - 1)
    msg_lines = _wrap_description(message, desc_width)

    out = Text()
    out.append(_pad(time_str, TIME_W), style=_TIME_COLOR)
    out.append(" " * GUTTER)
    out.append(_pad(label, AGENT_W), style=f"bold {_agent_color(agent)}")
    out.append(" " * GUTTER)
    out.append(_pad(badge_text, SEVERITY_W), style=f"bold {badge_color}")
    out.append(" " * GUTTER)
    # First message line sits next to the badge (no extra indent).
    out.append(msg_lines[0], style=_DESC_COLOR)
    indent_str = " " * DESC_INDENT
    for line in msg_lines[1:]:
        out.append("\n")
        out.append(indent_str)
        out.append(line, style=_DESC_COLOR)
    return out


__all__ = [
    "AGENT_W",
    "CODE_W",
    "DESC_INDENT",
    "SEVERITY_W",
    "TIME_W",
    "format_abstain_row",
    "format_finding_block",
    "format_finding_row",
    "format_question_row",
    "format_status_row",
]
