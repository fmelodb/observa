"""Render a finished case as a self-contained HTML diagnostic report.

The report is a single file with inline CSS/JS (no external assets) and an
AWR-style information architecture: a fixed left-hand table of contents and a
summary-first body that drills down into per-agent findings, methodology,
dialogue, hypotheses, and evidence-replay verification. Source data is the final
``CaseState`` only — nothing is fetched or recomputed here.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import jinja2

from observa.export.view_model import SEVERITY_COLORS, build_view_model
from observa.models import CaseState

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "report.html.j2"


def _severity_color(value: str) -> str:
    return SEVERITY_COLORS.get((value or "").lower(), SEVERITY_COLORS["info"])


@functools.lru_cache(maxsize=1)
def _environment() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=jinja2.select_autoescape(["html", "j2", "html.j2"], default=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["sev_color"] = _severity_color
    return env


def render_html(state: CaseState | dict, ledger: Any = None) -> str:
    """Render the case state into a complete HTML document string."""
    view = build_view_model(state, ledger)
    template = _environment().get_template(_TEMPLATE_NAME)
    return template.render(**view)
