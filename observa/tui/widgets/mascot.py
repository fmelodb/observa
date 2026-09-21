"""Shared mascot/title rendering used by NewCaseScreen and LiveRunScreen.

The mascot is a 4-line ASCII android in amber, paired with the wordmark
``observa`` and a two-line deadpan. The compact variant drops the punchline
so it fits in tighter top-bars.
"""
from __future__ import annotations

from rich.text import Text


_AMBER = "#F5B041"
_SUB_DIM = "#666666"


def build_mascot(compact: bool = False) -> Text:
    """Return a Rich ``Text`` containing the mascot, title, and subtitle.

    Parameters
    ----------
    compact:
        When True, the punchline is omitted so the mascot fits in a
        constrained top-bar (e.g. LiveRunScreen). The 4-line ASCII figure
        is preserved either way.
    """
    hero = Text()
    hero.append("   ╭────╮\n", style=_AMBER)
    hero.append("   │", style=_AMBER)
    hero.append("◉  ◉", style=f"bold {_AMBER}")
    hero.append("│   ", style=_AMBER)
    hero.append("observa\n", style=f"bold {_AMBER}")
    hero.append("   ╞════╡   ", style=_AMBER)
    hero.append('"Nothing changed."\n', style=_SUB_DIM)
    if compact:
        hero.append("    ╨  ╨    ", style=_AMBER)
    else:
        hero.append("    ╨  ╨    ", style=_AMBER)
        hero.append("— every incident, ever", style=f"italic {_SUB_DIM}")
    return hero
