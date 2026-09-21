"""Right-aligned footer line summarizing prompt-cache effectiveness.

Aggregates ``cached_tokens``, ``cache_creation_tokens``, and ``tokens``
across every ``AgentRuntimeState`` in a ``TokenTracker``. Renders one
short line:

    cache: 145.2k read · 12.4k written · 87% hit

* "read"     — total tokens served from the cache (cheap, ~10% of full).
* "written"  — total tokens written to the cache this session (full
                price plus a small write premium on Anthropic).
* "hit"      — cache_read / (cache_read + fresh input). Approximates the
                fraction of input tokens that came from cache.

When ``evidence_totals`` is provided (accesses, hits), an extra segment
is appended:

    ·  evidence 37 · ⚡9 cache

Pure function — no Textual imports. Returns a ``rich.text.Text``.
"""
from __future__ import annotations

from rich.text import Text

from observa.llm import AgentRuntimeState

_LABEL_COLOR = "#A0A0A0"      # bright gray — always readable on the dark bg
_VALUE_ACTIVE = "#A9DFBF"     # mint when a value is non-zero
_VALUE_IDLE = "#888888"       # medium gray when value is 0 (still visible)
_HIT_ACTIVE = "#F5B041"       # amber for the hit-rate badge
_SEP_COLOR = "#555555"        # subtle but visible separator


def _short(n: int) -> str:
    if n <= 0:
        return "0"
    if n < 1000:
        return f"{n}"
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.1f}M"


def aggregate_cache_stats(
    by_agent: dict[str, AgentRuntimeState],
) -> tuple[int, int, int]:
    """Return ``(total_billable, total_cache_read, total_cache_creation)``.

    ``total_billable`` is the sum of ``tokens`` (which is already the
    cache-MISS portion — fresh input + output). Cache-read is broken out
    separately because it is billed at a fraction of the rate.
    """
    total_billable = 0
    total_cache_read = 0
    total_cache_creation = 0
    for st in by_agent.values():
        total_billable += st.tokens
        total_cache_read += st.cached_tokens
        total_cache_creation += st.cache_creation_tokens
    return total_billable, total_cache_read, total_cache_creation


def render_cache_footer(
    by_agent: dict[str, AgentRuntimeState],
    evidence_totals: tuple[int, int] | None = None,
) -> Text:
    """Build the right-aligned footer line.

    Always renders with visible colors so the analyst sees the widget is
    alive even before any LLM call has reported cache stats. Numbers turn
    mint when non-zero; the hit-rate badge turns amber once cache reads
    actually happen.
    """
    billable, cache_read, cache_creation = aggregate_cache_stats(by_agent)

    fresh_input_proxy = max(billable, 0)
    denom = cache_read + fresh_input_proxy
    hit_pct = (100 * cache_read / denom) if denom > 0 else 0.0

    read_style = _VALUE_ACTIVE if cache_read else _VALUE_IDLE
    write_style = _VALUE_ACTIVE if cache_creation else _VALUE_IDLE
    hit_style = f"bold {_HIT_ACTIVE}" if cache_read else _VALUE_IDLE

    line = Text()
    line.append("cache  ", style=_LABEL_COLOR)
    line.append(_short(cache_read), style=read_style)
    line.append(" read  ", style=_LABEL_COLOR)
    line.append("·", style=_SEP_COLOR)
    line.append("  ", style=_LABEL_COLOR)
    line.append(_short(cache_creation), style=write_style)
    line.append(" written  ", style=_LABEL_COLOR)
    line.append("·", style=_SEP_COLOR)
    line.append("  ", style=_LABEL_COLOR)
    line.append(f"{hit_pct:.0f}%", style=hit_style)
    line.append(" hit", style=_LABEL_COLOR)
    if evidence_totals is not None:
        accesses, hits = evidence_totals
        line.append("  ·  ", style=_SEP_COLOR)
        line.append("evidence ", style=_LABEL_COLOR)
        line.append(str(accesses), style=_VALUE_ACTIVE if accesses else _VALUE_IDLE)
        line.append(" · ", style=_SEP_COLOR)
        line.append(f"⚡{hits}", style=f"bold {_HIT_ACTIVE}" if hits else _VALUE_IDLE)
        line.append(" cache", style=_LABEL_COLOR)
    return line


__all__ = ["aggregate_cache_stats", "render_cache_footer"]
