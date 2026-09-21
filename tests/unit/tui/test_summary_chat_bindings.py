"""Regression pins for SummaryChatScreen key bindings.

The chat ``Input`` is focused by default on this screen and Textual's Input
binds ``ctrl+e`` to "go to end of line" — without ``priority=True`` on the
screen binding, the HTML-export key is silently swallowed by the input.
"""
from __future__ import annotations

from textual.binding import Binding

from observa.tui.summary_chat import SummaryChatScreen


def _binding(key: str) -> Binding:
    for b in SummaryChatScreen.BINDINGS:
        if isinstance(b, Binding) and b.key == key:
            return b
    raise AssertionError(f"no binding for {key!r}")


def test_export_html_binding_beats_focused_input():
    assert _binding("ctrl+e").priority is True


def test_other_export_bindings_exist():
    assert _binding("ctrl+shift+m").action == "export_md"
    assert _binding("ctrl+shift+e").action == "export_chat"


def test_ctrl_d_evidence_binding_priority():
    binding = _binding("ctrl+d")
    assert binding.action == "toggle_evidence"
    assert binding.priority is True
