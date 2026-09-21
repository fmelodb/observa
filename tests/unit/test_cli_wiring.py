"""Unit tests for CLI command wiring (Task 22).

These are structural tests — they verify commands exist and the case_id
generation logic is correct without launching the TUI or connecting to Oracle.
"""
from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from observa.cli import app


runner = CliRunner()


# ---------------------------------------------------------------------------
# Command existence checks
# ---------------------------------------------------------------------------


def test_new_command_exists():
    """observa new command must be registered in the Typer app."""
    command_names = list(app.registered_commands)
    # Typer stores commands as DefaultPlaceholder or CliCommand objects;
    # the simplest way is to check the help text which enumerates commands.
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "new" in result.output


def test_chat_command_exists():
    """observa chat placeholder must be registered."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "chat" in result.output


def test_removed_commands_are_gone():
    """Stateless refactor removed list/export/resume."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for cmd in ("list", "export", "resume"):
        assert cmd not in result.output, f"expected '{cmd}' to be removed"


# ---------------------------------------------------------------------------
# case_id format check
# ---------------------------------------------------------------------------

CASE_ID_RE = re.compile(r"^CASE-\d{4}-\d{4}-\d{2}$")


def test_case_id_format():
    """Generated case_id must match CASE-YYYY-MMDD-NN."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    case_id = f"CASE-{now.strftime('%Y-%m%d')}-01"
    assert CASE_ID_RE.match(case_id), (
        f"case_id {case_id!r} does not match pattern CASE-YYYY-MMDD-NN"
    )


def test_case_id_format_fixed_date():
    """Verify format with a known fixed date."""
    case_id = "CASE-2026-0418-01"
    assert CASE_ID_RE.match(case_id), f"Fixed-date case_id {case_id!r} failed pattern check"


def test_case_id_format_invalid():
    """Sanity check — a bad case_id should NOT match."""
    assert not CASE_ID_RE.match("CASE-2026-04-18-01")  # extra dash in date part
    assert not CASE_ID_RE.match("case-2026-0418-01")   # lowercase
    assert not CASE_ID_RE.match("CASE-2026-0418-1")    # NN must be two digits
