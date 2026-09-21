"""Topology probe + deterministic agent skip filter."""
from __future__ import annotations

import pytest

from observa.graph.topology_probe import (
    TopologyFlags,
    filter_skipped_agents,
    probe_topology,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _ScriptedRunner:
    """Returns canned responses keyed by SQL fragment."""

    def __init__(self, responses: dict[str, dict | Exception]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    async def __call__(self, sql: str) -> dict:
        self.calls.append(sql)
        for fragment, response in self._responses.items():
            if fragment in sql:
                if isinstance(response, Exception):
                    raise response
                return response
        return {"rows": [{"C": 0}], "row_count": 1}


def _count_response(n: int) -> dict:
    return {"rows": [{"C": n}], "row_count": 1, "truncated": False}


# ---------------------------------------------------------------------------
# probe_topology
# ---------------------------------------------------------------------------

async def test_probe_returns_zero_dbid_error():
    flags = await probe_topology(0, _ScriptedRunner({}))
    assert flags.probe_succeeded is False
    assert flags.error == "dbid not set"


async def test_probe_single_instance_detects_no_rac():
    runner = _ScriptedRunner({
        "DBA_HIST_DATABASE_INSTANCE": _count_response(1),
        "DBA_HIST_PARAMETER": _count_response(0),
        "DBA_HIST_CELL_CONFIG": _count_response(0),
    })
    flags = await probe_topology(123, runner)
    assert flags.probe_succeeded is True
    assert flags.is_rac is False
    assert flags.has_dataguard is False
    assert flags.is_exadata is False
    assert flags.instance_count == 1


async def test_probe_rac_detected_at_two_instances():
    runner = _ScriptedRunner({
        "DBA_HIST_DATABASE_INSTANCE": _count_response(2),
        "DBA_HIST_PARAMETER": _count_response(0),
        "DBA_HIST_CELL_CONFIG": _count_response(0),
    })
    flags = await probe_topology(123, runner)
    assert flags.is_rac is True
    assert flags.instance_count == 2


async def test_probe_dataguard_detected_via_archive_dest():
    runner = _ScriptedRunner({
        "DBA_HIST_DATABASE_INSTANCE": _count_response(1),
        "DBA_HIST_PARAMETER": _count_response(2),
        "DBA_HIST_CELL_CONFIG": _count_response(0),
    })
    flags = await probe_topology(123, runner)
    assert flags.has_dataguard is True
    assert flags.dg_dest_count == 2


async def test_probe_exadata_detected_when_cell_config_has_rows():
    runner = _ScriptedRunner({
        "DBA_HIST_DATABASE_INSTANCE": _count_response(1),
        "DBA_HIST_PARAMETER": _count_response(0),
        "DBA_HIST_CELL_CONFIG": _count_response(1),
    })
    flags = await probe_topology(123, runner)
    assert flags.is_exadata is True
    assert flags.exadata_data_present is True


async def test_probe_treats_ora_00942_as_not_exadata():
    """View missing on non-Exadata installs — definitive negative, not failure."""
    runner = _ScriptedRunner({
        "DBA_HIST_DATABASE_INSTANCE": _count_response(1),
        "DBA_HIST_PARAMETER": _count_response(0),
        "DBA_HIST_CELL_CONFIG": RuntimeError("ORA-00942: table or view does not exist"),
    })
    flags = await probe_topology(123, runner)
    assert flags.probe_succeeded is True
    assert flags.is_exadata is False


async def test_probe_other_errors_mark_failure_and_keep_safe_defaults():
    runner = _ScriptedRunner({
        "DBA_HIST_DATABASE_INSTANCE": RuntimeError("connection lost"),
        "DBA_HIST_PARAMETER": _count_response(0),
        "DBA_HIST_CELL_CONFIG": _count_response(0),
    })
    flags = await probe_topology(123, runner)
    # Probe failed → safe defaults: don't skip anything.
    assert flags.probe_succeeded is False
    assert flags.is_rac is True
    assert flags.has_dataguard is True
    assert flags.is_exadata is True


# ---------------------------------------------------------------------------
# filter_skipped_agents
# ---------------------------------------------------------------------------

ALL_AGENTS = [
    "wait_time", "ash", "infra", "sql", "memory", "io_storage",
    "concurrency", "segment_object", "rac", "exadata", "dg",
]


def test_filter_skips_nothing_when_probe_failed():
    flags = TopologyFlags(probe_succeeded=False)
    skipped, reasons = filter_skipped_agents(flags, ALL_AGENTS)
    assert skipped == []
    assert reasons == {}


def test_filter_skips_only_conditional_agents():
    """Universal agents must NEVER be skipped, even if probe says zero of everything."""
    flags = TopologyFlags(
        probe_succeeded=True, is_rac=False, has_dataguard=False, is_exadata=False,
    )
    skipped, _ = filter_skipped_agents(flags, ALL_AGENTS)
    assert set(skipped) == {"rac", "dg", "exadata"}
    # Sanity: no universal agent appears.
    for universal in ("wait_time", "ash", "infra", "sql", "memory",
                      "io_storage", "concurrency", "segment_object"):
        assert universal not in skipped


def test_filter_keeps_rac_when_multi_instance():
    flags = TopologyFlags(
        probe_succeeded=True, is_rac=True, has_dataguard=False, is_exadata=False,
        instance_count=3,
    )
    skipped, _ = filter_skipped_agents(flags, ALL_AGENTS)
    assert "rac" not in skipped
    assert "dg" in skipped
    assert "exadata" in skipped


def test_filter_skips_only_disabled_targets():
    """If 'rac' isn't even in enabled_agents, don't list it as 'skipped'."""
    flags = TopologyFlags(
        probe_succeeded=True, is_rac=False, has_dataguard=False, is_exadata=False,
    )
    enabled = ["wait_time", "ash", "rac"]   # dg/exadata not enabled
    skipped, reasons = filter_skipped_agents(flags, enabled)
    assert skipped == ["rac"]
    assert "dg" not in reasons
    assert "exadata" not in reasons


def test_filter_reasons_reference_evidence():
    flags = TopologyFlags(
        probe_succeeded=True, is_rac=False, has_dataguard=False, is_exadata=False,
        instance_count=1, dg_dest_count=0,
    )
    _, reasons = filter_skipped_agents(flags, ALL_AGENTS)
    assert "instance_count=1" in reasons["rac"]
    assert "log_archive_dest" in reasons["dg"]
    assert "DBA_HIST_CELL_CONFIG" in reasons["exadata"]
