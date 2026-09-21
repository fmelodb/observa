"""Deterministic topology probe — runs once before turn_1.

Detects three boolean facts about the source database whose AWR snapshots
were imported into this repository:

* ``is_rac``           — more than one instance recorded in DBA_HIST_DATABASE_INSTANCE
* ``has_dataguard``    — Data Guard archive destinations OR DG status rows exist
* ``is_exadata``       — DBA_HIST_CELL_CONFIG has rows for this dbid

The result feeds ``filter_skipped_agents`` which deterministically excludes
agents whose pre-condition is mathematically false (no chance of finding
anything). Only the conditional agents (rac, dg, exadata) can be skipped by
this probe — every other enabled agent (wait_time, ash, infra, sql, memory,
io_storage, concurrency, segment_object, file_reader) ALWAYS runs.

Probe failures (no MCP, missing views, query errors) default to
"include all agents" — we never skip on uncertain signals.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

# Agents whose pre-condition can be deterministically evaluated.
_CONDITIONAL_AGENTS: tuple[str, ...] = ("rac", "dg", "exadata")


@dataclass(frozen=True)
class TopologyFlags:
    """Boolean facts inferred from the AWR repository for a given dbid."""
    is_rac: bool = True
    has_dataguard: bool = True
    is_exadata: bool = True
    instance_count: int = 0
    dg_dest_count: int = 0
    exadata_data_present: bool = False
    probe_succeeded: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "is_rac": self.is_rac,
            "has_dataguard": self.has_dataguard,
            "is_exadata": self.is_exadata,
            "instance_count": self.instance_count,
            "dg_dest_count": self.dg_dest_count,
            "exadata_data_present": self.exadata_data_present,
            "probe_succeeded": self.probe_succeeded,
            "error": self.error,
        }


# A query runner returns ``{"rows": [...], "row_count": N, ...}`` — same shape
# as ``McpClient.query_awr``. Kept as a callable so tests can inject a mock.
QueryRunner = Callable[[str], Awaitable[dict]]


# ---------------------------------------------------------------------------
# Probe SQL — three small queries, each filtered by dbid
# ---------------------------------------------------------------------------

def _instance_count_sql(dbid: int) -> str:
    return (
        f"SELECT COUNT(DISTINCT instance_number) AS C "
        f"FROM DBA_HIST_DATABASE_INSTANCE WHERE dbid = {int(dbid)}"
    )


def _dg_dest_count_sql(dbid: int) -> str:
    # Count log_archive_dest_N parameters that point to a remote service —
    # the canonical AWR signal for a Data Guard configuration. Filtering by
    # SERVICE= avoids counting local archive destinations.
    return (
        f"SELECT COUNT(*) AS C FROM DBA_HIST_PARAMETER "
        f"WHERE dbid = {int(dbid)} "
        f"AND parameter_name LIKE 'log_archive_dest_%' "
        f"AND value IS NOT NULL "
        f"AND UPPER(value) LIKE '%SERVICE=%'"
    )


def _exadata_data_sql(dbid: int) -> str:
    # If DBA_HIST_CELL_CONFIG doesn't exist (non-Exadata Oracle install), the
    # call raises ORA-00942; we treat that as "not Exadata".
    return (
        f"SELECT COUNT(*) AS C FROM DBA_HIST_CELL_CONFIG "
        f"WHERE dbid = {int(dbid)} AND ROWNUM <= 1"
    )


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def _scalar_count(result: dict) -> int:
    """Pull the first scalar count off a guarded_query result dict."""
    rows = result.get("rows") or []
    if not rows:
        return 0
    first = rows[0]
    if isinstance(first, dict):
        # Column may come back as 'C' / 'c' / 'count(*)' depending on driver.
        for k in ("C", "c"):
            if k in first:
                return int(first[k] or 0)
        # Fall back to first numeric value.
        for v in first.values():
            try:
                return int(v or 0)
            except (TypeError, ValueError):
                continue
    return 0


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

async def probe_topology(dbid: int, run_query: QueryRunner) -> TopologyFlags:
    """Run the three probe queries and return aggregated TopologyFlags.

    On any failure (probe didn't even reach SQLcl, dbid is invalid, etc.) we
    return flags with all booleans set to True so no agent is skipped — the
    fail-safe direction is "do the work, don't gamble on triage."

    Note: ORA-00942 on the Exadata probe is expected on non-Exadata systems
    and is treated as "not Exadata" (not as a probe failure).
    """
    if not dbid or dbid <= 0:
        return TopologyFlags(error="dbid not set")

    instance_count = 0
    dg_dest_count = 0
    exadata_data_present = False
    errors: list[str] = []

    try:
        result = await run_query(_instance_count_sql(dbid))
        instance_count = _scalar_count(result)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"instance_count: {exc}")

    try:
        result = await run_query(_dg_dest_count_sql(dbid))
        dg_dest_count = _scalar_count(result)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"dg_dest_count: {exc}")

    try:
        result = await run_query(_exadata_data_sql(dbid))
        exadata_data_present = _scalar_count(result) > 0
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        # ORA-00942 = view not present on this Oracle install. Definitive
        # signal that this is not an Exadata system.
        if "ORA-00942" in msg or "table or view does not exist" in msg.lower():
            exadata_data_present = False
        else:
            errors.append(f"exadata: {exc}")

    if errors:
        # Partial probe failure — be conservative, leave booleans=True.
        logger.warning("topology probe had errors: %s", "; ".join(errors))
        return TopologyFlags(
            is_rac=True,
            has_dataguard=True,
            is_exadata=True,
            instance_count=instance_count,
            dg_dest_count=dg_dest_count,
            exadata_data_present=exadata_data_present,
            probe_succeeded=False,
            error="; ".join(errors)[:300],
        )

    return TopologyFlags(
        is_rac=instance_count > 1,
        has_dataguard=dg_dest_count > 0,
        is_exadata=exadata_data_present,
        instance_count=instance_count,
        dg_dest_count=dg_dest_count,
        exadata_data_present=exadata_data_present,
        probe_succeeded=True,
    )


def filter_skipped_agents(
    flags: TopologyFlags,
    enabled_agents: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Return (skipped_agent_names, reasons) given probe flags.

    Only the three conditional agents (rac, dg, exadata) are eligible for
    skip. Universal agents always run. If the probe failed
    (``probe_succeeded=False``), we skip nothing — fail-safe.
    """
    if not flags.probe_succeeded:
        return [], {}
    skipped: list[str] = []
    reasons: dict[str, str] = {}
    if "rac" in enabled_agents and not flags.is_rac:
        skipped.append("rac")
        reasons["rac"] = (
            f"single-instance database (instance_count={flags.instance_count}) "
            "— RAC-specific findings are impossible"
        )
    if "dg" in enabled_agents and not flags.has_dataguard:
        skipped.append("dg")
        reasons["dg"] = (
            f"no Data Guard archive destinations recorded "
            f"(log_archive_dest_*=SERVICE entries: {flags.dg_dest_count})"
        )
    if "exadata" in enabled_agents and not flags.is_exadata:
        skipped.append("exadata")
        reasons["exadata"] = (
            "DBA_HIST_CELL_CONFIG has no rows for this dbid "
            "— not an Exadata system"
        )
    return skipped, reasons


__all__ = [
    "CONDITIONAL_AGENTS",
    "TopologyFlags",
    "probe_topology",
    "filter_skipped_agents",
]

CONDITIONAL_AGENTS = _CONDITIONAL_AGENTS
