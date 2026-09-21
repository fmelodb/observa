"""Turn-to-turn refinement — decide which agents run in turn N+1.

Turn 1 is breadth-first: every enabled agent runs (after Level-1 deterministic
skip). Turn 2+ becomes depth-first — only agents whose participation is
justified by the previous turn's findings keep running.

Promotion rules (any one keeps an agent active for turn N+1):

1. Universal anchors (``wait_time``, ``ash`` by default) — always run.
2. Produced a finding with severity in ``keep_severities`` last turn.
3. Was cross-referenced — another agent's finding mentioned its domain
   (keyword match against ``DOMAIN_KEYWORDS``).
4. Has a fresh analyst answer in the gate-chat after last turn.

Demotion: not universal AND none of (2, 3, 4) → dropped.

Floor safeguard: if the active set falls below ``min_active``, agents that
at least produced *something* (non-abstain) last turn are added back.

Detection is keyword-based and pure-Python — zero extra LLM calls. Bias is
deliberately toward false-positive (run an unneeded agent) over
false-negative (miss a finding).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

from observa.models import ChatMessage, TurnFindings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain keywords — case-insensitive substring match against findings'
# description + evidence + related_objects of OTHER agents. Generous by
# design; falsePositive cost (re-run an agent) ≪ falseNegative cost (miss
# a real finding).
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "wait_time": (
        "log file sync", "db file sequential", "db file scattered",
        "buffer busy", "enq:", "wait class", "top wait",
    ),
    "ash": (
        "active session history", "ASH sample", "session_state",
        "top sql by elapsed",
    ),
    "sql": (
        "SQL_ID", "SQL_PLAN", "PLAN_HASH", "execution plan",
        "literal SQL", "sql_text", "sql elapsed",
    ),
    "memory": (
        "SGA", "PGA", "shared pool", "buffer cache", "library cache",
        "ORA-04031", "memory advisor", "cursor: pin",
    ),
    "io_storage": (
        "ASM", "datafile", "tempfile", "redo log", "filesystemio",
        "I/O", "iops", "throughput", "storage",
    ),
    "concurrency": (
        "latch:", "mutex:", "TX -", "row lock", "blocker",
        "blocked session", "lock wait", "library cache lock",
    ),
    "segment_object": (
        "SEGMENT_NAME", "TABLESPACE", "INDEX", "PARTITION",
        "HWM", "high water mark", "extent",
    ),
    "rac": (
        "gc ", "global cache", "INST_ID", "interconnect",
        "cluster wait", "cache fusion", "instance crash",
    ),
    "exadata": (
        "cell ", "smart scan", "offload", "storage index",
        "cell single block", "cell multiblock",
    ),
    "dg": (
        "dataguard", "data guard", "log_archive_dest", "redo apply",
        "broker", "standby", "primary database",
    ),
    "infra": (
        "ALTER SYSTEM SET", "init.ora", "spfile", "parameter change",
        "underscore parameter",
    ),
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RefinementConfig:
    """Tunable knobs for the refinement policy."""
    enabled: bool = False
    universal_agents: tuple[str, ...] = ("wait_time", "ash")
    keep_severities: frozenset[str] = field(
        default_factory=lambda: frozenset({"high", "critical"})
    )
    enable_cross_reference: bool = True
    min_active: int = 3


# ---------------------------------------------------------------------------
# Cross-reference detection
# ---------------------------------------------------------------------------

def _lowered_haystack_for(tf: TurnFindings) -> str:
    """Concatenate searchable text from a TurnFindings into one lower-case blob."""
    pieces: list[str] = []
    for f in tf.findings:
        pieces.append(f.description or "")
        pieces.append(f.evidence_text())
        pieces.extend(o for o in f.related_objects)
    return " ".join(pieces).lower()


def find_cross_references(
    last_turn_findings: Iterable[TurnFindings],
) -> set[str]:
    """Return the set of agent names referenced by OTHER agents' findings.

    Pure function — no side effects. An agent name appears in the result if
    any keyword from ``DOMAIN_KEYWORDS[agent]`` is found in any other
    agent's finding text. The owning agent is never added by its own findings.
    """
    referenced: set[str] = set()
    findings_list = list(last_turn_findings)
    for tf in findings_list:
        if tf.abstained or not tf.findings:
            continue
        haystack = _lowered_haystack_for(tf)
        if not haystack:
            continue
        for other_agent, keywords in DOMAIN_KEYWORDS.items():
            if other_agent == tf.agent:
                continue
            if other_agent in referenced:
                continue
            for kw in keywords:
                if kw.lower() in haystack:
                    referenced.add(other_agent)
                    break
    return referenced


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def select_agents_for_turn(
    turn: int,
    enabled_agents: list[str],
    prior_turn_findings: list[TurnFindings],
    new_chat_messages: list[ChatMessage],
    cfg: RefinementConfig,
) -> tuple[list[str], dict[str, str]]:
    """Decide which agents run this turn.

    Returns ``(active_agent_names_sorted, drop_reasons)``.

    * ``turn == 1`` → all enabled agents run, no drop reasons.
    * ``cfg.enabled is False`` → all enabled agents run (refinement off).
    * Otherwise → apply promotion rules over the previous turn's findings.

    ``new_chat_messages`` is the slice of analyst-replies that arrived in the
    gate AFTER turn N-1 (used by rule 4: "premise changed").
    """
    enabled_set = set(enabled_agents)

    # First turn or refinement disabled → run everything.
    if turn <= 1 or not cfg.enabled:
        return sorted(enabled_set), {}

    last_turn = turn - 1
    last = [tf for tf in prior_turn_findings if tf.turn == last_turn]
    by_agent: dict[str, TurnFindings] = {tf.agent: tf for tf in last}

    keep: set[str] = set(cfg.universal_agents) & enabled_set
    promoted_via: dict[str, str] = {}
    for a in keep:
        promoted_via[a] = "universal anchor"

    # Rule 2: severity-based promotion
    for tf in last:
        if tf.agent not in enabled_set:
            continue
        for f in tf.findings:
            if f.severity in cfg.keep_severities:
                if tf.agent not in keep:
                    keep.add(tf.agent)
                    promoted_via[tf.agent] = (
                        f"produced {f.severity} finding turn {last_turn}"
                    )
                break

    # Rule 3: cross-reference
    if cfg.enable_cross_reference:
        referenced = find_cross_references(last)
        for a in (referenced & enabled_set):
            if a not in keep:
                keep.add(a)
                promoted_via[a] = f"cross-referenced by another agent turn {last_turn}"

    # Rule 4: agents whose questions got new analyst answers
    answered_agents = {
        m.agent for m in new_chat_messages
        if m.sender == "user" and m.agent
    }
    for a in (answered_agents & enabled_set):
        if a not in keep:
            keep.add(a)
            promoted_via[a] = f"analyst answered question turn {last_turn}"

    # Floor: never go below min_active. Promote producers (any non-abstain
    # agent) until the floor is met.
    if len(keep) < cfg.min_active:
        producers = [
            tf.agent for tf in last
            if not tf.abstained and tf.agent in enabled_set and tf.agent not in keep
        ]
        for agent in producers:
            if len(keep) >= cfg.min_active:
                break
            keep.add(agent)
            promoted_via[agent] = "floor safeguard (min_active)"

    # Compute drop reasons for everyone enabled but not kept.
    drop_reasons: dict[str, str] = {}
    for agent in enabled_agents:
        if agent in keep:
            continue
        tf = by_agent.get(agent)
        if tf is None:
            drop_reasons[agent] = "did not run last turn"
        elif tf.abstained:
            drop_reasons[agent] = (
                f"abstained turn {last_turn}: "
                f"{(tf.abstain_reason or 'no reason given')[:120]}"
            )
        else:
            sevs = sorted({f.severity for f in tf.findings}) or ["none"]
            drop_reasons[agent] = (
                f"only {','.join(sevs)} findings turn {last_turn}, "
                "no cross-references"
            )

    if drop_reasons or promoted_via:
        logger.info(
            "refinement turn %d: kept=%d (%s) dropped=%d",
            turn, len(keep), sorted(keep), len(drop_reasons),
        )

    return sorted(keep), drop_reasons


__all__ = [
    "DOMAIN_KEYWORDS",
    "RefinementConfig",
    "find_cross_references",
    "select_agents_for_turn",
]
