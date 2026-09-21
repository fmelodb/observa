"""Refinement policy — turn-to-turn agent selection."""
from __future__ import annotations

from datetime import datetime, timezone

from observa.graph.refinement import (
    DOMAIN_KEYWORDS,
    RefinementConfig,
    find_cross_references,
    select_agents_for_turn,
)
from observa.models import ChatMessage, Finding, TurnFindings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_AGENTS = [
    "wait_time", "ash", "infra", "sql", "memory", "io_storage",
    "concurrency", "segment_object", "rac", "exadata", "dg",
]


def _cfg(**overrides) -> RefinementConfig:
    base = dict(
        enabled=True,
        universal_agents=("wait_time", "ash"),
        keep_severities=frozenset({"high", "critical"}),
        enable_cross_reference=True,
        min_active=3,
    )
    base.update(overrides)
    return RefinementConfig(**base)


def _finding(severity: str = "low", description: str = "", evidence_note: str = "",
             related_objects: list[str] | None = None) -> Finding:
    return Finding(
        finding_id="x", agent="x", turn=1,
        severity=severity, code="C", description=description,
        evidence_note=evidence_note, related_objects=related_objects or [],
    )


def _tf(agent: str, turn: int = 1, *, abstained: bool = False,
        findings: list[Finding] | None = None,
        abstain_reason: str = "") -> TurnFindings:
    return TurnFindings(
        agent=agent, turn=turn, abstained=abstained,
        abstain_reason=abstain_reason,
        findings=findings or [],
    )


def _chat(agent: str, turn: int) -> ChatMessage:
    return ChatMessage(
        sender="user", agent=agent, turn=turn,
        text="...", timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Turn 1 / disabled — no-op behaviors
# ---------------------------------------------------------------------------

def test_turn_1_keeps_all_enabled_regardless_of_config():
    active, dropped = select_agents_for_turn(
        turn=1, enabled_agents=ALL_AGENTS,
        prior_turn_findings=[], new_chat_messages=[], cfg=_cfg(),
    )
    assert sorted(active) == sorted(ALL_AGENTS)
    assert dropped == {}


def test_disabled_config_keeps_all_even_for_turn_2():
    # No findings, no chat — but disabled => no-op.
    active, dropped = select_agents_for_turn(
        turn=2, enabled_agents=ALL_AGENTS,
        prior_turn_findings=[], new_chat_messages=[],
        cfg=_cfg(enabled=False),
    )
    assert sorted(active) == sorted(ALL_AGENTS)
    assert dropped == {}


# ---------------------------------------------------------------------------
# Promotion rules
# ---------------------------------------------------------------------------

def test_universal_agents_never_drop_even_when_abstain():
    last = [_tf(name, abstained=True) for name in ALL_AGENTS]
    active, dropped = select_agents_for_turn(
        turn=2, enabled_agents=ALL_AGENTS,
        prior_turn_findings=last, new_chat_messages=[], cfg=_cfg(),
    )
    assert "wait_time" in active
    assert "ash" in active


def test_high_severity_finding_keeps_agent():
    last = [
        _tf("memory", findings=[_finding(severity="high", description="ORA-04031")]),
        _tf("sql", abstained=True),
        _tf("infra", abstained=True),
    ]
    # min_active=2 to disable the floor's interference for this test
    active, _ = select_agents_for_turn(
        turn=2, enabled_agents=["memory", "sql", "infra", "wait_time", "ash"],
        prior_turn_findings=last, new_chat_messages=[],
        cfg=_cfg(min_active=2),
    )
    assert "memory" in active


def test_low_severity_alone_drops_agent():
    last = [
        _tf("memory", findings=[_finding(severity="low", description="generic note")]),
    ]
    active, dropped = select_agents_for_turn(
        turn=2, enabled_agents=["memory", "wait_time", "ash"],
        prior_turn_findings=last, new_chat_messages=[],
        cfg=_cfg(min_active=1),  # disable floor
    )
    assert "memory" not in active
    assert "memory" in dropped
    assert "low" in dropped["memory"]


def test_abstained_drops_with_reason():
    last = [_tf("memory", abstained=True, abstain_reason="no PGA pressure observed")]
    active, dropped = select_agents_for_turn(
        turn=2, enabled_agents=["memory", "wait_time", "ash"],
        prior_turn_findings=last, new_chat_messages=[],
        cfg=_cfg(min_active=1),
    )
    assert "memory" not in active
    assert "abstained" in dropped["memory"]
    assert "PGA pressure" in dropped["memory"]


def test_did_not_run_last_turn_yields_specific_reason():
    last = [_tf("wait_time", findings=[_finding(severity="medium")])]
    # 'rac' has no entry for turn 1.
    _, dropped = select_agents_for_turn(
        turn=2, enabled_agents=["wait_time", "rac", "ash"],
        prior_turn_findings=last, new_chat_messages=[],
        cfg=_cfg(min_active=1),
    )
    assert dropped.get("rac") == "did not run last turn"


# ---------------------------------------------------------------------------
# Cross-reference detection
# ---------------------------------------------------------------------------

def test_cross_reference_finds_other_agents_keywords():
    """SQL agent's finding mentions ORA-04031 → memory agent referenced."""
    last = [
        _tf("sql", findings=[
            _finding(description="Top SQL hit ORA-04031 during execution"),
        ]),
    ]
    referenced = find_cross_references(last)
    assert "memory" in referenced
    # Self-reference must not happen.
    assert "sql" not in referenced


def test_cross_reference_respects_related_objects():
    last = [
        _tf("infra", findings=[
            _finding(description="parameter changed",
                     related_objects=["log file sync"]),
        ]),
    ]
    referenced = find_cross_references(last)
    assert "wait_time" in referenced


def test_cross_reference_promotes_abstained_agent():
    last = [
        _tf("memory", abstained=True, abstain_reason="nothing seen"),
        _tf("sql", findings=[_finding(description="hit ORA-04031 in shared pool")]),
    ]
    active, _ = select_agents_for_turn(
        turn=2,
        enabled_agents=["memory", "sql", "wait_time", "ash"],
        prior_turn_findings=last, new_chat_messages=[],
        cfg=_cfg(min_active=1),
    )
    assert "memory" in active


def test_cross_reference_disabled_does_not_promote():
    last = [
        _tf("memory", abstained=True),
        _tf("sql", findings=[_finding(description="ORA-04031 again")]),
    ]
    active, _ = select_agents_for_turn(
        turn=2,
        enabled_agents=["memory", "sql", "wait_time", "ash"],
        prior_turn_findings=last, new_chat_messages=[],
        cfg=_cfg(enable_cross_reference=False, min_active=1),
    )
    assert "memory" not in active


def test_abstained_agent_does_not_contribute_to_xref_haystack():
    """If sql abstained, its (empty) finding text shouldn't promote anyone."""
    last = [_tf("sql", abstained=True, abstain_reason="ORA-04031 in reason")]
    referenced = find_cross_references(last)
    assert "memory" not in referenced


# ---------------------------------------------------------------------------
# Analyst-answer promotion (rule 4)
# ---------------------------------------------------------------------------

def test_analyst_answer_promotes_agent_back():
    last = [_tf("memory", abstained=True)]
    chat_after_turn_1 = [_chat("memory", turn=1)]
    active, _ = select_agents_for_turn(
        turn=2,
        enabled_agents=["memory", "wait_time", "ash"],
        prior_turn_findings=last, new_chat_messages=chat_after_turn_1,
        cfg=_cfg(min_active=1),
    )
    assert "memory" in active


# ---------------------------------------------------------------------------
# Min-active floor
# ---------------------------------------------------------------------------

def test_min_active_floor_promotes_producers():
    """When too few agents qualify, non-abstain agents get re-added."""
    last = [
        _tf("infra", findings=[_finding(severity="low")]),
        _tf("sql", findings=[_finding(severity="medium")]),
        _tf("memory", abstained=True),
    ]
    enabled = ["infra", "sql", "memory", "wait_time", "ash"]
    # universals = 2, no high/critical, no xref. Floor=4 forces 2 producers in.
    active, _ = select_agents_for_turn(
        turn=2, enabled_agents=enabled,
        prior_turn_findings=last, new_chat_messages=[],
        cfg=_cfg(min_active=4),
    )
    # 2 universals + at least 2 producers (infra, sql).
    assert len(active) >= 4
    assert {"wait_time", "ash"}.issubset(set(active))
    # memory abstained and no floor logic should pull it (only producers do).
    assert {"infra", "sql"}.issubset(set(active))


def test_floor_skips_abstained_agents():
    last = [
        _tf("memory", abstained=True),
        _tf("rac", abstained=True),
    ]
    active, _ = select_agents_for_turn(
        turn=2, enabled_agents=["memory", "rac", "wait_time", "ash"],
        prior_turn_findings=last, new_chat_messages=[],
        cfg=_cfg(min_active=4),  # floor wants 4 but only 2 universals exist
    )
    # Floor cannot promote abstainers — keep set is just the 2 universals.
    assert sorted(active) == ["ash", "wait_time"]


# ---------------------------------------------------------------------------
# Sanity — Level 1 / Level 5 interaction
# ---------------------------------------------------------------------------

def test_topology_skipped_agent_never_appears_in_active_set():
    """Caller passes already-filtered enabled_agents — refinement respects it."""
    # Imagine 'rac' was skipped by topology. It's not in enabled_agents.
    last = [
        _tf("sql", findings=[_finding(description="gc current block lost — INST_ID 2")]),
    ]
    active, _ = select_agents_for_turn(
        turn=2,
        enabled_agents=["sql", "wait_time", "ash"],   # rac NOT here
        prior_turn_findings=last, new_chat_messages=[],
        cfg=_cfg(min_active=1),
    )
    assert "rac" not in active   # even though sql's text would xref it
