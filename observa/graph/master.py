"""Master LangGraph — turn-based multi-specialist orchestration with chat gates.

Topology (dynamically built for N = turns.count)::

    init → turn_1 → gate_1 → turn_2 → gate_2 → … → turn_N →
          consolidate → synthesize_hypotheses → detect_contradictions →
          [resolve_contradiction →]? replay_verify → END

Each ``gate_i`` inspects the findings just produced in turn *i*. If any agent
raised a question, it calls ``langgraph.types.interrupt`` with the payload so
the TUI can pause, show the questions, and let the analyst respond. The
resume value is a dict ``{"messages": [{"sender": "user", "text": "..."}]}``
which is appended to ``chat_log`` for subsequent turns to see.

The MCP client (SQLcl + files) is constructed by the caller (TUI / CLI) and
passed via ``config["configurable"]["mcp_client"]``. Without it, agents still
run but have no tools — they will all abstain.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from observa.agents import REGISTRY
from observa.agents.base import ReflectiveAgent
from observa.config import get_settings
from observa.graph.contradiction_detector import ContradictionDetector
from observa.graph.diff_probe import run_diff_probe
from observa.graph.ora_timeline import build_ora_timeline
from observa.graph.master_consolidator import consolidate as _consolidate
from observa.graph.hypothesis_synth import synthesize_hypotheses
from observa.graph.contradiction_resolver import resolve_contradictions
from observa.graph.replay_verify import replay_verify, patch_confidence
from observa.graph.refinement import select_agents_for_turn
from observa.graph.topology_probe import filter_skipped_agents, probe_topology
from observa.graph.turn_controller import run_turn
from observa.llm import configure_llm_concurrency, make_llm
from observa.models import CaseState, ChatMessage

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM factories — cached per model id so a single ChatAnthropic / ChatOpenAI
# instance is reused across agents that share the same model.
# ---------------------------------------------------------------------------

_llm_by_model: dict[tuple[str, str], Any] = {}


def _llm_for_model(model: str, *, slot: str = "agent") -> Any:
    if not model:
        raise ValueError("Empty model id passed to _llm_for_model")
    key = (model, slot)
    inst = _llm_by_model.get(key)
    if inst is None:
        inst = make_llm(model, slot=slot)
        _llm_by_model[key] = inst
    return inst


def _llm_for_agent(agent_name: str) -> Any:
    return _llm_for_model(get_settings().model_for_agent(agent_name), slot="agent")


def _ensure_main_llm() -> Any:
    """Default LLM (used by callers that aren't agent-scoped). Treated as the
    consolidator slot — the main case-level LLM produces the FinalSummary."""
    return _llm_for_model(get_settings().llm_model, slot="consolidator")


def _ensure_hypotheses_llm() -> Any:
    return _llm_for_model(get_settings().model_for_hypotheses(), slot="hypotheses")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mcp_tools_from_config(config: Optional[RunnableConfig]) -> list[BaseTool]:
    if not config:
        return []
    client = (config.get("configurable") or {}).get("mcp_client")
    if client is None:
        return []
    return list(client.langchain_tools())


def _build_active_agents() -> list[ReflectiveAgent]:
    enabled = get_settings().agents_enabled
    return [cls() for name, cls in REGISTRY.items() if enabled.get(name, False)]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def init_node(state: CaseState, config: Optional[RunnableConfig] = None) -> dict:
    settings = get_settings()
    configure_llm_concurrency(settings.turns_max_concurrent_agents)
    logger.info(
        "init_node: case_id=%s dbid=%s turns=%d budget=%ds concurrency=%d",
        state.get("case_id"),
        state.get("dbid"),
        settings.turns_count,
        settings.turns_time_budget_seconds,
        settings.turns_max_concurrent_agents,
    )
    return {
        "total_turns": settings.turns_count,
        "current_turn": 0,
    }


async def topology_probe_node(
    state: CaseState, config: Optional[RunnableConfig] = None
) -> dict:
    """Run the deterministic topology probe and decide which agents to skip.

    Fail-safe: any probe error => no agents skipped.
    """
    dbid = state.get("dbid", 0) or 0
    settings = get_settings()
    enabled_agents = [n for n, on in settings.agents_enabled.items() if on]

    if not config:
        logger.info("topology_probe: no RunnableConfig — skipping")
        return {"topology_flags": {"probe_succeeded": False, "error": "no config"}}
    client = (config.get("configurable") or {}).get("mcp_client")
    if client is None:
        logger.info("topology_probe: no MCP client — skipping")
        return {"topology_flags": {"probe_succeeded": False, "error": "no mcp client"}}

    # Internal fixed statements — exempt from case-predicate enforcement,
    # same as the diff probe and intake catalog queries.
    flags = await probe_topology(int(dbid), lambda sql: client.query_catalog(sql, requester="topology_probe"))
    skipped, reasons = filter_skipped_agents(flags, enabled_agents)
    logger.info(
        "topology_probe: instance_count=%d dg_dest=%d exadata=%s — skipped=%s",
        flags.instance_count, flags.dg_dest_count, flags.exadata_data_present, skipped,
    )
    return {
        "topology_flags": flags.to_dict(),
        "agents_skipped": skipped,
        "agents_skip_reasons": reasons,
    }


async def diff_probe_node(
    state: CaseState, config: Optional[RunnableConfig] = None
) -> dict:
    """Deterministic problem-vs-baseline deltas seeded before turn 1. Fail-open."""
    problem = state.get("problem_window")
    if problem is None:
        logger.info("diff_probe: no problem window — skipping (legacy/manual case)")
        return {}
    if not config:
        logger.info("diff_probe: no RunnableConfig — skipping")
        return {}
    client = (config.get("configurable") or {}).get("mcp_client")
    if client is None:
        logger.info("diff_probe: no MCP client — skipping")
        return {}

    result = await run_diff_probe(
        int(state.get("dbid") or 0),
        problem,
        state.get("baseline_window"),
        lambda sql: client.query_catalog(sql, requester="diff_probe"),
    )
    if result.digest is None:
        logger.warning("diff_probe: probe produced no digest (errors=%s)", result.errors)
        return {}
    logger.info(
        "diff_probe: digest ready (baseline=%s suspect=%s sections_failed=%d)",
        result.digest["has_baseline"], result.suspect, len(result.errors),
    )
    return {"baseline_diff": result.digest, "baseline_suspect": result.suspect}


async def ora_timeline_node(
    state: CaseState, config: Optional[RunnableConfig] = None
) -> dict:
    """Deterministic ORA-incident timeline from alert logs. Fail-open."""
    alert_logs = [f for f in (state.get("input_files") or []) if f.type == "alertlog"]
    if not alert_logs:
        logger.info("ora_timeline: no alert log attached — skipping")
        return {}
    if not config:
        logger.info("ora_timeline: no RunnableConfig — skipping")
        return {}
    client = (config.get("configurable") or {}).get("mcp_client")
    if client is None:
        logger.info("ora_timeline: no MCP client — skipping")
        return {}
    timeline = await build_ora_timeline(
        alert_logs,
        state.get("problem_window"),
        state.get("baseline_window"),
        client.scan_alertlog,
    )
    if timeline is None:
        logger.info("ora_timeline: no incidents / no coverage — skipping")
        return {}
    logger.info(
        "ora_timeline: %d incidents (problem=%d baseline=%d outside=%d)",
        timeline["counts"]["incidents"], timeline["counts"]["problem"],
        timeline["counts"]["baseline"], timeline["counts"]["outside"],
    )
    return {"ora_timeline": timeline}


def _make_turn_node(turn: int):
    async def _turn_node(state: CaseState, config: Optional[RunnableConfig] = None) -> dict:
        settings = get_settings()
        mcp_tools = _mcp_tools_from_config(config)
        if not mcp_tools:
            logger.warning("turn %d: no MCP tools in config — agents will likely abstain", turn)

        client = (config.get("configurable") or {}).get("mcp_client") if config else None
        ledger = getattr(client, "ledger", None)
        tools_for_agent: Optional[Callable[[str, int], list[BaseTool]]] = (
            (lambda name, t: list(client.langchain_tools(requester=name, turn=t)))
            if client is not None else None
        )

        # Build the candidate set: enabled minus topology-skipped (Level 1).
        all_agents = _build_active_agents()
        topology_skipped = state.get("agents_skipped") or []
        candidates = [a for a in all_agents if a.name not in topology_skipped]
        candidate_names = [a.name for a in candidates]

        # Refinement (Level 5): turn 1 runs everyone, turn 2+ may drop agents.
        new_chat = [
            m for m in (state.get("chat_log") or [])
            if m.turn == turn - 1 and m.sender == "user"
        ]
        active_names, drop_reasons = select_agents_for_turn(
            turn=turn,
            enabled_agents=candidate_names,
            prior_turn_findings=state.get("turn_findings") or [],
            new_chat_messages=new_chat,
            cfg=settings.refinement_cfg(),
        )
        agents = [a for a in candidates if a.name in active_names]
        if drop_reasons:
            logger.info(
                "turn %d refinement: kept %d, dropped %d (%s)",
                turn, len(active_names), len(drop_reasons), drop_reasons,
            )
        logger.info("turn %d: running %d agents", turn, len(agents))

        turn_results = await run_turn(
            state,
            turn,
            agents,
            _ensure_main_llm(),
            mcp_tools,
            settings.turns_time_budget_seconds,
            settings.turns_max_concurrent_agents,
            llm_for_agent=_llm_for_agent,
            tools_for_agent=tools_for_agent,
            ledger=ledger,
        )
        logger.info(
            "turn %d complete: %d findings, %d abstains, %d questions",
            turn,
            sum(len(tf.findings) for tf in turn_results),
            sum(1 for tf in turn_results if tf.abstained),
            sum(len(tf.questions) for tf in turn_results),
        )
        # Persist refinement audit — merge with any prior turns' entries.
        active_log = dict(state.get("agents_active_per_turn") or {})
        active_log[turn] = active_names
        refine_log = dict(state.get("agents_refinement_log") or {})
        if drop_reasons:
            refine_log[turn] = drop_reasons

        # Reducer fields append; scalars overwrite.
        return {
            "turn_findings": turn_results,
            "agent_questions": [q for tf in turn_results for q in tf.questions],
            "current_turn": turn,
            "agents_active_per_turn": active_log,
            "agents_refinement_log": refine_log,
        }

    _turn_node.__name__ = f"turn_{turn}_node"
    return _turn_node


def _make_gate_node(turn: int):
    """Emit an interrupt if this turn produced questions. No-op otherwise."""
    async def _gate_node(state: CaseState, config: Optional[RunnableConfig] = None) -> dict:
        # Look only at questions raised in THIS turn.
        tfs = [tf for tf in state.get("turn_findings", []) if tf.turn == turn]
        questions = [
            {"agent": q.agent, "question": q.question}
            for tf in tfs
            for q in tf.questions
        ]
        if not questions:
            logger.info("gate %d: no questions — skipping chat", turn)
            return {}

        logger.info("gate %d: pausing for %d analyst responses", turn, len(questions))
        payload = {
            "type": "chat_gate",
            "turn": turn,
            "questions": questions,
        }
        # Suspend here. On resume, `interrupt` returns whatever the caller sent
        # via ``Command(resume=...)``.
        response = interrupt(payload)
        if not isinstance(response, dict):
            return {}
        messages_in = response.get("messages") or []
        if not messages_in:
            return {}
        now = datetime.now(timezone.utc)
        new_chat: list[ChatMessage] = []
        for m in messages_in:
            sender = m.get("sender", "user")
            text = m.get("text", "")
            if not text:
                continue
            new_chat.append(
                ChatMessage(sender=sender, turn=turn, text=text, timestamp=now)
            )
        return {"chat_log": new_chat} if new_chat else {}

    _gate_node.__name__ = f"gate_{turn}_node"
    return _gate_node


async def consolidate_node(state: CaseState, config: Optional[RunnableConfig] = None) -> dict:
    llm = _llm_for_model(get_settings().model_for_consolidator(), slot="consolidator")
    client = (config.get("configurable") or {}).get("mcp_client") if config else None
    summary = await _consolidate(state, llm, ledger=getattr(client, "ledger", None))
    return {"final_summary": summary}


async def synthesize_hypotheses_node(state: CaseState, config: Optional[RunnableConfig] = None) -> dict:
    try:
        settings = get_settings()
        if not settings.agents_enable_hypotheses:
            return {"hypotheses": []}
        llm = _ensure_hypotheses_llm()
        hyps = await synthesize_hypotheses(state, llm)
        logger.info("synthesize_hypotheses: %d hypotheses", len(hyps))
        return {"hypotheses": hyps}
    except Exception:
        logger.warning("synthesize_hypotheses_node: failed — no hypotheses", exc_info=True)
        return {"hypotheses": []}


async def detect_contradictions_node(state: CaseState, config: Optional[RunnableConfig] = None) -> dict:
    settings = get_settings()
    hyps = state.get("hypotheses") or []
    if not settings.agents_enable_hypotheses or not hyps:
        return {"contradictions": []}
    advocate_llm = _ensure_hypotheses_llm()
    detector = ContradictionDetector(
        llm=advocate_llm,
        threshold=settings.agents_contradiction_threshold,
    )
    contradictions = await detector.run(hyps)
    return {"contradictions": contradictions}


async def resolve_contradiction_node(state: CaseState, config: Optional[RunnableConfig] = None) -> dict:
    client = (config.get("configurable") or {}).get("mcp_client") if config else None
    ledger = getattr(client, "ledger", None)
    llm = _ensure_hypotheses_llm()
    try:
        resolutions = await resolve_contradictions(state, ledger, llm)
    except Exception:
        logger.warning("resolve_contradiction_node: resolver failed — no resolutions", exc_info=True)
        return {"contradiction_resolutions": []}
    return {"contradiction_resolutions": resolutions}


async def replay_verify_node(state: CaseState, config: Optional[RunnableConfig] = None) -> dict:
    try:
        settings = get_settings()
        if not settings.agents_enable_cove:
            return {}
        client = (config.get("configurable") or {}).get("mcp_client") if config else None
        ledger = getattr(client, "ledger", None)
        llm = _llm_for_model(settings.model_for_cove(), slot="cove")
        result = await replay_verify(
            state, ledger, llm,
            support_threshold=settings.agents_verification_support_threshold,
        )
        if result is None:
            return {}
        out: dict = {"verification": result}
        summary = state.get("final_summary")
        if summary is not None and result.applied_confidence_factor != 1.0:
            out["final_summary"] = patch_confidence(summary, result.applied_confidence_factor)
        return out
    except Exception:
        logger.warning("replay_verify_node: failed — no verification", exc_info=True)
        return {}


def _route_after_detection(state: CaseState) -> str:
    if state.get("contradictions"):
        return "resolve_contradiction"
    return "replay_verify"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph(checkpointer: Any | None = None) -> "CompiledStateGraph":
    """Build the graph. Uses an in-memory ``MemorySaver`` when no checkpointer is
    provided — required so ``interrupt()`` in the gates can persist/resume state.
    """
    settings = get_settings()
    n_turns = max(1, settings.turns_count)

    builder: StateGraph = StateGraph(CaseState)
    builder.add_node("init", init_node)
    builder.add_node("topology_probe", topology_probe_node)
    builder.add_node("diff_probe", diff_probe_node)
    builder.add_node("ora_timeline", ora_timeline_node)
    for t in range(1, n_turns + 1):
        builder.add_node(f"turn_{t}", _make_turn_node(t))
        if t < n_turns:
            builder.add_node(f"gate_{t}", _make_gate_node(t))
    builder.add_node("consolidate", consolidate_node)
    builder.add_node("synthesize_hypotheses", synthesize_hypotheses_node)
    builder.add_node("detect_contradictions", detect_contradictions_node)
    builder.add_node("resolve_contradiction", resolve_contradiction_node)
    builder.add_node("replay_verify", replay_verify_node)

    builder.set_entry_point("init")
    builder.add_edge("init", "topology_probe")
    builder.add_edge("topology_probe", "diff_probe")
    builder.add_edge("diff_probe", "ora_timeline")
    builder.add_edge("ora_timeline", "turn_1")
    for t in range(1, n_turns):
        builder.add_edge(f"turn_{t}", f"gate_{t}")
        builder.add_edge(f"gate_{t}", f"turn_{t + 1}")
    builder.add_edge(f"turn_{n_turns}", "consolidate")
    builder.add_edge("consolidate", "synthesize_hypotheses")
    builder.add_edge("synthesize_hypotheses", "detect_contradictions")
    builder.add_conditional_edges(
        "detect_contradictions",
        _route_after_detection,
        {"resolve_contradiction": "resolve_contradiction", "replay_verify": "replay_verify"},
    )
    builder.add_edge("resolve_contradiction", "replay_verify")
    builder.add_edge("replay_verify", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def reset_llm_cache() -> None:
    """Test helper — drop memoized LLM instances so settings changes take effect."""
    _llm_by_model.clear()


def initialize_agents(_pool: Any = None) -> None:  # pragma: no cover — compat shim
    logger.warning(
        "initialize_agents() is a no-op in the turn-based architecture; agents are now stateless classes."
    )
