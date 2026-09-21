"""Runs one turn: every enabled agent in parallel, each with its own time budget.

Each agent call is wrapped in an extensible watchdog: the base deadline is
``time_budget_seconds + grace``, but it is *pushed forward* by any rate-limit
sleep time the agent's LLM calls accumulated. An agent that had to wait out a
60s 429 still gets its full working budget — rate-limit waits never cause a
premature watchdog cancel.

An ``asyncio.Semaphore`` caps how many agents can run at once (matches the LLM
concurrency cap in ``observa.llm.rate_limited``) so we don't slam the provider
with 11 parallel calls.

If the watchdog fires or the agent raises, we return a ``TurnFindings`` marked
``abstained=True`` with a clear reason, so the blackboard always has one entry
per agent per turn.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from typing import Any, Callable, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from observa.agents.base import ReflectiveAgent
from observa.llm import RateLimitTracker, rate_limit_sleep
from observa.models import CaseState, TurnFindings


async def _dispatch_agent_done(result: TurnFindings) -> None:
    """Stream one agent's finished output to any custom-event listeners.

    Falls back to a no-op if we are not running inside a LangGraph callback
    context (e.g. unit tests). Never raises.
    """
    try:
        from langchain_core.callbacks.manager import adispatch_custom_event
    except Exception:  # noqa: BLE001
        return
    payload = {
        "agent": result.agent,
        "turn": result.turn,
        "abstained": result.abstained,
        "abstain_reason": result.abstain_reason,
        "notes": result.notes,
        "findings": [
            {
                "severity": f.severity,
                "code": f.code,
                "description": f.description,
                "evidence_refs": list(f.evidence_refs),
                "evidence_note": f.evidence_note,
                "uncited": f.uncited,
                "related_objects": list(f.related_objects),
            }
            for f in result.findings
        ],
        "questions": [{"question": q.question} for q in result.questions],
    }
    try:
        await adispatch_custom_event("agent_done", payload)
    except Exception:  # noqa: BLE001
        # Not inside a callback context (tests) — silently skip.
        pass

logger = logging.getLogger(__name__)


def validate_evidence_refs(result: TurnFindings, ledger: Any) -> TurnFindings:
    """Drop hallucinated evidence refs and (re)flag uncited findings.

    ``ledger`` needs only ``known_ids()``; None (tests, no MCP) is a no-op.
    Mutates in place and returns the same object.
    """
    if ledger is None:
        return result
    known = ledger.known_ids()
    for f in result.findings:
        valid = [r for r in f.evidence_refs if r in known]
        dropped = [r for r in f.evidence_refs if r not in known]
        if dropped:
            logger.info(
                "%s turn %d: dropped hallucinated evidence refs %s (finding %s)",
                result.agent, result.turn, dropped, f.code,
            )
        f.evidence_refs = valid
        f.uncited = not valid
    return result


# Grace period on top of the agent's internal deadline before the controller
# force-cancels it.
_WATCHDOG_GRACE_SECONDS = 10
# Poll cadence for the extensible-deadline watchdog.
_WATCHDOG_POLL_SECONDS = 1.0


async def _run_agent_with_watchdog(
    agent: ReflectiveAgent,
    state: CaseState,
    turn: int,
    llm: BaseChatModel,
    mcp_tools: Sequence[BaseTool],
    time_budget_seconds: int,
) -> TurnFindings:
    """Run one agent under an extensible deadline.

    The deadline is ``budget + grace + tracker.sleep_seconds``, recomputed each
    poll cycle. This means a rate-limit sleep doesn't "eat" the agent's budget:
    while the agent is stuck waiting on a 429, the deadline moves with it.
    """
    tracker = RateLimitTracker()
    ctx = contextvars.copy_context()
    ctx.run(rate_limit_sleep.set, tracker)

    async def _coro() -> TurnFindings:
        return await agent.run(state, turn, llm, list(mcp_tools), time_budget_seconds)

    # Create the task inside the copied context so the tracker propagates.
    task: asyncio.Task[TurnFindings] = asyncio.Task(_coro(), context=ctx)

    start = time.monotonic()
    base_timeout = time_budget_seconds + _WATCHDOG_GRACE_SECONDS
    timed_out = False
    try:
        while True:
            deadline = start + base_timeout + tracker.sleep_seconds
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                break
            try:
                return await asyncio.wait_for(
                    asyncio.shield(task), timeout=min(remaining, _WATCHDOG_POLL_SECONDS)
                )
            except asyncio.TimeoutError:
                if task.done():
                    return task.result()
                continue
    except Exception as exc:  # noqa: BLE001
        logger.exception("%s: turn %d failed", agent.name, turn)
        return TurnFindings(
            agent=agent.name,
            turn=turn,
            abstained=True,
            abstain_reason=f"Agent error: {type(exc).__name__}: {exc}",
        )

    if timed_out:
        elapsed = time.monotonic() - start
        logger.warning(
            "%s: watchdog timeout after %.0fs (rate-limit sleep=%.0fs)",
            agent.name, elapsed, tracker.sleep_seconds,
        )
        return TurnFindings(
            agent=agent.name,
            turn=turn,
            abstained=True,
            abstain_reason=(
                f"Watchdog timeout after {elapsed:.0f}s "
                f"(rate-limit sleep={tracker.sleep_seconds:.0f}s)"
            ),
        )
    # Defensive: should be unreachable.
    return TurnFindings(
        agent=agent.name, turn=turn, abstained=True,
        abstain_reason="watchdog exited without result",
    )


async def _run_one_agent(
    agent: ReflectiveAgent,
    state: CaseState,
    turn: int,
    llm: BaseChatModel,
    mcp_tools: Sequence[BaseTool],
    time_budget_seconds: int,
    semaphore: asyncio.Semaphore,
    ledger: Any = None,
) -> TurnFindings:
    async with semaphore:
        logger.info("%s: turn %d starting", agent.name, turn)
        result = await _run_agent_with_watchdog(
            agent, state, turn, llm, mcp_tools, time_budget_seconds,
        )
        result = validate_evidence_refs(result, ledger)
        await _dispatch_agent_done(result)
        return result


async def run_turn(
    state: CaseState,
    turn: int,
    agents: Sequence[ReflectiveAgent],
    llm: BaseChatModel,
    mcp_tools: Sequence[BaseTool],
    time_budget_seconds: int,
    max_concurrent_agents: int = 4,
    llm_for_agent: Optional[Callable[[str], BaseChatModel]] = None,
    tools_for_agent: Optional[Callable[[str, int], Sequence[BaseTool]]] = None,
    ledger: Any = None,
) -> list[TurnFindings]:
    """Execute one turn concurrently across all provided agents.

    ``llm_for_agent`` (when provided) is called once per agent to resolve the
    LLM instance for that agent — enabling per-agent model tiers (Level 3).
    Falls back to the shared ``llm`` argument when not supplied.

    ``tools_for_agent`` (when provided) is called once per agent to resolve the
    MCP tools for that agent and turn — enabling per-agent tool sets (Level 4).
    Falls back to the shared ``mcp_tools`` argument when not supplied.

    ``ledger`` (when provided) is used to validate evidence refs in each agent's
    findings after the watchdog returns, dropping hallucinated refs.
    """
    if not agents:
        return []
    tool_names = [t.name for t in mcp_tools]
    concurrency = max(1, min(max_concurrent_agents, len(agents)))
    logger.info(
        "run_turn %d: %d agent(s), %d MCP tool(s)=%s, budget=%ds, concurrency=%d",
        turn, len(agents), len(tool_names), tool_names,
        time_budget_seconds, concurrency,
    )
    if not tool_names and tools_for_agent is None:
        logger.warning(
            "run_turn %d: NO MCP tools available — every agent will almost certainly abstain. "
            "Check MCP bootstrap in the main-app log.",
            turn,
        )
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    for a in agents:
        agent_llm = llm_for_agent(a.name) if llm_for_agent is not None else llm
        agent_tools = list(tools_for_agent(a.name, turn)) if tools_for_agent is not None else list(mcp_tools)
        tasks.append(
            _run_one_agent(a, state, turn, agent_llm, agent_tools,
                           time_budget_seconds, semaphore, ledger=ledger)
        )
    return list(await asyncio.gather(*tasks))
