"""Tests for the TokenTracker / AgentRuntimeState plumbing."""
from __future__ import annotations

import asyncio
import contextvars
from types import SimpleNamespace

from observa.llm.rate_limited import (
    AgentRuntimeState,
    TokenTracker,
    _record_sleep,
    _record_usage,
    current_agent,
    rate_limit_sleep,
    RateLimitTracker,
    token_tracker,
)


def _set_tracker(tt: TokenTracker, agent: str | None) -> None:
    token_tracker.set(tt)
    current_agent.set(agent)


def test_record_usage_sums_total_tokens() -> None:
    tt = TokenTracker()
    _set_tracker(tt, "wait_time")
    response = SimpleNamespace(
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    )
    _record_usage(response)

    assert tt.by_agent["wait_time"].tokens == 150
    assert tt.by_agent["wait_time"].last_activity > 0


def test_record_usage_falls_back_to_input_plus_output() -> None:
    tt = TokenTracker()
    _set_tracker(tt, "ash")
    response = SimpleNamespace(
        usage_metadata={"input_tokens": 80, "output_tokens": 20}
    )
    _record_usage(response)

    assert tt.by_agent["ash"].tokens == 100


def test_record_usage_accumulates_across_calls() -> None:
    tt = TokenTracker()
    _set_tracker(tt, "sql")
    for total in (10, 25, 7):
        _record_usage(SimpleNamespace(usage_metadata={"total_tokens": total}))

    assert tt.by_agent["sql"].tokens == 42


def test_record_usage_isolates_per_agent() -> None:
    tt = TokenTracker()
    _set_tracker(tt, "rac")
    _record_usage(SimpleNamespace(usage_metadata={"total_tokens": 100}))
    current_agent.set("dg")
    _record_usage(SimpleNamespace(usage_metadata={"total_tokens": 30}))

    assert tt.by_agent["rac"].tokens == 100
    assert tt.by_agent["dg"].tokens == 30


def test_record_usage_unwraps_include_raw_dict() -> None:
    """include_raw=True returns {"raw": AIMessage, "parsed": ...}; we read raw."""
    tt = TokenTracker()
    _set_tracker(tt, "sql")
    raw = SimpleNamespace(usage_metadata={"total_tokens": 1234})
    _record_usage({"raw": raw, "parsed": object(), "parsing_error": None})
    assert tt.by_agent["sql"].tokens == 1234


def test_record_usage_extracts_cache_read_and_creation() -> None:
    """Cache details from input_token_details accumulate separately."""
    tt = TokenTracker()
    _set_tracker(tt, "sql")
    response = SimpleNamespace(usage_metadata={
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "input_token_details": {"cache_read": 5000, "cache_creation": 200},
    })
    _record_usage(response)
    st = tt.by_agent["sql"]
    assert st.tokens == 150            # billable (input + output, EXCLUDES cache_read)
    assert st.cached_tokens == 5000
    assert st.cache_creation_tokens == 200


def test_record_usage_handles_missing_cache_fields() -> None:
    """Providers without prompt cache → cache fields stay 0."""
    tt = TokenTracker()
    _set_tracker(tt, "sql")
    _record_usage(SimpleNamespace(usage_metadata={"total_tokens": 100}))
    assert tt.by_agent["sql"].cached_tokens == 0
    assert tt.by_agent["sql"].cache_creation_tokens == 0


def test_record_usage_cache_only_response_still_records() -> None:
    """If a response has only cache_read (rare edge case), don't drop it."""
    tt = TokenTracker()
    _set_tracker(tt, "sql")
    _record_usage(SimpleNamespace(usage_metadata={
        "input_tokens": 0, "output_tokens": 0,
        "input_token_details": {"cache_read": 500},
    }))
    assert tt.by_agent["sql"].cached_tokens == 500


def test_record_usage_no_tracker_is_safe() -> None:
    token_tracker.set(None)
    _record_usage(SimpleNamespace(usage_metadata={"total_tokens": 9000}))


def test_record_usage_no_current_agent_is_safe() -> None:
    tt = TokenTracker()
    token_tracker.set(tt)
    current_agent.set(None)
    _record_usage(SimpleNamespace(usage_metadata={"total_tokens": 1}))
    assert tt.by_agent == {}


def test_record_usage_missing_metadata_is_safe() -> None:
    tt = TokenTracker()
    _set_tracker(tt, "infra")
    _record_usage(SimpleNamespace())  # no usage_metadata
    assert "infra" not in tt.by_agent or tt.by_agent["infra"].tokens == 0


def test_record_sleep_marks_agent_rate_limited() -> None:
    tt = TokenTracker()
    _set_tracker(tt, "memory")
    rl = RateLimitTracker()
    rate_limit_sleep.set(rl)

    _record_sleep(60.0)

    assert rl.sleep_seconds == 60.0
    assert tt.by_agent["memory"].status == "rate_limited"


def test_agent_runtime_state_defaults() -> None:
    s = AgentRuntimeState()
    assert s.tokens == 0
    assert s.status == "idle"
    assert s.last_activity == 0.0


# ---------------------------------------------------------------------------
# Regression: 4 agents in parallel must not trample each other's attribution.
# Prior bug — tt.current_agent was a shared attribute, so concurrent agents
# overwrote it and tokens got attributed to whoever set it last.
# ---------------------------------------------------------------------------

async def test_parallel_agents_do_not_trample_token_attribution() -> None:
    tt = TokenTracker()
    token_tracker.set(tt)

    async def agent_work(name: str, n_calls: int, tokens_per_call: int) -> None:
        # Each task gets its OWN copy of `current_agent` because the spawning
        # `asyncio.Task(...context=...)` snapshots the parent context.
        token = current_agent.set(name)
        try:
            for _ in range(n_calls):
                # Yield to scheduler so all 4 tasks interleave their work.
                await asyncio.sleep(0)
                _record_usage(
                    SimpleNamespace(usage_metadata={"total_tokens": tokens_per_call})
                )
        finally:
            current_agent.reset(token)

    async def spawn(name: str, n_calls: int, per: int) -> asyncio.Task:
        ctx = contextvars.copy_context()
        return asyncio.Task(agent_work(name, n_calls, per), context=ctx)

    tasks = await asyncio.gather(
        spawn("ash", 100, 10),
        spawn("sql", 100, 10),
        spawn("io_storage", 100, 10),
        spawn("memory", 100, 10),
    )
    await asyncio.gather(*tasks)

    # Each agent's bucket must hold EXACTLY the tokens it generated.
    for name in ("ash", "sql", "io_storage", "memory"):
        assert tt.by_agent[name].tokens == 1000, (
            f"{name} should hold 1000 tokens, got {tt.by_agent[name].tokens}"
        )
