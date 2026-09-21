"""Rate-limit-aware wrapper around LangChain chat models.

Three mechanisms work together to keep N parallel agents from being murdered by
per-minute TPM caps:

1. A global ``asyncio.Semaphore`` caps the number of in-flight LLM calls. This
   is the primary pressure valve — by default only 4 agents can be talking to
   the provider at once, so 11 agents run in waves instead of all slamming the
   API simultaneously.

2. When a 429/rate-limit still slips through, we sleep 60s (TPM reset window)
   and retry, up to 3 attempts.

3. While sleeping, we accumulate the wait time in a ``ContextVar``
   (:data:`rate_limit_sleep`). The agent's internal deadline-check and the
   turn-controller watchdog both subtract that value, so a rate-limit sleep
   never causes an agent to be cancelled before it had a chance to do real
   work.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass, field
from typing import Any

# Module-level imports — patchable via observa.llm.rate_limited.* in tests.
from observa.llm.oci_capabilities import validate_oci_model_for_slot
from observa.llm.oci_factory import build_oci_chat

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate-limit sleep tracker (per-agent, propagated through asyncio tasks)
# ---------------------------------------------------------------------------

@dataclass
class RateLimitTracker:
    """Accumulates rate-limit sleep seconds observed in this context."""
    sleep_seconds: float = field(default=0.0)


rate_limit_sleep: contextvars.ContextVar[RateLimitTracker | None] = contextvars.ContextVar(
    "observa_rate_limit_sleep", default=None,
)


def current_tracker() -> RateLimitTracker | None:
    return rate_limit_sleep.get()


def _record_sleep(seconds: float) -> None:
    tracker = rate_limit_sleep.get()
    if tracker is not None:
        tracker.sleep_seconds += seconds
    tt = token_tracker.get()
    agent = current_agent.get()
    if tt is not None and agent is not None:
        st = tt.by_agent.setdefault(agent, AgentRuntimeState())
        st.status = "rate_limited"
        st.last_activity = time.monotonic()


# ---------------------------------------------------------------------------
# Agent runtime state (tokens + status) shared with the TUI for telemetry.
# ---------------------------------------------------------------------------

@dataclass
class AgentRuntimeState:
    """Live snapshot of a single agent's run.

    Updated by the LLM wrapper (tokens, rate_limited) and by the agent base
    class (status transitions). Read by the LiveRunScreen to render the
    per-agent grid every ~150 ms.

    Token accounting:
      * ``tokens`` — usage_metadata.total_tokens (= input + output, where
        input is only the FRESH part — already excludes cache reads).
      * ``cached_tokens`` — sum of input_token_details.cache_read across
        calls. These are the tokens served from the prompt cache; they do
        NOT count toward ``tokens`` because Anthropic / OpenAI report them
        separately. Tracked so the UI can show cache-hit savings.
      * ``cache_creation_tokens`` — input_token_details.cache_creation,
        i.e. tokens written to the cache this call (charged at full price
        plus a small write premium on Anthropic). Useful to attribute
        the cost of warming the cache.
    """
    tokens: int = 0
    cached_tokens: int = 0
    cache_creation_tokens: int = 0
    status: str = "idle"  # idle | thinking | tool | done | abstained | error | rate_limited
    last_activity: float = 0.0


@dataclass
class TokenTracker:
    """Aggregated per-agent stats. The ``by_agent`` dict is shared across tasks
    (each entry is owned by a single agent name), but the *current agent name*
    must NOT live here — see ``current_agent`` ContextVar below.
    """
    by_agent: dict[str, AgentRuntimeState] = field(default_factory=dict)


token_tracker: contextvars.ContextVar[TokenTracker | None] = contextvars.ContextVar(
    "observa_token_tracker", default=None,
)

# The name of the agent that owns LLM calls in the *current asyncio task*.
# Lives in its own ContextVar so N agents running in parallel don't trample
# each other's attribution. Each task that calls ``contextvars.copy_context()``
# (the turn controller does) gets its own independent value.
current_agent: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "observa_current_agent", default=None,
)


def current_token_tracker() -> TokenTracker | None:
    return token_tracker.get()


def _extract_usage(response: Any) -> dict | None:
    """Return the usage_metadata dict for a response, handling both shapes:

    * Plain ``AIMessage`` — ``response.usage_metadata`` is the dict.
    * ``with_structured_output(..., include_raw=True)`` — response is
      ``{"raw": AIMessage, "parsed": ..., "parsing_error": ...}``; we look at
      ``raw.usage_metadata``.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return usage
    if isinstance(response, dict):
        raw = response.get("raw")
        if raw is not None:
            return getattr(raw, "usage_metadata", None)
    return None


def _record_usage(response: Any) -> None:
    """Pull token usage off a response and add it to the agent owning this task.

    Records both the billable total (input + output) and the cache breakdown
    from ``input_token_details`` when the provider exposes it.
    """
    tt = token_tracker.get()
    if tt is None:
        return
    agent = current_agent.get()
    if agent is None:
        return
    usage = _extract_usage(response)
    if not isinstance(usage, dict):
        return

    total = usage.get("total_tokens")
    if total is None:
        inp = usage.get("input_tokens") or 0
        out = usage.get("output_tokens") or 0
        total = (inp or 0) + (out or 0)

    details = usage.get("input_token_details") or {}
    cache_read = int(details.get("cache_read") or 0) if isinstance(details, dict) else 0
    cache_creation = (
        int(details.get("cache_creation") or 0) if isinstance(details, dict) else 0
    )

    if not total and not cache_read and not cache_creation:
        return

    st = tt.by_agent.setdefault(agent, AgentRuntimeState())
    if total:
        st.tokens += int(total)
    st.cached_tokens += cache_read
    st.cache_creation_tokens += cache_creation
    st.last_activity = time.monotonic()


# ---------------------------------------------------------------------------
# Global LLM concurrency limit
# ---------------------------------------------------------------------------

_llm_semaphore: asyncio.Semaphore | None = None
_llm_semaphore_limit: int = 4


def configure_llm_concurrency(max_concurrent: int) -> None:
    """Set the global cap on concurrent LLM calls. Called once at startup."""
    global _llm_semaphore, _llm_semaphore_limit
    limit = max(1, int(max_concurrent))
    _llm_semaphore_limit = limit
    _llm_semaphore = asyncio.Semaphore(limit)


def _get_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(_llm_semaphore_limit)
    return _llm_semaphore


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def _is_rate_limit_exc(exc: BaseException) -> bool:
    try:
        import anthropic

        if isinstance(exc, anthropic.RateLimitError):
            return True
    except Exception:
        pass
    try:
        import openai

        if isinstance(exc, openai.RateLimitError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
        return True
    return "ratelimit" in type(exc).__name__.lower()


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class RateLimitedLLM:
    """Delegating wrapper: global semaphore + rate-limit retry + sleep tracking."""

    _MAX_ATTEMPTS = 3
    _SLEEP_SECONDS = 60.0

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        last_exc: BaseException | None = None
        sem = _get_semaphore()
        for attempt in range(self._MAX_ATTEMPTS):
            try:
                async with sem:
                    response = await self._inner.ainvoke(*args, **kwargs)
                    _record_usage(response)
                    return response
            except Exception as exc:
                last_exc = exc
                if not _is_rate_limit_exc(exc) or attempt == self._MAX_ATTEMPTS - 1:
                    raise
                logger.warning(
                    "rate limit on ainvoke (attempt %d/%d) — sleeping %.0fs",
                    attempt + 1, self._MAX_ATTEMPTS, self._SLEEP_SECONDS,
                )
                _record_sleep(self._SLEEP_SECONDS)
                await asyncio.sleep(self._SLEEP_SECONDS)
        if last_exc:
            raise last_exc

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        last_exc: BaseException | None = None
        for attempt in range(self._MAX_ATTEMPTS):
            try:
                response = self._inner.invoke(*args, **kwargs)
                _record_usage(response)
                return response
            except Exception as exc:
                last_exc = exc
                if not _is_rate_limit_exc(exc) or attempt == self._MAX_ATTEMPTS - 1:
                    raise
                logger.warning(
                    "rate limit on invoke (attempt %d/%d) — sleeping %.0fs",
                    attempt + 1, self._MAX_ATTEMPTS, self._SLEEP_SECONDS,
                )
                _record_sleep(self._SLEEP_SECONDS)
                time.sleep(self._SLEEP_SECONDS)
        if last_exc:
            raise last_exc

    def bind_tools(self, *args: Any, **kwargs: Any) -> "RateLimitedLLM":
        return RateLimitedLLM(self._inner.bind_tools(*args, **kwargs))

    def with_structured_output(self, *args: Any, **kwargs: Any) -> "RateLimitedLLM":
        return RateLimitedLLM(self._inner.with_structured_output(*args, **kwargs))


def make_llm(model: str, *, slot: str = "agent") -> RateLimitedLLM:
    """Create a chat model wrapped with rate-limit retry.

    Routing rules:
      * ``oci/<model_id>`` → ChatOCIGenAI via observa.llm.oci_factory.
        ``slot`` selects the capability requirements applied at validation.
      * model contains ``claude``  → ChatAnthropic.
      * everything else            → ChatOpenAI.

    Defaults ``slot`` to ``"agent"`` (most restrictive: requires tool calling
    + structured output). Callers in non-agent contexts (consolidator, cove,
    hypotheses, master_chat) MUST pass the matching slot.
    """
    if model.startswith("oci/"):
        model_id = model[len("oci/"):]
        validate_oci_model_for_slot(model_id, slot)
        base: Any = build_oci_chat(model_id)
        return RateLimitedLLM(base)

    if "claude" in model.lower():
        from langchain_anthropic import ChatAnthropic

        base = ChatAnthropic(model=model)
    else:
        from langchain_openai import ChatOpenAI

        base = ChatOpenAI(model=model)
    return RateLimitedLLM(base)
