"""Robust structured-output parsing across LLM providers.

LangChain's ``with_structured_output(method="function_calling")`` parser
expects the model to emit the schema as a tool call. Some models (notably
xAI's Grok-4-fast-non-reasoning via OCI) instead emit the JSON payload as
plain text in ``content``, leaving ``tool_calls`` empty. The parser then
returns ``None`` and any caller that accesses fields on the result crashes.

``parse_structured`` wraps ``with_structured_output(..., include_raw=True)``
and adds two fallbacks before giving up:

1. If ``parsed`` is already a ``schema`` instance or a dict, validate and
   return it.
2. If the raw assistant ``content`` parses as JSON (with optional ```json
   fence stripping), validate the dict against ``schema`` and return.

Anything else raises :class:`StructuredOutputError` with diagnostic context
so callers can decide whether to abstain, fall back, or surface the error.
"""
from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """The model produced no parseable output for the requested schema."""

    def __init__(
        self,
        message: str,
        *,
        raw_text: str = "",
        raw_tool_calls: Any = None,
        parsing_error: Any = None,
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.raw_tool_calls = raw_tool_calls
        self.parsing_error = parsing_error


def _strip_markdown_fence(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()
    return candidate


async def parse_structured(
    llm: BaseChatModel,
    schema: type[T],
    messages: list[BaseMessage],
) -> T:
    """Invoke ``llm`` with structured output for ``schema`` and return a
    validated instance, falling back through text-JSON if needed.

    Raises :class:`StructuredOutputError` if no fallback succeeds. Other
    exceptions from the underlying ``ainvoke`` propagate unchanged so the
    caller can distinguish transport/parser failures from "model emitted
    nothing usable".
    """
    structured = llm.with_structured_output(schema, include_raw=True)
    result = await structured.ainvoke(messages)

    parsed: Any = result
    parsing_error: Any = None
    raw_msg: Any = None
    if isinstance(result, dict):
        parsed = result.get("parsed")
        parsing_error = result.get("parsing_error")
        raw_msg = result.get("raw")

    if isinstance(parsed, schema):
        return parsed
    if isinstance(parsed, dict):
        return schema.model_validate(parsed)

    full_raw_text = ""
    raw_tool_calls: Any = None
    if raw_msg is not None:
        content = getattr(raw_msg, "content", "") or ""
        full_raw_text = content if isinstance(content, str) else str(content)
        raw_tool_calls = getattr(raw_msg, "tool_calls", None)

    candidate = _strip_markdown_fence(full_raw_text)
    if candidate:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            try:
                return schema.model_validate(payload)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "text-JSON fallback failed validation for %s: %s; payload=%r",
                    schema.__name__, exc, payload,
                )

    raise StructuredOutputError(
        f"model did not produce a parseable {schema.__name__}",
        raw_text=full_raw_text[:300],
        raw_tool_calls=raw_tool_calls,
        parsing_error=parsing_error,
    )
