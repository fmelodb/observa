"""Tests for the shared text-JSON-fallback structured-output helper."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from observa.llm.structured import StructuredOutputError, parse_structured


class _Out(BaseModel):
    name: str
    count: int = 0


def _llm_returning(raw_content: str, parsed: Any = None, parsing_error: Any = None) -> MagicMock:
    raw = AIMessage(content=raw_content, tool_calls=[])
    structured = MagicMock()

    async def fake_ainvoke(_prompt: Any) -> dict:
        return {"raw": raw, "parsed": parsed, "parsing_error": parsing_error}

    structured.ainvoke = fake_ainvoke
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


async def test_returns_parsed_when_provider_already_validated():
    expected = _Out(name="ok", count=1)
    llm = _llm_returning("", parsed=expected)
    result = await parse_structured(llm, _Out, [])
    assert result is expected


async def test_validates_dict_payload_from_parsed():
    llm = _llm_returning("", parsed={"name": "x", "count": 2})
    result = await parse_structured(llm, _Out, [])
    assert result == _Out(name="x", count=2)


async def test_text_json_fallback_when_parsed_is_none():
    llm = _llm_returning('{"name": "y"}')
    result = await parse_structured(llm, _Out, [])
    assert result == _Out(name="y", count=0)


async def test_text_json_fallback_strips_markdown_fence():
    llm = _llm_returning('```json\n{"name": "z", "count": 5}\n```')
    result = await parse_structured(llm, _Out, [])
    assert result == _Out(name="z", count=5)


async def test_raises_with_diagnostics_when_no_fallback_succeeds():
    llm = _llm_returning("not json at all", parsing_error=ValueError("nope"))
    with pytest.raises(StructuredOutputError) as exc:
        await parse_structured(llm, _Out, [])
    assert "_Out" in str(exc.value)
    assert exc.value.raw_text.startswith("not json")
    assert isinstance(exc.value.parsing_error, ValueError)
