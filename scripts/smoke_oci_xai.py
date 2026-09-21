"""Manual smoke test for the OCI XaiProvider.

Run by hand:
    uv run python scripts/smoke_oci_xai.py

Requirements:
    - llm.providers.oci configured in config/config.yaml
    - ~/.oci/config with API key for the configured profile
    - uv sync --extra oci

Exits 0 if every check passes, 1 if any fails. Prints PASS/FAIL per check.
"""
from __future__ import annotations

import json
import sys
import traceback
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from pydantic import BaseModel, Field


MODEL_ID = "xai.grok-4-fast"


def _build_chat() -> Any:
    from observa.llm.oci_factory import build_oci_chat
    return build_oci_chat(MODEL_ID)


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


class Answer(BaseModel):
    """A short answer with a confidence score."""
    text: str = Field(description="the answer text")
    confidence: float = Field(description="0.0 to 1.0", ge=0.0, le=1.0)


def check(name: str, fn: Any) -> bool:
    print(f"\n=== {name} ===")
    try:
        ok = fn()
    except Exception:
        traceback.print_exc()
        print(f"FAIL: {name}")
        return False
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
    return ok


def check_plain_chat() -> bool:
    chat = _build_chat()
    msg = chat.invoke([
        SystemMessage(content="Be very concise."),
        HumanMessage(content="Say the single word: pong."),
    ])
    text = msg.content if isinstance(msg.content, str) else str(msg.content)
    print(f"  response: {text!r}")
    return bool(text.strip())


def check_tool_call() -> bool:
    chat = _build_chat().bind_tools([add])
    msg = chat.invoke([
        HumanMessage(content="Use the add tool to compute 2 + 3. Do not answer directly."),
    ])
    print(f"  tool_calls: {msg.tool_calls}")
    if not msg.tool_calls:
        return False
    args = msg.tool_calls[0]["args"]
    return args == {"a": 2, "b": 3} or args == {"a": 3, "b": 2}


def check_structured_output() -> bool:
    chat = _build_chat().with_structured_output(Answer)
    out = chat.invoke([
        HumanMessage(content="What color is the sky on a clear day? Be brief."),
    ])
    print(f"  structured: {out!r}")
    return isinstance(out, Answer) and bool(out.text)


def check_tool_use_loop() -> bool:
    chat = _build_chat().bind_tools([add])
    history: list[HumanMessage | AIMessage | ToolMessage] = [
        HumanMessage(content="Use the add tool to compute 2 + 3, then state the result.")
    ]
    first = chat.invoke(history)
    if not first.tool_calls:
        print("  no tool call from model")
        return False
    call = first.tool_calls[0]
    print(f"  first tool_call: {call}")
    history.append(AIMessage(content=first.content, tool_calls=first.tool_calls))
    history.append(ToolMessage(content="5", tool_call_id=call["id"]))
    second = chat.invoke(history)
    text = second.content if isinstance(second.content, str) else json.dumps(second.content)
    print(f"  final: {text!r}")
    return "5" in text


def main() -> int:
    results = [
        check("plain chat",        check_plain_chat),
        check("tool call",         check_tool_call),
        check("structured output", check_structured_output),
        check("tool-use loop",     check_tool_use_loop),
    ]
    print()
    print("---")
    print(f"{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
