"""Post-analysis chat with the Master.

The Master LLM retains full context of the case (FinalSummary, blackboard)
and the analyst can ask follow-up questions. The Master is given the same MCP
tools as the specialists so it can run new queries or re-read files when the
analyst asks.

Unlike the per-turn specialist loop, this is a single round-trip: analyst
message in, Master answer out (with tool calls interleaved by the LLM runtime
until it produces a final text answer).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from observa.models import CaseState, ChatMessage, Finding, FinalSummary, TurnFindings

logger = logging.getLogger(__name__)


_MASTER_CHAT_SYSTEM = """You are the Master consolidator from a just-completed
Oracle diagnostic case. The analyst is asking follow-up questions about the
Final Summary or probing into details.

Rules:
- Ground every answer in the blackboard (specialist findings and abstains) and
  the Final Summary that were produced during the case. Do not invent new
  findings.
- When the analyst asks something that requires fresh data (a new SQL query,
  reading a new slice of a file), use the MCP tools. Always cite what you ran.
- Keep answers concise. If the analyst's question is out of scope for Oracle
  diagnostics, say so and steer back to the case.
- If the analyst asks for recommendations, restrict yourself to actions
  supported by the evidence already collected plus any new queries you run.
  Do not speculate.

Output format:
- Respond in GitHub-flavored Markdown.
- Render tabular data (query results, comparisons) as Markdown tables with
  pipe syntax. Round large numbers and abbreviate when helpful.
- Render SQL in fenced ```sql code blocks. Render other code or file
  excerpts in plain fenced code blocks.
- Use bullet lists for enumerations of findings or steps.
- Keep prose terse — analysts read this in a narrow side panel."""


def _summary_block(summary: FinalSummary | None) -> str:
    if summary is None:
        return "(no final summary available)"
    lines = [
        f"Problem: {summary.problem_restated}",
        f"Root cause: {summary.root_cause}",
        f"Confidence: {summary.confidence:.2f}",
        "Top findings:",
    ]
    for f in summary.top_findings:
        lines.append(f"  - [{f.severity.upper()}] ({f.agent}) {f.description}")
    if summary.unknowns:
        lines.append("Unknowns:")
        lines.extend(f"  - {u}" for u in summary.unknowns)
    return "\n".join(lines)


def _findings_block(state: CaseState) -> str:
    turn_findings: list[TurnFindings] = list(state.get("turn_findings") or [])
    all_findings: list[Finding] = []
    for tf in turn_findings:
        all_findings.extend(tf.findings)
    if not all_findings:
        return "(no findings recorded)"
    return "\n".join(
        f"- [{f.severity.upper()}] ({f.agent}, turn {f.turn}) {f.code}: {f.description}"
        + (f" — evidence: {f.evidence_text()}" if f.evidence_text() else "")
        for f in all_findings
    )


def _history_as_messages(history: Sequence[ChatMessage]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in history:
        if m.sender == "user":
            out.append(HumanMessage(content=m.text))
        else:
            out.append(AIMessage(content=m.text))
    return out


def _build_context_human_message(state: CaseState) -> HumanMessage:
    content = (
        f"=== CASE CONTEXT ===\n"
        f"Problem statement: {state.get('problem_statement', '')}\n"
        f"DBID: {state.get('dbid', '?')} · "
        f"Oracle: {state.get('oracle_version', '?')} · "
        f"Topology: {state.get('topology', '?')}\n\n"
        f"=== FINAL SUMMARY ===\n{_summary_block(state.get('final_summary'))}\n\n"
        f"=== ALL FINDINGS ===\n{_findings_block(state)}"
    )
    return HumanMessage(content=content)


async def _run_tool_call(
    call: dict, tools_by_name: dict[str, BaseTool]
) -> str:
    name = call.get("name", "")
    args = call.get("args", {}) or {}
    tool = tools_by_name.get(name)
    if tool is None:
        return f"ERROR: unknown tool '{name}'"
    try:
        result = await tool.ainvoke(args)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {type(exc).__name__}: {exc}"
    if isinstance(result, (dict, list)):
        return json.dumps(result, default=str)[:8000]
    return str(result)[:8000]


async def master_chat(
    state: CaseState,
    history: Sequence[ChatMessage],
    user_message: str,
    llm: BaseChatModel,
    mcp_tools: Sequence[BaseTool],
    max_tool_iterations: int = 10,
) -> str:
    """Run a single analyst question through the master chat loop."""
    tools_by_name = {t.name: t for t in mcp_tools}
    llm_with_tools = llm.bind_tools(list(mcp_tools)) if mcp_tools else llm

    messages: list[BaseMessage] = [
        SystemMessage(content=_MASTER_CHAT_SYSTEM),
        _build_context_human_message(state),
        *_history_as_messages(history),
        HumanMessage(content=user_message),
    ]

    for _ in range(max_tool_iterations):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            content = getattr(response, "content", "")
            return content if isinstance(content, str) else str(content)
        for call in tool_calls:
            tool_result = await _run_tool_call(call, tools_by_name)
            messages.append(ToolMessage(content=tool_result, tool_call_id=call.get("id", "")))

    return "(Master exceeded the tool-iteration budget while answering — please rephrase or ask a narrower question.)"
