"""Reflective agent base class.

Each specialist subclass defines a short ``name``, a human-readable
``display_name``, and a ``specialty_prompt`` describing what it analyzes and
— crucially — what it must NOT analyze. The shared system prompt wraps this
with universal rules (objectivity, abstain-if-no-evidence, strict scope).

The ``run()`` method executes a time-budgeted tool-use loop against a LangChain
LLM bound to the subset of MCP tools the subclass declares in ``allowed_tools``
(by default only ``query_awr`` — AWR specialists do NOT read diagnostic files;
that is the ``file_reader`` agent's exclusive remit). The loop ends when the LLM
returns a response without tool calls OR when the deadline passes. A final
structured prompt asks the LLM to emit JSON matching ``_ReflectiveOutput`` which
we convert into ``TurnFindings``.

The agent NEVER enforces its own wall-clock timeout — the caller
(``turn_controller``) is responsible for ``asyncio.wait_for``. This class just
checks the deadline between LLM turns so it stops hitting the API once time
expires.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC
from typing import Any, ClassVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

from observa.llm import (
    AgentRuntimeState,
    current_agent,
    current_token_tracker,
    current_tracker,
)
from observa.llm.structured import StructuredOutputError, parse_structured
from observa.models import (
    AgentQuestion,
    CaseState,
    Finding,
    Severity,
    TurnFindings,
)

logger = logging.getLogger(__name__)


def _format_abstain_reason(model: str, exc: BaseException, stage: str) -> str:
    """Build an abstain reason that names the offending model.

    When the LLM rejects bind_tools or with_structured_output (capability
    mismatch), produce a message that points the user at config.yaml. For
    every other exception, still prefix the model id so the FinalSummary
    surfaces which model failed.

    ``stage`` is one of "tool_loop" or "structured_output" — included in the
    fallback message for diagnostic clarity.
    """
    msg = str(exc)
    low = msg.lower()
    capability_markers = (
        "tools not supported",
        "function calling not supported",
        "tool calling is not supported",
        "unsupported response_format",
        "tool_use is not supported",
    )
    if any(marker in low for marker in capability_markers):
        return (
            f'model "{model}" rejected bind_tools/with_structured_output: {msg}. '
            f'Remove this model from config.yaml or pick one with both capabilities.'
        )
    return f'model "{model}" failed during {stage}: {type(exc).__name__}: {msg}'


# ---------------------------------------------------------------------------
# Structured output schema used by the LLM at the end of the reflection loop.
# ---------------------------------------------------------------------------

class _Finding(BaseModel):
    """A single evidence-based finding produced by a specialist agent."""
    severity: Severity = Field(description="Severity of this finding")
    code: str = Field(description="Short identifier, e.g. LOG_FILE_SYNC_HIGH")
    description: str = Field(description="One- or two-sentence explanation of the finding")
    evidence_refs: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of the evidence records that support this finding — copy the "
            "`evidence_id` values (e.g. \"Q-014\", \"F-003\") exactly as they "
            "appeared in the tool results you used"
        ),
    )
    evidence_note: str = Field(
        default="",
        description="One short sentence tying the cited evidence to the finding (which rows/values matter and why)",
    )
    related_objects: list[str] = Field(default_factory=list, description="Schema-qualified object names involved (tables, indexes, segments)")

    @field_validator("related_objects", "evidence_refs", mode="before")
    @classmethod
    def _none_to_empty_list(cls, v: Any) -> Any:
        # Some models (Gemini in particular) emit ``"related_objects": null``
        # when there are no related objects. Coerce to empty list so validation
        # succeeds — the field is semantically optional.
        return [] if v is None else v


class _Question(BaseModel):
    """A single clarifying question the agent wants to ask the analyst."""
    question: str = Field(
        description=(
            "A question to the analyst about CONTEXT THAT IS NOT IN THE DATABASE — "
            "e.g. recent changes, external incidents, whether the symptom is new or "
            "recurring, business context, workloads that are not yet in AWR. NEVER "
            "ask the analyst for data you can obtain yourself via query_awr / "
            "read_file (sessions, wait events, SQL history, blocker chains, etc. "
            "are all in DBA_HIST_*). If the only way to answer your question is to "
            "run another query, do not ask — run the query."
        )
    )


class _ReflectiveOutput(BaseModel):
    """Final structured report a specialist agent emits at the end of a turn.

    Carries the agent's findings, questions for the analyst, an explicit
    abstain flag (set when nothing relevant to the scope was found), and a
    short notes field summarizing what was explored.
    """
    abstained: bool = Field(description="True when no evidence-based finding was reached in your specialty area this turn")
    abstain_reason: str = Field(default="", description="One sentence explaining why the agent abstained, if abstained=True")
    findings: list[_Finding] = Field(default_factory=list, description="Evidence-based findings tied directly to the investigation scope")
    questions: list[_Question] = Field(default_factory=list, description="Questions for the analyst about context not present in the database")
    notes: str = Field(default="", description="One-sentence summary of what you explored this turn")


# ---------------------------------------------------------------------------
# Shared rules that apply to every specialist.
# ---------------------------------------------------------------------------

_UNIVERSAL_RULES = """
You are a specialist Oracle Database diagnostic agent. Follow these rules strictly:

0. ENVIRONMENT — CRITICAL CONTEXT:
   You are analyzing an **AWR REPOSITORY database**. This is NOT the database that has the performance problem. The problem database's AWR snapshots were IMPORTED into this repository via awrload/awrxtrb. All data you see is historical AWR data loaded from a remote production instance.

   Consequences you MUST internalize:
   - NEVER query V$ or GV$ views — they describe the LOCAL repository instance and tell you nothing about the problem database. Using them is always wrong. The allowlist will reject them anyway.
   - The repository may contain snapshots from multiple source databases. EVERY query against a DBA_HIST_* table MUST include a filter `WHERE dbid = <the DBID you were given>` (or join through a CTE that enforces it). Without the dbid filter you will mix data from unrelated databases.
   - If a query returns zero rows, that means "no data in this window for this DBID matching these conditions" — it does NOT mean "the tool is broken" or "the connection is down". Never ask the analyst to "restore the connection" or "provide an export" — the connection is fine. Instead: widen the snap range, drop optional predicates, or abstain with a clear reason.
   - The source database's Oracle version, topology and instance numbers are derived from DBA_HIST_DATABASE_INSTANCE (not V$INSTANCE). Query that view when you need topology info.

1. OBJECTIVITY: Report ONLY what the evidence (query results, file contents) directly shows. Do not speculate, extrapolate, or infer beyond the data. No "maionese" — if you don't have a specific observation backed by a specific query or file read, do not invent one.

2. SCOPE DISCIPLINE — THE HARDEST RULE: The analyst's INVESTIGATION SCOPE is the ONLY thing you are here to investigate. Every finding you report MUST have a direct, explained causal relationship to the scoped problem (the specific SQL, session, time window, object, or symptom named in the scope). A real issue in an unrelated area of the database is NOT a finding for this case — discard it. Before emitting a finding, write one sentence in `evidence_note` or `description` that explicitly ties it to the scope: e.g. "this contention affects segment X which is accessed by the scoped SQL_ID Y" or "this wait class dominates during the time window mentioned in the scope". If you cannot make that connection from the data, do NOT report it — abstain instead. Tangential observations are noise, not signal.

3. STRICT SPECIALTY: Stay within your specialty area defined below. If the data points to issues outside your area, DO NOT claim them — another specialist will handle them. You may, however, cite cross-domain observations from the blackboard as context.

4. ABSTAIN EXPLICITLY: If, after exploring the data, you find nothing relevant TO THE SCOPE in your specialty this turn, set abstained=true and state briefly why. "Nothing relevant to the scope" is a first-class outcome — report it cleanly instead of padding with tangential observations.

5. USE MCP TOOLS: A small set of MCP tools is bound to your session — exactly the ones your specialty needs. Use only those; do not assume access to tools that are not bound. Craft your tool calls to reflect the INVESTIGATION SCOPE first, then problem statement + known facts + blackboard findings. Your specialty prompt below describes the tools you actually have and how to use them.

   When ``query_awr`` IS bound to your session: it runs read-only SQL against DBA_HIST_* tables ONLY. No V$/GV$. Always include `dbid = <case DBID>` in the WHERE clause. Prefer queries that filter by the scoped SQL_ID / time window / object when the scope names one.
     * DO NOT prefix tables with a schema. Write `DBA_HIST_SNAPSHOT`, not `"OBSERVA"."DBA_HIST_SNAPSHOT"` or `SYS.DBA_HIST_SNAPSHOT`. The connection already resolves unqualified names correctly; schema-qualified references will be rejected or produce wrong results.
     * DO NOT invent column names. If you are unsure what columns a view exposes, query `SELECT column_name, data_type FROM DBA_TAB_COLUMNS WHERE owner='SYS' AND table_name='DBA_HIST_<name>'` FIRST to discover them, then write your real query. The column is called `column_name` only in DBA_TAB_COLUMNS itself — it is NOT a generic column on every view. Common columns on most DBA_HIST_* views are `snap_id`, `dbid`, `instance_number`. Beyond those, verify before you guess.
     * If a query returns `ORA-00904: "X": invalid identifier`, STOP re-trying variants. Use the DBA_TAB_COLUMNS probe above to find the actual column names, then rewrite.

6. BLACKBOARD AWARENESS: Earlier turns are shown below. Incorporate other agents' findings when they correlate with yours. Do NOT duplicate another agent's finding — cite it if useful, but don't re-report it.

7. TIME: You have a bounded budget per turn. Prioritize high-signal queries tied to the scope. Don't run many redundant queries.

8. QUESTIONS TO THE ANALYST — STRICT RULE: You may only ask the analyst for things that are NOT available in the database or the input files. Valid questions: "was there a recent change to X?", "is this symptom new or recurring?", "what was happening at the application layer at this time?", "can you provide <external log / ticket / config>?". INVALID questions: anything whose answer is in DBA_HIST_* (session details, blocker chains, wait-event distribution, SQL execution history, segment stats, etc.). If the only way to answer is to run a query, DO NOT ask — run the query yourself next turn or this turn. Asking the analyst to "allow a follow-up turn" or "provide SQL execution history" is forbidden; those are your job.

9. FINAL OUTPUT: After exploring, produce a single structured JSON response matching the schema requested. NO prose outside that JSON.

10. TIME WINDOWS: When the case defines a PROBLEM window (and optionally a BASELINE window), restrict every DBA_HIST_* query to those snap ranges (`snap_id BETWEEN <begin> AND <end>`). The guard warns or rejects unbounded scans. If a restricted query returns zero rows, abstain (Rule 4) rather than widening — the diff probe already covers the broader window. A precomputed BASELINE DIFF may be provided below — treat it as deterministic ground truth for orientation, and interpret your findings relative to those deltas; re-verify any delta you build a finding on.

11. EVIDENCE CITATION: Every query_awr / read_file result includes an `evidence_id` (e.g. "Q-014", "F-003"). For each finding, list in `evidence_refs` the evidence_ids of the tool results that support it — copy the IDs exactly; never invent one. Use `evidence_note` for a one-line pointer to the relevant rows/values. A finding with no valid evidence_refs is flagged "uncited" and carries less weight in consolidation. The precomputed BASELINE DIFF lists its own probe evidence IDs — you may cite those for deltas you build on.
"""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ReflectiveAgent(ABC):
    #: Short machine-identifier. Matches the config key in ``agents.enabled``.
    name: ClassVar[str]

    #: Human-readable label used in UI.
    display_name: ClassVar[str]

    #: Per-specialty instructions. MUST describe both the area of focus and
    #: explicit exclusions. Subclasses override this classvar.
    specialty_prompt: ClassVar[str]

    #: Subset of MCP tool names this agent is allowed to bind to its LLM.
    #: Default reflects the architectural rule: every specialist operates
    #: exclusively on the AWR repository via ``query_awr``. The diagnostic
    #: files MCP (``list_files`` / ``read_file``) is reserved for the
    #: dedicated ``file_reader`` agent, which overrides this classvar.
    allowed_tools: ClassVar[frozenset[str]] = frozenset({"query_awr"})

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        return (
            f"{_UNIVERSAL_RULES}\n\n"
            f"=== YOUR SPECIALTY: {self.display_name} ===\n"
            f"{self.specialty_prompt}"
        )

    def _blackboard_digest(self, state: CaseState, current_turn: int) -> str:
        """Render prior-turn findings from all agents as a compact text digest."""
        prior = [tf for tf in state.get("turn_findings", []) if tf.turn < current_turn]
        if not prior:
            return "(no prior turns yet — this is the first turn)"
        lines = []
        for tf in sorted(prior, key=lambda t: (t.turn, t.agent)):
            header = f"[turn {tf.turn} · {tf.agent}]"
            if tf.abstained:
                lines.append(f"{header} ABSTAINED — {tf.abstain_reason or 'nothing relevant found'}")
                continue
            for f in tf.findings:
                lines.append(f"{header} {f.severity.upper()} · {f.code}: {f.description}")
            if tf.notes:
                lines.append(f"{header} notes: {tf.notes}")
        return "\n".join(lines) if lines else "(prior turns produced no findings)"

    def _chat_digest(self, state: CaseState, current_turn: int) -> str:
        chat = [m for m in state.get("chat_log", []) if (m.turn is None or m.turn < current_turn)]
        if not chat:
            return "(no analyst chat yet)"
        return "\n".join(
            f"[{m.sender}{'/' + m.agent if m.agent else ''} · turn {m.turn}] {m.text}"
            for m in chat
        )

    def _files_digest(self, state: CaseState) -> str:
        files = state.get("input_files", []) or []
        if not files:
            return "(no files provided)"
        return "\n".join(f"- [{f.type}] {f.path}" + (f" ({f.label})" if f.label else "") for f in files)

    def _ora_timeline_block(self, state: CaseState) -> str:
        ora = state.get("ora_timeline")
        if not ora:
            return ""
        from observa.graph.ora_timeline import render_ora_timeline_digest
        digest = render_ora_timeline_digest(ora)
        if not digest:
            return ""
        return (
            "ORA INCIDENT TIMELINE (deterministic scan of the attached alert logs):\n"
            f"{digest}\n\n"
        )

    def _windows_digest(self, state: CaseState) -> str:
        problem = state.get("problem_window")
        if problem is None:
            return (
                "TIME WINDOWS: no problem/baseline windows defined for this case. "
                "Discover the relevant snap range via DBA_HIST_SNAPSHOT first."
            )
        # function-local: agents.base must not depend on graph at import time
        from observa.graph.diff_probe import render_diff_digest

        baseline = state.get("baseline_window")
        lines = [
            "TIME WINDOWS (fixed for this case — restrict queries to these snap ranges):",
            f"- PROBLEM: snaps {problem.begin_snap}→{problem.end_snap}",
        ]
        if baseline is not None:
            lines.append(f"- BASELINE: snaps {baseline.begin_snap}→{baseline.end_snap}")
        else:
            lines.append("- BASELINE: none (absolute profile)")
        lines.append("")
        lines.append("BASELINE DIFF (precomputed by a deterministic probe at case start):")
        lines.append(render_diff_digest(state.get("baseline_diff")))
        return "\n".join(lines)

    def _user_prompt_static(self, state: CaseState) -> str:
        """Case-level context that does NOT change across turns or tool-loop iterations.

        Kept as a separate method so it can be cached as a single prefix when
        the underlying provider supports it (Anthropic ``cache_control``).
        """
        return (
            f"INVESTIGATION SCOPE (this is what you are here to investigate — every finding must tie back to this):\n{state.get('investigation_scope', '')}\n\n"
            f"PROBLEM STATEMENT (background context):\n{state.get('problem_statement', '')}\n\n"
            f"KNOWN FACTS:\n"
            + "\n".join(f"- {fact}" for fact in (state.get('known_facts') or [])) + "\n\n"
            f"DBID: {state.get('dbid', 'unknown')}\n"
            f"ORACLE: {state.get('oracle_version', 'unknown')} · topology={state.get('topology', 'unknown')}\n\n"
            f"{self._windows_digest(state)}\n\n"
            f"{self._ora_timeline_block(state)}"
            f"AVAILABLE FILES:\n{self._files_digest(state)}\n"
            f"(Only the `file_reader` agent opens these files. If `read_file` / "
            f"`list_files` are not bound to your session, treat the list above as "
            f"context only — do NOT speculate about file contents.)"
        )

    def _user_prompt_variable(self, state: CaseState, current_turn: int) -> str:
        """Per-turn context (turn header, blackboard, chat) — NOT cached."""
        return (
            f"=== Turn {current_turn} of {state.get('total_turns', '?')} ===\n\n"
            f"BLACKBOARD (prior turns):\n{self._blackboard_digest(state, current_turn)}\n\n"
            f"ANALYST CHAT:\n{self._chat_digest(state, current_turn)}\n\n"
            f"Begin your reflection. Use tools as needed."
        )

    def _user_prompt(self, state: CaseState, current_turn: int) -> str:
        # Convenience for non-Anthropic providers and tests — returns the
        # full prompt as a single string, in the same order the cached
        # variant would assemble it.
        return (
            self._user_prompt_static(state)
            + "\n\n"
            + self._user_prompt_variable(state, current_turn)
        )

    def _final_request(self) -> str:
        schema = json.dumps(_ReflectiveOutput.model_json_schema(), indent=2)
        return (
            "Time is up — or you've gathered enough. Emit your structured output "
            "now as a single JSON object matching this schema. No prose, no code "
            "fences, just the JSON.\n\n"
            f"Schema:\n{schema}"
        )

    # ------------------------------------------------------------------
    # Tool-use loop
    # ------------------------------------------------------------------

    @staticmethod
    def _is_anthropic_llm(llm: BaseChatModel) -> bool:
        """Detect ChatAnthropic (possibly wrapped by RateLimitedLLM)."""
        candidate: Any = llm
        # Unwrap one level of delegation if present (RateLimitedLLM.__getattr__
        # passes attribute reads through, but ``_inner`` is a real attribute).
        inner = getattr(llm, "_inner", None)
        if inner is not None:
            candidate = inner
        return type(candidate).__name__ == "ChatAnthropic"

    def _build_messages(
        self,
        state: CaseState,
        turn: int,
        cache_eligible: bool,
    ) -> list[BaseMessage]:
        """Construct system + user messages, with cache breakpoints when supported.

        Anthropic's prompt cache rewards stable prefixes: the system prompt and
        the case-level user context (scope, problem, files) are identical for
        every ``ainvoke`` inside the tool-use loop AND across turns. Marking
        them as cacheable means the second call onward in a session pays
        ~10 % of the prompt-token price for the cached portion.
        """
        if not cache_eligible:
            return [
                SystemMessage(content=self._system_prompt()),
                HumanMessage(content=self._user_prompt(state, turn)),
            ]
        return [
            SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": self._system_prompt(),
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            ),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": self._user_prompt_static(state),
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": self._user_prompt_variable(state, turn),
                    },
                ]
            ),
        ]

    def _set_status(self, status: str) -> None:
        tt = current_token_tracker()
        if tt is None:
            return
        st = tt.by_agent.setdefault(self.name, AgentRuntimeState())
        st.status = status
        st.last_activity = time.monotonic()

    async def _execute_tool_call(
        self,
        call: dict[str, Any],
        tools_by_name: dict[str, BaseTool],
    ) -> str:
        name = call.get("name", "")
        args = call.get("args", {}) or {}
        tool = tools_by_name.get(name)
        if tool is None:
            logger.warning("%s: LLM requested unknown tool %r", self.name, name)
            return f"ERROR: unknown tool '{name}'"
        logger.info("%s: tool_call %s args=%s", self.name, name, str(args)[:300])
        self._set_status("tool")
        try:
            result = await tool.ainvoke(args)
        except Exception as exc:  # noqa: BLE001 — surface to LLM so it adapts
            logger.warning("%s: tool %s failed: %s", self.name, name, exc)
            return f"ERROR: {type(exc).__name__}: {exc}"
        finally:
            self._set_status("thinking")
        if isinstance(result, (dict, list)):
            serialized = json.dumps(result, default=str)
            logger.debug(
                "%s: tool %s ← %d chars (%.0f%% truncated)",
                self.name, name, len(serialized),
                100 * min(len(serialized), 8000) / max(len(serialized), 1),
            )
            return serialized[:8000]
        return str(result)[:8000]

    async def _tool_loop(
        self,
        messages: list[BaseMessage],
        llm_with_tools: BaseChatModel,
        tools_by_name: dict[str, BaseTool],
        deadline: float,
        max_iterations: int = 20,
    ) -> list[BaseMessage]:
        for _ in range(max_iterations):
            # Rate-limit sleep time is not "working time" — push the deadline
            # forward by whatever the tracker has accumulated so far.
            tracker = current_tracker()
            effective_deadline = deadline + (tracker.sleep_seconds if tracker else 0.0)
            if time.monotonic() >= effective_deadline:
                logger.info("%s: deadline reached in tool loop", self.name)
                break
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                break
            # Gemini's API rejects the next turn with HTTP 400 if the count of
            # function-response parts differs from the count of function-call
            # parts in the assistant turn. Guarantee a ToolMessage per tool_call
            # even if execution raises something unexpected — a missing response
            # would corrupt the conversation for any subsequent OCI request.
            for call in tool_calls:
                tool_call_id = call.get("id") or ""
                try:
                    content = await self._execute_tool_call(call, tools_by_name)
                except Exception as exc:  # noqa: BLE001 — never break call/response parity
                    logger.exception(
                        "%s: tool execution raised unexpectedly for call %r",
                        self.name, call.get("name"),
                    )
                    content = f"ERROR: {type(exc).__name__}: {exc}"
                messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
        return messages

    async def _force_structured_output(
        self,
        messages: list[BaseMessage],
        llm: BaseChatModel,
        model_id: str,
    ) -> _ReflectiveOutput:
        prompt = messages + [HumanMessage(content=self._final_request())]
        try:
            return await parse_structured(llm, _ReflectiveOutput, prompt)
        except StructuredOutputError as exc:
            logger.warning(
                "%s: unexpected structured output shape — parsing_error=%r raw_text=%r raw_tool_calls=%r",
                self.name, exc.parsing_error, exc.raw_text, exc.raw_tool_calls,
            )
            empty_response = (
                not exc.raw_text
                and not exc.raw_tool_calls
                and exc.parsing_error is None
            )
            if empty_response:
                # Most likely a safety filter, max_tokens, or transient model
                # blank — see the OCI provider's "choice.message is None"
                # warning for the finish_reason. Surface a clearer reason than
                # "no parseable output" since there's literally nothing to parse.
                reason = (
                    f'model "{model_id}" returned an empty response '
                    "(no content, no tool calls). Likely blocked by a safety "
                    "filter or hit max_tokens — check the provider log for "
                    "finish_reason."
                )
            else:
                reason = (
                    f'model "{model_id}" did not produce a parseable _ReflectiveOutput. '
                    f'parsing_error={exc.parsing_error!r} raw_text={exc.raw_text!r} '
                    f'tool_calls={exc.raw_tool_calls!r}'
                )
            return _ReflectiveOutput(abstained=True, abstain_reason=reason)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: structured output failed (%s); abstaining", self.name, exc)
            return _ReflectiveOutput(
                abstained=True,
                abstain_reason=_format_abstain_reason(model_id, exc, "structured_output"),
            )

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    async def run(
        self,
        state: CaseState,
        turn: int,
        llm: BaseChatModel,
        mcp_tools: list[BaseTool],
        time_budget_seconds: int,
    ) -> TurnFindings:
        deadline = time.monotonic() + time_budget_seconds
        from observa.config import get_settings
        model_id = get_settings().model_for_agent(self.name)
        mcp_tools = [t for t in mcp_tools if t.name in self.allowed_tools]
        tools_by_name = {t.name: t for t in mcp_tools}
        llm_with_tools = llm.bind_tools(mcp_tools) if mcp_tools else llm

        cache_eligible = self._is_anthropic_llm(llm)
        if cache_eligible:
            logger.debug("%s: prompt caching enabled (Anthropic)", self.name)
        messages: list[BaseMessage] = self._build_messages(state, turn, cache_eligible)

        # Bind the agent name to THIS task's context so concurrent agents don't
        # trample each other's token attribution. ContextVar.set() returns a
        # token we restore in finally — this is the asyncio-safe pattern.
        tt = current_token_tracker()
        if tt is not None:
            tt.by_agent.setdefault(self.name, AgentRuntimeState())
        agent_token = current_agent.set(self.name)
        self._set_status("thinking")

        try:
            try:
                messages = await self._tool_loop(messages, llm_with_tools, tools_by_name, deadline)
            except Exception as exc:  # noqa: BLE001
                logger.exception("%s: tool loop failed", self.name)
                self._set_status("error")
                return TurnFindings(
                    agent=self.name,
                    turn=turn,
                    abstained=True,
                    abstain_reason=_format_abstain_reason(model_id, exc, "tool_loop"),
                )

            output = await self._force_structured_output(messages, llm, model_id)
            self._set_status("abstained" if output.abstained else "done")
            return self._to_turn_findings(output, turn)
        finally:
            current_agent.reset(agent_token)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _to_turn_findings(self, output: _ReflectiveOutput, turn: int) -> TurnFindings:
        findings = [
            Finding(
                finding_id=f"{self.name}-{turn}-{uuid.uuid4().hex[:8]}",
                agent=self.name,
                turn=turn,
                severity=f.severity,
                code=f.code,
                description=f.description,
                evidence_refs=list(f.evidence_refs),
                evidence_note=f.evidence_note,
                uncited=not f.evidence_refs,
                related_objects=f.related_objects,
            )
            for f in output.findings
        ]
        questions = [
            AgentQuestion(agent=self.name, turn=turn, question=q.question)
            for q in output.questions
        ]
        if not findings and not output.abstained:
            # LLM produced no findings but didn't mark abstain — normalize.
            return TurnFindings(
                agent=self.name,
                turn=turn,
                abstained=True,
                abstain_reason=output.abstain_reason or "No evidence-based findings this turn",
                notes=output.notes,
                questions=questions,
            )
        return TurnFindings(
            agent=self.name,
            turn=turn,
            findings=findings,
            questions=questions,
            abstained=output.abstained,
            abstain_reason=output.abstain_reason,
            notes=output.notes,
        )
