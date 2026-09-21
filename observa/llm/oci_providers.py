"""Observa-shipped OCI providers for langchain-community's ChatOCIGenAI.

langchain-community 0.4.1's ChatOCIGenAI._provider_map only registers
CohereProvider and MetaProvider. This module ships providers for the OCI
GENERIC chat-request API with full tool-calling support — currently
``XaiProvider`` (xAI Grok) and ``GoogleProvider`` (Gemini) — and a
ChatOCIGenAI subclass that injects them under the matching keys. Both
providers share the same wire format, so the conversion logic lives in a
private ``_GenericChatProvider`` base; subclasses exist only as distinct
identities for the provider lookup. Future GENERIC-format providers extend
by adding a new subclass and one more entry to
``ObservaChatOCIGenAI._provider_map``.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any, Mapping

logger = logging.getLogger(__name__)

from langchain_community.chat_models.oci_generative_ai import (
    ChatOCIGenAI,
    Provider,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel


class _GenericChatProvider(Provider):
    """OCI GENERIC chat-request provider with tool calling.

    Shared by xAI Grok and Google Gemini — both speak the same GENERIC wire
    format (OpenAI-style messages, OCI-shaped ``FunctionCall`` /
    ``FunctionDefinition`` objects). Subclasses exist only so the upstream
    ``_provider_map`` lookup (keyed on ``model_id.split('.')[0]``) can resolve
    distinct identities to the same implementation.

    ``allow_parallel_tool_calls`` controls whether the request leaves OCI's
    ``isParallelToolCalls`` unset (True, default — model decides) or forces it
    off (False). Gemini's translator collapses consecutive function-call
    assistant turns when the model emits parallel calls, breaking the
    function_call/function_response part-count parity that Gemini's API
    enforces; forcing serial calls eliminates the multi-call assistant turn
    entirely.
    """

    allow_parallel_tool_calls: bool = True

    @property
    def stop_sequence_key(self) -> str:
        return "stop"

    def __init__(self) -> None:
        from oci.generative_ai_inference import models

        self.oci_chat_request = models.GenericChatRequest
        self.oci_user_message = models.UserMessage
        self.oci_system_message = models.SystemMessage
        self.oci_assistant_message = models.AssistantMessage
        self.oci_tool_message = models.ToolMessage
        self.oci_text_content = models.TextContent
        self.oci_function_call = models.FunctionCall
        self.oci_function_definition = models.FunctionDefinition
        self.oci_tool_choice_auto = models.ToolChoiceAuto
        self.chat_api_format = models.BaseChatRequest.API_FORMAT_GENERIC

    def get_role(self, message: BaseMessage) -> str:
        if isinstance(message, HumanMessage):
            return "USER"
        if isinstance(message, AIMessage):
            return "ASSISTANT"
        if isinstance(message, SystemMessage):
            return "SYSTEM"
        if isinstance(message, ToolMessage):
            return "TOOL"
        raise ValueError(f"Got unknown type {message}")

    def messages_to_oci_params(
        self, messages: Any, **kwargs: Any
    ) -> dict[str, Any]:
        import json

        oci_messages: list[Any] = []
        for msg in messages:
            role = self.get_role(msg)
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            content_items = [self.oci_text_content(text=text or " ")]

            if role == "USER":
                oci_messages.append(self.oci_user_message(content=content_items))
            elif role == "SYSTEM":
                oci_messages.append(self.oci_system_message(content=content_items))
            elif role == "ASSISTANT":
                tool_calls = getattr(msg, "tool_calls", None) or None
                if tool_calls:
                    oci_tool_calls = [
                        self.oci_function_call(
                            id=tc["id"],
                            name=tc["name"],
                            arguments=json.dumps(tc["args"]),
                        )
                        for tc in tool_calls
                    ]
                    # OCI's AssistantMessage docstring is explicit: "When
                    # responding to a tool call, set ``content`` to null (not
                    # ``""``)." Gemini enforces this — sending a padded text
                    # part alongside tool_calls makes its translator count
                    # mismatched function-call vs function-response parts and
                    # rejects the next turn with HTTP 400. xAI tolerated it,
                    # which is why this only surfaced after enabling Gemini.
                    # Pass real text only when the assistant actually emitted
                    # text content; otherwise omit content entirely.
                    real_text = text.strip()
                    assistant_kwargs: dict[str, Any] = {"tool_calls": oci_tool_calls}
                    if real_text:
                        assistant_kwargs["content"] = [self.oci_text_content(text=real_text)]
                    oci_messages.append(self.oci_assistant_message(**assistant_kwargs))
                else:
                    oci_messages.append(
                        self.oci_assistant_message(content=content_items)
                    )
            elif role == "TOOL":
                oci_messages.append(
                    self.oci_tool_message(
                        tool_call_id=getattr(msg, "tool_call_id", ""),
                        content=content_items,
                    )
                )

        out: dict[str, Any] = {
            "messages": oci_messages,
            "api_format": self.chat_api_format,
        }
        tools = kwargs.get("tools")
        if tools:
            out["tools"] = list(tools)
            out["tool_choice"] = self.oci_tool_choice_auto()
            if not self.allow_parallel_tool_calls:
                out["is_parallel_tool_calls"] = False
        return out

    def convert_to_oci_tool(self, tool: Any) -> Any:
        if isinstance(tool, BaseTool) or (isinstance(tool, type) and issubclass(tool, BaseModel)) or callable(tool):
            schema = convert_to_openai_function(tool)
            name = schema.get("name") or getattr(tool, "name", None) or getattr(tool, "__name__", None)
            description = schema.get("description") or name or ""
            parameters = schema.get("parameters") or {"type": "object", "properties": {}, "required": []}
            return self.oci_function_definition(
                name=name,
                description=description,
                parameters=parameters,
            )
        if isinstance(tool, dict):
            if not all(k in tool for k in ("title", "description", "properties")):
                raise ValueError(
                    "Unsupported dict type. Tool dict must have keys: title, description, properties."
                )
            parameters = {
                "type": "object",
                "properties": tool["properties"],
                "required": tool.get("required", []),
            }
            return self.oci_function_definition(
                name=tool["title"],
                description=tool["description"],
                parameters=parameters,
            )
        raise ValueError(
            f"Unsupported tool type {type(tool)}. Tool must be a BaseTool, JSON schema dict, BaseModel subclass, or callable."
        )

    def chat_response_to_text(self, response: Any) -> str:
        # Gemini (and occasionally other GENERIC providers) can return a choice
        # with a null ``message`` — typically when the response was blocked by
        # a safety filter, hit max_tokens with no content, or otherwise didn't
        # produce any output. Treat that as empty content; the caller logs the
        # finish_reason via ``chat_generation_info``.
        message = response.data.chat_response.choices[0].message
        if message is None:
            return ""
        content = message.content or []
        return content[0].text if content else ""

    def chat_generation_info(self, response: Any) -> dict[str, Any]:
        choice = response.data.chat_response.choices[0]
        info: dict[str, Any] = {
            "finish_reason": choice.finish_reason,
            "time_created": str(response.data.chat_response.time_created),
        }
        if choice.message is None:
            return info
        tool_calls = getattr(choice.message, "tool_calls", None)
        if tool_calls:
            info["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in tool_calls
            ]
        return info

    def chat_stream_to_text(self, event_data: dict) -> str:
        message = event_data.get("message") or {}
        content = message.get("content") or []
        if not content:
            return ""
        first = content[0]
        if isinstance(first, dict):
            return first.get("text") or ""
        return getattr(first, "text", "") or ""

    def is_chat_stream_end(self, event_data: dict) -> bool:
        return "finishReason" in event_data

    def chat_stream_generation_info(self, event_data: dict) -> dict[str, Any]:
        return {"finish_reason": event_data.get("finishReason")}


class XaiProvider(_GenericChatProvider):
    """OCI GENERIC chat-request provider for xAI Grok models (``xai.*``)."""


class GoogleProvider(_GenericChatProvider):
    """OCI GENERIC chat-request provider for Google Gemini models (``google.*``).

    Forces ``is_parallel_tool_calls=False`` because Gemini's API rejects the
    next turn with HTTP 400 ("function response parts != function call parts")
    when the model emits parallel tool calls — OCI's translator into Gemini's
    native part-based format collapses consecutive turns and the part-count
    parity check fails.
    """

    allow_parallel_tool_calls = False


class OpenaiProvider(_GenericChatProvider):
    """OCI GENERIC chat-request provider for OpenAI models (``openai.*``)."""


class ObservaChatOCIGenAI(ChatOCIGenAI):
    """ChatOCIGenAI subclass that adds GENERIC-format providers and a fixed _generate.

    Two overrides:

    1. ``_provider_map`` — adds ``"xai": XaiProvider()`` and
       ``"google": GoogleProvider()`` so the ChatOCIGenAI provider lookup
       (keyed on ``model_id.split('.')[0]``) resolves xAI / Gemini model ids
       to our shipped providers. Future GENERIC-format providers extend by
       appending entries.

    2. ``_generate`` — upstream's tool-call extraction at lines 800-803 of
       langchain_community/chat_models/oci_generative_ai.py reads
       ``response.data.chat_response.tool_calls`` (the Cohere shape). The
       GENERIC chat-request schema used by xAI and Gemini puts tool calls at
       ``choices[0].message.tool_calls`` with flat ``tc.id`` / ``tc.name`` /
       ``tc.arguments`` (JSON string). When the active provider is a
       ``_GenericChatProvider``, this override extracts the LangChain ToolCall
       list from the GENERIC shape; for other providers it delegates to
       ``super()._generate``.

    ``_astream`` is overridden to bypass the broken streaming path for
    ``_GenericChatProvider``. See the ``_astream`` docstring for the full
    rationale.
    """

    @property
    def _provider_map(self) -> Mapping[str, Any]:
        return {
            **super()._provider_map,
            "xai": XaiProvider(),
            "google": GoogleProvider(),
            "openai": OpenaiProvider(),
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(self._provider, _GenericChatProvider):
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        # GENERIC chat-request path for xAI Grok.
        # We mirror the upstream _generate body but extract tool calls from
        # `choices[0].message.tool_calls` instead of the Cohere-only
        # `chat_response.tool_calls` path.
        import json

        from langchain_community.llms.utils import enforce_stop_tokens
        from langchain_core.messages import AIMessage, ToolCall
        from langchain_core.outputs import ChatGeneration, ChatResult

        if self.is_stream:
            stream_iter = self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            from langchain_core.language_models.chat_models import generate_from_stream
            return generate_from_stream(stream_iter)

        request = self._prepare_request(messages, stop=stop, stream=False, **kwargs)
        try:
            response = self.client.chat(request)
        except Exception as exc:
            # Surface the conversation shape on failures so 400s like Gemini's
            # "function response parts != function call parts" can be diagnosed
            # without re-running. Logs counts only — no message content.
            try:
                shape = []
                for m in messages:
                    role = type(m).__name__
                    n_calls = len(getattr(m, "tool_calls", None) or [])
                    has_id = bool(getattr(m, "tool_call_id", None))
                    shape.append(f"{role}(calls={n_calls},tool_id={has_id})")
                logger.warning(
                    "OCI chat() failed: %s — conversation shape: [%s]",
                    type(exc).__name__, ", ".join(shape),
                )
            except Exception:  # noqa: BLE001
                pass
            raise

        content = self._provider.chat_response_to_text(response)
        if stop is not None:
            content = enforce_stop_tokens(content, stop)

        generation_info = self._provider.chat_generation_info(response)

        llm_output = {
            "model_id": response.data.model_id,
            "model_version": response.data.model_version,
            "request_id": response.request_id,
            "content-length": response.headers["content-length"],
        }

        choice = response.data.chat_response.choices[0]
        if choice.message is None:
            logger.warning(
                "OCI GENERIC: choice.message is None — finish_reason=%r model=%s",
                choice.finish_reason, response.data.model_id,
            )
        oci_tool_calls = getattr(choice.message, "tool_calls", None) or []
        # LangChain's ToolMessage requires a string ``tool_call_id`` and the
        # agent tool-loop reads it back as ``call["id"]`` — propagating ``None``
        # there fails pydantic validation. Some GENERIC-format providers
        # (notably Google Gemini via OCI) don't emit an ``id`` on their tool
        # calls, so synthesize one when missing. The synthesized id is mirrored
        # back to the model through the matching ToolMessage and returns via
        # ``messages_to_oci_params``, which is sufficient for round-tripping.
        lc_tool_calls = [
            ToolCall(
                name=tc.name,
                args=json.loads(tc.arguments) if tc.arguments else {},
                id=tc.id or f"call_{uuid.uuid4().hex[:12]}",
            )
            for tc in oci_tool_calls
        ]

        # OCI GENERIC chat-response carries an optional ``usage`` object with
        # ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` and an
        # optional ``prompt_tokens_details.cached_tokens``. LangChain Core's
        # standard usage_metadata shape uses ``input_tokens`` / ``output_tokens``
        # / ``total_tokens`` plus ``input_token_details.cache_read``. Translate
        # so observa's RateLimitedLLM token tracker (which reads usage_metadata
        # off the AIMessage) records this call.
        usage = getattr(response.data.chat_response, "usage", None)
        usage_metadata: dict[str, Any] | None = None
        if usage is not None:
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(
                getattr(usage, "total_tokens", 0) or (input_tokens + output_tokens)
            )
            usage_metadata = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
            details = getattr(usage, "prompt_tokens_details", None)
            cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
            if cached:
                usage_metadata["input_token_details"] = {"cache_read": cached}

        message = AIMessage(
            content=content,
            additional_kwargs=generation_info,
            tool_calls=lc_tool_calls,
            usage_metadata=usage_metadata,
        )
        return ChatResult(
            generations=[ChatGeneration(message=message, generation_info=generation_info)],
            llm_output=llm_output,
        )

    async def _astream(  # type: ignore[override]
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Async path: bypass upstream's streaming for GENERIC-format providers.

        LangChain Core's `_agenerate_with_cache` always iterates `_astream`
        when invoked via `ainvoke`. Upstream `_astream` (inherited from
        `BaseChatModel`) runs `_stream` in an executor; that path calls
        `chat_stream_to_text` which assumes a Meta-shaped event payload and
        crashes on GENERIC-format tool-call/metadata deltas (KeyError: 'content').

        Observa does not depend on streaming UX — agents await the full
        AIMessage. So for `_GenericChatProvider` (xAI, Gemini) we run our
        (correct) `_generate` in an executor and emit its result as a single
        chunk. Other providers fall back to upstream's default `_astream`.
        """
        if not isinstance(self._provider, _GenericChatProvider):
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk
            return

        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_core.runnables.config import run_in_executor

        sync_run_manager = run_manager.get_sync() if run_manager else None
        result = await run_in_executor(
            None,
            self._generate,
            messages,
            stop,
            sync_run_manager,
            **kwargs,
        )
        gen = result.generations[0]
        msg = gen.message
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=msg.content,
                additional_kwargs=msg.additional_kwargs,
                tool_calls=msg.tool_calls,
                usage_metadata=getattr(msg, "usage_metadata", None),
            ),
            generation_info=gen.generation_info,
        )
