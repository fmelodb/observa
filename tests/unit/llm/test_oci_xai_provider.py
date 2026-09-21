"""Unit tests for the Observa-shipped XaiProvider for ChatOCIGenAI."""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def _install_fake_oci_models(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Install a fake `oci.generative_ai_inference.models` module.

    Returns a namespace whose attributes are the MagicMock classes the test can
    inspect to assert XaiProvider built the right OCI request shape. The fake
    module is installed in `sys.modules` BEFORE XaiProvider is imported so its
    late-import inside `__init__` resolves to these mocks.
    """
    fake_models = types.SimpleNamespace(
        GenericChatRequest=MagicMock(name="GenericChatRequest"),
        UserMessage=MagicMock(name="UserMessage"),
        SystemMessage=MagicMock(name="SystemMessage"),
        AssistantMessage=MagicMock(name="AssistantMessage"),
        ToolMessage=MagicMock(name="ToolMessage"),
        TextContent=MagicMock(name="TextContent"),
        ToolCall=MagicMock(name="ToolCall"),
        FunctionCall=MagicMock(name="FunctionCall"),
        ToolDefinition=MagicMock(name="ToolDefinition"),
        FunctionDefinition=MagicMock(name="FunctionDefinition"),
        ToolChoiceAuto=MagicMock(name="ToolChoiceAuto"),
        BaseChatRequest=types.SimpleNamespace(API_FORMAT_GENERIC="GENERIC"),
    )
    fake_pkg = types.ModuleType("oci")
    fake_inf = types.ModuleType("oci.generative_ai_inference")
    fake_models_mod = types.ModuleType("oci.generative_ai_inference.models")
    for name, value in vars(fake_models).items():
        setattr(fake_models_mod, name, value)
    monkeypatch.setitem(sys.modules, "oci", fake_pkg)
    monkeypatch.setitem(sys.modules, "oci.generative_ai_inference", fake_inf)
    monkeypatch.setitem(sys.modules, "oci.generative_ai_inference.models", fake_models_mod)
    return fake_models


def test_xai_provider_get_role(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    p = XaiProvider()
    assert p.get_role(HumanMessage(content="x")) == "USER"
    assert p.get_role(SystemMessage(content="x")) == "SYSTEM"
    assert p.get_role(AIMessage(content="x")) == "ASSISTANT"
    assert p.get_role(ToolMessage(content="x", tool_call_id="t1")) == "TOOL"


def test_xai_provider_stop_sequence_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    assert XaiProvider().stop_sequence_key == "stop"


def test_messages_to_oci_params_system_and_user(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    p = XaiProvider()
    out = p.messages_to_oci_params(
        [SystemMessage(content="be brief"), HumanMessage(content="hello")]
    )

    assert out["api_format"] == "GENERIC"
    assert "tools" not in out and "tool_choice" not in out
    assert len(out["messages"]) == 2
    fakes.SystemMessage.assert_called_once()
    fakes.UserMessage.assert_called_once()
    fakes.TextContent.assert_any_call(text="be brief")
    fakes.TextContent.assert_any_call(text="hello")


def test_messages_to_oci_params_assistant_text_only(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    p = XaiProvider()
    p.messages_to_oci_params(
        [HumanMessage(content="hi"), AIMessage(content="hello back")]
    )

    fakes.AssistantMessage.assert_called_once()
    kwargs = fakes.AssistantMessage.call_args.kwargs
    # No tool_calls when none on the AIMessage.
    assert "tool_calls" not in kwargs or kwargs["tool_calls"] is None


def test_messages_to_oci_params_unknown_role_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_oci_models(monkeypatch)
    from langchain_core.messages import ChatMessage
    from observa.llm.oci_providers import XaiProvider

    with pytest.raises(ValueError, match="unknown type"):
        XaiProvider().messages_to_oci_params([ChatMessage(role="oracle", content="x")])


def test_messages_to_oci_params_assistant_with_tool_calls_omits_empty_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the assistant emits only tool calls (no text), OCI requires
    ``content`` to be omitted (not padded with whitespace). Gemini enforces
    this — see AssistantMessage docstring. xAI tolerated padding, which is
    why this regression only surfaced after enabling Gemini."""
    fakes = _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    ai = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "add", "args": {"a": 1, "b": 2}}],
    )
    XaiProvider().messages_to_oci_params([HumanMessage(content="2+1?"), ai])

    asst_kwargs = fakes.AssistantMessage.call_args.kwargs
    assert "tool_calls" in asst_kwargs and len(asst_kwargs["tool_calls"]) == 1
    assert "content" not in asst_kwargs, "must not pad tool-only assistant messages with whitespace content"
    fakes.FunctionCall.assert_called_once_with(
        id="call_1", name="add", arguments='{"a": 1, "b": 2}'
    )


def test_messages_to_oci_params_assistant_with_text_and_tool_calls_keeps_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the assistant emitted real text alongside tool calls, the text is preserved."""
    fakes = _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    ai = AIMessage(
        content="calling add",
        tool_calls=[{"id": "call_1", "name": "add", "args": {"a": 1, "b": 2}}],
    )
    XaiProvider().messages_to_oci_params([HumanMessage(content="2+1?"), ai])

    asst_kwargs = fakes.AssistantMessage.call_args.kwargs
    assert "tool_calls" in asst_kwargs
    assert "content" in asst_kwargs and len(asst_kwargs["content"]) == 1
    fakes.TextContent.assert_any_call(text="calling add")


def test_messages_to_oci_params_tool_message(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    XaiProvider().messages_to_oci_params(
        [
            HumanMessage(content="2+1?"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "add", "args": {"a": 2, "b": 1}}],
            ),
            ToolMessage(content="3", tool_call_id="call_1"),
        ]
    )

    fakes.ToolMessage.assert_called_once()
    tm_kwargs = fakes.ToolMessage.call_args.kwargs
    assert tm_kwargs["tool_call_id"] == "call_1"


def test_messages_to_oci_params_passes_tools_and_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fakes = _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    fake_tool_def = object()
    out = XaiProvider().messages_to_oci_params(
        [HumanMessage(content="hi")], tools=[fake_tool_def]
    )
    assert out["tools"] == [fake_tool_def]
    fakes.ToolChoiceAuto.assert_called_once_with()
    assert out["tool_choice"] is fakes.ToolChoiceAuto.return_value


def test_messages_to_oci_params_xai_does_not_disable_parallel_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """xAI tolerates parallel tool calls — leave ``is_parallel_tool_calls`` unset
    so the model can choose."""
    _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    out = XaiProvider().messages_to_oci_params(
        [HumanMessage(content="?")],
        tools=[MagicMock(name="ToolDef")],
    )
    assert "is_parallel_tool_calls" not in out


def test_messages_to_oci_params_google_disables_parallel_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini's translator breaks the function_call/function_response part
    parity when the model emits parallel calls. GoogleProvider forces
    ``is_parallel_tool_calls=False`` to keep the conversation strictly
    alternating."""
    _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import GoogleProvider

    out = GoogleProvider().messages_to_oci_params(
        [HumanMessage(content="?")],
        tools=[MagicMock(name="ToolDef")],
    )
    assert out["is_parallel_tool_calls"] is False


def test_messages_to_oci_params_google_omits_parallel_flag_when_no_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without tools bound, ``is_parallel_tool_calls`` is meaningless and must
    not be sent — OCI may reject the field outside a tool-calling request."""
    _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import GoogleProvider

    out = GoogleProvider().messages_to_oci_params([HumanMessage(content="?")])
    assert "is_parallel_tool_calls" not in out


def test_messages_to_oci_params_no_tools_kwarg_omits_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    out = XaiProvider().messages_to_oci_params([HumanMessage(content="hi")])
    assert "tools" not in out
    assert "tool_choice" not in out


def test_convert_to_oci_tool_basetool(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = _install_fake_oci_models(monkeypatch)
    from langchain_core.tools import tool
    from observa.llm.oci_providers import XaiProvider

    @tool
    def add(a: int, b: int = 0) -> int:
        """Add two integers."""
        return a + b

    result = XaiProvider().convert_to_oci_tool(add)

    fakes.FunctionDefinition.assert_called_once()
    fd_kwargs = fakes.FunctionDefinition.call_args.kwargs
    assert fd_kwargs["name"] == "add"
    assert "Add two integers" in fd_kwargs["description"]
    params = fd_kwargs["parameters"]
    assert params["type"] == "object"
    assert "a" in params["properties"]
    assert "b" in params["properties"]
    assert params["required"] == ["a"]
    assert result is fakes.FunctionDefinition.return_value
    fakes.ToolDefinition.assert_not_called()


def test_convert_to_oci_tool_pydantic_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = _install_fake_oci_models(monkeypatch)
    from pydantic import BaseModel, Field
    from observa.llm.oci_providers import XaiProvider

    class Answer(BaseModel):
        """A short answer."""
        text: str = Field(description="the answer")
        confidence: float = 0.5

    XaiProvider().convert_to_oci_tool(Answer)
    fd_kwargs = fakes.FunctionDefinition.call_args.kwargs
    assert fd_kwargs["name"] == "Answer"
    params = fd_kwargs["parameters"]
    assert "text" in params["properties"]
    assert "text" in params["required"]
    assert "confidence" not in params["required"]


def test_convert_to_oci_tool_invalid_dict_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    with pytest.raises(ValueError, match="Unsupported"):
        XaiProvider().convert_to_oci_tool({"foo": "bar"})


def test_convert_to_oci_tool_unsupported_type_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    with pytest.raises(ValueError, match="Unsupported tool type"):
        XaiProvider().convert_to_oci_tool(42)


def test_chat_response_to_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    response = MagicMock()
    response.data.chat_response.choices = [
        MagicMock(message=MagicMock(content=[MagicMock(text="hello world")]))
    ]
    assert XaiProvider().chat_response_to_text(response) == "hello world"


def test_chat_generation_info_no_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    response = MagicMock()
    choice = MagicMock(finish_reason="stop")
    choice.message.tool_calls = None
    response.data.chat_response.choices = [choice]
    response.data.chat_response.time_created = "2026-04-30T00:00:00Z"

    info = XaiProvider().chat_generation_info(response)
    assert info["finish_reason"] == "stop"
    assert "tool_calls" not in info


def test_chat_generation_info_with_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    # Real OCI FunctionCall is flat: tc.name / tc.arguments are direct
    # attributes (no tc.function nesting). Use spec= to ensure MagicMock
    # rejects access to a nonexistent .function attribute — otherwise this
    # test would silently mask the regression we're guarding against.
    tc = MagicMock(spec=["id", "type", "name", "arguments"])
    tc.id = "call_1"
    tc.type = "FUNCTION"
    tc.name = "add"
    tc.arguments = '{"a": 1, "b": 2}'
    choice = MagicMock(finish_reason="tool_calls")
    choice.message.tool_calls = [tc]
    response = MagicMock()
    response.data.chat_response.choices = [choice]
    response.data.chat_response.time_created = "2026-04-30T00:00:00Z"

    info = XaiProvider().chat_generation_info(response)
    assert info["finish_reason"] == "tool_calls"
    calls = info["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["id"] == "call_1"
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "add"
    assert calls[0]["function"]["arguments"] == '{"a": 1, "b": 2}'


def test_chat_stream_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    p = XaiProvider()
    streaming = {"message": {"content": [{"text": "hi"}]}}
    assert p.is_chat_stream_end(streaming) is False
    assert p.chat_stream_to_text(streaming) == "hi"
    end = {"finishReason": "stop"}
    assert p.is_chat_stream_end(end) is True
    assert p.chat_stream_generation_info(end) == {"finish_reason": "stop"}


def _stub_upstream_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub CohereProvider and MetaProvider __init__ so super()._provider_map
    doesn't blow up when the fake OCI models module lacks Cohere/Meta classes."""
    from langchain_community.chat_models.oci_generative_ai import (
        CohereProvider,
        MetaProvider,
    )

    monkeypatch.setattr(CohereProvider, "__init__", lambda self: None)
    monkeypatch.setattr(MetaProvider, "__init__", lambda self: None)


def test_chat_response_to_text_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    response = MagicMock()
    response.data.chat_response.choices = [MagicMock(message=MagicMock(content=[]))]
    assert XaiProvider().chat_response_to_text(response) == ""


def test_observa_chat_oci_genai_provider_map(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_oci_models(monkeypatch)
    _stub_upstream_providers(monkeypatch)
    from observa.llm.oci_providers import (
        GoogleProvider,
        ObservaChatOCIGenAI,
        XaiProvider,
        _GenericChatProvider,
    )

    instance = ObservaChatOCIGenAI.model_construct()
    pmap = instance._provider_map
    assert set(pmap.keys()) >= {"cohere", "meta", "xai", "google"}
    assert isinstance(pmap["xai"], XaiProvider)
    assert isinstance(pmap["google"], GoogleProvider)
    # Both subclasses share the GENERIC chat-request implementation.
    assert isinstance(pmap["xai"], _GenericChatProvider)
    assert isinstance(pmap["google"], _GenericChatProvider)


def test_observa_generate_extracts_tool_calls_from_generic_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_generate must read tool calls from choices[0].message.tool_calls (GENERIC)
    and convert FunctionCall.{name, arguments} → LangChain ToolCall(name, args)."""
    _install_fake_oci_models(monkeypatch)
    _stub_upstream_providers(monkeypatch)
    from observa.llm.oci_providers import ObservaChatOCIGenAI, XaiProvider

    chat = ObservaChatOCIGenAI.model_construct()
    # Force the provider so _provider returns XaiProvider without going through
    # the model_id derivation path.
    object.__setattr__(chat, "provider", "xai")

    fake_request = MagicMock(name="prepared_request")
    fake_response = MagicMock(name="oci_response")
    fake_response.request_id = "req_1"
    fake_response.headers = {"content-length": "42"}
    fake_response.data.model_id = "xai.grok-4-fast"
    fake_response.data.model_version = "v1"

    # Plain text content present too — assistant text + tool call mixed.
    text_chunk = MagicMock()
    text_chunk.text = "calling add"
    fake_response.data.chat_response.choices = [
        MagicMock(
            finish_reason="tool_calls",
            message=MagicMock(content=[text_chunk]),
        )
    ]
    fake_response.data.chat_response.time_created = "2026-04-30T00:00:00Z"
    tc = MagicMock(id="call_42")
    tc.name = "add"
    tc.arguments = '{"a": 2, "b": 3}'
    fake_response.data.chat_response.choices[0].message.tool_calls = [tc]

    monkeypatch.setattr(chat, "_prepare_request", lambda *a, **k: fake_request)
    monkeypatch.setattr(chat, "client", MagicMock(chat=MagicMock(return_value=fake_response)))
    monkeypatch.setattr(chat, "is_stream", False)

    result = chat._generate([HumanMessage(content="2 + 3?")])
    msg = result.generations[0].message

    assert msg.content == "calling add"
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["name"] == "add"
    assert msg.tool_calls[0]["args"] == {"a": 2, "b": 3}
    assert msg.tool_calls[0]["id"] == "call_42"


def test_observa_generate_handles_tool_only_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Grok returns ONLY tool calls (empty content list), _generate must not crash."""
    _install_fake_oci_models(monkeypatch)
    _stub_upstream_providers(monkeypatch)
    from observa.llm.oci_providers import ObservaChatOCIGenAI

    chat = ObservaChatOCIGenAI.model_construct()
    object.__setattr__(chat, "provider", "xai")

    fake_response = MagicMock()
    fake_response.request_id = "req_1"
    fake_response.headers = {"content-length": "0"}
    fake_response.data.model_id = "xai.grok-4-fast"
    fake_response.data.model_version = "v1"
    fake_response.data.chat_response.choices = [
        MagicMock(
            finish_reason="tool_calls",
            message=MagicMock(content=[]),  # tool-only response
        )
    ]
    fake_response.data.chat_response.time_created = "2026-04-30T00:00:00Z"
    tc = MagicMock(id="call_x")
    tc.name = "add"
    tc.arguments = '{"a": 1, "b": 1}'
    fake_response.data.chat_response.choices[0].message.tool_calls = [tc]

    monkeypatch.setattr(chat, "_prepare_request", lambda *a, **k: MagicMock())
    monkeypatch.setattr(chat, "client", MagicMock(chat=MagicMock(return_value=fake_response)))
    monkeypatch.setattr(chat, "is_stream", False)

    result = chat._generate([HumanMessage(content="1+1?")])
    msg = result.generations[0].message
    assert msg.content == ""
    assert len(msg.tool_calls) == 1


def test_chat_response_to_text_handles_none_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """When Gemini blocks/aborts a response, ``choices[0].message`` can be None.
    The provider must return empty content rather than crashing on attribute access."""
    _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    provider = XaiProvider()
    fake_response = MagicMock()
    fake_response.data.chat_response.choices = [MagicMock(message=None)]
    assert provider.chat_response_to_text(fake_response) == ""


def test_chat_generation_info_handles_none_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """``chat_generation_info`` must still surface finish_reason when message is None."""
    _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    provider = XaiProvider()
    fake_response = MagicMock()
    fake_response.data.chat_response.choices = [
        MagicMock(message=None, finish_reason="SAFETY")
    ]
    fake_response.data.chat_response.time_created = "2026-04-30T00:00:00Z"
    info = provider.chat_generation_info(fake_response)
    assert info["finish_reason"] == "SAFETY"
    assert "tool_calls" not in info


def test_observa_generate_handles_none_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: a None-message response yields an empty AIMessage, not a crash."""
    _install_fake_oci_models(monkeypatch)
    _stub_upstream_providers(monkeypatch)
    from observa.llm.oci_providers import ObservaChatOCIGenAI

    chat = ObservaChatOCIGenAI.model_construct()
    object.__setattr__(chat, "provider", "google")

    fake_response = MagicMock()
    fake_response.request_id = "req"
    fake_response.headers = {"content-length": "0"}
    fake_response.data.model_id = "google.gemini-2.5-pro"
    fake_response.data.model_version = "v1"
    fake_response.data.chat_response.choices = [
        MagicMock(message=None, finish_reason="SAFETY")
    ]
    fake_response.data.chat_response.time_created = "2026-04-30T00:00:00Z"
    fake_response.data.chat_response.usage = None

    monkeypatch.setattr(chat, "_prepare_request", lambda *a, **k: MagicMock())
    monkeypatch.setattr(chat, "client", MagicMock(chat=MagicMock(return_value=fake_response)))
    monkeypatch.setattr(chat, "is_stream", False)

    msg = chat._generate([HumanMessage(content="?")]).generations[0].message
    assert msg.content == ""
    assert msg.tool_calls == []


def test_observa_generate_synthesizes_id_when_tool_call_id_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini via OCI omits ``id`` on tool calls — propagating ``None`` breaks
    the agent's ToolMessage validator. Synthesize a stable id instead."""
    _install_fake_oci_models(monkeypatch)
    _stub_upstream_providers(monkeypatch)
    from observa.llm.oci_providers import ObservaChatOCIGenAI

    chat = ObservaChatOCIGenAI.model_construct()
    object.__setattr__(chat, "provider", "google")

    fake_response = MagicMock()
    fake_response.request_id = "req"
    fake_response.headers = {"content-length": "0"}
    fake_response.data.model_id = "google.gemini-2.5-flash"
    fake_response.data.model_version = "v1"
    fake_response.data.chat_response.choices = [
        MagicMock(finish_reason="tool_calls", message=MagicMock(content=[]))
    ]
    fake_response.data.chat_response.time_created = "2026-04-30T00:00:00Z"
    tc = MagicMock(id=None)
    tc.name = "add"
    tc.arguments = '{"a": 1, "b": 2}'
    fake_response.data.chat_response.choices[0].message.tool_calls = [tc]
    fake_response.data.chat_response.usage = None

    monkeypatch.setattr(chat, "_prepare_request", lambda *a, **k: MagicMock())
    monkeypatch.setattr(chat, "client", MagicMock(chat=MagicMock(return_value=fake_response)))
    monkeypatch.setattr(chat, "is_stream", False)

    msg = chat._generate([HumanMessage(content="?")]).generations[0].message
    assert msg.tool_calls[0]["id"], "tool_call id must be a non-empty string"
    assert isinstance(msg.tool_calls[0]["id"], str)


def test_observa_generate_uses_generic_path_for_google_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GoogleProvider must take the same GENERIC tool-call extraction path as xAI."""
    _install_fake_oci_models(monkeypatch)
    _stub_upstream_providers(monkeypatch)
    from observa.llm.oci_providers import ObservaChatOCIGenAI

    chat = ObservaChatOCIGenAI.model_construct()
    object.__setattr__(chat, "provider", "google")

    fake_response = MagicMock()
    fake_response.request_id = "req"
    fake_response.headers = {"content-length": "0"}
    fake_response.data.model_id = "google.gemini-2.5-flash"
    fake_response.data.model_version = "v1"
    text_chunk = MagicMock()
    text_chunk.text = "calling add"
    fake_response.data.chat_response.choices = [
        MagicMock(finish_reason="tool_calls", message=MagicMock(content=[text_chunk]))
    ]
    fake_response.data.chat_response.time_created = "2026-04-30T00:00:00Z"
    tc = MagicMock(id="call_g")
    tc.name = "add"
    tc.arguments = '{"a": 7, "b": 5}'
    fake_response.data.chat_response.choices[0].message.tool_calls = [tc]
    fake_response.data.chat_response.usage = None

    monkeypatch.setattr(chat, "_prepare_request", lambda *a, **k: MagicMock())
    monkeypatch.setattr(chat, "client", MagicMock(chat=MagicMock(return_value=fake_response)))
    monkeypatch.setattr(chat, "is_stream", False)

    msg = chat._generate([HumanMessage(content="7+5?")]).generations[0].message
    assert msg.content == "calling add"
    assert msg.tool_calls[0]["name"] == "add"
    assert msg.tool_calls[0]["args"] == {"a": 7, "b": 5}


def test_observa_generate_delegates_for_non_xai_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For Cohere/Meta, _generate must delegate to super()._generate unchanged."""
    _install_fake_oci_models(monkeypatch)
    _stub_upstream_providers(monkeypatch)
    from observa.llm.oci_providers import ObservaChatOCIGenAI

    chat = ObservaChatOCIGenAI.model_construct()
    object.__setattr__(chat, "provider", "cohere")

    sentinel = object()
    called = {}

    def fake_super_generate(self, messages, stop=None, run_manager=None, **kwargs):
        called["yes"] = True
        return sentinel

    from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI as _Upstream
    monkeypatch.setattr(_Upstream, "_generate", fake_super_generate)

    result = chat._generate([HumanMessage(content="hi")])
    assert result is sentinel
    assert called.get("yes")


def test_chat_stream_to_text_handles_missing_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """xAI tool-call deltas lack 'content' — must return '' not KeyError."""
    _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    p = XaiProvider()
    # Only a "message" key with no "content" inside (tool-call delta shape).
    assert p.chat_stream_to_text({"message": {"role": "assistant"}}) == ""
    # Empty content list.
    assert p.chat_stream_to_text({"message": {"content": []}}) == ""
    # Completely empty event.
    assert p.chat_stream_to_text({}) == ""


def test_is_chat_stream_end_uses_finish_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_oci_models(monkeypatch)
    from observa.llm.oci_providers import XaiProvider

    p = XaiProvider()
    assert p.is_chat_stream_end({"finishReason": "stop"}) is True
    assert p.is_chat_stream_end({"message": {"content": [{"text": "hi"}]}}) is False
    # Tool-call delta: has message, no finishReason → still streaming.
    assert p.is_chat_stream_end({"message": {"role": "assistant"}}) is False


async def test_observa_astream_yields_single_chunk_via_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For XaiProvider, _astream delegates to _generate and yields one chunk."""
    _install_fake_oci_models(monkeypatch)
    _stub_upstream_providers(monkeypatch)
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from observa.llm.oci_providers import ObservaChatOCIGenAI

    chat = ObservaChatOCIGenAI.model_construct()
    object.__setattr__(chat, "provider", "xai")

    sentinel_message = AIMessage(content="hello", tool_calls=[])
    fake_result = ChatResult(
        generations=[ChatGeneration(message=sentinel_message, generation_info={"finish_reason": "stop"})],
    )

    def fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
        return fake_result

    monkeypatch.setattr(ObservaChatOCIGenAI, "_generate", fake_generate)

    chunks = []
    async for chunk in chat._astream([HumanMessage(content="hi")]):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].message.content == "hello"
    assert chunks[0].generation_info == {"finish_reason": "stop"}


def test_real_oci_sdk_constructs_match_xai_provider_calls() -> None:
    """Smoke test against real OCI SDK to catch shape mismatches.

    XaiProvider builds OCI request objects in convert_to_oci_tool and
    messages_to_oci_params. If the kwarg shapes diverge from the actual
    SDK schema, init_model_state_from_kwargs raises TypeError. This test
    builds the same shapes XaiProvider builds and asserts they construct
    cleanly. Skipped when oci extra is not installed.
    """
    pytest.importorskip("oci.generative_ai_inference.models")
    from oci.generative_ai_inference import models

    # Mirror convert_to_oci_tool's output shape:
    fd = models.FunctionDefinition(
        name="add", description="Add two ints", parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )
    assert fd.name == "add"

    # Mirror messages_to_oci_params's tool_calls list shape:
    fc = models.FunctionCall(id="call_1", name="add", arguments='{"a": 1, "b": 2}')
    assert fc.id == "call_1"
    assert fc.name == "add"

    # FunctionCall must be assignable to AssistantMessage.tool_calls (list[ToolCall]):
    msg = models.AssistantMessage(
        content=[models.TextContent(text=" ")],
        tool_calls=[fc],
    )
    assert msg.tool_calls == [fc]

    # FunctionDefinition must be assignable to GenericChatRequest.tools:
    req = models.GenericChatRequest(
        messages=[models.UserMessage(content=[models.TextContent(text="x")])],
        tools=[fd],
    )
    assert req.tools == [fd]


def _build_chat_with_response(
    monkeypatch: pytest.MonkeyPatch, fake_response: MagicMock
) -> Any:
    _install_fake_oci_models(monkeypatch)
    _stub_upstream_providers(monkeypatch)
    from observa.llm.oci_providers import ObservaChatOCIGenAI

    chat = ObservaChatOCIGenAI.model_construct()
    object.__setattr__(chat, "provider", "xai")
    monkeypatch.setattr(chat, "_prepare_request", lambda *a, **k: MagicMock())
    monkeypatch.setattr(chat, "client", MagicMock(chat=MagicMock(return_value=fake_response)))
    monkeypatch.setattr(chat, "is_stream", False)
    return chat


def _make_response_with_usage(
    *, prompt: int | None, completion: int | None, total: int | None, cached: int | None = None
) -> MagicMock:
    fake_response = MagicMock()
    fake_response.request_id = "req"
    fake_response.headers = {"content-length": "0"}
    fake_response.data.model_id = "xai.grok-4-fast"
    fake_response.data.model_version = "v1"
    fake_response.data.chat_response.choices = [
        MagicMock(finish_reason="stop", message=MagicMock(content=[]))
    ]
    fake_response.data.chat_response.choices[0].message.tool_calls = []
    fake_response.data.chat_response.time_created = "2026-04-30T00:00:00Z"

    usage = MagicMock(spec=["prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details"])
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = total
    if cached is not None:
        details = MagicMock(spec=["cached_tokens"])
        details.cached_tokens = cached
        usage.prompt_tokens_details = details
    else:
        usage.prompt_tokens_details = None
    fake_response.data.chat_response.usage = usage
    return fake_response


def test_observa_generate_populates_usage_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """OCI ``usage`` (prompt/completion/total) → AIMessage.usage_metadata."""
    fake_response = _make_response_with_usage(prompt=120, completion=40, total=160)
    chat = _build_chat_with_response(monkeypatch, fake_response)
    msg = chat._generate([HumanMessage(content="hi")]).generations[0].message
    assert msg.usage_metadata == {
        "input_tokens": 120,
        "output_tokens": 40,
        "total_tokens": 160,
    }


def test_observa_generate_includes_cache_read_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_response = _make_response_with_usage(prompt=200, completion=50, total=250, cached=80)
    chat = _build_chat_with_response(monkeypatch, fake_response)
    msg = chat._generate([HumanMessage(content="hi")]).generations[0].message
    assert msg.usage_metadata is not None
    assert msg.usage_metadata.get("input_token_details") == {"cache_read": 80}


def test_observa_generate_omits_usage_when_response_has_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_response = _make_response_with_usage(prompt=0, completion=0, total=0)
    fake_response.data.chat_response.usage = None
    chat = _build_chat_with_response(monkeypatch, fake_response)
    msg = chat._generate([HumanMessage(content="hi")]).generations[0].message
    assert msg.usage_metadata is None


def test_observa_generate_falls_back_to_sum_when_total_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_response = _make_response_with_usage(prompt=10, completion=7, total=None)
    chat = _build_chat_with_response(monkeypatch, fake_response)
    msg = chat._generate([HumanMessage(content="hi")]).generations[0].message
    assert msg.usage_metadata == {
        "input_tokens": 10,
        "output_tokens": 7,
        "total_tokens": 17,
    }
