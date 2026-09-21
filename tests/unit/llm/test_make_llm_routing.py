"""Unit tests for make_llm routing across the three providers."""

from unittest.mock import MagicMock, patch

import pytest


def test_make_llm_routes_claude_to_anthropic():
    fake_chat = MagicMock(name="ChatAnthropic_inst")
    with patch("langchain_anthropic.ChatAnthropic", return_value=fake_chat) as cls:
        from observa.llm.rate_limited import make_llm
        wrapped = make_llm("claude-sonnet-4-6", slot="agent")
    cls.assert_called_once_with(model="claude-sonnet-4-6")
    assert wrapped._inner is fake_chat


def test_make_llm_routes_gpt_to_openai():
    fake_chat = MagicMock(name="ChatOpenAI_inst")
    with patch("langchain_openai.ChatOpenAI", return_value=fake_chat) as cls:
        from observa.llm.rate_limited import make_llm
        wrapped = make_llm("gpt-5.4-mini", slot="agent")
    cls.assert_called_once_with(model="gpt-5.4-mini")
    assert wrapped._inner is fake_chat


def test_make_llm_routes_oci_prefix_to_factory():
    fake_chat = MagicMock(name="ChatOCIGenAI_inst")
    with patch(
        "observa.llm.rate_limited.validate_oci_model_for_slot", return_value=None
    ) as validate_mock, patch(
        "observa.llm.rate_limited.build_oci_chat", return_value=fake_chat
    ) as factory_mock:
        from observa.llm.rate_limited import make_llm
        wrapped = make_llm("oci/xai.grok-4-fast", slot="agent")
    validate_mock.assert_called_once_with("xai.grok-4-fast", "agent")
    factory_mock.assert_called_once_with("xai.grok-4-fast")
    assert wrapped._inner is fake_chat


def test_make_llm_oci_unknown_model_raises_config_error():
    from observa.llm.oci_capabilities import ConfigError
    from observa.llm.rate_limited import make_llm
    with pytest.raises(ConfigError):
        make_llm("oci/totally.invented-model", slot="agent")


def test_make_llm_oci_capability_mismatch_raises_config_error():
    from observa.llm import oci_capabilities as caps_mod
    from observa.llm.oci_capabilities import ConfigError, ModelCaps
    from observa.llm.rate_limited import make_llm
    caps_mod.OCI_CAPS["fake.no-tools"] = ModelCaps(tools=False, structured=True)
    try:
        with pytest.raises(ConfigError) as exc:
            make_llm("oci/fake.no-tools", slot="agent")
        assert "tool" in str(exc.value).lower()
    finally:
        del caps_mod.OCI_CAPS["fake.no-tools"]


def test_make_llm_default_slot_is_agent():
    """A model lacking tools must be rejected when slot defaults to 'agent'."""
    from observa.llm import oci_capabilities as caps_mod
    from observa.llm.oci_capabilities import ConfigError, ModelCaps
    from observa.llm.rate_limited import make_llm
    caps_mod.OCI_CAPS["fake.no-tools-2"] = ModelCaps(tools=False, structured=True)
    try:
        with pytest.raises(ConfigError):
            make_llm("oci/fake.no-tools-2")
    finally:
        del caps_mod.OCI_CAPS["fake.no-tools-2"]
