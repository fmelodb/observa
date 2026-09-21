"""Unit tests for eager LLM config validation."""

from unittest.mock import MagicMock

import pytest

from observa.llm.oci_capabilities import ConfigError


def _settings(
    *,
    llm_model="gpt-5.4-mini",
    agent_models=None,
    consolidator="",
    cove="",
    master_chat="",
    hypotheses="",
):
    return MagicMock(
        llm_model=llm_model,
        llm_agent_models=agent_models or {},
        llm_consolidator_model=consolidator,
        llm_cove_model=cove,
        llm_master_chat_model=master_chat,
        llm_hypotheses_model=hypotheses,
    )


def test_validate_passes_for_pure_anthropic_openai_config():
    from observa.llm.validate_config import validate_llm_config
    s = _settings()
    validate_llm_config(s)  # must not raise


def test_validate_passes_for_compatible_oci_assignments():
    from observa.llm.validate_config import validate_llm_config
    s = _settings(
        agent_models={"infra": "oci/cohere.command-r-08-2024"},
        consolidator="oci/cohere.command-r-plus-08-2024",
        cove="oci/cohere.command-a-03-2025",
    )
    validate_llm_config(s)


def test_validate_aggregates_multiple_errors():
    """Two bad slots -> single ConfigError naming both."""
    from observa.llm.validate_config import validate_llm_config
    s = _settings(
        agent_models={"infra": "oci/meta.llama-unknown"},
        cove="oci/also.unknown",
    )
    with pytest.raises(ConfigError) as exc_info:
        validate_llm_config(s)
    msg = str(exc_info.value)
    assert "agent_models.infra" in msg
    assert "cove_model" in msg
    assert "oci/meta.llama-unknown" in msg
    assert "oci/also.unknown" in msg


def test_validate_rejects_non_oci_garbage_silently():
    """Non-oci model strings are NOT validated by the OCI matrix - that is the
    job of make_llm at call time. Strings without oci/ prefix pass."""
    from observa.llm.validate_config import validate_llm_config
    s = _settings(agent_models={"infra": "some-random-openai-model"})
    validate_llm_config(s)


def test_validate_uses_correct_slot_per_field():
    """consolidator slot only requires structured output, so a model without
    tools (if matrix had one) would pass for consolidator but fail for agent."""
    from observa.llm import oci_capabilities as caps_mod
    from observa.llm.oci_capabilities import ModelCaps
    from observa.llm.validate_config import validate_llm_config
    caps_mod.OCI_CAPS["fake.no-tools-3"] = ModelCaps(tools=False, structured=True)
    try:
        s_ok = _settings(consolidator="oci/fake.no-tools-3")
        validate_llm_config(s_ok)
        s_bad = _settings(agent_models={"infra": "oci/fake.no-tools-3"})
        with pytest.raises(ConfigError):
            validate_llm_config(s_bad)
    finally:
        del caps_mod.OCI_CAPS["fake.no-tools-3"]
