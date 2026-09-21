"""Unit tests for the OCI capability matrix and slot validator."""

import pytest

from observa.llm.oci_capabilities import (
    OCI_CAPS,
    SLOT_REQUIREMENTS,
    ModelCaps,
    ConfigError,
    validate_oci_model_for_slot,
    compatible_models_for_slot,
)


def test_matrix_contains_verified_models():
    """Verified models include cohere, xai, and google (via observa-shipped providers)."""
    assert "cohere.command-r-plus-08-2024" in OCI_CAPS
    assert "cohere.command-r-08-2024" in OCI_CAPS
    assert "cohere.command-a-03-2025" in OCI_CAPS
    assert "xai.grok-4-fast-non-reasoning" in OCI_CAPS
    assert "xai.grok-4-fast-reasoning" in OCI_CAPS
    assert "google.gemini-2.5-flash" in OCI_CAPS
    assert "google.gemini-2.5-pro" in OCI_CAPS
    for caps in OCI_CAPS.values():
        assert isinstance(caps, ModelCaps)


def test_slot_requirements_present():
    assert SLOT_REQUIREMENTS["agent"] == ModelCaps(tools=True, structured=True)
    assert SLOT_REQUIREMENTS["master_chat"] == ModelCaps(tools=True, structured=False)
    assert SLOT_REQUIREMENTS["consolidator"] == ModelCaps(tools=False, structured=True)
    assert SLOT_REQUIREMENTS["cove"] == ModelCaps(tools=False, structured=True)
    assert SLOT_REQUIREMENTS["hypotheses"] == ModelCaps(tools=False, structured=True)


def test_validate_known_compatible_returns_none():
    assert validate_oci_model_for_slot("cohere.command-r-plus-08-2024", "agent") is None
    assert validate_oci_model_for_slot("cohere.command-r-plus-08-2024", "consolidator") is None


def test_validate_unknown_model_raises():
    with pytest.raises(ConfigError) as exc_info:
        validate_oci_model_for_slot("meta.llama-3.3-70b-instruct", "agent")
    msg = str(exc_info.value)
    assert "not in OCI_CAPS" in msg
    assert "meta.llama-3.3-70b-instruct" in msg


def test_validate_unknown_slot_raises():
    with pytest.raises(ConfigError):
        validate_oci_model_for_slot("cohere.command-r-plus-08-2024", "nonexistent_slot")


def test_compatible_models_for_agent_slot():
    models = compatible_models_for_slot("agent")
    assert "cohere.command-r-plus-08-2024" in models
    assert "cohere.command-r-08-2024" in models
    assert "cohere.command-a-03-2025" in models


def test_capability_mismatch_message_lists_alternatives():
    """If we add a hypothetical structured-only model, agent slot should reject it
    with a helpful list."""
    from observa.llm import oci_capabilities as caps_mod
    caps_mod.OCI_CAPS["fake.structured-only"] = ModelCaps(tools=False, structured=True)
    try:
        with pytest.raises(ConfigError) as exc_info:
            validate_oci_model_for_slot("fake.structured-only", "agent")
        msg = str(exc_info.value)
        assert "tool calling" in msg.lower() or "tools" in msg.lower()
        assert "cohere.command-r-plus-08-2024" in msg
    finally:
        del caps_mod.OCI_CAPS["fake.structured-only"]


def test_xai_grok_4_fast_non_reasoning_is_in_oci_caps() -> None:
    from observa.llm.oci_capabilities import OCI_CAPS

    assert "xai.grok-4-fast-non-reasoning" in OCI_CAPS
    caps = OCI_CAPS["xai.grok-4-fast-non-reasoning"]
    assert caps.tools is True
    assert caps.structured is True


def test_xai_grok_4_fast_reasoning_is_in_oci_caps() -> None:
    from observa.llm.oci_capabilities import OCI_CAPS

    assert "xai.grok-4-fast-reasoning" in OCI_CAPS
    caps = OCI_CAPS["xai.grok-4-fast-reasoning"]
    assert caps.tools is True
    assert caps.structured is True


def test_xai_grok_4_fast_non_reasoning_validates_for_agent_slot() -> None:
    from observa.llm.oci_capabilities import validate_oci_model_for_slot

    validate_oci_model_for_slot("xai.grok-4-fast-non-reasoning", "agent")  # must not raise
