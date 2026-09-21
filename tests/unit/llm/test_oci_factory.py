"""Unit tests for build_oci_chat."""

from unittest.mock import MagicMock, patch

import pytest


def _settings_stub(**overrides):
    """Build a Settings-like object good enough for build_oci_chat."""
    base = dict(
        oci_enabled=True,
        oci_default_model="xai.grok-4-fast",
        oci_compartment_id="ocid1.compartment.oc1..aaa",
        oci_service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        oci_auth_profile="DEFAULT",
        oci_auth_type="API_KEY",
    )
    base.update(overrides)
    return MagicMock(**base)


def test_build_oci_chat_passes_config_to_chatocigenai():
    fake_settings = _settings_stub()
    fake_chat = MagicMock(name="ChatOCIGenAI_instance")

    with patch("observa.llm.oci_factory.get_settings", return_value=fake_settings), \
         patch("observa.llm.oci_factory._import_chat_oci_genai") as import_mock:
        chat_cls = MagicMock(return_value=fake_chat)
        import_mock.return_value = chat_cls

        from observa.llm.oci_factory import build_oci_chat
        result = build_oci_chat("cohere.command-r-plus-08-2024")

    assert result is fake_chat
    chat_cls.assert_called_once()
    kwargs = chat_cls.call_args.kwargs
    assert kwargs["model_id"] == "cohere.command-r-plus-08-2024"
    assert kwargs["compartment_id"] == "ocid1.compartment.oc1..aaa"
    assert kwargs["service_endpoint"].startswith("https://inference.generativeai.")
    assert kwargs["auth_profile"] == "DEFAULT"
    assert kwargs["auth_type"] == "API_KEY"


def test_build_oci_chat_raises_when_provider_disabled():
    fake_settings = _settings_stub(oci_enabled=False, oci_compartment_id="")

    with patch("observa.llm.oci_factory.get_settings", return_value=fake_settings):
        from observa.llm.oci_factory import build_oci_chat
        with pytest.raises(Exception) as exc_info:
            build_oci_chat("xai.grok-4-fast")
    msg = str(exc_info.value).lower()
    assert "oci" in msg
    assert "providers.oci" in msg or "compartment" in msg or "configured" in msg


def test_build_oci_chat_raises_when_extras_missing():
    fake_settings = _settings_stub()

    def _import_raises():
        raise ImportError("No module named 'langchain_community'")

    with patch("observa.llm.oci_factory.get_settings", return_value=fake_settings), \
         patch("observa.llm.oci_factory._import_chat_oci_genai", side_effect=_import_raises):
        from observa.llm.oci_factory import build_oci_chat
        with pytest.raises(Exception) as exc_info:
            build_oci_chat("xai.grok-4-fast")
    msg = str(exc_info.value).lower()
    assert "uv sync --extra oci" in msg or "extras" in msg


def test_import_chat_oci_genai_returns_observa_subclass():
    from observa.llm.oci_factory import _import_chat_oci_genai
    from observa.llm.oci_providers import ObservaChatOCIGenAI

    assert _import_chat_oci_genai() is ObservaChatOCIGenAI
